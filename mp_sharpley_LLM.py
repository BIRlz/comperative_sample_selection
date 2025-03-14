import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import json
import numpy as np
import argparse
from prompt import *
import re
import os

from collections import defaultdict

def _obtain_messages(system_content, prompt):
    messages = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    return messages

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", help="cuda_index", type=int, default=0)
    parser.add_argument("--ds", help="dataset name", type=str, default='huatuo')
    parser.add_argument("--round", help="round", type=int, default=1)
    parser.add_argument("--a_ws", help="window size of A pool", type=int, default=10)
    parser.add_argument("--b_ws", help="window size of B pool", type=int, default=10)
    parser.add_argument("--n", help="number of samples", type=int, default=10)
    
    args = parser.parse_args()

    device = f'cuda:{args.split}'
    if args.ds == 'huatuo':
        ds = load_dataset("FreedomIntelligence/HuatuoGPT-sft-data-v1")
            
    model_path = "Qwen/Qwen2-72B-Instruct"

    save_root = model_path.split('/')[-1]+f"_{args.a_ws}_{args.b_ws}_{args.round}"
    os.makedirs(save_root, exist_ok=True)

    model = LLM(model_path, tensor_parallel_size=8, dtype=torch.bfloat16, device="cuda", max_model_len=4096*4, gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    sampling_params = SamplingParams(n=1, temperature=0.0, top_p=1, max_tokens=1024)

    system_content = system_prompt()
    shapley_prompt = prompt_0
    
    ori_input_ds = ds["train"]
    #split the input_ds into 4 parts
    parts_len = len(ori_input_ds) // 4
    #obtain the input_ds according to args.split
    input_ds = ori_input_ds[args.split * parts_len: (args.split + 1) * parts_len]
    length_of_input = len(input_ds['data'])

    seed_subset_pool = np.random.choice(length_of_input, args.a_ws, replace=False)
    rest_subset_pool = np.setdiff1d(np.arange(length_of_input), seed_subset_pool)
    ori_input_data = ds['train'][args.split * parts_len: (args.split + 1) * parts_len]

    torch.save({'ori_input_data': ori_input_data}, os.path.join(save_root, f"ori_input_data_pool_{args.split}.pt"))

    #start MC
    i = 0
    #56510 / 2 = 28255
    #56510 / 3 = 18838
    while len(seed_subset_pool) <= 120000 // 4:
        # sample a subset of the data
        import pdb;pdb.set_trace()
        seed_subset_index = np.random.choice(seed_subset_pool, args.a_ws, replace=False)
        seed_subset = ['\t'.join(input_ds['data'][x]).replace('\n', '') for x in seed_subset_index]
        seed_subset = "\n".join([f'{i + 1}. {s}' for i, s in enumerate(seed_subset)])

        candidate_subset_index = np.random.choice(rest_subset_pool, args.b_ws, replace=False)
        candidate_subset = ['\t'.join(input_ds['data'][x]).replace('\n', '') for x in candidate_subset_index]
        #start with [[A]], [[B]], ..., 
        candidate_subset = "\n".join([f'[[{chr(65 + i)}]]: {s}' for i, s in enumerate(candidate_subset)])

        query = _obtain_messages(system_content, shapley_prompt(seed_subset, candidate_subset))
        query = tokenizer.apply_chat_template(query, tokenize=False, add_generation_prompt=True)
        
        # generate the response
        response = model.generate(query, sampling_params)
        resp = response[0].outputs[0].text
        try:
            
            final_decision = re.findall(r'\[\[(.*?)\]\]', resp)[0]
            _i = ord(final_decision) - 65

            selected_index = candidate_subset_index[_i]
            #update seed_subset_pool and rest_subset_pool
            seed_subset_pool = np.append(seed_subset_pool, selected_index)
            seed_subset_pool = list(set(seed_subset_pool))
            rest_subset_pool = np.setdiff1d(rest_subset_pool, selected_index)
            print(f"Number of Selected: {len(seed_subset_pool)}")
            
        except:
            continue
        
        #all = 226042
        # 10% = 22604

        if len(seed_subset_pool) == 22604 // 4:
            np.save(os.path.join(save_root, f"10%_curr_selected_index_{args.split}.npy"), seed_subset_pool)
            torch.save({'selected_index': seed_subset_pool}, os.path.join(save_root, f"10%_curr_selected_index_{args.split}.pt"))

        # 20% = 45208
        if len(seed_subset_pool) == 45208 // 4:
            np.save(os.path.join(save_root, f"20%_curr_selected_index_{args.split}.npy"), seed_subset_pool)
            torch.save({'selected_index': seed_subset_pool}, os.path.join(save_root, f"20%_curr_selected_index_{args.split}.pt"))

        # 30% = 67812
        if len(seed_subset_pool) == 67812 // 4:
            np.save(os.path.join(save_root, f"30%_curr_selected_index_{args.split}.npy"), seed_subset_pool)
            torch.save({'selected_index': seed_subset_pool}, os.path.join(save_root, f"30%_curr_selected_index_{args.split}.pt"))

        # 40% = 90416
        if len(seed_subset_pool) == 90416 // 4:
            np.save(os.path.join(save_root, f"40%_curr_selected_index_{args.split}.npy"), seed_subset_pool)
            torch.save({'selected_index': seed_subset_pool}, os.path.join(save_root, f"40%_curr_selected_index_{args.split}.pt"))

        # 50% = 113020
        if len(seed_subset_pool) == 113020 // 4:
            np.save(os.path.join(save_root, f"50%_curr_selected_index_{args.split}.npy"), seed_subset_pool)
            torch.save({'selected_index': seed_subset_pool}, os.path.join(save_root, f"50%_curr_selected_index_{args.split}.pt"))
            break