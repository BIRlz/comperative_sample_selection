
import torch
from datasets import load_dataset
import json
import numpy as np
import argparse
from openai import OpenAI
import numpy as np
from prompt import *
import os
import re
import time

api_key = "Your_API_Key"
api_base = "Your_API_Base"

client = OpenAI(api_key=api_key, base_url=api_base) 

def get_openai_response(query):
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages=query,
        temperature=0.0,
        top_p=1.0,
    )
    return response

def query_GPT4(query):
    try:
        response = get_openai_response(query)
        answer = response.choices[0].message.content
    except:
        time.sleep(3)
        try:
            response = get_openai_response(query)
            answer = response.choices[0].message.content
        except:
            answer = 'failed'
    
    return answer

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
    args = parser.parse_args()

    save_root = "chatgpt-turbo"
    os.makedirs(save_root, exist_ok=True)

    device = f'cuda:{args.split}'
    ds = load_dataset("tatsu-lab/alpaca")
    # ds = load_dataset("databricks/databricks-dolly-15k")
    # ds = load_dataset("OpenAssistant/oasst1")
    # ds = load_dataset("SirNeural/flan_v2")
       
    system_content = system_prompt()
    shapley_prompt = prompt_0
    
    ori_input_ds = ds["train"]
    #split the input_ds into 4 parts
    parts_len = len(ori_input_ds) // 4
    #obtain the input_ds according to args.split
    input_ds = ori_input_ds[args.split * parts_len: (args.split + 1) * parts_len]
    length_of_input = len(input_ds['instruction'])

    seed_subset_pool = np.random.choice(length_of_input, 100, replace=False)
    rest_subset_pool = np.setdiff1d(np.arange(length_of_input), seed_subset_pool)
    ori_input_data = ds['train'][args.split * parts_len: (args.split + 1) * parts_len]

    #start MC
    i = 0

    while len(seed_subset_pool) <= 20000 // 4:
        # sample a subset of the data
        seed_subset_index = np.random.choice(seed_subset_pool, 20, replace=False)
        seed_subset = [input_ds['text'][x].replace('\n', '') for x in seed_subset_index]
        seed_subset = "\n".join([f'{i + 1}. {s}' for i, s in enumerate(seed_subset)])

        candidate_subset_index = np.random.choice(rest_subset_pool, 10, replace=False)
        candidate_subset = [input_ds['text'][x].replace('\n', '') for x in candidate_subset_index]
        #start with [[A]], [[B]], ..., 
        candidate_subset = "\n".join([f'[[{chr(65 + i)}]]: {s}' for i, s in enumerate(candidate_subset)])

        query = _obtain_messages(system_content, shapley_prompt(seed_subset, candidate_subset))
        # generate the response
        resp = query_GPT4(query)
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
            np.save(os.path.join(save_root, f"_curr_selected_index_{args.split}.npy"), seed_subset_pool)
            torch.save({'selected_index': seed_subset_pool, 'ori_input_data': ori_input_data}, os.path.join(save_root, f"_curr_selected_index_{args.split}.pt"))
            continue
            
        if len(seed_subset_pool) == 9232 // 4:
            np.save(os.path.join(save_root, f"9232_curr_selected_index_{args.split}.npy"), seed_subset_pool)
            torch.save({'selected_index': seed_subset_pool, 'ori_input_data': ori_input_data}, os.path.join(save_root, f"9232_curr_selected_index_{args.split}.pt"))
    
    #save the selected index into a numpy file
    np.save(os.path.join(save_root, f"selected_index_{args.split}.npy"), seed_subset_pool)
    torch.save({'selected_index': seed_subset_pool, 'ori_input_data': ori_input_data}, os.path.join(save_root, f"selected_index_{args.split}.pt"))
    
    