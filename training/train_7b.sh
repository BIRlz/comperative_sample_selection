#!/bin/bash
# 计费账号
# 任务名称
#SBATCH --job-name Train_7b
# 申请的分区
#SBATCH --partition p-A100
# 申请的节点数量
#SBATCH --nodes=1                  
# 每个节点的任务数，固定为 1，我们只需要在每个节点启动一个任务
#SBATCH --ntasks-per-node=1
# 每个节点申请的显卡数量
#SBATCH --gpus-per-node=8
# 每个 GPU 搭配申请的 CPU 数量
#SBATCH --cpus-per-gpu=32
# 使用预留的资源
#SBATCH --reservation root_140
# 批处理脚本自身的日志文件
#SBATCH --output full_SFT_selected_alpaca_72B_5_percent.training.txt

module load cuda12.2/toolkit/12.2.2
module load gcc/11.2.0
conda activate cvllm

DATA_DIR=../data/qwen2.5_72B_5_percent_data_value.json
MODEL_DIR=../cache/Llama-2-7b-hf/
OUTPUT_DIR=./models/7b-Qwen2.5-72B-5percent/

torchrun --nproc_per_node=8 --master_port=54321 train_alpaca.py \
    --model_name_or_path ${MODEL_DIR} \
    --data_path ${DATA_DIR} \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --save_only_model True \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --report_to "none" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \