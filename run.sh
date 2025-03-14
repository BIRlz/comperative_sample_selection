#!/bin/bash
# 计费账号
# 任务名称
#SBATCH --job-name 72B_select
# 申请的分区
#SBATCH --partition p-A100
# 申请的节点数量
#SBATCH --nodes=4
# 每个节点的任务数，固定为1，我们只需要在每个节点启动一个任务
#SBATCH --ntasks-per-node=1
# 每一个节点申请的显卡数量
#SBATCH --gpus-per-node=8
# 每个 GPU 搭配申请的 CPU 数量
#SBATCH --cpus-per-gpu=32
# 使用预留的资源
#SBATCH --reservation root_140
# 输出日志文件，使用 %j（作业 ID）和 %N（节点名称）来区分不同节点的日志
#SBATCH --output all_72B_test_alpaca_9232_split_%j_%N_aws10_bws10_round2_percent.log.txt

module load cuda12.1/toolkit/12.1.1
module load gcc-10.3.0/10.3.0
conda activate cvllm

export HDF5_USE_FILE_LOCKING=FALSE

# 获取节点列表
NODELIST=$(scontrol show hostnames $SLURM_NODELIST)
NODEARRAY=($NODELIST)
a_ws=10
b_ws=10
round=2

# 循环遍历每个节点，分配对应的 split
for (( i=0; i<${#NODEARRAY[@]}; i++ )); do
    NODE=${NODEARRAY[$i]}
    srun --nodes=1 --ntasks=1 --nodelist=$NODE \
        python mp_sharpley_LLM.py --split $i --ds alpaca --round ${round} --a_ws ${a_ws} --b_ws ${b_ws} &
done

# 等待所有任务完成
wait

# python merge.py --root_path ./Qwen2.5-72B-Instruct_${a_ws}_${b_ws}_${round}