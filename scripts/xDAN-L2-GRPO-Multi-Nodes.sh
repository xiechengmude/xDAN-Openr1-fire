#!/bin/bash

# 设置项目根目录
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

# 设置环境变量
GPUS=${GPUS:-8}
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
export MASTER_PORT=34228
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# 创建日志目录
LOG_DIR="${ROOT_DIR}/logs/grpo"
mkdir -p "${LOG_DIR}"

# 获取当前时间戳
TIMESTAMP=$(TZ=Asia/Shanghai date +'%Y%m%d-%H-%M-%S')
LOG_FILE="${LOG_DIR}/GRPO_Training_xDAN-Testing-R1-Distill-Qwen-32-${TIMESTAMP}.log"

# 启动训练
{
    torchrun \
        --nnodes=2 \
        --node_rank=0 \
        --master_addr=10.110.10.3 \
        --nproc_per_node=${GPUS} \
        --master_port=${MASTER_PORT} \
        src/open_r1/grpo.py \
        --model_name_or_path /data/vayu/train/models/DeepSeek-R1-Distill-Qwen-32B \
        --dataset_name AI-MO/NuminaMath-TIR \
        --output_dir xDAN-Testing-R1-Distill-Qwen-32-GRPO \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --logging_steps 10 \
        --bf16 \
        --reward_funcs accuracy format \
        --save_strategy steps \
        --save_steps 0.1 \
        --num_train_epochs 5 \
        --save_total_limit 10
} > ${LOG_FILE} 2>&1 &

echo "Training started. Log file: tail -f  ${LOG_FILE}"
