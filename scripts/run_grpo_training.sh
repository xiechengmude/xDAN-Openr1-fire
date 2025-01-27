#!/bin/bash

# 创建日志目录
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR="${ROOT_DIR}/logs/grpo"
mkdir -p ${LOG_DIR}

# 获取时间戳
TIMESTAMP=$(TZ=Asia/Shanghai date +'%Y%m%d-%H-%M-%S')
LOG_FILE="${LOG_DIR}/GRPO_Training_DeepSeek-R1-Distill-Qwen-7B-${TIMESTAMP}.log"

# 启动训练
{
accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
    --model_name_or_path /data/vayu/train/models/DeepSeek-R1-Distill-Qwen-7B \
    --dataset_name AI-MO/NuminaMath-TIR \
    --output_dir xDAN-Testing-R1-Distill-Qwen-7B-GRPO \
    --max_prompt_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --bf16 \
    --reward_funcs accuracy format \
    --save_strategy steps \
    --save_steps 500 \
    --num_train_epochs 5 \
    --save_total_limit 10 
} 2>&1 | tee "${LOG_FILE}" &

echo "Training started. Log file: ${LOG_FILE}"
