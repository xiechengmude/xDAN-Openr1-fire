#!/bin/bash

# 设置CUDA内存相关环境变量
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 创建日志目录
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR="${ROOT_DIR}/logs/grpo"
if [ ! -d "${LOG_DIR}" ]; then
    mkdir -p ${LOG_DIR}
fi

# 获取时间戳
TIMESTAMP=$(TZ=Asia/Shanghai date +'%Y%m%d-%H-%M-%S')
LOG_FILE="${LOG_DIR}/GRPO_Training_DeepSeek-R1-Distill-Qwen-32B-${TIMESTAMP}.log"

# 启动训练
{

OUTPUT_DIR="/data/vayu/train/trained/xDAN-L2-Testing-Distill-32B"
if [ ! -d "${OUTPUT_DIR}" ]; then
    mkdir -p ${OUTPUT_DIR}
fi

accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path /data/vayu/train/models/DeepSeek-R1-Distill-Qwen-32B \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --max_completion_length 512 \
    --gradient_accumulation_steps 16 \
    --logging_steps 10 \
    --bf16 \
    --ds3_gather_for_generation false \
    --save_strategy steps \
    --save_steps 0.05 \
    --num_train_epochs 5 \
    --save_total_limit 10 \
    --gradient_checkpointing
} 2>&1 | tee "${LOG_FILE}" &

echo "Training started. Log file: ${LOG_FILE}"
