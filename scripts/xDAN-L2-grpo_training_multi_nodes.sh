#!/bin/bash

set -x
GPUS=${GPUS:-8}
NODE_RANK=${NODE_RANK:-0}
NNODES=${NNODES:-3}
MASTER_ADDR=${MASTER_ADDR:-"10.110.10.3"}

# 设置项目根目录
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
export MASTER_PORT=34228
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# 检查端口占用
check_and_kill_port() {
    local port=34228
    local pid=$(lsof -ti:$port)
    if [ ! -z "$pid" ]; then
        echo "Port $port is in use by process $pid"
        echo "Will kill the process in 10 seconds..."
        for i in {10..1}; do
            echo -ne "\rCountdown: $i seconds..."
            sleep 1
        done
        echo -e "\nKilling process $pid..."
        kill -9 $pid
        sleep 2
    fi
}

# 检查已有的训练进程
check_and_kill_process() {
    local process_name="torchrun"
    local pids=$(ps aux | grep "$process_name" | grep "grpo.py" | grep -v grep | awk '{print $2}')
    if [ ! -z "$pids" ]; then
        echo "Found existing training processes: $pids"
        echo "Will kill the processes in 10 seconds..."
        for i in {10..1}; do
            echo -ne "\rCountdown: $i seconds..."
            sleep 1
        done
        echo -e "\nKilling processes $pids..."
        echo $pids | xargs kill -9
        sleep 2
    fi
}

# 执行检查
check_and_kill_port
check_and_kill_process

# 创建日志目录
LOG_DIR="${ROOT_DIR}/logs/grpo"
mkdir -p ${LOG_DIR}

# 获取当前时间戳
TIMESTAMP=$(TZ=Asia/Shanghai date +'%Y%m%d-%H-%M-%S')
LOG_FILE="${LOG_DIR}/GRPO_Training_xDAN-Testing-R1-Distill-Qwen-32-${TIMESTAMP}.log"

# 启动训练
{
    torchrun \
        --nnodes=${NNODES} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --nproc_per_node=${GPUS} \
        --master_port=${MASTER_PORT} \
        src/open_r1/grpo.py \
        --model_name_or_path /data/vayu/train/models/DeepSeek-R1-Distill-Qwen-32B \
        --dataset_name AI-MO/NuminaMath-TIR \
        --output_dir xDAN-Testing-R1-Distill-Qwen-32-GRPO \
        --max_prompt_length 256 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --logging_steps 10 \
        --bf16 \
        --reward_funcs accuracy format \
        --save_strategy steps \
        --save_steps 0.1 \
        --num_train_epochs 5 \
        --save_total_limit 10 \
        --gradient_checkpointing true \
        --torch_compile false \
        --trust_remote_code true \
        --ddp_find_unused_parameters false \
        --dataloader_num_workers 4 \
        --max_grad_norm 1.0
} 2>&1 | tee "${LOG_FILE}" &

echo "Training started. Log file: ${LOG_FILE}"
echo "Node rank: ${NODE_RANK}, Total nodes: ${NNODES}, GPUs per node: ${GPUS}"
