#!/bin/bash

# Import port and other settings from config.py
CONFIG_PORT=$(python -c "import sys; sys.path.append('NOVER'); from config import VLLM_PORT; print(VLLM_PORT)")
CONFIG_MODEL=$(python -c "import sys; sys.path.append('NOVER'); from config import MODEL_NAME_VLLM; print(MODEL_NAME_VLLM)")
CONFIG_GPU_MEM_UTIL=$(python -c "import sys; sys.path.append('NOVER'); from config import VLLM_GPU_MEMORY_UTILIZATION; print(VLLM_GPU_MEMORY_UTILIZATION)")

MODEL_NAME="${CONFIG_MODEL}"
GPU_MEM_UTIL="${CONFIG_GPU_MEM_UTIL}"
MAX_MODEL_LEN=8192
USED_PORT="${CONFIG_PORT}"
TENSOR_PARALLEL_SIZE=1
VLLM_GPUS="1"

echo "Starting vLLM server for model: $MODEL_NAME"
echo "Using GPU memory utilization: $GPU_MEM_UTIL"
echo "Using port: $USED_PORT"
echo "Using GPUs $VLLM_GPUS for vLLM server (tensor parallel size: $TENSOR_PARALLEL_SIZE)"

export CUDA_VISIBLE_DEVICES=$VLLM_GPUS
export VLLM_ATTENTION_BACKEND=triton
export MASTER_PORT=29505 
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

echo "Using model path: $MODEL_NAME"
NCCL_DEBUG=INFO trl vllm-serve --model "$MODEL_NAME" --tensor_parallel_size $TENSOR_PARALLEL_SIZE --gpu_memory_utilization "$GPU_MEM_UTIL" --max_model_len "$MAX_MODEL_LEN" --port "$USED_PORT"