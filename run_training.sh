#!/bin/bash

# Import settings from config.py
CONFIG_VLLM_PORT=$(python -c "import sys; sys.path.append('NOVER'); from config import VLLM_PORT; print(VLLM_PORT)")
CONFIG_VLLM_HOST=$(python -c "import sys; sys.path.append('NOVER'); from config import VLLM_HOST; print(VLLM_HOST)")
DEFAULT_INTERMEDIATE_TAG=$(python -c "import sys; sys.path.append('NOVER'); from config import INTERMEDIATE_TAG; print(INTERMEDIATE_TAG)")
DEFAULT_FINAL_TAG=$(python -c "import sys; sys.path.append('NOVER'); from config import FINAL_TAG; print(FINAL_TAG)")

# Process command line arguments
INTERMEDIATE_TAG="${DEFAULT_INTERMEDIATE_TAG}"
FINAL_TAG="${DEFAULT_FINAL_TAG}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --intermediate-tag)
      INTERMEDIATE_TAG="$2"
      shift 2
      ;;
    --final-tag)
      FINAL_TAG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Available options: --intermediate-tag <tag>, --final-tag <tag>"
      exit 1
      ;;
  esac
done

TRAINING_GPU_IDS="0"
TRAINING_GPUS=1
MAIN_PROCESS_PORT=28890
USED_VLLM_PORT="${CONFIG_VLLM_PORT}"
CONFIG_VLLM_HOST="${CONFIG_VLLM_HOST}"

export CUDA_VISIBLE_DEVICES="$TRAINING_GPU_IDS"
echo "Using GPUs $TRAINING_GPU_IDS for training"

echo "Note: vLLM server should be started separately using run_vllm_server.sh"
echo "Training will connect to vLLM server at ${CONFIG_VLLM_HOST}:${USED_VLLM_PORT}"
echo "Using intermediate tag: <${INTERMEDIATE_TAG}>"
echo "Using final tag: <${FINAL_TAG}>"

export NCCL_P2P_DISABLE=1

NCCL_DEBUG=INFO accelerate launch \
    --num_processes=$TRAINING_GPUS \
    --main_process_port $MAIN_PROCESS_PORT \
    main.py --intermediate-tag "${INTERMEDIATE_TAG}" --final-tag "${FINAL_TAG}" 