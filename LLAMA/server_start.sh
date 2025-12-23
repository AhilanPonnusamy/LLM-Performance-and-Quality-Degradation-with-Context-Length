#!/bin/bash
# -----------------------------------------------------------------------------
# Script Name : start_vllm_server.sh
# Purpose     : Launch vLLM OpenAI-compatible API server for Llama-3.1-70B-Instruct
# Requirements:
#   - NVIDIA GPU(s) available
#   - nvidia-smi installed
#   - HF_TOKEN environment variable set
#   - vLLM installed in Python environment
# -----------------------------------------------------------------------------

echo "Starting vLLM server for model"
echo "Checking HuggingFace token..."

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set!"
    echo "Please set it with: export HF_TOKEN=your_token"
    exit 1
fi

echo "HF_TOKEN is set. Launching vLLM..."

# -----------------------------------------------------------------------------
# Configuration Section
# -----------------------------------------------------------------------------
MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"
HOST="0.0.0.0"
PORT=8000
MAX_MODEL_LEN=128000

echo "Model            : $MODEL_NAME"
echo "Max Model Length : $MAX_MODEL_LEN"
echo "Server           : $HOST:$PORT"
echo "----------------------------------------"


# Detect number of GPUs
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "ERROR: No GPUs detected!"
    exit 1
fi

# Display GPU info
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Set tensor parallel size to use all GPUs
TENSOR_PARALLEL_SIZE=$GPU_COUNT
echo "Setting tensor-parallel-size to $TENSOR_PARALLEL_SIZE"


# -----------------------------------------------------------------------------
# Launch vLLM OpenAI-Compatible API Server
# -----------------------------------------------------------------------------
echo "🚦 Launching vLLM API server..."

python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --max-model-len 128000 \
    --host 0.0.0.0 \
    --port 8000 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 128000 \
    --tensor-parallel-size 8 \
    --kv-cache-memory-bytes 14000000000

