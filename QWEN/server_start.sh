#!/bin/bash

echo "Starting vLLM server for model"
echo "Checking HuggingFace token..."

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set!"
    echo "Please set it with: export HF_TOKEN=your_token"
    exit 1
fi

echo "HF_TOKEN is set. Launching vLLM..."

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

python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen1.5-14B-Chat \
    --max-model-len 32768 \
    --host 0.0.0.0 \
    --port 8000 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 32768 \
    --tensor-parallel-size 8 \
    --kv-cache-memory-bytes 14000000000

