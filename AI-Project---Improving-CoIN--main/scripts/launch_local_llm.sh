#!/usr/bin/env bash
# Launch local LLM server using vLLM with OpenAI-compatible API
# This replaces the need for Groq API

# Default configuration
export LOCAL_LLM_PORT=${LOCAL_LLM_PORT:-8000}
export LOCAL_LLM_MODEL=${LOCAL_LLM_MODEL:-/home/ensta/data/Qwen2.5-Coder-32B-Instruct}
export LOCAL_LLM_MODEL_NAME=${LOCAL_LLM_MODEL_NAME:-Qwen2.5-Coder-32B-Instruct}
export LOCAL_LLM_GPU=${LOCAL_LLM_GPU:-1}  # GPU to use for LLM (separate from VLM)

# Available models:
# - /home/ensta/data/Qwen2.5-Coder-32B-Instruct
# - /home/ensta/data/Llama-3.1-70B-Instruct
# - /home/ensta/data/Ministral-8B-Instruct-2410

echo "Starting local LLM server..."
echo "Model: ${LOCAL_LLM_MODEL}"
echo "Port: ${LOCAL_LLM_PORT}"
echo "GPU: ${LOCAL_LLM_GPU}"

# Check if model exists
if [ ! -d "${LOCAL_LLM_MODEL}" ]; then
    echo "Error: Model directory not found: ${LOCAL_LLM_MODEL}"
    exit 1
fi

# Launch vLLM with OpenAI-compatible API
CUDA_VISIBLE_DEVICES=${LOCAL_LLM_GPU} python -m vllm.entrypoints.openai.api_server \
    --model ${LOCAL_LLM_MODEL} \
    --served-model-name ${LOCAL_LLM_MODEL_NAME} \
    --port ${LOCAL_LLM_PORT} \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --trust-remote-code \
    --dtype auto
