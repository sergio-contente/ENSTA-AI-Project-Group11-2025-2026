#!/bin/bash
# Interactive script to run CoIN with local LLMs
# Use this on an allocated node (after salloc) or interactive session
#
# Usage:
#   salloc --partition=ENSTA-h100 --gres=gpu:3 --cpus-per-task=16 --time=4:00:00
#   ./scripts/run_local_llm_eval.sh

set -e

echo "=== CoIN with Local LLM - Interactive Mode ==="

# Check if GPUs are available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Are you on a GPU node?"
    exit 1
fi

echo "Available GPUs:"
nvidia-smi -L
echo ""

# Configuration
export USE_LOCAL_LLM=true
export LOCAL_LLM_PORT=8000

# Model selection - uncomment one:
# Large model (requires 2-4 GPUs, ~140GB+ VRAM total)
export LOCAL_LLM_MODEL=/home/ensta/data/Qwen2.5-Coder-32B-Instruct
export LOCAL_LLM_MODEL_NAME=Qwen2.5-Coder-32B-Instruct

# Medium model (1-2 GPUs)
# export LOCAL_LLM_MODEL=/home/ensta/data/Ministral-8B-Instruct-2410
# export LOCAL_LLM_MODEL_NAME=Ministral-8B-Instruct-2410

# Small model (1 GPU, ~16GB VRAM)
# export LOCAL_LLM_MODEL=/home/ensta/data/Qwen2.5-Coder-32B-Instruct
# export LOCAL_LLM_MODEL_NAME=Qwen2.5-Coder-32B-Instruct

echo "Using model: ${LOCAL_LLM_MODEL_NAME}"

# Start the LLM server on GPU 2 (leaving 0-1 for VLM servers)
echo ""
echo "Starting local LLM server on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model ${LOCAL_LLM_MODEL} \
    --served-model-name ${LOCAL_LLM_MODEL_NAME} \
    --port ${LOCAL_LLM_PORT} \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --trust-remote-code \
    --dtype auto &

LLM_PID=$!
echo "LLM server PID: ${LLM_PID}"

# Wait for LLM server to start
echo "Waiting for LLM server to initialize..."
sleep 30

# Test LLM server
echo "Testing LLM server..."
curl -s http://localhost:${LOCAL_LLM_PORT}/v1/models | python -m json.tool || {
    echo "Warning: LLM server might not be ready yet, waiting longer..."
    sleep 30
}

echo ""
echo "=== LLM Server Ready ==="
echo "LLM server running at: http://localhost:${LOCAL_LLM_PORT}/v1"
echo ""
echo "Now start the VLM servers in another terminal:"
echo "  export USE_LOCAL_LLM=true"
echo "  export LOCAL_LLM_PORT=${LOCAL_LLM_PORT}"
echo "  ./scripts/launch_vlm_servers.sh"
echo ""
echo "Then run your evaluation:"
echo "  python -m vlfm.run"
echo ""
echo "To stop the LLM server: kill ${LLM_PID}"

# Keep script running
wait ${LLM_PID}
