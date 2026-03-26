#!/bin/bash
#SBATCH --job-name=coin_eval
#SBATCH --partition=ENSTA-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/coin_eval_%j.out
#SBATCH --error=logs/coin_eval_%j.err

set -e

echo "=== CoIN Evaluation with Local LLM ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

# Create logs directory
mkdir -p logs

# Activate conda
source /home/ensta/ensta-arous/miniconda3/etc/profile.d/conda.sh
conda activate coin

# Change to project directory
cd /home/ensta/ensta-arous/projetia/coin

# Environment configuration
export PYTHONPATH=/home/ensta/ensta-arous/projetia/coin:${PYTHONPATH}
export VLFM_PYTHON=$(which python)

# LLM configuration
export USE_LOCAL_LLM=true
export LOCAL_LLM_MODEL=/home/ensta/data/Qwen2.5-Coder-32B-Instruct
export LOCAL_LLM_MODEL_NAME=Qwen2.5-Coder-32B-Instruct
export LOCAL_LLM_PORT=8000

# VLM/Detection server ports
export GROUNDING_DINO_PORT=12181
export BLIP2ITM_PORT=12182
export SAM_PORT=12183
export LLava_PORT=12189

# Model paths
export MOBILE_SAM_CHECKPOINT=data/mobile_sam.pt
export GROUNDING_DINO_CONFIG=GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
export GROUNDING_DINO_WEIGHTS=data/groundingdino_swint_ogc.pth
export CLASSES_PATH=vlfm/vlm/classes.txt

echo ""
echo "=== Starting LLM Server on GPU 3 ==="
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model ${LOCAL_LLM_MODEL} \
    --served-model-name ${LOCAL_LLM_MODEL_NAME} \
    --port ${LOCAL_LLM_PORT} \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --trust-remote-code \
    --dtype auto \
    --disable-log-requests \
    > logs/llm_server_${SLURM_JOB_ID}.log 2>&1 &
LLM_PID=$!
echo "LLM server started with PID: ${LLM_PID}"

echo ""
echo "=== Starting VLM Servers on GPUs 0-1 ==="

# Start GroundingDINO on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT} \
    > logs/grounding_dino_${SLURM_JOB_ID}.log 2>&1 &
GDINO_PID=$!

# Start BLIP2ITM on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT} \
    > logs/blip2itm_${SLURM_JOB_ID}.log 2>&1 &
BLIP_PID=$!

# Start SAM on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m vlfm.vlm.sam --port ${SAM_PORT} \
    > logs/sam_${SLURM_JOB_ID}.log 2>&1 &
SAM_PID=$!

# Start LLaVA on GPUs 0-1
CUDA_VISIBLE_DEVICES=0,1 python -m vlfm.vlm.llava_next --port ${LLava_PORT} \
    > logs/llava_${SLURM_JOB_ID}.log 2>&1 &
LLAVA_PID=$!

echo "VLM servers started: GDINO=${GDINO_PID}, BLIP=${BLIP_PID}, SAM=${SAM_PID}, LLaVA=${LLAVA_PID}"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "=== Cleaning up server processes ==="
    kill $LLM_PID $GDINO_PID $BLIP_PID $SAM_PID $LLAVA_PID 2>/dev/null || true
    echo "Cleanup complete"
}
trap cleanup EXIT

echo ""
echo "=== Waiting for LLM server to be ready ==="
MAX_RETRIES=60
RETRY_COUNT=0
while ! curl -s http://localhost:${LOCAL_LLM_PORT}/v1/models > /dev/null 2>&1; do
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $((RETRY_COUNT % 6)) -eq 0 ]; then
        echo "Waiting... ($((RETRY_COUNT * 5))s elapsed)"
    fi
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: LLM server failed to start in $((MAX_RETRIES * 5)) seconds"
        echo "Check logs/llm_server_${SLURM_JOB_ID}.log for details"
        tail -50 logs/llm_server_${SLURM_JOB_ID}.log
        exit 1
    fi
done
echo "✓ LLM Server is ready at http://localhost:${LOCAL_LLM_PORT}"

# Wait for VLM servers (they take ~60-90 seconds to load models)
echo ""
echo "=== Waiting for VLM servers to load models (90 seconds) ==="
sleep 90

echo ""
echo "=== Starting CoIN Evaluation ==="
# Use GPU 2 for the main evaluation process (separate from servers)
CUDA_VISIBLE_DEVICES=2 python -m vlfm.run

echo ""
echo "=== Evaluation Complete ==="
date
