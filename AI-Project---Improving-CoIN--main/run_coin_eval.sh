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

# This script runs CoIN evaluation using local LLMs on the cluster
# 
# GPU allocation:
#   GPU 0: GroundingDINO, BLIP2, SAM (shared)
#   GPU 1: LLaVA-Next (VLM)
#   GPU 2-3: Local LLM server (if using large model like 70B)
#
# For smaller models (8B), only GPUs 0-2 are needed

set -e

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=== CoIN Evaluation with Local LLM ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L)"
echo "Date: $(date)"
echo ""

# Activate conda environment (works in non-interactive SLURM jobs)
# Option 1: Use conda.sh directly
source /home/ensta/ensta-arous/miniconda3/etc/profile.d/conda.sh
conda activate coin  # or your environment name

# Option 2: If the above path doesn't work, try:
# eval "$(conda shell.bash hook)"
# conda activate coin

# Environment configuration
export VLFM_PYTHON=$(which python)
export USE_LOCAL_LLM=true

# Add project root to PYTHONPATH so vlfm module can be found
export PYTHONPATH=/home/ensta/ensta-arous/projetia/coin:${PYTHONPATH}

# Model paths (from /home/ensta/data/)
# Choose one of the following:
# - Small (faster, fits on 1 GPU): Meta-Llama-3-8B-Instruct
# - Medium: Ministral-8B-Instruct-2410  
# - Large (needs multiple GPUs): Qwen2.5-Coder-32B-Instruct
export LOCAL_LLM_MODEL=/home/ensta/data/Qwen2.5-Coder-32B-Instruct
export LOCAL_LLM_MODEL_NAME=Qwen2.5-Coder-32B-Instruct
export LOCAL_LLM_PORT=8000

# VLM and detection server ports
export GROUNDING_DINO_PORT=12181
export BLIP2ITM_PORT=12182
export SAM_PORT=12183
export LLava_PORT=12189

# GPU allocation
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Starting servers..."
cd /home/ensta/ensta-arous/projetia/coin

# Launch all servers (runs in tmux, you can attach to monitor)
./scripts/launch_vlm_servers.sh

# Wait for servers to be ready
echo "Waiting for servers to start (90 seconds)..."
sleep 90

# Check if LLM server is up
echo "Checking LLM server status..."
curl -s http://localhost:${LOCAL_LLM_PORT}/v1/models || echo "Warning: LLM server may not be ready yet"

# Run the evaluation
echo ""
echo "Starting CoIN evaluation..."
python -m vlfm.run

echo ""
echo "=== Evaluation complete ==="
