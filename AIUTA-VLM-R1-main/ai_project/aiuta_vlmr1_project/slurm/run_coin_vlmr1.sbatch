#!/bin/bash
#SBATCH --job-name=coin_vlmr1
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=ENSTA-h100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aiuta
whoami && hostname && pwd
cd "$HOME/ai_project/aiuta_vlmr1_project"
mkdir -p "$HOME/hf_cache"
export HF_HOME="$HOME/hf_cache"
export HF_HUB_CACHE="$HOME/hf_cache"
export TRANSFORMERS_CACHE="$HOME/hf_cache"
if [ -f "$HOME/.hf_env" ]; then source "$HOME/.hf_env"; fi
if [ -z "${HF_TOKEN:-}" ]; then echo "ERROR: HF_TOKEN not set"; exit 1; fi
set -x
mkdir -p results/logs
echo "=== RUN VLM-R1 + KG ==="
python3 -m pytest tests/ -v --tb=short
echo "=== TESTS PASSED ==="
