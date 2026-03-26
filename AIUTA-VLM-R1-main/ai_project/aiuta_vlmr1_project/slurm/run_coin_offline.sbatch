#!/bin/bash
#SBATCH --job-name=coin_offline
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=ENSTA-h100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aiuta
whoami && hostname && pwd

cd "$HOME/ai_project/aiuta_vlmr1_project"

COIN_MODE="${COIN_MODE:-offline_static_coin}"

mkdir -p "$HOME/hf_cache"
export HF_HOME="$HOME/hf_cache"
export HF_HUB_CACHE="$HOME/hf_cache"
export TRANSFORMERS_CACHE="$HOME/hf_cache"

if [ -f "$HOME/.hf_env" ]; then source "$HOME/.hf_env"; fi
if [ -z "${HF_TOKEN:-}" ]; then echo "ERROR: HF_TOKEN not set"; exit 1; fi
set -x

pip install huggingface_hub --break-system-packages -q 2>/dev/null || true

mkdir -p results/logs results/coin_offline

echo "=== DOWNLOAD CoIN-Bench ==="
python3 -c "
from aiuta_vlmr1.evaluation.coin_loader import CoINBenchLoader
loader = CoINBenchLoader('./data/CoIN-Bench')
loader.download()
print(loader.summary())
"

STRICT_ARGS=()
if [ "$COIN_MODE" = "offline_static_coin" ]; then
  STRICT_ARGS+=(--strict_coin_images)
fi

echo "=== RUN OFFLINE EVALUATION (COIN_MODE=${COIN_MODE}) ==="
python3 -m aiuta_vlmr1.pipeline.episode_runner \
  --config configs/vlmr1_coin.yaml \
  --mode "$COIN_MODE" \
  --coin_bench_dir ./data/CoIN-Bench \
  --output "results/coin_offline/${COIN_MODE}_${SLURM_JOB_ID}.json" \
  --seed 42 \
  --user_mode description_based \
  "${STRICT_ARGS[@]}"

echo "=== DONE ==="
