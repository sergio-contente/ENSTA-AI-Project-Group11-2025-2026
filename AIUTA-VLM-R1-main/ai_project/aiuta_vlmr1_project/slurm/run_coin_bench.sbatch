#!/bin/bash
#SBATCH --job-name=coin_bench_eval
#SBATCH --output=results/logs/coin_bench-%j.out
#SBATCH --error=results/logs/coin_bench-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=ENSTA-h100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aiuta
whoami && hostname && pwd

# Override with: export COIN_PROJECT_ROOT=/path/to/aiuta_vlmr1_project
cd "${COIN_PROJECT_ROOT:-$HOME/ai_project/aiuta_vlmr1_project}"

  # shellcheck source=/dev/null
fi

MODE="${MODE:-entropy}"
TAU="${TAU:-0.15}"
SPLIT="${SPLIT:-val_seen}"
LIMIT_ARG="${LIMIT_ARG:-}"

mkdir -p "$HOME/hf_cache"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HOME/hf_cache}"

if [ -f "$HOME/.hf_env" ]; then source "$HOME/.hf_env"; fi
set -x

mkdir -p results/logs results/coin_bench

OUT="results/coin_bench/${MODE}_${SLURM_JOB_ID}.json"

ARGS=(
  python3 -m aiuta_vlmr1.pipeline.coin_bench_runner
  --coin-bench-path "${COIN_BENCH_PATH:-$HOME/data/coin_bench}"
  --mode "$MODE"
  --tau "$TAU"
  --split "$SPLIT"
  --output "$OUT"
)
if [ -n "$LIMIT_ARG" ]; then
  ARGS+=(--limit "$LIMIT_ARG")
fi
if [ -n "${CONFIG:-}" ]; then
  ARGS+=(--config "$CONFIG")
fi

echo "=== CoIN-Bench eval MODE=${MODE} TAU=${TAU} SPLIT=${SPLIT} OUT=${OUT} ==="
"${ARGS[@]}"

echo "=== DONE ==="
