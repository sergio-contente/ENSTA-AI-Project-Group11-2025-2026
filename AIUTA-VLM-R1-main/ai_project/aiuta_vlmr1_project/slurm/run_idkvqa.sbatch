#!/bin/bash
#SBATCH --job-name=idkvqa_eval
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=ENSTA-l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aiuta
whoami && hostname && pwd

cd "$HOME/ai_project/aiuta_vlmr1_project"

MODE="${MODE:-raw}"
GLOBAL_KG_PATH="${GLOBAL_KG_PATH:-}"

mkdir -p "$HOME/hf_cache"
export HF_HOME="$HOME/hf_cache"
export HF_HUB_CACHE="$HOME/hf_cache"
export TRANSFORMERS_CACHE="$HOME/hf_cache"

if [ -f "$HOME/.hf_env" ]; then source "$HOME/.hf_env"; fi
if [ -z "${HF_TOKEN:-}" ]; then echo "ERROR: HF_TOKEN not set"; exit 1; fi
set -x

pip install datasets --break-system-packages -q 2>/dev/null || true

mkdir -p results/logs results/idkvqa

EXTRA_ARGS=()
if [ -n "$GLOBAL_KG_PATH" ]; then
  EXTRA_ARGS+=(--global-kg-path "$GLOBAL_KG_PATH")
fi

echo "=== RUN IDKVQA EVALUATION (mode=${MODE}) ==="
python3 -m aiuta_vlmr1.evaluation.idkvqa_eval \
  --config ${CONFIG:-configs/idkvqa_eval.yaml} \
  --mode "$MODE" \
  --output "results/idkvqa/${MODE}_${SLURM_JOB_ID}.json" \
  "${EXTRA_ARGS[@]}"

echo "=== DONE ==="
