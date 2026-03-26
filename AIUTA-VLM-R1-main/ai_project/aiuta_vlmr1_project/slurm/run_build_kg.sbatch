#!/bin/bash
#SBATCH --job-name=build_global_kg
#SBATCH --output=results/logs/build_global_kg-%j.out
#SBATCH --error=results/logs/build_global_kg-%j.err
#SBATCH --time=08:00:00
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

mkdir -p "$HOME/hf_cache"
export HF_HOME="$HOME/hf_cache"
export HF_HUB_CACHE="$HOME/hf_cache"
export TRANSFORMERS_CACHE="$HOME/hf_cache"
export PYTHONUNBUFFERED=1

if [ -f "$HOME/.hf_env" ]; then source "$HOME/.hf_env"; fi
set -x

pip install datasets --break-system-packages -q 2>/dev/null || true

mkdir -p results/logs results

OUT="${GLOBAL_KG_OUTPUT:-results/global_kg_${SLURM_JOB_ID}.json}"

echo "=== BUILD GLOBAL KG → ${OUT} ==="

python3 -m aiuta_vlmr1.knowledge_graph.build_global_kg \
  --config "${CONFIG:-configs/idkvqa_eval.yaml}" \
  --split "${SPLIT:-val}" \
  --seed "${SEED:-42}" \
  --output "$OUT"

echo "=== DONE ==="
