#!/bin/bash
#SBATCH --job-name=kg_bench
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err
#SBATCH --time=06:00:00
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

pip install datasets --break-system-packages -q 2>/dev/null || true

mkdir -p results/logs results/kg_comparison "results/kg_comparison/paper_${SLURM_JOB_ID}"

echo "=== KG vs RAW VLM-R1 BENCHMARK (full bundle + paper artifacts) ==="
python3 -m aiuta_vlmr1.evaluation.kg_vs_raw_benchmark \
  --config configs/idkvqa_eval.yaml \
  --output "results/kg_comparison/benchmark_${SLURM_JOB_ID}.json" \
  --paper-artifacts-dir "results/kg_comparison/paper_${SLURM_JOB_ID}"

echo "=== DONE ==="
