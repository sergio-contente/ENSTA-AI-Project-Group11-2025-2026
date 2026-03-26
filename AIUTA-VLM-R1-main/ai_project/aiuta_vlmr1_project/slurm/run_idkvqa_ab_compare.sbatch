#!/bin/bash
#SBATCH --job-name=idkvqa_ab_compare
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=ENSTA-h100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
set -x
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aiuta
whoami && hostname && pwd

cd "$HOME/ai_project/aiuta_vlmr1_project"

INPUT_A="${INPUT_A:-}"
INPUT_B="${INPUT_B:-}"
LABEL_A="${LABEL_A:-baseline}"
LABEL_B="${LABEL_B:-variant}"
TAU_STEP="${TAU_STEP:-0.05}"
ABSTENTION_RULE="${ABSTENTION_RULE:-entropy_above_tau_to_idk}"
OUT_JSON="${OUT_JSON:-results/idkvqa/ab_compare_${SLURM_JOB_ID}.json}"
SWEEP_DIR="${SWEEP_DIR:-results/idkvqa/ab_sweeps_${SLURM_JOB_ID}}"

if [ -z "$INPUT_A" ] || [ -z "$INPUT_B" ]; then
  echo "ERROR: INPUT_A and INPUT_B must be set."
  echo "Example:"
  echo "  sbatch --export=ALL,INPUT_A=results/idkvqa/a.json,INPUT_B=results/idkvqa/b.json slurm/run_idkvqa_ab_compare.sbatch"
  exit 1
fi

mkdir -p results/logs results/idkvqa "$SWEEP_DIR"

python3 -m aiuta_vlmr1.evaluation.idkvqa_ab_compare \
  --input-a "$INPUT_A" \
  --input-b "$INPUT_B" \
  --label-a "$LABEL_A" \
  --label-b "$LABEL_B" \
  --output "$OUT_JSON" \
  --export-threshold-sweep-dir "$SWEEP_DIR" \
  --tau-step "$TAU_STEP" \
  --abstention-rule "$ABSTENTION_RULE"

echo "=== DONE ==="
