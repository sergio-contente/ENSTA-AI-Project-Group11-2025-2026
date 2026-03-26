#!/bin/bash
#SBATCH --job-name=coin_vlmr1
#SBATCH --partition=ENSTA-l40s
#SBATCH --output=logs/coin_vlmr1_%j.out
#SBATCH --error=logs/coin_vlmr1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G

set -euo pipefail
set -x

source ~/miniconda3/etc/profile.d/conda.sh
conda activate coin-hab

# ── Paths ──
COIN_FORK="$HOME/ai_project/CoIN-fork"
AIUTA_PROJECT="$HOME/ai_project/aiuta_vlmr1_project"

cd "$COIN_FORK"
mkdir -p logs

# ── Make aiuta_vlmr1 importable (no pip install, just PYTHONPATH) ──
export PYTHONPATH="${AIUTA_PROJECT}:${PYTHONPATH:-}"

# ── Fix torch LD_LIBRARY_PATH ──
export TORCH_LIB_DIR="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# ── HuggingFace cache ──
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
if [ -f "$HOME/.hf_env" ]; then source "$HOME/.hf_env"; fi

# ── VLM-R1 mode ──
export COIN_USE_VLMR1=1
export AIUTA_VLMR1_CONFIG="${AIUTA_VLMR1_CONFIG:-$AIUTA_PROJECT/configs/vlmr1_coin.yaml}"
export COIN_VLMR1_DETECT_EVERY="${COIN_VLMR1_DETECT_EVERY:-5}"

# ── Ensure dummy policy exists ──
if [ ! -f data/dummy_policy.pth ]; then
    mkdir -p data
    python -c "import torch; torch.save({'config': None, 'extra_state': {'step': 0}, 'state_dict': {}}, 'data/dummy_policy.pth')"
fi

# ── Ensure CoIN-Bench data exists ──
SPLIT="${SPLIT:-val_seen}"
COIN_BENCH_DIR="${COIN_BENCH_DIR:-$HOME/ai_project/CoIN-Bench}"
if [ ! -d "$COIN_BENCH_DIR/$SPLIT" ]; then
    echo "ERROR: CoIN-Bench not found at $COIN_BENCH_DIR/$SPLIT"
    exit 1
fi

# # ── Scene data check ──
# SCENE_DIR="${SCENE_DIR:-$HOME/ai_project/data/scene_datasets/hm3d/eval}"
# if [ ! -d "$SCENE_DIR" ]; then
#     echo "ERROR: HM3D scenes not found at $SCENE_DIR"
#     exit 1
# fi

# ══════════════════════════════════════════════════════════
#  STEP 1: Launch MobileSAM server (background, same env)
# ══════════════════════════════════════════════════════════
export SAM_PORT="${SAM_PORT:-12183}"
echo "=== Starting MobileSAM server on port $SAM_PORT ==="

python -m vlfm.vlm.sam --port "$SAM_PORT" &
SAM_PID=$!
sleep 10

if ! kill -0 $SAM_PID 2>/dev/null; then
    echo "ERROR: MobileSAM server failed to start"
    exit 1
fi
echo "MobileSAM running (PID=$SAM_PID)"
trap "echo 'Stopping SAM...'; kill $SAM_PID 2>/dev/null || true" EXIT

# ══════════════════════════════════════════════════════════
#  STEP 2: Run CoIN-Bench evaluation
# ══════════════════════════════════════════════════════════
LIMIT="${LIMIT:--1}"

echo "=== CoIN-Bench VLM-R1 Evaluation ==="
echo "  Split:       $SPLIT"
echo "  Detect every: $COIN_VLMR1_DETECT_EVERY steps"
echo "  Config:      $AIUTA_VLMR1_CONFIG"
echo "  Episodes:    $LIMIT"
echo ""

python -m vlfm.run \
    habitat.task.measurements.success.success_distance="0.25" \
    habitat_baselines.eval.split="$SPLIT" \
    habitat.dataset.data_path="CoIN-Bench/${SPLIT}/${SPLIT}.json.gz" \
    habitat_baselines.test_episode_count="$LIMIT"

echo "=== DONE (job $SLURM_JOB_ID) ==="
