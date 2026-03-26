#!/bin/bash
#SBATCH --job-name=coin_eval_final
#SBATCH --partition=ENSTA-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/coin_eval_%j.out
#SBATCH --error=logs/coin_eval_%j.err

set -e

# --- Environment Setup (THREE PYTHON ENVS) ---
cd /home/ensta/ensta-arous/projetia/coin

# Python 3.9 for CoIN evaluation + Detection stack (habitat_sim, LAVIS compatible)
PY39=/home/ensta/ensta-arous/miniconda3/envs/coin_vllm_clone/bin/python
# Python 3.10 for vLLM server only (required for vLLM v0.11+)
PY310=/home/ensta/ensta-arous/miniconda3/envs/vllm_py310/bin/python
# Python 3.10 for LLaVA-NeXT only (requires transformers >=4.45)
PY_LLAVA=/home/ensta/ensta-arous/miniconda3/envs/llava_next_py310/bin/python

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export VLFM_PYTHON=$PY39

# --- Model & Port Config ---
export USE_LOCAL_LLM=true
export LOCAL_LLM_MODEL=/home/ensta/data/gpt-oss/gpt-oss-20b
export LOCAL_LLM_MODEL_NAME=gpt-oss-20b
export LOCAL_LLM_PORT=8000
export GROUNDING_DINO_PORT=12181
export BLIP2ITM_PORT=12182
export SAM_PORT=12183
export LLava_PORT=12189

# Model file paths
export MOBILE_SAM_CHECKPOINT=data/mobile_sam.pt
export GROUNDING_DINO_CONFIG=GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
export GROUNDING_DINO_WEIGHTS=data/groundingdino_swint_ogc.pth
export CLASSES_PATH=vlfm/vlm/classes.txt

mkdir -p logs

echo "=== CoIN Evaluation with Local LLM ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

# --- 1. Start LLM Server on GPU 3 (Python 3.10) ---
# LOGIC: H100 95GB - optimized for bfloat16 with 4K context
echo "Starting LLM Server (Qwen2.5-Coder-32B-Instruct) on GPU 3 with Python 3.10..."
CUDA_VISIBLE_DEVICES=3 $PY310 -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 \
    --model ${LOCAL_LLM_MODEL} \
  --served-model-name ${LOCAL_LLM_MODEL_NAME} \
    --port ${LOCAL_LLM_PORT} \
    --gpu-memory-utilization 0.92 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --dtype bfloat16 \
    --disable-log-requests > logs/llm_server_${SLURM_JOB_ID}.log 2>&1 &
LLM_PID=$!
echo "  → LLM server PID: ${LLM_PID}"

# --- 2. Start Detection Stack on GPU 0 (Python 3.9) ---
echo "Starting Detection Stack on GPU 0..."
CUDA_VISIBLE_DEVICES=0 $PY39 -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT} > logs/gdino_${SLURM_JOB_ID}.log 2>&1 &
GDINO_PID=$!
CUDA_VISIBLE_DEVICES=0 $PY39 -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT} > logs/blip_${SLURM_JOB_ID}.log 2>&1 &
BLIP_PID=$!
CUDA_VISIBLE_DEVICES=0 $PY39 -m vlfm.vlm.sam --port ${SAM_PORT} > logs/sam_${SLURM_JOB_ID}.log 2>&1 &
SAM_PID=$!
echo "  → Detection PIDs: GDINO=${GDINO_PID}, BLIP=${BLIP_PID}, SAM=${SAM_PID}"

# --- 3. Start LLaVA-Next on GPU 1 (dedicated env: transformers >=4.45) ---
echo "Starting LLaVA-Next on GPU 1 (llava_next_py310 env)..."
CUDA_VISIBLE_DEVICES=1 $PY_LLAVA -m vlfm.vlm.llava_next --port ${LLava_PORT} > logs/llava_${SLURM_JOB_ID}.log 2>&1 &
LLAVA_PID=$!
echo "  → LLaVA PID: ${LLAVA_PID}"

# --- Cleanup Logic ---
# LOGIC: Ensures no "zombie" processes stay alive if the job fails.
cleanup() {
    echo ""
    echo "=== Shutting down all servers ==="
    kill $LLM_PID $GDINO_PID $BLIP_PID $SAM_PID $LLAVA_PID 2>/dev/null || true
    echo "Cleanup complete"
}
trap cleanup EXIT

# --- 4. Health Checks ---
echo ""
echo "Waiting for LLM server to be healthy..."
RETRY_COUNT=0
until curl -s http://localhost:${LOCAL_LLM_PORT}/v1/models > /dev/null 2>&1; do
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $((RETRY_COUNT % 6)) -eq 0 ]; then
        echo "  LLM still booting... ($((RETRY_COUNT * 10))s elapsed, check logs/llm_server_${SLURM_JOB_ID}.log)"
    fi
    if [ $RETRY_COUNT -ge 60 ]; then
        echo "ERROR: LLM server failed to start in 600s"
        tail -50 logs/llm_server_${SLURM_JOB_ID}.log
        exit 1
    fi
done
echo "✓ LLM is UP at http://localhost:${LOCAL_LLM_PORT}"
echo "Models available:"
MODEL_LIST_JSON=$(curl -s http://127.0.0.1:${LOCAL_LLM_PORT}/v1/models)
MODEL_LIST_JSON_PATH="logs/model_list_${SLURM_JOB_ID}.json"
printf '%s' "${MODEL_LIST_JSON}" > "${MODEL_LIST_JSON_PATH}"

echo "Raw /v1/models response (for validation):"
cat "${MODEL_LIST_JSON_PATH}"

if [ ! -s "${MODEL_LIST_JSON_PATH}" ]; then
  echo "ERROR: /v1/models returned an empty response; cannot validate model-id compatibility."
  exit 1
fi

echo "Validating LLM model-id compatibility..."
python - "${LOCAL_LLM_MODEL_NAME}" "${MODEL_LIST_JSON_PATH}" <<'PY'
import json
import sys

expected_model = sys.argv[1]
json_path = sys.argv[2]

with open(json_path, "r", encoding="utf-8") as f:
  raw_response = f.read()

if not raw_response.strip():
  raise SystemExit("ERROR: /v1/models response is empty; cannot parse JSON.")

try:
  with open(json_path, "r", encoding="utf-8") as f:
    payload = json.load(f)
except json.JSONDecodeError as exc:
  raise SystemExit(
    f"ERROR: Failed to parse /v1/models response as JSON: {exc}. Raw response: {raw_response!r}"
  ) from exc

ids = [entry.get("id") for entry in payload.get("data", []) if isinstance(entry, dict)]
print(f"Advertised model ids: {ids}")
if expected_model not in ids:
  raise SystemExit(
    f"ERROR: LOCAL_LLM_MODEL_NAME='{expected_model}' not present in /v1/models ids: {ids}"
  )
print(f"Model id match OK: {expected_model}")
PY

echo ""
echo "Waiting for VLM ports to open..."
for i in {1..120}; do
  if nc -z 127.0.0.1 ${GROUNDING_DINO_PORT} && \
     nc -z 127.0.0.1 ${BLIP2ITM_PORT} && \
     nc -z 127.0.0.1 ${SAM_PORT} && \
     nc -z 127.0.0.1 ${LLava_PORT}; then
    echo "✓ All VLM ports are open"
    break
  fi
  sleep 2
  if [ $i -eq 120 ]; then
    echo "ERROR: One or more VLM servers did not open ports in time."
    tail -30 logs/gdino_${SLURM_JOB_ID}.log || true
    tail -30 logs/blip_${SLURM_JOB_ID}.log || true
    tail -30 logs/sam_${SLURM_JOB_ID}.log || true
    tail -30 logs/llava_${SLURM_JOB_ID}.log || true
    exit 1
  fi
done

# --- 5. Run Evaluation on GPU 2 (Python 3.9) ---
# Running SMALL TEST: 10 episodes to verify pipeline works
echo ""
echo "=== Starting CoIN Evaluation on GPU 2 (10 episodes test) ==="
CUDA_VISIBLE_DEVICES=2 $PY39 -m vlfm.run habitat_baselines.test_episode_count=10

echo ""
echo "=== Evaluation Finished ==="
echo "Date: $(date)"
