#!/bin/bash
#SBATCH --job-name=coin_eval_sanity
#SBATCH --partition=ENSTA-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/home/ensta/ensta-arous/projetia/coin/logs/coin_eval_%j.out
#SBATCH --error=/home/ensta/ensta-arous/projetia/coin/logs/coin_eval_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

# --- Always ensure logs dir exists (SLURM writes stdout/err before script runs) ---
mkdir -p /home/ensta/ensta-arous/projetia/coin/logs
cd /home/ensta/ensta-arous/projetia/coin

echo "=== TOP OF SCRIPT ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "PWD : $(pwd)"
echo "Job : ${SLURM_JOB_ID}"
echo ""

# --- Environment Setup (DUAL PYTHON: 3.10 for vLLM, 3.9 for CoIN) ---
# Python 3.9 for CoIN evaluation + VLM servers (habitat_sim compatible)
PY39=/home/ensta/ensta-arous/miniconda3/envs/coin_vllm/bin/python
# Python 3.10 for vLLM server only (required for vLLM v0.11+)
PY310=/home/ensta/ensta-arous/miniconda3/envs/vllm_py310/bin/python

echo "Python 3.9 (CoIN):  $($PY39 --version)"
echo "Python 3.10 (vLLM): $($PY310 --version)"
echo ""

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export VLFM_PYTHON=$PY39

# --- Model & Port Config ---
export USE_LOCAL_LLM=true
export LOCAL_LLM_MODEL=/home/ensta/data/Qwen2.5-Coder-32B-Instruct
export LOCAL_LLM_MODEL_NAME=Qwen2.5-Coder-32B-Instruct
export LOCAL_LLM_PORT=8000

export GROUNDING_DINO_PORT=12181
export BLIP2ITM_PORT=12182
export SAM_PORT=12183
export LLava_PORT=12189

# Optional but often prevents “still calling remote” bugs in OpenAI-style clients
export OPENAI_API_BASE="http://127.0.0.1:${LOCAL_LLM_PORT}/v1"
export OPENAI_BASE_URL="http://127.0.0.1:${LOCAL_LLM_PORT}/v1"

# Model file paths
export MOBILE_SAM_CHECKPOINT=data/mobile_sam.pt
export GROUNDING_DINO_CONFIG=GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
export GROUNDING_DINO_WEIGHTS=data/groundingdino_swint_ogc.pth
export CLASSES_PATH=vlfm/vlm/classes.txt

# --- Preflight checks (fail fast) ---
[ -d "${LOCAL_LLM_MODEL}" ] || { echo "ERROR: LOCAL_LLM_MODEL not found: ${LOCAL_LLM_MODEL}"; exit 1; }
[ -f "${MOBILE_SAM_CHECKPOINT}" ] || { echo "ERROR: MOBILE_SAM_CHECKPOINT missing: ${MOBILE_SAM_CHECKPOINT}"; exit 1; }
[ -f "${GROUNDING_DINO_WEIGHTS}" ] || { echo "ERROR: GROUNDING_DINO_WEIGHTS missing: ${GROUNDING_DINO_WEIGHTS}"; exit 1; }
[ -f "${GROUNDING_DINO_CONFIG}" ] || { echo "ERROR: GROUNDING_DINO_CONFIG missing: ${GROUNDING_DINO_CONFIG}"; exit 1; }
[ -f "${CLASSES_PATH}" ] || { echo "ERROR: CLASSES_PATH missing: ${CLASSES_PATH}"; exit 1; }

echo ""
echo "=== GPU STATUS (nvidia-smi) ==="
nvidia-smi || true
echo ""

# --- Helpers: port check with nc or python fallback ---
if command -v nc >/dev/null 2>&1; then
  port_open() { nc -z 127.0.0.1 "$1" >/dev/null 2>&1; }
else
  port_open() {
    python - "$1" <<'PY'
import socket, sys
port=int(sys.argv[1])
s=socket.socket()
s.settimeout(1.0)
try:
    s.connect(("127.0.0.1", port))
    print("open", port)
    sys.exit(0)
except Exception:
    sys.exit(1)
finally:
    try: s.close()
    except Exception: pass
PY
  }
fi

cleanup() {
  echo ""
  echo "=== Shutting down all servers ==="
  kill "${LLM_PID:-}" "${GDINO_PID:-}" "${BLIP_PID:-}" "${SAM_PID:-}" "${LLAVA_PID:-}" 2>/dev/null || true
  echo "Cleanup complete"
}
trap cleanup EXIT

echo "=== CoIN Evaluation with Local LLM (SANITY RUN) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

echo "Smoke test (Py3.10, GPU 3): start $(date)"
timeout 30s bash -lc "CUDA_VISIBLE_DEVICES=3 $PY310 -u - <<'PY'
print('hello from py310')
import sys
print('python ok', sys.version)
PY"
echo "Smoke test exit code: $?"

# --- 2) Start LLM server on GPU 3 (Python 3.10) ---
echo "Starting vLLM (Qwen2.5-Coder-32B-Instruct) on GPU 3 with Python 3.10..."
CUDA_VISIBLE_DEVICES=3 \
PYTHONUNBUFFERED=1 \
$PY310 -u -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --model "${LOCAL_LLM_MODEL}" \
  --served-model-name "${LOCAL_LLM_MODEL_NAME}" \
  --port "${LOCAL_LLM_PORT}" \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype bfloat16 \
  --disable-log-requests \
  > "logs/llm_server_${SLURM_JOB_ID}.log" 2>&1 &
LLM_PID=$!

# Check if vLLM process died immediately
sleep 2
if ! kill -0 "$LLM_PID" 2>/dev/null; then
  echo "ERROR: vLLM process exited immediately (PID ${LLM_PID}). Showing log:"
  ls -lh "logs/llm_server_${SLURM_JOB_ID}.log" || true
  cat "logs/llm_server_${SLURM_JOB_ID}.log" || true
  echo ""
  echo "This usually means:"
  echo "  - PyTorch has wrong CUDA build for this driver"
  echo "  - Missing CUDA libraries"
  echo "  - vLLM dependencies incompatible with compute node"
  exit 1
fi
echo "  → vLLM PID: ${LLM_PID} (alive)"

# --- 2) Start detection stack on GPU 0 (Python 3.9) ---
echo "Starting Detection Stack (GroundingDINO, BLIP2ITM, SAM) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 $PY39 -m vlfm.vlm.grounding_dino --port "${GROUNDING_DINO_PORT}" \
  > "logs/gdino_${SLURM_JOB_ID}.log" 2>&1 &
GDINO_PID=$!
CUDA_VISIBLE_DEVICES=0 $PY39 -m vlfm.vlm.blip2itm --port "${BLIP2ITM_PORT}" \
  > "logs/blip_${SLURM_JOB_ID}.log" 2>&1 &
BLIP_PID=$!
CUDA_VISIBLE_DEVICES=0 $PY39 -m vlfm.vlm.sam --port "${SAM_PORT}" \
  > "logs/sam_${SLURM_JOB_ID}.log" 2>&1 &
SAM_PID=$!
echo "  → Detection PIDs: GDINO=${GDINO_PID}, BLIP=${BLIP_PID}, SAM=${SAM_PID}"

# --- 3) Start LLaVA-Next on GPU 1 (Python 3.9) ---
echo "Starting LLaVA-Next on GPU 1..."
CUDA_VISIBLE_DEVICES=1 $PY39 -m vlfm.vlm.llava_next --port "${LLava_PORT}" \
  > "logs/llava_${SLURM_JOB_ID}.log" 2>&1 &
LLAVA_PID=$!
echo "  → LLaVA PID: ${LLAVA_PID}"

# --- 4) Health checks ---
echo ""
echo "Waiting for vLLM to be healthy (/v1/models)..."
RETRY=0
until curl -s "http://127.0.0.1:${LOCAL_LLM_PORT}/v1/models" >/dev/null 2>&1; do
  sleep 5
  RETRY=$((RETRY+1))
  if [ $((RETRY % 6)) -eq 0 ]; then
    echo "  vLLM still booting... ($((RETRY*5))s) tail logs/llm_server_${SLURM_JOB_ID}.log"
  fi
  if [ "${RETRY}" -ge 120 ]; then
    echo "ERROR: vLLM failed to start within 600s"
    tail -80 "logs/llm_server_${SLURM_JOB_ID}.log" || true
    exit 1
  fi
done
echo "✓ vLLM is UP at http://127.0.0.1:${LOCAL_LLM_PORT}"
echo "Models available:"
curl -s "http://127.0.0.1:${LOCAL_LLM_PORT}/v1/models" | head || true

echo ""
echo "Waiting for VLM ports to open..."
for i in {1..180}; do
  if port_open "${GROUNDING_DINO_PORT}" && \
     port_open "${BLIP2ITM_PORT}" && \
     port_open "${SAM_PORT}" && \
     port_open "${LLava_PORT}"; then
    echo "✓ All VLM ports are open"
    break
  fi
  sleep 2
  if [ $i -eq 180 ]; then
    echo "ERROR: One or more VLM servers did not open ports in time."
    tail -60 "logs/gdino_${SLURM_JOB_ID}.log" || true
    tail -60 "logs/blip_${SLURM_JOB_ID}.log" || true
    tail -60 "logs/sam_${SLURM_JOB_ID}.log" || true
    tail -60 "logs/llava_${SLURM_JOB_ID}.log" || true
    exit 1
  fi
done

# --- 5) Run sanity evaluation on GPU 2 (Python 3.9) ---
echo ""
echo "=== Starting CoIN Evaluation on GPU 2 (10-episode sanity run) ==="
CUDA_VISIBLE_DEVICES=2 $PY39 -u -m vlfm.run habitat_baselines.test_episode_count=10

echo ""
echo "=== Evaluation Finished ==="
echo "Date: $(date)"