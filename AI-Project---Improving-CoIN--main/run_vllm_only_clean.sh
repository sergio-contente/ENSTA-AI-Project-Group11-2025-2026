#!/bin/bash
#SBATCH --job-name=vllm_only_clean
#SBATCH --partition=ENSTA-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/home/ensta/ensta-arous/projetia/coin/logs/vllm_only_clean_%j.out
#SBATCH --error=/home/ensta/ensta-arous/projetia/coin/logs/vllm_only_clean_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

mkdir -p /home/ensta/ensta-arous/projetia/coin/logs
cd /home/ensta/ensta-arous/projetia/coin

PY310=/home/ensta/ensta-arous/miniconda3/envs/vllm_py310_clean/bin/python
export LOCAL_LLM_MODEL=/home/ensta/data/Qwen2.5-Coder-32B-Instruct
export LOCAL_LLM_MODEL_NAME=Qwen2.5-Coder-32B-Instruct
export LOCAL_LLM_PORT=8000

echo "=== vLLM-only CLEAN debug job ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "PWD : $(pwd)"
echo "PY310: $($PY310 --version)"
echo ""

echo "=== GPU STATUS (nvidia-smi) ==="
nvidia-smi || true
echo ""

# These can help on some HPC setups (safe to keep for debugging)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "=== Torch CUDA smoke test (Py3.10 clean env, GPU 0) ==="
timeout 90s bash -lc "CUDA_VISIBLE_DEVICES=0 $PY310 -u - <<'PY'
import time, os
print('pid', os.getpid())
print('step1: start')

t=time.time()
print('step2: importing torch...')
import torch
print('step2 done in', round(time.time()-t,3), 's')
print('torch', torch.__version__, 'torch.cuda', torch.version.cuda)

t=time.time()
print('step3: torch.cuda.is_available() ...')
avail = torch.cuda.is_available()
print('step3 done in', round(time.time()-t,3), 's')
print('is_available', avail)

if avail:
    print('device_count', torch.cuda.device_count())
    print('device_name', torch.cuda.get_device_name(0))
    x = torch.randn(1, device='cuda')
    torch.cuda.synchronize()
    print('cuda tensor OK')
PY"
TORCH_RC=$?
echo "Torch smoke exit code: ${TORCH_RC}"
if [ "${TORCH_RC}" -ne 0 ]; then
  echo "ERROR: Torch CUDA smoke test failed/hung. vLLM will not start."
  exit 1
fi
echo ""

echo "=== Starting vLLM in FOREGROUND (timeout 10 min) ==="
export VLLM_LOGGING_LEVEL=DEBUG
export PYTHONFAULTHANDLER=1

timeout 600s bash -lc "CUDA_VISIBLE_DEVICES=0 $PY310 -u -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --model ${LOCAL_LLM_MODEL} \
  --served-model-name ${LOCAL_LLM_MODEL_NAME} \
  --port ${LOCAL_LLM_PORT} \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.92 \
  --uvicorn-log-level debug"
VLLM_RC=$?
echo "vLLM exit code: ${VLLM_RC}"
exit "${VLLM_RC}"