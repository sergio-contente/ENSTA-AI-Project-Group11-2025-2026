#!/bin/bash
#SBATCH --job-name=torch_threadfix_l40
#SBATCH --partition=ENSTA-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/home/ensta/ensta-arous/projetia/coin/logs/torch_threadfix_l40_%j.out
#SBATCH --error=/home/ensta/ensta-arous/projetia/coin/logs/torch_threadfix_l40_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

mkdir -p /home/ensta/ensta-arous/projetia/coin/logs
cd /home/ensta/ensta-arous/projetia/coin

PY310=/home/ensta/ensta-arous/miniconda3/envs/vllm_py310_clean/bin/python

echo "Host: $(hostname)"
echo "PY310: $($PY310 --version)"
echo ""

echo "=== GPU STATUS (nvidia-smi) ==="
nvidia-smi || true
echo ""

# --- Thread/runtime deadlock mitigations (very common fix for torch import hangs) ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export KMP_INIT_AT_FORK=FALSE
export KMP_DUPLICATE_LIB_OK=TRUE

# Keep these too (they don't hurt and help later for CUDA/NCCL)
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

echo "Thread env:"
env | egrep 'OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|KMP_' || true
echo ""

echo "Test A: import torch with NO CUDA + thread fixes..."
timeout 60s bash -lc "CUDA_VISIBLE_DEVICES='' $PY310 -u - <<'PY'
import time
print('import torch (NO CUDA) with thread fixes...')
t=time.time()
import torch
print('imported torch in', time.time()-t)
print('torch', torch.__version__, 'cuda', torch.version.cuda)
PY"
RC_A=$?
echo "Test A exit code: ${RC_A}"
echo ""

echo "Test B: torch.cuda.is_available on L40..."
timeout 60s bash -lc "CUDA_VISIBLE_DEVICES=0 $PY310 -u - <<'PY'
import torch, time
print('torch', torch.__version__, 'cuda', torch.version.cuda)
t=time.time()
print('cuda available:', torch.cuda.is_available(), 'check in', time.time()-t, 's')
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    x=torch.randn(1, device='cuda')
    torch.cuda.synchronize()
    print('cuda tensor OK')
PY"
RC_B=$?
echo "Test B exit code: ${RC_B}"

if [ "${RC_A}" -ne 0 ] || [ "${RC_B}" -ne 0 ]; then
  echo "ERROR: torch still hanging/failing even with thread fixes."
  exit 1
fi

echo "SUCCESS: torch import + CUDA OK with thread fixes."