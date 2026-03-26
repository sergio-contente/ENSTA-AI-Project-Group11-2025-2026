#!/bin/bash
#SBATCH --job-name=torch_import_diag
#SBATCH --partition=ENSTA-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=00:10:00
#SBATCH --output=/home/ensta/ensta-arous/projetia/coin/logs/torch_import_diag_%j.out
#SBATCH --error=/home/ensta/ensta-arous/projetia/coin/logs/torch_import_diag_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

mkdir -p /home/ensta/ensta-arous/projetia/coin/logs
cd /home/ensta/ensta-arous/projetia/coin

PY310=/home/ensta/ensta-arous/miniconda3/envs/vllm_py310_clean/bin/python

echo "=== Torch import diagnostic job (NO GPU) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "PWD : $(pwd)"
echo "PY310: $($PY310 --version)"
echo ""

echo "=== Test 1: import torch with CUDA_VISIBLE_DEVICES='' (should be CPU-only) ==="
timeout 60s bash -lc "CUDA_VISIBLE_DEVICES='' $PY310 -u - <<'PY'
import time
print('import torch (NO CUDA_VISIBLE_DEVICES)...')
t=time.time()
import torch
print('imported torch in', time.time()-t)
print('torch', torch.__version__, 'cuda', torch.version.cuda)
PY"
RC1=$?
echo "NO-CUDA torch import exit code: $RC1"
echo ""

echo "=== Test 2: same test but with LD_PRELOAD of system libstdc++/libgcc (common HPC fix) ==="
timeout 60s bash -lc "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH; \
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libgcc_s.so.1; \
CUDA_VISIBLE_DEVICES='' $PY310 -u - <<'PY'
import time
print('import torch (NO CUDA_VISIBLE_DEVICES) with LD_PRELOAD...')
t=time.time()
import torch
print('imported torch in', time.time()-t)
print('torch', torch.__version__, 'cuda', torch.version.cuda)
PY"
RC2=$?
echo "LD_PRELOAD torch import exit code: $RC2"
echo ""

# Exit non-zero if both tests fail/time out (timeout usually returns 124)
if [ "$RC1" -ne 0 ] && [ "$RC2" -ne 0 ]; then
  echo "ERROR: Both torch import tests failed (likely dynamic linker / libstdc++ issue)."
  exit 1
fi

echo "Done."