#!/bin/bash
#SBATCH --job-name=torch_preload_only
#SBATCH --partition=ENSTA-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --output=/home/ensta/ensta-arous/projetia/coin/logs/torch_preload_only_%j.out
#SBATCH --error=/home/ensta/ensta-arous/projetia/coin/logs/torch_preload_only_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

mkdir -p /home/ensta/ensta-arous/projetia/coin/logs
cd /home/ensta/ensta-arous/projetia/coin

PY310=/home/ensta/ensta-arous/miniconda3/envs/vllm_py310_clean/bin/python

echo "Host: $(hostname)"
echo "PY310: $($PY310 --version)"
echo ""

echo "Finding system libstdc++ and libgcc paths..."
LIBSTDCPP=$(ldconfig -p | awk '/libstdc\+\+\.so\.6/{print $NF; exit}')
LIBGCC=$(ldconfig -p | awk '/libgcc_s\.so\.1/{print $NF; exit}')
echo "LIBSTDCPP=${LIBSTDCPP}"
echo "LIBGCC=${LIBGCC}"
echo ""

if [ -z "${LIBSTDCPP}" ] || [ -z "${LIBGCC}" ]; then
  echo "ERROR: Could not find system libstdc++/libgcc via ldconfig."
  exit 1
fi

export LD_PRELOAD="${LIBSTDCPP}:${LIBGCC}"
echo "LD_PRELOAD=${LD_PRELOAD}"
echo ""

echo "Test: import torch with NO CUDA (and LD_PRELOAD)..."
timeout 60s bash -lc "CUDA_VISIBLE_DEVICES='' $PY310 -u - <<'PY'
import time
print('import torch with LD_PRELOAD...')
t=time.time()
import torch
print('imported torch in', time.time()-t)
print('torch', torch.__version__, 'cuda', torch.version.cuda)
PY"
RC=$?
echo "Exit code: $RC"
exit $RC