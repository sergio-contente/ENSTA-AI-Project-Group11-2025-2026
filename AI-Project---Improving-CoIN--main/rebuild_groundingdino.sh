#!/bin/bash
#SBATCH --job-name=rebuild_gdino
#SBATCH --partition=ENSTA-l40s
#SBATCH --output=logs/rebuild_gdino_%j.out
#SBATCH --error=logs/rebuild_gdino_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G

echo "======================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================"

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate coin

# Verify environment
echo "Python path: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Navigate to GroundingDINO directory
cd /home/ensta/ensta-arous/projetia/coin/GroundingDINO

# Set compilation targets for both L40s (sm_89) and H100 (sm_90)
export TORCH_CUDA_ARCH_LIST="8.9;9.0"
echo "Compiling for GPU architectures: $TORCH_CUDA_ARCH_LIST"
echo ""

# Rebuild GroundingDINO with CUDA support
echo "Rebuilding GroundingDINO with CUDA support..."
pip uninstall -y groundingdino
pip install --no-build-isolation -e . --no-deps

# Test that CUDA ops loaded successfully
echo ""
echo "======================================"
echo "Testing CUDA ops..."

# Set library paths for PyTorch
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH

python -c "
try:
    from groundingdino import _C
    print('✓ CUDA ops loaded successfully!')
except Exception as e:
    print(f'✗ Failed to load CUDA ops: {e}')
    exit(1)
"

echo ""
echo "======================================"
echo "Job completed at: $(date)"
echo "======================================"
