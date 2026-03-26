# Patch: replace .venv with conda activate aiuta in all sbatch files
for f in slurm/*.sbatch; do
  # Remove lines with .venv/bin/activate
  sed -i '/\.venv\/bin\/activate/d' "$f"
  # Remove lines with VENV_ACTIVATE
  sed -i '/VENV_ACTIVATE/d' "$f"
  # Add conda activate after "set -x" (or "set -euo pipefail")
  # Skip if already has conda activate
  if ! grep -q "conda activate" "$f"; then
    sed -i '/^set -x$/a source ~/miniconda3/etc/profile.d/conda.sh\nconda activate aiuta' "$f"
  fi
done
