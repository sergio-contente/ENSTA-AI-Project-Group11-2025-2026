#!/bin/bash
#SBATCH --job-name=offline_v2_DEPRECATED
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=ENSTA-h100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail
set -x
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aiuta

cat <<'EOF' >&2
================================================================================
DEPRECATED: slurm/run_offline_v2.sbatch

This job name previously ran episode_runner with the legacy --mode offline alias.
That flow is superseded by explicit CoIN offline modes:

  - offline_static_coin
  - offline_proxy
  - habitat (not implemented in episode_runner)

Use instead:

  sbatch --export=ALL,COIN_MODE=offline_static_coin slurm/run_coin_offline.sbatch
  sbatch --export=ALL,COIN_MODE=offline_proxy slurm/run_coin_offline.sbatch

See slurm/run_coin_offline.sbatch for HF cache, venv, and CoIN-Bench download.
================================================================================
EOF

exit 1
