#!/bin/bash
#SBATCH --job-name=coverage-diversity
#SBATCH --partition=ice-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_FLAX=0
exec "$PWD/.venv/bin/python" analysis/scripts/full_campaign_analysis.py \
  --campaign "${1:?Campaign directory}" --output "${2:?New analysis directory}" \
  --final-generation "${3:-10}" --workers "${SLURM_CPUS_PER_TASK:-8}"
