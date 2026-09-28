#!/bin/bash
# sbatch run_semantic_projection.sbatch [existing_prompt_diversity_directory]
#SBATCH --job-name=semantic_projection
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=semantic_projection_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
ANALYSIS_DIRECTORY="${1:-results/analysis/jb_5956355/prompt_diversity}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMBA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export USE_TF=0 USE_FLAX=0 PYTHONUNBUFFERED=1
srun .venv/bin/python analysis/scripts/semantic_projection.py "$ANALYSIS_DIRECTORY"
