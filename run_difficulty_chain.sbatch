#!/bin/bash
#SBATCH --job-name=difficulty-chain
#SBATCH --account=coc
#SBATCH --qos=coe-ice
#SBATCH --partition=ice-gpu,coe-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:5
#SBATCH --constraint=H200
#SBATCH --exclude=atl1-1-03-014-16-0
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --no-requeue
#SBATCH --output=difficulty-chain-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Missing submission directory}"
# exec makes Python receive Slurm's batch signals and forward them to the worker.
exec "${DT_PYTHON:-$PWD/.venv/bin/python}" -u scripts/difficulty_chain.py "$@"
