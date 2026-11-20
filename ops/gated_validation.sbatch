#!/bin/bash
#SBATCH --job-name=gpt20_pipeline_7001848
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -C H100|H200
#SBATCH --mem=192G
#SBATCH -c 24
#SBATCH -t 02:00:00
#SBATCH --no-requeue
#SBATCH -o pipeline_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ops/run.env
chain=$1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$JB_EXPERIMENT_PYTHON" ops/launch_checks.py --chain "$chain" --phase preflight
"$JB_EXPERIMENT_PYTHON" ops/launch_checks.py --chain "$chain" --phase serving
bash ops/validation.sbatch
# The supervisor must finish successfully before its results are accepted.
"$JB_EXPERIMENT_PYTHON" ops/launch_checks.py --chain "$chain" --phase pipeline
