#!/bin/bash
# Resource requests are supplied by the dispatcher for each stage.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
python_bin="${COVERAGE_PYTHON:-$PWD/.venv/bin/python}"
exec "$python_bin" -m analysis.experiments.campaign work "${1:?Campaign root}" --stage "${2:?Stage ID}"
