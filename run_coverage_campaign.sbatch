#!/bin/bash
#SBATCH --job-name=coverage-controller
#SBATCH --partition=ice-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=18:00:00
#SBATCH --output=coverage-controller-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
root="${1:?Usage: sbatch run_coverage_campaign.sbatch CAMPAIGN_ROOT [initialization options]}"
shift
python_bin="${COVERAGE_PYTHON:-$PWD/.venv/bin/python}"
# init is idempotent: subsequent controllers load the same saved configuration.
"$python_bin" -m analysis.experiments.campaign init "$root" "$@"
exec "$python_bin" -m analysis.experiments.campaign control "$root"
