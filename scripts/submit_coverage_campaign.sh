#!/bin/bash
# Run on a login node. Downloads/stages inputs without doing embeddings or GPU work.
# Usage: bash scripts/submit_coverage_campaign.sh ROOT [campaign initialization options]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
root="${1:?Provide a campaign output directory}"
shift
python_bin="${COVERAGE_PYTHON:-$PWD/.venv/bin/python}"
"$python_bin" -m analysis.experiments.campaign init "$root" "$@"
"$python_bin" -m analysis.experiments.campaign stage-inputs "$root"
"$python_bin" -m analysis.experiments.campaign submit "$root"
