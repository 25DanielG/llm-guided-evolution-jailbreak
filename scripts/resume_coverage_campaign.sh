#!/bin/bash
# Restore the recorded runtime in an isolated copy, validate, then resume.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${COVERAGE_PYTHON:-$PWD/.venv/bin/python}"
exec "$python_bin" -m analysis.experiments.recovery "${1:?Provide a campaign directory}" --resume
