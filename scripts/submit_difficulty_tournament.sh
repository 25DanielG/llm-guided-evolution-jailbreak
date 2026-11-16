#!/bin/bash
# Explicitly submit all three initial populations, then all six experiments.
set -euo pipefail
campaign="$(realpath -e "${1:?Usage: SCRIPT CAMPAIGN}")"
python_bin="${DT_PYTHON:-$PWD/.venv/bin/python}"
# Validate immutable inputs and determine array sizes before submitting anything.
seeds=$("$python_bin" - "$campaign" <<'PY'
from pathlib import Path
import sys
from analysis.experiments.difficulty_tournament import load_manifest
print(len(load_manifest(Path(sys.argv[1]))['seeds']))
PY
)
initial_job=$(sbatch --parsable --array="0-$((seeds - 1))" \
    run_difficulty_tournament.sbatch initial "$campaign" new)
initial_job="${initial_job%%;*}"
printf 'Initial population job: %s\n' "$initial_job"
run_job=$(sbatch --parsable --array="0-$((seeds * 2 - 1))" --dependency="afterok:$initial_job" \
    run_difficulty_tournament.sbatch run "$campaign" new)
printf 'Experiment job: %s\n' "$run_job"
