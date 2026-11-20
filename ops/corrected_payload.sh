#!/bin/bash
set -euo pipefail
gpus=$("$JB_EXPERIMENT_PYTHON" -c 'import json; print(json.load(open("experiment.json"))["resources"]["gpus"])')
"$JB_EXPERIMENT_PYTHON" -B ops/resource_sampler.py "$RUN_ID/gpu-resources.jsonl" --expected "$gpus" &
sampler=$!
pipeline=''
cleanup() {
    if [[ -n "$pipeline" ]]; then kill -TERM "$pipeline" 2>/dev/null || true; wait "$pipeline" 2>/dev/null || true; fi
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
}
trap cleanup EXIT
for ((attempt=0; attempt<45; attempt++)); do
    kill -0 "$sampler" 2>/dev/null || { wait "$sampler"; exit 1; }
    [[ -s "$RUN_ID/gpu-resources.jsonl" && -s "$RUN_ID/gpu-resources.allocation.json" ]] && break
    sleep 1
done
[[ -s "$RUN_ID/gpu-resources.allocation.json" ]] || { echo 'GPU discovery timed out' >&2; exit 1; }
bash run_jailbreak.sbatch &
pipeline=$!
while kill -0 "$pipeline" 2>/dev/null; do
    kill -0 "$sampler" 2>/dev/null || { echo 'GPU monitoring failed during pipeline' >&2; exit 1; }
    sleep 1
done
wait "$pipeline"
pipeline=''
kill -0 "$sampler" 2>/dev/null || { echo 'GPU monitoring stopped before checks' >&2; exit 1; }
"$JB_EXPERIMENT_PYTHON" ops/corrected_checks.py completed
