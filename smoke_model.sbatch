#!/bin/bash
# One serving-only acceptance test; never starts evolution or submits more jobs.
#SBATCH --job-name=vip_smoke
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -C H100|H200
#SBATCH --mem=96G
#SBATCH -c 8
#SBATCH -t 00:45:00
#SBATCH --no-requeue
#SBATCH -o vip_smoke_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${JB_SERVING_ENV:?Set the separate, installed vLLM environment}"
: "${JB_EXPERIMENT_PYTHON:?Set existing experiment Python 3.12 for the /generate proxy}"
: "${JB_SHARED_MODEL_ROOT:?Set the verified writable/staged model root}"
: "${JB_SERVE_MODEL:?Test 20B, then Devstral, then 120B in separate allocations}"
: "${JB_SMOKE_RESULTS:?Persistent, permitted result directory outside checkpoints}"
: "${JB_CUDA_MODULE:?Verified compatible CUDA module}"
module load "$JB_CUDA_MODULE"
export JB_VLLM_BIN="$JB_SERVING_ENV/bin/vllm"
TP_ARGS=()
[[ -z "${JB_SERVE_TP:-}" ]] || TP_ARGS=(--tp "$JB_SERVE_TP")
timeout --kill-after=45s 40m "$JB_SERVING_ENV/bin/python" -m src.serving_smoke \
    --model "$JB_SERVE_MODEL" --experiment-python "$JB_EXPERIMENT_PYTHON" \
    --output "$JB_SMOKE_RESULTS/${JB_SERVE_MODEL}_${SLURM_JOB_ID}" "${TP_ARGS[@]}"
