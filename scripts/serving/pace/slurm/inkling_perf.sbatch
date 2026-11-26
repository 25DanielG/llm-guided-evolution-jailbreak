#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --tmp=96G
#SBATCH --time=01:15:00
#SBATCH --no-requeue
set -euo pipefail

repo=${1:?Pass the absolute PACE source path}
export PACE_EXPECTED_GPU_FAMILY=${2:?Pass H100 or H200}
export PACE_EXPECTED_GPUS=1
shift 2
: "${PACE_INFERENCE_STATE:?Set the external state directory}"
python=${PACE_PYTHON:-python3}
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACEHUB_API_TOKEN
umask 077
output=$(mktemp -d "$PACE_INFERENCE_STATE/perf-$SLURM_JOB_ID.XXXXXXXX")
printf 'Performance results %s\n' "$output"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
"$python" -B "$repo/pace_inference.py" preflight inkling 2>&1 | tee "$output/preflight.log"

"$python" -B "$repo/inkling_benchmark.py" "$output" "$@" &
manager=$!
trap 'kill -TERM "$manager" 2>/dev/null || true; wait "$manager" || true; exit 143' TERM INT
wait "$manager"
