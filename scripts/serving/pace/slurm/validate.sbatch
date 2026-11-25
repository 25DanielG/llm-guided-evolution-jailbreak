#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --tmp=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail

repo=${1:?Pass the absolute checkout path}
model=${2:?Pass a model alias}
mode=${3:-text}
tp=${4:-1}
source_mode=${5:-shared}
gpu_family=${6:-}
operation=${7:-validate}
port=${8:-0}
: "${PACE_INFERENCE_STATE:?Set the external state directory before submission}"
python=${PACE_PYTHON:-python3}
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACEHUB_API_TOKEN
case "$source_mode" in
  shared) fetch_flags=() ;;
  node-fetch) fetch_flags=(--node-fetch) ;;
  *) echo "Unknown checkpoint source $source_mode" >&2; exit 2 ;;
esac
export PACE_EXPECTED_GPU_FAMILY="$gpu_family"
export PACE_EXPECTED_GPUS="$tp"
case "$model" in
  lightning|super-vl)
    # Match compiler and CUDA headers for FlashInfer's Hopper Mamba SSU JIT.
    if [[ -z "${PACE_CUDA_TOOLKIT:-}" ]]; then
      if ! type module >/dev/null 2>&1; then
        source "${PACE_MODULE_INIT:-/etc/profile.d/z00_lmod_pace.sh}"
      fi
      module load "${PACE_CUDA_MODULE:-cuda/13.0.1}"
      export PACE_CUDA_TOOLKIT="$CUDA_HOME"
    fi
    ;;
esac

umask 077
run_dir=$(mktemp -d "$PACE_INFERENCE_STATE/run-${SLURM_JOB_ID}.XXXXXXXX")
printf 'Host %s, Slurm GPU list %s, CUDA mask %s\n' \
  "$(hostname)" "${SLURM_JOB_GPUS:-unset}" "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,compute_cap --format=csv
nvidia-smi topo -m
findmnt -T "${TMPDIR:-/tmp}"
df -h "${TMPDIR:-/tmp}" /dev/shm
[[ ! -r /sys/fs/cgroup/memory.max ]] || cat /sys/fs/cgroup/memory.max
job_root=$(mktemp -d "${TMPDIR:-/tmp}/pace-job-${SLURM_JOB_ID}.XXXXXXXX")
export TMPDIR="$job_root"
cleanup() {
  local status=$?
  if [[ -n "${manager_pid:-}" ]]; then
    kill -TERM "$manager_pid" 2>/dev/null || true
    wait "$manager_pid" 2>/dev/null || true
  fi
  if (( status != 0 )); then
    find "$job_root" -path '*/cache/humming/*/stderr.log' -type f -size -1M \
      -print -exec cat {} \; > "$run_dir/humming-errors.log" || true
  fi
  rm -rf -- "$job_root"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
"$python" -B "$repo/pace_inference.py" preflight "$model" 2>&1 | tee "$run_dir/preflight.log"

"$python" -B "$repo/managed_job.py" "$model" "$mode" "$tp" "$operation" "$port" "$run_dir" "${fetch_flags[@]}" &
manager_pid=$!
wait "$manager_pid"
