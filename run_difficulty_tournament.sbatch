#!/bin/bash
#SBATCH --job-name=difficulty-tournament
#SBATCH --account=coc
#SBATCH --qos=coe-ice
#SBATCH --partition=ice-gpu,coe-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:5
#SBATCH --constraint=H200
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --no-requeue
#SBATCH --output=difficulty-%A_%a.out
set -euo pipefail
umask 027
stage="${1:?Usage: SCRIPT initial|run CAMPAIGN [new|required]}"
campaign="${2:?Supply the campaign directory}"
resume="${3:-new}"
: "${SLURM_ARRAY_TASK_ID:?Submit as a job array}"
: "${SLURM_SUBMIT_DIR:?Missing submission directory}"
: "${CUDA_VISIBLE_DEVICES:?Missing allocated GPUs}"
case "$stage" in initial|run) ;; *) exit 2 ;; esac
case "$resume" in new|required) ;; *) exit 2 ;; esac
cd "$SLURM_SUBMIT_DIR"
campaign="$(realpath -e "$campaign")"
python_bin="${DT_PYTHON:-$SLURM_SUBMIT_DIR/.venv/bin/python}"
[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 2; }
module load cuda
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
attempt="$campaign/attempts/$stage-$SLURM_ARRAY_TASK_ID/$SLURM_JOB_ID"
mkdir -p "$attempt"
export DT_STOP_FILE="$attempt/stop-requested" DT_ATTEMPT_DIR="$attempt"
child_pid=""
request_stop() { touch "$DT_STOP_FILE"; }
terminate() {
    request_stop
    if [[ -n "$child_pid" ]]; then kill -TERM -- "-$child_pid" 2>/dev/null || true; fi
}
cleanup() {
    if [[ -n "$child_pid" ]]; then kill -TERM -- "-$child_pid" 2>/dev/null || true; fi
}
trap request_stop USR1
trap terminate TERM INT
trap cleanup EXIT
args=(-m analysis.experiments.difficulty_tournament "$stage"
      --campaign "$campaign" --index "$SLURM_ARRAY_TASK_ID" --resume "$resume")
setsid "$python_bin" "${args[@]}" >"$attempt/driver.log" 2>&1 &
child_pid=$!
set +e
while true; do
    wait "$child_pid"
    result=$?
    kill -0 "$child_pid" 2>/dev/null || break
done
set -e
if (( result == 75 )); then
    echo "Committed checkpoint; resume this array index with required."
else
    echo "Stage exit code: $result; logs: $attempt"
fi
exit "$result"
