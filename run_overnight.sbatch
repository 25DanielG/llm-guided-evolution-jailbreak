#!/bin/bash
# Override --time=12:00:00 only after checking live partition AND account/QoS limits.
# The 3-hour default fits five GPUs within the 960 GPU-minute coe-ice limit.
#SBATCH --job-name=jb_supervised
#SBATCH -N 1
#SBATCH -G 5
#SBATCH --mem=192G
#SBATCH -c 24
#SBATCH -t 3:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --no-requeue
#SBATCH -o jb_supervised_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${SLURM_JOB_ID:?This launcher requires an existing Slurm allocation}"
: "${RUN_ID:?Set a stable RUN_ID; do not generate a new ID when resuming}"
: "${JB_EXPERIMENT_PYTHON:?Absolute Python 3.12 from the existing experiment environment}"
if [[ -n "${JB_PREVIOUS_STATUS:-}" ]]; then
    "$JB_EXPERIMENT_PYTHON" -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s.get("run_id")==sys.argv[2] and s.get("state")=="COMPLETED" else 1)' \
        "$JB_PREVIOUS_STATUS" "${JB_PREVIOUS_RUN_ID:?Set JB_PREVIOUS_RUN_ID with JB_PREVIOUS_STATUS}" || {
        echo "Previous run has not completed: $JB_PREVIOUS_STATUS" >&2
        exit 2
    }
fi
if [[ "${JB_RESUME_MODE:-}" == required && -f "$RUN_ID/status.json" ]] &&
   "$JB_EXPERIMENT_PYTHON" -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s.get("run_id")==sys.argv[2] and s.get("state")=="COMPLETED" else 1)' "$RUN_ID/status.json" "$RUN_ID"; then
    echo "Run $RUN_ID is already complete; skipping redundant continuation"
    exit 0
fi
CHECKPOINT_ARGS=()
[[ -z "${JB_CHECKPOINT_FILE:-}" ]] || CHECKPOINT_ARGS=(--checkpoint-file "$JB_CHECKPOINT_FILE")
case "${JB_RESUME_MODE:-}" in
    new)
        if [[ -e "$RUN_ID/status.json" || -d "$RUN_ID/ckpt" ]]; then
            echo 'New run ID already exists; inspect it and use required resume' >&2
            exit 2
        fi
        ;;
    required)
        "$JB_EXPERIMENT_PYTHON" -m src.checkpointing --checkpoints "$RUN_ID/ckpt" \
            --global-path "$RUN_ID/global" "${CHECKPOINT_ARGS[@]}"
        ;;
    *) echo 'Set JB_RESUME_MODE=new or required explicitly; auto is forbidden here' >&2; exit 2 ;;
esac
export LLMGE_NUM_GENERATIONS="${LLMGE_NUM_GENERATIONS:-17}"
export JB_SMOKE_ONLY=0 JB_BOOTSTRAP=0
exec "$JB_EXPERIMENT_PYTHON" -m src.run_status supervise --status "$RUN_ID/status.json" \
    --run-id "$RUN_ID" --interval 300 --max-hours 12 -- bash run_jailbreak.sbatch
