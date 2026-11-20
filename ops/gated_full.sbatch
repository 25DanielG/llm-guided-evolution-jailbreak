#!/bin/bash
#SBATCH --job-name=gpt20_full_7001848
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -C H100|H200
#SBATCH --mem=192G
#SBATCH -c 24
#SBATCH -t 12:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --no-requeue
#SBATCH -o full_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ops/run.env
chain=$1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export JB_PIPELINE_PROBE=0 JB_VALIDATE_CONTINUATION=0 JB_CHECKPOINT_FILE=''
"$JB_EXPERIMENT_PYTHON" ops/launch_checks.py --chain "$chain" --phase pipeline
# The batch PID becomes the supervisor so it receives Slurm's USR1 warning.
[[ ! -e "$RUN_ID/status.json" && ! -d "$RUN_ID/ckpt" ]] || {
    echo 'Fresh full RUN_ID already exists; refusing to restart' >&2; exit 2;
}
exec "$JB_EXPERIMENT_PYTHON" -m src.run_status supervise --status "$RUN_ID/status.json" \
    --run-id "$RUN_ID" --interval 300 --max-hours 12 -- bash ops/full_payload.sh
