#!/bin/bash
# Resources, account, partition, QoS and output are supplied explicitly by launch_corrected.py.
#SBATCH --nodes=1
#SBATCH --signal=B:USR1@900
#SBATCH --no-requeue
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ops/corrected_common.env
source ops/run.env
"$JB_EXPERIMENT_PYTHON" ops/corrected_checks.py preflight
exec "$JB_EXPERIMENT_PYTHON" -m src.run_status supervise --status "$RUN_ID/status.json" \
    --run-id "$RUN_ID" --interval 60 --max-hours "$JB_MAX_HOURS" -- bash ops/corrected_payload.sh
