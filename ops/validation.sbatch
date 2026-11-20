#!/bin/bash
# Submit from a separately prepared pilot checkout after serving acceptance.
#SBATCH --job-name=gptoss_pipeline
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -C H100|H200
#SBATCH --mem=192G
#SBATCH -c 24
#SBATCH -t 02:00:00
#SBATCH --no-requeue
#SBATCH -o validation_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ops/run.env
export JB_PIPELINE_PROBE=1 JB_VALIDATE_CONTINUATION=1 JB_RESUME_MODE=new
export LLMGE_START_POPULATION_SIZE=8 LLMGE_POPULATION_SIZE=8 LLMGE_NUM_GENERATIONS=2
export JB_SMOKE_ONLY=0
timeout --signal=TERM --kill-after=45s 110m "$JB_EXPERIMENT_PYTHON" -m src.run_status supervise \
    --status "$RUN_ID/status.json" --run-id "$RUN_ID" --interval 300 --max-hours 1.8 \
    -- bash run_jailbreak.sbatch
