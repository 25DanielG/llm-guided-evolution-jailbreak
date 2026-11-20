#!/bin/bash
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
out="$(dirname "$PWD")/output/job-$SLURM_JOB_ID"
mkdir "$out"
hostname -f > "$out/hostname.log"

export JB_SERVING_ENV=/storage/ice1/8/3/pablo3/vllm-0.17-py312-f88321e
export JB_VLLM_BIN="$JB_SERVING_ENV/bin/vllm"
export JB_EXPERIMENT_PYTHON=/storage/ice1/8/3/pablo3/llmge-redsea/.venv/bin/python
export PYTHONPATH="$PWD:$PWD/sota/Jailbreak"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH=/opt/slurm/current/bin:/usr/local/bin:/usr/bin:/bin
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_TORCH=1
export VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_USE_FLASHINFER_SAMPLER=0
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export JB_SERVER_READY_TIMEOUT=3600 VIP_STAGE_TIMEOUT=3600 VIP_AUX_READY_TIMEOUT=1800
export JB_REQUEST_TIMEOUT=120 JB_MAX_CONCURRENCY=1
export JB_JUDGE_MODE=llm JB_JUDGE_PROMPT=guide JB_RUN_JUDGE_TP=2 JB_RUN_JUDGE_MAX_LEN=4096
export JB_TARGET_MAX_NEW_TOKENS=512 JB_TARGET_TEMPERATURE=0.0 JB_MUTATOR_MAX_NEW_TOKENS=850
export JB_SELECTION=nsga2 JB_STRATIFIED=1 JB_FULL_BEHAVIORS=0 JB_N_BEHAVIORS_PER_EVAL=50
export JB_BEHAVIORS_PATH="$PWD/sota/Jailbreak/behaviors/curated_tiered_jb_strat_base1.csv"
export JB_SMOKE_ONLY=1 RUN_ID="$out"
export JB_TARGET_PROFILE=gpt-oss-20b
exec "$JB_SERVING_ENV/bin/python" -uB ops/guide70b_validation/validate.py "$out"
