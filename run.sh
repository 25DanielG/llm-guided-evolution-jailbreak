#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 8:00:00
#SBATCH --mem 16G
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --output=run_job_outputs/islands/slurm-%j.out
echo "launching LLM Guided Evolution"
hostname
module load uv

: "${JB_VLLM_API_KEY:?JB_VLLM_API_KEY must be set}"
export JB_JUDGE_MODE="${JB_JUDGE_MODE:-llm}"
export JB_TARGET_SERVED_NAME="${JB_TARGET_SERVED_NAME:-target}"
export JB_JUDGE_SERVED_NAME="${JB_JUDGE_SERVED_NAME:-judge}"
: "${JB_BEHAVIORS_PATH:?JB_BEHAVIORS_PATH must identify an approved benchmark CSV}"
export JB_BEHAVIORS_PATH
export JB_N_BEHAVIORS_PER_EVAL="${JB_N_BEHAVIORS_PER_EVAL:-4}"
export JB_MAX_CONCURRENCY="${JB_MAX_CONCURRENCY:-2}"
export JB_CACHE_DIR="${JB_CACHE_DIR:-${HOME}/scratch/jb_cache}"
export JB_REQUEST_TIMEOUT="${JB_REQUEST_TIMEOUT:-120}"
export JB_SERVER_READY_TIMEOUT="${JB_SERVER_READY_TIMEOUT:-1200}"

export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

export SERVER_HOSTNAME=$(hostname)
uv run python run_improved.py jailbreak_test
