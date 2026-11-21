#!/bin/bash
# Grade a finished run with the team's fixed post-hoc judge panel (guide P(full) + scored checklist on
# Llama-3.3-70B, next to the run's HarmBench and in-loop verdicts). Writes numbers only, never response text.
# Two steps, from the repo root:
#   1. sbatch run_reporting_judge.sbatch --candidates <run>/candidates.jsonl --out <run>/reporting.csv
#   2. sbatch run_grade_run.sbatch --reporting <run>/reporting.csv --out-dir <run>/grades
# Step 2's arguments pass straight through to sota/Jailbreak/grade_run.py (--generations last|all|0,8,16,
# --sample-per-generation K, --seed, --workers, --run-id). The default grades the last generation; grading
# every generation of a full run can take hours, so sample or ask for a longer -t.
# The judge is served like run_jailbreak.sbatch's 70B guide judge (served name "judge", TP 2, eager,
# --max-model-len 4096), but on a free localhost port so it cannot collide with another job on the node.
# Override the weights with JB_GRADE_JUDGE_MODEL_PATH (the panel was validated only on this 70B).
#SBATCH --job-name=jb_grade
#SBATCH -N 1
#SBATCH -G 2
#SBATCH -C "A100-80GB|H100|H200"
#SBATCH -t 4:00:00
#SBATCH --mem=160G
#SBATCH -c 16
#SBATCH -o jb_grade_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"  # uv + vllm CLIs
# FlashInfer's sampler and multi-GPU all-reduce JIT-build with 'ninja' (absent on nodes): use vLLM's own.
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 HF_HUB_OFFLINE=1

PY="$HOME/.local/share/uv/tools/vllm/bin/python"
MODEL="${JB_GRADE_JUDGE_MODEL_PATH:-/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct}"
[[ -d "$MODEL" ]] || { echo "ERROR: judge weights not found at $MODEL" >&2; exit 1; }
# Check the arguments and the selection before loading 140GB of weights.
"$PY" sota/Jailbreak/grade_run.py "$@" --dry-run
# --help / --dry-run asked for only that: stop before holding 2 GPUs for the judge.
for a in "$@"; do case "$a" in -h|--help|--dry-run) exit 0;; esac; done

PORT="${JB_GRADE_JUDGE_PORT:-$("$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')}"
LOG="jb_grade_judge_vllm_${SLURM_JOB_ID}.log"
"${JB_VLLM_BIN:-vllm}" serve "$MODEL" --served-model-name judge --host 127.0.0.1 --port "$PORT" \
    --gpu-memory-utilization 0.9 --tensor-parallel-size 2 --enforce-eager --max-model-len 4096 > "$LOG" 2>&1 &
JUDGE_PID=$!
trap 'kill "$JUDGE_PID" 2>/dev/null || true' EXIT

READY_TIMEOUT="${JB_SERVER_READY_TIMEOUT:-2400}"
deadline=$((SECONDS + READY_TIMEOUT))
until curl -sf --max-time 10 "http://127.0.0.1:$PORT/v1/models" >/dev/null; do  # vLLM binds the port before loading
    if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
        echo "ERROR: judge vLLM exited before it was ready. Last lines of $LOG:" >&2
        tail -n 30 "$LOG" >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "ERROR: judge vLLM not ready after ${READY_TIMEOUT}s (JB_SERVER_READY_TIMEOUT). Last lines of $LOG:" >&2
        tail -n 30 "$LOG" >&2
        exit 1
    fi
    sleep 10
done
echo "Judge ready on port $PORT after ${SECONDS}s: $MODEL"

export JB_JUDGE_VLLM_URL="http://127.0.0.1:$PORT/v1" JB_JUDGE_SERVED_NAME=judge JB_JUDGE_MODEL_PATH="$MODEL"
export JB_CACHE_DIR="${TMPDIR:-/tmp}/jb_grade_cache_${SLURM_JOB_ID}"  # fresh per job: no verdict is reused
export JB_REQUEST_TIMEOUT="${JB_REQUEST_TIMEOUT:-300}"
"$PY" sota/Jailbreak/grade_run.py "$@"
echo "=== grading complete ==="
