#!/bin/bash
# Production topology: 1 GPU target, 1 GPU judge, 1 GPU reporting (HarmBench,
# verification/audit only), 1 GPU mutator.
# Adjust Slurm allocation and groups together if using a different GPU type.
#SBATCH --job-name=jb_run
#SBATCH -G 4
#SBATCH -t 4:00:00
#SBATCH --mem=192G
#SBATCH -c 24
#SBATCH -o jb_run_%j.out

# Target weaker model, for a first ASR>0 validation; judge + reporting are
# separate models from the target (and from each other) to avoid a model
# judging its own outputs. Each get their own vLLM server on a different GPU.
# Mutator is HF server.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"          # uv + vllm CLIs
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"

# FlashInfer's sampler JIT-compiles with 'ninja' (absent on nodes) -> use native sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0

TARGET_MODEL="$HOME/scratch/llm_storage/Mistral-7B-Instruct-v0.2"
JUDGE_MODEL="$HOME/scratch/llm_storage/Llama-3.1-8B-Instruct"
REPORTING_MODEL="$HOME/scratch/llm_storage/HarmBench-Llama-2-13b-cls"
export LOCAL=true LOCAL_LLM=true JB_JUDGE_MODE=llm JB_FULL_BEHAVIORS=1
export JB_TARGET_MODEL_PATH="$TARGET_MODEL"
export JB_JUDGE_MODEL_PATH="$JUDGE_MODEL"
export JB_REPORTING_MODEL_PATH="$REPORTING_MODEL"
export JB_REPORTING_SERVED_NAME="cais/HarmBench-Llama-2-13b-cls"
export JB_MUTATOR_MODEL_PATH="$JUDGE_MODEL"

HOST=$(hostname)

IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [ "${#GPU_IDS[@]}" -lt 4 ]; then
    echo "ERROR: expected 4 GPUs from Slurm, got: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
TARGET_DEV="${GPU_IDS[0]}"
JUDGE_DEV="${GPU_IDS[1]}"
REPORTING_DEV="${GPU_IDS[2]}"
MUTATOR_DEV="${GPU_IDS[3]}"
echo "Slurm gave GPUs: $CUDA_VISIBLE_DEVICES -> target=$TARGET_DEV judge=$JUDGE_DEV reporting=$REPORTING_DEV mutator=$MUTATOR_DEV"

# Retry a service's own startup a few times before giving up
start_with_retry() {
    local label="$1" logfile="$2"; shift 2
    local attempt pid
    for attempt in 1 2 3; do
        : > "$logfile"
        ( "$@" ) > "$logfile" 2>&1 &
        pid=$!
        sleep 20
        if kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
        echo "$label died on attempt $attempt (prob CUDA-init race); retrying..." >&2
        wait "$pid" 2>/dev/null
        sleep 10
    done
    echo "ERROR: $label failed to start after 3 attempts" >&2
    return 1
}

# target vLLM (weaker model)
TARGET_PID=$(start_with_retry "target vLLM" sota/Jailbreak/target_vllm.out \
    env CUDA_VISIBLE_DEVICES="$TARGET_DEV" vllm serve "$TARGET_MODEL" \
    --served-model-name target --host 0.0.0.0 --port 8001 \
    --gpu-memory-utilization 0.9) || exit 1
echo "$HOST:8001" > sota/Jailbreak/target_host.log
export JB_TARGET_VLLM_URL="http://$HOST:8001/v1"

# judge vLLM (in-loop rubric judge; fitness-relevant)
JUDGE_PID=$(start_with_retry "judge vLLM" sota/Jailbreak/judge_vllm.out \
    env CUDA_VISIBLE_DEVICES="$JUDGE_DEV" vllm serve "$JUDGE_MODEL" \
    --served-model-name judge --host 0.0.0.0 --port 8002 \
    --gpu-memory-utilization 0.9) || exit 1
echo "$HOST:8002" > sota/Jailbreak/judge_host.log
export JB_JUDGE_VLLM_URL="http://$HOST:8002/v1"
export JB_JUDGE_SERVED_NAME=judge

# reporting vLLM (HarmBench classifier; verification/audit only, never fitness)
REPORTING_PID=$(start_with_retry "reporting vLLM" sota/Jailbreak/reporting_vllm.out \
    env CUDA_VISIBLE_DEVICES="$REPORTING_DEV" vllm serve "$REPORTING_MODEL" \
    --served-model-name "$JB_REPORTING_SERVED_NAME" --host 0.0.0.0 --port 8003 \
    --gpu-memory-utilization 0.9) || exit 1
echo "$HOST:8003" > sota/Jailbreak/reporting_host.log
export JB_REPORTING_VLLM_URL="http://$HOST:8003/v1"

MUT_PID=$(start_with_retry "mutator" mutator.log \
    env CUDA_VISIBLE_DEVICES="$MUTATOR_DEV" uv run python -m uvicorn server:app \
    --host "$HOST" --port 8137) || exit 1

cleanup() { kill "$TARGET_PID" "$JUDGE_PID" "$REPORTING_PID" "$MUT_PID" 2>/dev/null || true; }
trap cleanup EXIT

# wait for all four to be ready; fail fast if a server process dies instead of
# spinning on curl for the whole walltime (bit us on a MIG node above).
wait_or_die() {
    local url="$1" pid="$2" label="$3" logfile="$4"
    while ! curl -sf "$url" >/dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$label died before readiness. Recent log:" >&2
            tail -n 35 "$logfile" >&2
            exit 1
        fi
        sleep 5
    done
}
echo "Waiting for target vLLM..."
wait_or_die "http://$HOST:8001/v1/models" "$TARGET_PID" "target vLLM" sota/Jailbreak/target_vllm.out
echo "Target ready. Waiting for judge vLLM..."
wait_or_die "http://$HOST:8002/v1/models" "$JUDGE_PID" "judge vLLM" sota/Jailbreak/judge_vllm.out
echo "Judge ready. Waiting for reporting vLLM..."
wait_or_die "http://$HOST:8003/v1/models" "$REPORTING_PID" "reporting vLLM" sota/Jailbreak/reporting_vllm.out
echo "Reporting ready. Waiting for mutator..."
wait_or_die "http://$HOST:8137/" "$MUT_PID" "mutator" mutator.log
echo "Mutator ready. Starting evolution."

# run's checkpoint/global dirs
RUN_ID="${RUN_ID:-jb_${SLURM_JOB_ID}}"
uv run python run_improved.py --checkpoints "${RUN_ID}/ckpt" --global_path "${RUN_ID}/global" \
    --llm_model llama3 --prompt_group Jailbreak/traits --run-id "$RUN_ID"

echo "=== run complete ==="
