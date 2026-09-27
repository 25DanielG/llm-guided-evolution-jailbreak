#!/bin/bash
#SBATCH --job-name=jb_run
#SBATCH -G 3
#SBATCH -t 4:00:00
#SBATCH --mem=144G
#SBATCH -c 18
#SBATCH -o jb_run_%j.out

# Target weaker model, for a first ASR>0 validation and judge (a more capable model,
# to avoid a model judging its own outputs. Each get their own vLLM server on a different
# GPU. Mutator is HF server.
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
export LOCAL=true LOCAL_LLM=true JB_JUDGE_MODE=llm JB_FULL_BEHAVIORS=1
export JB_TARGET_MODEL_PATH="$TARGET_MODEL"
export JB_JUDGE_MODEL_PATH="$JUDGE_MODEL"
export JB_MUTATOR_MODEL_PATH="$JUDGE_MODEL"

HOST=$(hostname)

IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [ "${#GPU_IDS[@]}" -lt 3 ]; then
    echo "ERROR: expected 3 GPUs from Slurm, got: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
TARGET_DEV="${GPU_IDS[0]}"
JUDGE_DEV="${GPU_IDS[1]}"
MUTATOR_DEV="${GPU_IDS[2]}"
echo "Slurm gave GPUs: $CUDA_VISIBLE_DEVICES -> target=$TARGET_DEV judge=$JUDGE_DEV mutator=$MUTATOR_DEV"

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

# judge vLLM
JUDGE_PID=$(start_with_retry "judge vLLM" sota/Jailbreak/judge_vllm.out \
    env CUDA_VISIBLE_DEVICES="$JUDGE_DEV" vllm serve "$JUDGE_MODEL" \
    --served-model-name judge --host 0.0.0.0 --port 8002 \
    --gpu-memory-utilization 0.9) || exit 1
echo "$HOST:8002" > sota/Jailbreak/judge_host.log
export JB_JUDGE_VLLM_URL="http://$HOST:8002/v1"
export JB_JUDGE_SERVED_NAME=judge

# mutator (HF server)
export SERVER_HOSTNAME=$HOST
echo "$HOST" > hostname.log
MUT_PID=$(start_with_retry "mutator" mutator.log \
    env CUDA_VISIBLE_DEVICES="$MUTATOR_DEV" uv run python -m uvicorn server:app \
    --host "$HOST" --port 8137) || exit 1

cleanup() { kill "$TARGET_PID" "$JUDGE_PID" "$MUT_PID" 2>/dev/null || true; }
trap cleanup EXIT

# wait for all three to be ready; fail fast if a server process dies instead of
# spinning on curl for the whole walltime (bit us on a MIG node above).
wait_or_die() {
    local url="$1" pid="$2" label="$3"
    while ! curl -sf "$url" >/dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: $label process ($pid) died before becoming ready. See its log." >&2
            exit 1
        fi
        sleep 5
    done
}
echo "Waiting for target vLLM..."
wait_or_die "http://$HOST:8001/v1/models" "$TARGET_PID" "target vLLM"
echo "Target ready. Waiting for judge vLLM..."
wait_or_die "http://$HOST:8002/v1/models" "$JUDGE_PID" "judge vLLM"
echo "Judge ready. Waiting for mutator..."
wait_or_die "http://$HOST:8137/" "$MUT_PID" "mutator"
echo "Mutator ready. Starting evolution."

# run's checkpoint/global dirs
RUN_ID="${RUN_ID:-jb_${SLURM_JOB_ID}}"
uv run python run_improved.py --checkpoints "${RUN_ID}/ckpt" --global_path "${RUN_ID}/global" \
    --llm_model llama3 --prompt_group Jailbreak/traits

echo "=== run complete ==="
