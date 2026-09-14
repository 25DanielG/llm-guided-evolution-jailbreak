#!/bin/bash
#SBATCH --job-name=jb_run
#SBATCH -G 2
#SBATCH -t 4:00:00
#SBATCH --mem=96G
#SBATCH -c 12
#SBATCH -o jb_run_%j.out

# single vLLM both target and judge
# on GPU0, mutator server on GPU1, then the evolution loop
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load cuda 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"          # uv + vllm CLIs
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"

# FlashInfer's sampler JIT-compiles with 'ninja' (absent on nodes) -> use native sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0

MODELS="$HOME/scratch/llm_storage/Llama-3.1-8B-Instruct"
export LOCAL=true LOCAL_LLM=true JB_JUDGE_MODE=llm
export JB_TARGET_MODEL_PATH="$MODELS"
export JB_MUTATOR_MODEL_PATH="$MODELS"

HOST=$(hostname)

IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [ "${#GPU_IDS[@]}" -lt 2 ]; then
    echo "ERROR: expected 2 GPUs from Slurm, got: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
TARGET_DEV="${GPU_IDS[0]}"
MUTATOR_DEV="${GPU_IDS[1]}"
echo "Slurm gave GPUs: $CUDA_VISIBLE_DEVICES -> target=$TARGET_DEV mutator=$MUTATOR_DEV"

# single vLLM (target); judge reuses the same endpoint
CUDA_VISIBLE_DEVICES="$TARGET_DEV" vllm serve "$MODELS" \
    --served-model-name target --host 0.0.0.0 --port 8001 \
    --gpu-memory-utilization 0.9 > sota/Jailbreak/target_vllm.out 2>&1 &
VLLM_PID=$!
echo "$HOST:8001" > sota/Jailbreak/target_host.log
export JB_TARGET_VLLM_URL="http://$HOST:8001/v1"
export JB_JUDGE_VLLM_URL="http://$HOST:8001/v1"
export JB_JUDGE_SERVED_NAME=target

# mutator (HF server)
export SERVER_HOSTNAME=$HOST
echo "$HOST" > hostname.log
CUDA_VISIBLE_DEVICES="$MUTATOR_DEV" uv run python -m uvicorn server:app \
    --host "$HOST" --port 8137 > mutator.log 2>&1 &
MUT_PID=$!

cleanup() { kill "$VLLM_PID" "$MUT_PID" 2>/dev/null || true; }
trap cleanup EXIT

# wait for both to be ready; fail fast if a server process dies instead of
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
echo "Waiting for vLLM..."
wait_or_die "http://$HOST:8001/v1/models" "$VLLM_PID" "vLLM"
echo "vLLM ready. Waiting for mutator..."
wait_or_die "http://$HOST:8137/" "$MUT_PID" "mutator"
echo "Mutator ready. Starting evolution."

RUN_ID="jb_${SLURM_JOB_ID}"
uv run python run_improved.py --checkpoints "${RUN_ID}/ckpt" --global_path "${RUN_ID}/global" \
    --llm_model llama3 --prompt_group Jailbreak/traits

echo "=== run complete ==="
