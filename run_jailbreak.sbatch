#!/bin/bash
# Production topology: 2 GPUs mutator, 2 GPUs judge, 1 target, 1 reporter.
# Adjust Slurm allocation and groups together if using a different GPU type.
#SBATCH --job-name=jb_run
#SBATCH -G 6
#SBATCH -t 2:40:00
#SBATCH --mem=192G
#SBATCH -c 24
#SBATCH -o jb_run_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
VLLM_TOOL_BIN="$(dirname "$(readlink -f "$(command -v vllm)")")"
export PATH="$VLLM_TOOL_BIN:$PATH"
if ! command -v ninja >/dev/null 2>&1; then
    echo "vLLM requires ninja to compile GPU kernels, but ninja is not on PATH" >&2
    exit 1
fi
echo "Using ninja: $(command -v ninja) ($(ninja --version))"
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LOCAL=true LOCAL_LLM=true
export LLMGE_CONFIG=jailbreak
export LLMGE_PORT=8137
export JB_BEHAVIORS_PATH="$PWD/sota/Jailbreak/behaviors/curated.csv"
export JB_FULL_BEHAVIORS=1
BEHAVIOR_COUNT="$(python - "$JB_BEHAVIORS_PATH" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="") as file:
    rows = list(csv.DictReader(file))
print(len(rows) if all(row.get("source") == "advbench" for row in rows) else 0)
PY
)"
if [[ "$BEHAVIOR_COUNT" -ne 30 ]]; then
    echo "Expected 30 AdvBench behaviors in $JB_BEHAVIORS_PATH; found $BEHAVIOR_COUNT" >&2
    exit 1
fi
RUN_ID="${JB_RESUME_RUN_ID:-jb_${SLURM_JOB_ID}}"
EVAL_SCOPE="full30:$(sha256sum "$JB_BEHAVIORS_PATH" | cut -d' ' -f1)"
SCOPE_FILE="$RUN_ID/evaluation_scope.txt"
if [[ -n "${JB_RESUME_RUN_ID:-}" ]]; then
    if ! compgen -G "$RUN_ID/ckpt/checkpoint_gen_*.pkl" >/dev/null ||
       ! compgen -G "$RUN_ID/global/global_gen_*.pkl" >/dev/null; then
        echo "Missing population or global checkpoint for requested resume run: $RUN_ID" >&2
        exit 1
    fi
    if [[ ! -f "$SCOPE_FILE" || "$(cat "$SCOPE_FILE")" != "$EVAL_SCOPE" ]]; then
        echo "Resume run $RUN_ID was not checkpointed with this full 30-behavior evaluation set" >&2
        exit 1
    fi
else
    mkdir -p "$RUN_ID"
    printf '%s\n' "$EVAL_SCOPE" > "$SCOPE_FILE"
fi
echo "Evolution run ID: $RUN_ID; evaluating all $BEHAVIOR_COUNT AdvBench behaviors"
export LLMGE_RUN_ID="$RUN_ID"
export JB_CACHE_DIR="$PWD/$RUN_ID/cache"
mkdir -p "$JB_CACHE_DIR"

export JB_TARGET_MODEL_PATH="${JB_TARGET_MODEL_PATH:-$HOME/scratch/llm_storage/Llama-3.1-8B-Instruct}"
export JB_MUTATOR_MODEL_PATH="${JB_MUTATOR_MODEL_PATH:-/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct}"
export JB_JUDGE_MODEL_PATH="${JB_JUDGE_MODEL_PATH:-/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct}"
export JB_REPORTING_MODEL_PATH="${JB_REPORTING_MODEL_PATH:-$HOME/scratch/llm_storage/HarmBench-Llama-2-13b-cls}"
export JB_TARGET_SERVED_NAME="${JB_TARGET_SERVED_NAME:-target}"
export JB_JUDGE_SERVED_NAME="${JB_JUDGE_SERVED_NAME:-meta-llama/Llama-3.3-70B-Instruct}"
export JB_REPORTING_SERVED_NAME="${JB_REPORTING_SERVED_NAME:-cais/HarmBench-Llama-2-13b-cls}"

for model_path in "$JB_TARGET_MODEL_PATH" "$JB_MUTATOR_MODEL_PATH" "$JB_JUDGE_MODEL_PATH" "$JB_REPORTING_MODEL_PATH"; do
    if [[ ! -f "$model_path/config.json" ]]; then
        echo "Missing checkpoint: $model_path" >&2
        exit 1
    fi
done
if [[ "$JB_TARGET_MODEL_PATH" == "$JB_JUDGE_MODEL_PATH" || "$JB_TARGET_MODEL_PATH" == "$JB_REPORTING_MODEL_PATH" || "$JB_JUDGE_MODEL_PATH" == "$JB_REPORTING_MODEL_PATH" ]]; then
    echo "Target, judge, and reporting checkpoints must be distinct" >&2
    exit 1
fi

IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#GPU_IDS[@]}" -lt 6 ]]; then
    echo "Expected six allocated GPUs, got: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
echo "Slurm GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,uuid,compute_mode --format=csv
fi
VLLM_PYTHON="$VLLM_TOOL_BIN/python"
for device in "${GPU_IDS[@]}"; do
    echo "Checking CUDA device $device"
    CUDA_VISIBLE_DEVICES="$device" "$VLLM_PYTHON" -c 'import torch; torch.cuda.set_device(0); print("GPU:", torch.cuda.get_device_name(0), "free/total bytes:", torch.cuda.mem_get_info(), flush=True)' || {
        echo "CUDA preflight failed for device $device; no model servers were started" >&2
        exit 1
    }
done
HOST=$(hostname)
export JB_TARGET_VLLM_URL="http://$HOST:8001/v1"
export JB_JUDGE_VLLM_URL="http://$HOST:8002/v1"
export JB_REPORTING_VLLM_URL="http://$HOST:8003/v1"
echo "$HOST:8001" > sota/Jailbreak/target_host.log
echo "$HOST:8002" > sota/Jailbreak/judge_host.log
echo "$HOST:8003" > sota/Jailbreak/reporting_host.log

CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" vllm serve "$JB_TARGET_MODEL_PATH" --served-model-name "$JB_TARGET_SERVED_NAME" --host 0.0.0.0 --port 8001 --max-model-len "${JB_TARGET_MAX_MODEL_LEN:-4096}" --gpu-memory-utilization 0.9 > sota/Jailbreak/target_vllm.out 2>&1 &
TARGET_PID=$!
CUDA_VISIBLE_DEVICES="${GPU_IDS[1]},${GPU_IDS[2]}" vllm serve "$JB_JUDGE_MODEL_PATH" --served-model-name "$JB_JUDGE_SERVED_NAME" --tensor-parallel-size 2 --host 0.0.0.0 --port 8002 --max-model-len "${JB_JUDGE_MAX_MODEL_LEN:-4096}" --gpu-memory-utilization 0.9 > sota/Jailbreak/judge_vllm.out 2>&1 &
JUDGE_PID=$!
CUDA_VISIBLE_DEVICES="${GPU_IDS[3]}" vllm serve "$JB_REPORTING_MODEL_PATH" --served-model-name "$JB_REPORTING_SERVED_NAME" --host 0.0.0.0 --port 8003 --gpu-memory-utilization 0.9 > sota/Jailbreak/reporting_vllm.out 2>&1 &
REPORTING_PID=$!
export SERVER_HOSTNAME=$HOST
echo "$HOST" > hostname.log
CUDA_VISIBLE_DEVICES="${GPU_IDS[4]},${GPU_IDS[5]}" uv run python -m uvicorn server:app --host "$HOST" --port 8137 > mutator.log 2>&1 &
MUTATOR_PID=$!
cleanup() { kill "$TARGET_PID" "$JUDGE_PID" "$REPORTING_PID" "$MUTATOR_PID" 2>/dev/null || true; }
trap cleanup EXIT
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
wait_or_die "$JB_TARGET_VLLM_URL/models" "$TARGET_PID" target sota/Jailbreak/target_vllm.out
wait_or_die "$JB_JUDGE_VLLM_URL/models" "$JUDGE_PID" judge sota/Jailbreak/judge_vllm.out
wait_or_die "$JB_REPORTING_VLLM_URL/models" "$REPORTING_PID" reporter sota/Jailbreak/reporting_vllm.out
wait_or_die "http://$HOST:8137/" "$MUTATOR_PID" mutator mutator.log
uv run python - <<'PY'
import sys
sys.path.insert(0, "sota/Jailbreak")
import jb_config, vllm_client
for url, model in ((jb_config.target_base_url(), jb_config.target_model_name()),
                   (jb_config.judge_base_url(), jb_config.judge_model_name()),
                   (jb_config.reporting_base_url(), jb_config.reporting_model_name())):
    vllm_client.assert_model(url, model)
PY
uv run python run_improved.py --checkpoints "$RUN_ID/ckpt" --global_path "$RUN_ID/global" \
    --llm_model llama3 --prompt_group Jailbreak/traits --run-id "$RUN_ID"
