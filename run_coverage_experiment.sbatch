#!/bin/bash
# Production topology: 1 GPU target, 2 GPUs judge, 1 GPU reporting (HarmBench,
# verification/audit only), 2 GPUs mutator.
# Adjust Slurm allocation and groups together if using a different GPU type.
#SBATCH --job-name=coverage_run
#SBATCH -G 6
#SBATCH -t 2:40:00
#SBATCH --mem=192G
#SBATCH -c 24
#SBATCH -o coverage_%j.out

# Llama-3.1-8B target, Llama-3.3-70B judge/mutator, HarmBench reporter.
# Each service has its own GPU group; the mutator uses the HF server.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"          # uv + vllm CLIs
# uv exposes vllm through a symlink; its sibling tools include ninja, which
# FlashInfer's tensor-parallel kernels invoke during judge startup.
VLLM_TOOL_BIN="$(dirname "$(readlink -f "$(command -v vllm)")")"
export PATH="$VLLM_TOOL_BIN:$PATH"
if ! command -v ninja >/dev/null 2>&1; then
    echo "vLLM requires ninja to compile GPU kernels, but ninja is not on PATH" >&2
    exit 1
fi
echo "Using ninja: $(command -v ninja) ($(ninja --version))"
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"
# Match Slurm's PCI device ordering, including on shared GPU nodes.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Compiler artifacts are node-local and isolated per job/service. Shared-home
# caches produced stale file handles when workers started on several nodes.
KERNEL_CACHE_BASE="/tmp/coverage-kernels-${USER}-${SLURM_JOB_ID}"
mkdir -p "$KERNEL_CACHE_BASE"
echo "Kernel caches: $KERNEL_CACHE_BASE"

# FlashInfer's sampler JIT-compiles with 'ninja' (absent on nodes) -> use native sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0

TARGET_MODEL="${COVERAGE_TARGET_MODEL:-$HOME/scratch/llm_storage/Llama-3.1-8B-Instruct}"
JUDGE_MODEL="${COVERAGE_JUDGE_MODEL:-/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct}"
REPORTING_MODEL="${COVERAGE_REPORTING_MODEL:-$HOME/scratch/llm_storage/HarmBench-Llama-2-13b-cls}"
# Required experiment locations are absolute paths, supplied at submission.
: "${COVERAGE_MANIFEST:?Set COVERAGE_MANIFEST}"
: "${COVERAGE_INITIAL:?Set COVERAGE_INITIAL}"
: "${COVERAGE_RUN_ROOT:?Set COVERAGE_RUN_ROOT}"
export COVERAGE_RUN_ROOT="$(realpath -m "$COVERAGE_RUN_ROOT")"
SERVICE_DIR="$COVERAGE_RUN_ROOT/services/${SLURM_JOB_ID:-manual}"
mkdir -p "$SERVICE_DIR"
# Fixed service ports must not be shared by two of our jobs on the same node.
exec 9>"/tmp/coverage-services-${USER}.lock"
flock -n 9 || { echo "Another coverage service group is using this node" >&2; exit 1; }
TARGET_PID="" JUDGE_PID="" REPORTING_PID="" MUT_PID=""
cleanup() {
    local pid
    for pid in "$TARGET_PID" "$JUDGE_PID" "$REPORTING_PID" "$MUT_PID"; do
        if [[ -n "$pid" ]]; then kill -- "-$pid" 2>/dev/null || true; fi
    done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

export LOCAL=true LOCAL_LLM=true JB_JUDGE_MODE=llm
export LLMGE_HOSTNAME_PATH="$SERVICE_DIR/hostname.log"
export JB_TARGET_MODEL_PATH="$TARGET_MODEL"
export JB_JUDGE_MODEL_PATH="$JUDGE_MODEL"
export JB_REPORTING_MODEL_PATH="$REPORTING_MODEL"
export JB_REPORTING_SERVED_NAME="cais/HarmBench-Llama-2-13b-cls"
export JB_MUTATOR_MODEL_PATH="${COVERAGE_MUTATOR_MODEL:-/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct}"

HOST=$(hostname)

IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [ "${#GPU_IDS[@]}" -lt 6 ]; then
    echo "ERROR: expected 6 GPUs from Slurm, got: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
TARGET_DEV="${GPU_IDS[0]}"
JUDGE_DEV="${GPU_IDS[1]},${GPU_IDS[2]}"
REPORTING_DEV="${GPU_IDS[3]}"
MUTATOR_DEV="${GPU_IDS[4]},${GPU_IDS[5]}"
echo "Slurm gave GPUs: $CUDA_VISIBLE_DEVICES -> target=$TARGET_DEV judge=$JUDGE_DEV reporting=$REPORTING_DEV mutator=$MUTATOR_DEV"

source "$PWD/scripts/coverage_services.sh"

# target vLLM
TARGET_CMD=(env CUDA_VISIBLE_DEVICES="$TARGET_DEV" vllm serve "$TARGET_MODEL" \
    --served-model-name target --host 0.0.0.0 --port 8001 \
    --max-model-len "${JB_TARGET_MAX_MODEL_LEN:-4096}" \
    --gpu-memory-utilization 0.9)
TARGET_PID=$(start_service "target vLLM" "$SERVICE_DIR/target_vllm.out" "${TARGET_CMD[@]}")
echo "$HOST:8001" > "$SERVICE_DIR/target_host.log"
export JB_TARGET_VLLM_URL="http://$HOST:8001/v1"

# judge vLLM (in-loop rubric judge; fitness-relevant)
JUDGE_CMD=(env CUDA_VISIBLE_DEVICES="$JUDGE_DEV" vllm serve "$JUDGE_MODEL" \
    --served-model-name judge --host 0.0.0.0 --port 8002 \
    --tensor-parallel-size 2 --max-model-len "${JB_JUDGE_MAX_MODEL_LEN:-4096}" \
    --gpu-memory-utilization 0.9)
JUDGE_PID=$(start_service "judge vLLM" "$SERVICE_DIR/judge_vllm.out" "${JUDGE_CMD[@]}")
echo "$HOST:8002" > "$SERVICE_DIR/judge_host.log"
export JB_JUDGE_VLLM_URL="http://$HOST:8002/v1"
export JB_JUDGE_SERVED_NAME=judge

# reporting vLLM (HarmBench classifier; verification/audit only, never fitness)
REPORTING_CMD=(env CUDA_VISIBLE_DEVICES="$REPORTING_DEV" vllm serve "$REPORTING_MODEL" \
    --served-model-name "$JB_REPORTING_SERVED_NAME" --host 0.0.0.0 --port 8003 \
    --gpu-memory-utilization 0.9)
REPORTING_PID=$(start_service "reporting vLLM" "$SERVICE_DIR/reporting_vllm.out" "${REPORTING_CMD[@]}")
echo "$HOST:8003" > "$SERVICE_DIR/reporting_host.log"
export JB_REPORTING_VLLM_URL="http://$HOST:8003/v1"

export SERVER_HOSTNAME=$HOST
echo "$HOST" > "$LLMGE_HOSTNAME_PATH"
MUTATOR_CMD=(env CUDA_VISIBLE_DEVICES="$MUTATOR_DEV" uv run --frozen python \
    -m analysis.experiments.mutator_service --host "$HOST" --port 8137)
MUT_PID=$(start_service "mutator" "$SERVICE_DIR/mutator.log" "${MUTATOR_CMD[@]}")

# All services start concurrently. Retry failures throughout model loading,
# including failures that occur minutes after the initial process launch.
echo "Waiting for target vLLM..."
TARGET_PID=$(wait_or_die "http://$HOST:8001/v1/models" "$TARGET_PID" "target vLLM" "$SERVICE_DIR/target_vllm.out" "${TARGET_CMD[@]}") || exit 1
echo "Target ready. Waiting for judge vLLM..."
JUDGE_PID=$(wait_or_die "http://$HOST:8002/v1/models" "$JUDGE_PID" "judge vLLM" "$SERVICE_DIR/judge_vllm.out" "${JUDGE_CMD[@]}") || exit 1
echo "Judge ready. Waiting for reporting vLLM..."
REPORTING_PID=$(wait_or_die "http://$HOST:8003/v1/models" "$REPORTING_PID" "reporting vLLM" "$SERVICE_DIR/reporting_vllm.out" "${REPORTING_CMD[@]}") || exit 1
echo "Reporting ready. Waiting for mutator..."
MUT_PID=$(wait_or_die "http://$HOST:8137/" "$MUT_PID" "mutator" "$SERVICE_DIR/mutator.log" "${MUTATOR_CMD[@]}") || exit 1
echo "Mutator ready. Starting evolution."

# One job at a time per node: service ports follow the base topology.
if [ "${COVERAGE_ACTION:-run}" = "freeze-initial" ]; then
    uv run --frozen --extra diversity python analysis/scripts/coverage_experiment.py freeze-initial \
        --manifest "$COVERAGE_MANIFEST" --initial-population "$COVERAGE_INITIAL" \
        --run-root "$COVERAGE_RUN_ROOT" --seed "${COVERAGE_SEED:-42}"
else
    : "${COVERAGE_SUBSET:?Set COVERAGE_SUBSET}"
    uv run --frozen --extra diversity python analysis/scripts/coverage_experiment.py run \
        --manifest "$COVERAGE_MANIFEST" --initial-population "$COVERAGE_INITIAL" \
        --run-root "$COVERAGE_RUN_ROOT" --subset "$COVERAGE_SUBSET" \
        --seed "${COVERAGE_SEED:-42}" --final-generation "${COVERAGE_FINAL_GENERATION:-5}"
fi
