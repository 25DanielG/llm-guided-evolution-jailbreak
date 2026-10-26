#!/bin/bash
# Production topology: 1 GPU target, 1 GPU judge, 1 GPU reporting (HarmBench,
# verification/audit only), 1 GPU mutator.
# Adjust Slurm allocation and groups together if using a different GPU type.
# Judge override (defaults unchanged): JB_RUN_JUDGE_MODEL, JB_RUN_JUDGE_TP (judge
# GPUs, default 1), JB_RUN_JUDGE_MAX_LEN (vllm --max-model-len); needs
# 1 + JB_RUN_JUDGE_TP + 2 GPUs. JB_RUN_MUTATOR_MODEL is independent of the judge.
# Guide judge on the shared 70B (bf16, 2 x 80GB GPUs; sota/Jailbreak/README.md):
#   sbatch -N 1 -G 5 -C "A100-80GB|H100|H200" --export=ALL,JB_JUDGE_PROMPT=guide,JB_RUN_JUDGE_MODEL=/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct,JB_RUN_JUDGE_TP=2,JB_RUN_JUDGE_MAX_LEN=4096 run_jailbreak.sbatch
#SBATCH --job-name=jb_run
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --constraint=H100
#SBATCH -G 4
#SBATCH -t 4:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --mem=192G
#SBATCH -c 24
#SBATCH -o jb_run_%j.out

# Target weaker model, for a first ASR>0 validation; judge + reporting are
# separate models from the target (and from each other) to avoid a model
# judging its own outputs. Each get their own vLLM server on a different GPU.
# Mutator is HF server.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN_ID="${RUN_ID:-jb_${SLURM_JOB_ID}}"
JB_NUM_GENERATIONS="${JB_NUM_GENERATIONS:-17}"
export RUN_ID JB_NUM_GENERATIONS
if ! [[ "$JB_NUM_GENERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: JB_NUM_GENERATIONS must be a positive integer: $JB_NUM_GENERATIONS" >&2
    exit 1
fi
final_gen=$((JB_NUM_GENERATIONS - 1))
if [ -s "$RUN_ID/ckpt/checkpoint_gen_${final_gen}.pkl" ] &&
   [ -s "$RUN_ID/global/global_gen_${final_gen}.pkl" ]; then
    echo "Run $RUN_ID already completed generation $final_gen; no continuation needed."
    exit 0
fi
module load cuda 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"          # uv + vllm CLIs
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"

# FlashInfer's sampler JIT-compiles with 'ninja' (absent on nodes) -> use native sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL_ROOT="${JB_RUN_MODEL_ROOT:-$(dirname "$PWD")/llm_storage}"
TARGET_MODEL="${JB_RUN_TARGET_MODEL:-$MODEL_ROOT/Mistral-7B-Instruct-v0.2}"
LLAMA_8B="${JB_RUN_LLAMA_8B_MODEL:-$MODEL_ROOT/Llama-3.1-8B-Instruct}"
JUDGE_MODEL="${JB_RUN_JUDGE_MODEL:-$LLAMA_8B}"
JUDGE_TP="${JB_RUN_JUDGE_TP:-1}"
JUDGE_MAX_LEN="${JB_RUN_JUDGE_MAX_LEN:-}"
JUDGE_PROMPT="${JB_JUDGE_PROMPT-rubric}"  # unset -> rubric, same as jb_config.judge_prompt()
MUTATOR_MODEL="${JB_RUN_MUTATOR_MODEL:-$LLAMA_8B}"
REPORTING_MODEL="${JB_RUN_REPORTING_MODEL:-$MODEL_ROOT/HarmBench-Llama-2-13b-cls}"
export LOCAL=true LOCAL_LLM=true JB_JUDGE_MODE=llm JB_FULL_BEHAVIORS=0
# 50 behaviors per gene, equal easy/medium/hard; needs `tier` in curated.csv
# (sota/Jailbreak/stratify_behaviors.py) or eval.py raises.
export JB_STRATIFIED=1 JB_N_BEHAVIORS_PER_EVAL=50
export JB_BEHAVIORS_PATH="${JB_BEHAVIORS_PATH:-$PWD/sota/Jailbreak/behaviors/curated_tiered_jb_strat_base1.csv}"
export JB_GENE_RESULTS_DIR=results
# Fresh response/verdict cache per run so cached failures from older runs are not reused
export JB_CACHE_DIR="${JB_CACHE_DIR:-$HOME/scratch/jb_cache_${RUN_ID:-jb_${SLURM_JOB_ID}}}"
export JB_TARGET_MODEL_PATH="$TARGET_MODEL"
export JB_JUDGE_MODEL_PATH="$JUDGE_MODEL"
export JB_REPORTING_MODEL_PATH="$REPORTING_MODEL"
export JB_REPORTING_SERVED_NAME="cais/HarmBench-Llama-2-13b-cls"
export JB_MUTATOR_MODEL_PATH="$MUTATOR_MODEL"

uv run python sota/Jailbreak/preflight_behaviors.py
uv run python sota/Jailbreak/preflight_models.py \
    "$TARGET_MODEL" "$JUDGE_MODEL" "$REPORTING_MODEL" "$MUTATOR_MODEL"

HOST=$(hostname)

if ! [[ "$JUDGE_TP" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: JB_RUN_JUDGE_TP must be a positive integer, got: $JUDGE_TP" >&2
    exit 1
fi
case "$JUDGE_PROMPT" in
    rubric|guide) ;;
    *) echo "ERROR: JB_JUDGE_PROMPT must be rubric or guide, got: '$JUDGE_PROMPT'" >&2; exit 1 ;;
esac
if [ "$JUDGE_PROMPT" = guide ] && [ "$JUDGE_MODEL" = "$LLAMA_8B" ]; then
    echo "WARNING: JB_JUDGE_PROMPT=guide on the default 8B judge; it was validated on the 70B (set JB_RUN_JUDGE_MODEL)" >&2
fi
# Rubric verdict cache keys carry no model identity (served name is always "judge"), so a
# non-default rubric judge gets its own cache dir instead of reusing the 8B's verdicts.
if [ "$JUDGE_PROMPT" = rubric ] && [ "$JUDGE_MODEL" != "$LLAMA_8B" ]; then
    export JB_CACHE_DIR="${JB_CACHE_DIR:-$HOME/scratch/jb_cache}/judge_$(basename "$JUDGE_MODEL")"
    echo "Non-default rubric judge: verdict + response cache at $JB_CACHE_DIR"
fi
# target, JUDGE_TP judge GPUs, reporting, mutator
NEED_GPUS=$((JUDGE_TP + 3))
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [ "${#GPU_IDS[@]}" -lt "$NEED_GPUS" ]; then
    echo "ERROR: expected $NEED_GPUS GPUs from Slurm (judge tp=$JUDGE_TP; use sbatch -G $NEED_GPUS), got: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
TARGET_DEV="${GPU_IDS[0]}"
JUDGE_DEV=$(IFS=','; echo "${GPU_IDS[*]:1:JUDGE_TP}")
REPORTING_DEV="${GPU_IDS[JUDGE_TP + 1]}"
MUTATOR_DEV="${GPU_IDS[JUDGE_TP + 2]}"
echo "Slurm gave GPUs: $CUDA_VISIBLE_DEVICES -> target=$TARGET_DEV judge=$JUDGE_DEV reporting=$REPORTING_DEV mutator=$MUTATOR_DEV"
echo "Judge: $JUDGE_MODEL (tp=$JUDGE_TP, prompt=$JUDGE_PROMPT); mutator: $MUTATOR_MODEL"
for gpu_id in "${GPU_IDS[@]:0:NEED_GPUS}"; do
    echo "Checking CUDA allocation on GPU $gpu_id"
    env CUDA_VISIBLE_DEVICES="$gpu_id" uv run python sota/Jailbreak/preflight_cuda.py
done
JUDGE_EXTRA_ARGS=""
# Multi-GPU vLLM JIT-builds FlashInfer all-reduce kernels with 'ninja' (not on PATH here): use vLLM's own
# all-reduce, and eager mode so the compile-time all-reduce fusion pass doesn't build one either.
if [ "$JUDGE_TP" -gt 1 ]; then
    export VLLM_ALLREDUCE_USE_FLASHINFER=0
    JUDGE_EXTRA_ARGS="--tensor-parallel-size $JUDGE_TP --enforce-eager"
fi
if [ -n "$JUDGE_MAX_LEN" ]; then JUDGE_EXTRA_ARGS="$JUDGE_EXTRA_ARGS --max-model-len $JUDGE_MAX_LEN"; fi

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
    --gpu-memory-utilization 0.9 $JUDGE_EXTRA_ARGS) || exit 1
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

export SERVER_HOSTNAME=$HOST
echo "$HOST" > hostname.log
MUT_PID=$(start_with_retry "mutator" mutator.log \
    env CUDA_VISIBLE_DEVICES="$MUTATOR_DEV" uv run python -m uvicorn server:app \
    --host "$HOST" --port 8137) || exit 1

cleanup() { kill "$TARGET_PID" "$JUDGE_PID" "$REPORTING_PID" "$MUT_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Slurm warns the batch shell five minutes before the walltime. Queue the next
# allocation then; it starts after this job exits and resumes from the latest
# generation checkpoint. If this job finishes meanwhile, the next job exits
# at the completion check above without starting model servers.
queue_continuation() {
    if [ -s "$RUN_ID/ckpt/checkpoint_gen_${final_gen}.pkl" ] &&
       [ -s "$RUN_ID/global/global_gen_${final_gen}.pkl" ]; then
        return
    fi
    if [ "${CONTINUATION_QUEUED:-0}" = 1 ]; then
        return
    fi
    local attempt
    local submit_args=(--dependency="afterany:${SLURM_JOB_ID}"
                       --export="ALL,RUN_ID=$RUN_ID,JB_NUM_GENERATIONS=$JB_NUM_GENERATIONS")
    if [ -n "${JB_RUN_EXCLUDE_NODES:-}" ]; then
        submit_args+=(--exclude="$JB_RUN_EXCLUDE_NODES")
    fi
    for attempt in 1 2 3; do
        if sbatch "${submit_args[@]}" "$SLURM_SUBMIT_DIR/run_jailbreak.sbatch"; then
            CONTINUATION_QUEUED=1
            echo "Queued continuation for $RUN_ID after job $SLURM_JOB_ID"
            return
        fi
        echo "Continuation submission failed (attempt $attempt/3)" >&2
        sleep 10
    done
    echo "ERROR: could not queue continuation for $RUN_ID" >&2
}
trap queue_continuation USR1

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
echo "Mutator ready. Running smoke test."
uv run python sota/Jailbreak/smoke_test.py
echo "Smoke test passed. Starting evolution."

# run's checkpoint/global dirs
uv run python run_improved.py --checkpoints "${RUN_ID}/ckpt" --global_path "${RUN_ID}/global" \
    --llm_model llama3 --prompt_group Jailbreak/traits --run-id "$RUN_ID" &
EVOLUTION_PID=$!
# Bash handles USR1 promptly while waiting for a background child. A trapped
# signal interrupts wait, so continue waiting for evolution to finish or time out.
while true; do
    if wait "$EVOLUTION_PID"; then
        break
    else
        wait_status=$?
        if kill -0 "$EVOLUTION_PID" 2>/dev/null; then
            continue
        fi
        exit "$wait_status"
    fi
done

echo "=== run complete ==="
