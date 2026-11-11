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
# All four model services communicate over localhost and must share one node.
#SBATCH -N 1
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
if [[ -n "${JB_RUN_TARGET_PROFILE:-}${JB_RUN_JUDGE_PROFILE:-}${JB_RUN_MUTATOR_PROFILE:-}" ]]; then
    : "${JB_VLLM_BIN:?Profiles require the approved shared serving environment bin/vllm}"
    : "${JB_CUDA_MODULE:?Profiles require the CUDA module verified for that environment}"
    module load "$JB_CUDA_MODULE"
else
    module load cuda 2>/dev/null || true
fi
export PATH="$HOME/.local/bin:$PATH"          # uv + vllm CLIs
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"

# FlashInfer's sampler JIT-compiles with 'ninja' (absent on nodes) -> use native sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0

TARGET_MODEL="${JB_RUN_TARGET_MODEL:-$HOME/scratch/llm_storage/Mistral-7B-Instruct-v0.2}"
LLAMA_8B="$HOME/scratch/llm_storage/Llama-3.1-8B-Instruct"
JUDGE_MODEL="${JB_RUN_JUDGE_MODEL:-$LLAMA_8B}"
JUDGE_TP="${JB_RUN_JUDGE_TP:-1}"
JUDGE_MAX_LEN="${JB_RUN_JUDGE_MAX_LEN:-}"
JUDGE_PROMPT="${JB_JUDGE_PROMPT-rubric}"  # unset -> rubric, same as jb_config.judge_prompt()
MUTATOR_MODEL="${JB_RUN_MUTATOR_MODEL:-$LLAMA_8B}"
REPORTING_MODEL="$HOME/scratch/llm_storage/HarmBench-Llama-2-13b-cls"
# Opt-in registry profiles. Legacy path overrides retain their original behavior.
TARGET_PROFILE="${JB_RUN_TARGET_PROFILE:-}"
JUDGE_PROFILE="${JB_RUN_JUDGE_PROFILE:-}"
MUTATOR_PROFILE="${JB_RUN_MUTATOR_PROFILE:-}"
TARGET_TP=1
MUTATOR_TP=1
if [[ -n "$TARGET_PROFILE" ]]; then
    TARGET_MODEL=$(python3 -m src.model_serving path --model "$TARGET_PROFILE")
    TARGET_TP="${JB_RUN_TARGET_TP:-$(python3 -m src.model_serving tp --model "$TARGET_PROFILE")}"
    export JB_TARGET_PROFILE="$TARGET_PROFILE"
fi
if [[ -n "$JUDGE_PROFILE" ]]; then
    JUDGE_MODEL=$(python3 -m src.model_serving path --model "$JUDGE_PROFILE")
    JUDGE_TP="${JB_RUN_JUDGE_TP:-$(python3 -m src.model_serving tp --model "$JUDGE_PROFILE")}"
    export JB_JUDGE_PROFILE="$JUDGE_PROFILE"
    if [[ "$JUDGE_PROMPT" == guide && "$JUDGE_PROFILE" != llama-3.3-70b ]]; then
        echo "ERROR: guide judge is validated only with llama-3.3-70b; use rubric for new models" >&2
        exit 1
    fi
fi
if [[ -n "$MUTATOR_PROFILE" ]]; then
    MUTATOR_MODEL=$(python3 -m src.model_serving path --model "$MUTATOR_PROFILE")
    MUTATOR_TP="${JB_RUN_MUTATOR_TP:-$(python3 -m src.model_serving tp --model "$MUTATOR_PROFILE")}"
    export JB_MUTATOR_PROFILE="$MUTATOR_PROFILE"
fi

export LOCAL=true LOCAL_LLM=true JB_JUDGE_MODE=llm

# A fresh checkout has no earlier candidates from which to derive behavior
# difficulty. Bootstrap mode evaluates one small, unstratified starting
# population. Those candidates can then be used by calibrate_behaviors.sbatch
# and stratify_behaviors.py before the real MAP-Elites/baseline comparison.
BOOTSTRAP="${JB_BOOTSTRAP:-0}"
case "$BOOTSTRAP" in
    1|true|TRUE|True|yes|YES|Yes)
        export JB_FULL_BEHAVIORS=0
        export JB_STRATIFIED=0
        export JB_N_BEHAVIORS_PER_EVAL="${JB_N_BEHAVIORS_PER_EVAL:-12}"
        export LLMGE_NUM_GENERATIONS="${LLMGE_NUM_GENERATIONS:-1}"
        export LLMGE_START_POPULATION_SIZE="${LLMGE_START_POPULATION_SIZE:-12}"
        export LLMGE_POPULATION_SIZE="${LLMGE_POPULATION_SIZE:-12}"
        export JB_SELECTION="${JB_SELECTION:-nsga2}"
        RUN_ID="${RUN_ID:-bootstrap_${SLURM_JOB_ID}}"
        ;;
    0|false|FALSE|False|no|NO|No|"")
        export JB_FULL_BEHAVIORS="${JB_FULL_BEHAVIORS:-0}"
        # Production default: 50 behaviors per gene, split as evenly as
        # possible across easy/medium/hard. Requires tier values in curated.csv.
        export JB_STRATIFIED="${JB_STRATIFIED:-1}"
        export JB_N_BEHAVIORS_PER_EVAL="${JB_N_BEHAVIORS_PER_EVAL:-50}"
        export JB_SELECTION="${JB_SELECTION:-map_elites}"
        RUN_ID="${RUN_ID:-jb_${SLURM_JOB_ID}}"
        ;;
    *)
        echo "ERROR: JB_BOOTSTRAP must be 0/1, false/true, or no/yes; got '$BOOTSTRAP'" >&2
        exit 1
        ;;
esac
export RUN_ID
echo "Run configuration: run_id=$RUN_ID bootstrap=$BOOTSTRAP selection=$JB_SELECTION "\
     "stratified=$JB_STRATIFIED behaviors=$JB_N_BEHAVIORS_PER_EVAL "\
     "generations=${LLMGE_NUM_GENERATIONS:-17} start_population=${LLMGE_START_POPULATION_SIZE:-24} "\
     "population=${LLMGE_POPULATION_SIZE:-24}"
# Fresh response/verdict cache per run so cached failures from older runs are not reused
export JB_CACHE_DIR="${JB_CACHE_DIR:-$HOME/scratch/jb_cache_${RUN_ID}}"
export JB_TARGET_MODEL_PATH="$TARGET_MODEL"
export JB_JUDGE_MODEL_PATH="$JUDGE_MODEL"
export JB_REPORTING_MODEL_PATH="$REPORTING_MODEL"
export JB_REPORTING_SERVED_NAME="cais/HarmBench-Llama-2-13b-cls"
export JB_MUTATOR_MODEL_PATH="$MUTATOR_MODEL"

HOST=$(hostname)

if ! [[ "$JUDGE_TP" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: JB_RUN_JUDGE_TP must be a positive integer, got: $JUDGE_TP" >&2
    exit 1
fi
if ! [[ "$TARGET_TP" =~ ^[1-9][0-9]*$ && "$MUTATOR_TP" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: profile target and mutator TP must be positive integers" >&2
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
NEED_GPUS=$((TARGET_TP + JUDGE_TP + 1 + MUTATOR_TP))
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [ "${#GPU_IDS[@]}" -lt "$NEED_GPUS" ]; then
    echo "ERROR: expected $NEED_GPUS GPUs from Slurm (target=$TARGET_TP judge=$JUDGE_TP reporting=1 mutator=$MUTATOR_TP), got: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
TARGET_DEV=$(IFS=','; echo "${GPU_IDS[*]:0:TARGET_TP}")
JUDGE_DEV=$(IFS=','; echo "${GPU_IDS[*]:TARGET_TP:JUDGE_TP}")
REPORTING_DEV="${GPU_IDS[TARGET_TP + JUDGE_TP]}"
MUTATOR_DEV=$(IFS=','; echo "${GPU_IDS[*]:TARGET_TP+JUDGE_TP+1:MUTATOR_TP}")
echo "Slurm gave GPUs: $CUDA_VISIBLE_DEVICES -> target=$TARGET_DEV judge=$JUDGE_DEV reporting=$REPORTING_DEV mutator=$MUTATOR_DEV"
echo "Judge: $JUDGE_MODEL (tp=$JUDGE_TP, prompt=$JUDGE_PROMPT); mutator: $MUTATOR_MODEL"
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
        wait "$pid" 2>/dev/null || true
        sleep 10
    done
    echo "ERROR: $label failed to start after 3 attempts" >&2
    return 1
}

# Validate every selected checkpoint before loading any model onto GPUs.
for selected_profile in "$TARGET_PROFILE" "$JUDGE_PROFILE" "$MUTATOR_PROFILE"; do
    if [[ -n "$selected_profile" ]]; then
        python3 -m src.model_serving check --model "$selected_profile"
    fi
done
TARGET_PID="" JUDGE_PID="" REPORTING_PID="" MUT_PID="" MUT_VLLM_PID=""
cleanup() { for pid in "$TARGET_PID" "$JUDGE_PID" "$REPORTING_PID" "$MUT_PID" "$MUT_VLLM_PID"; do
    [[ -z "$pid" ]] || kill "$pid" 2>/dev/null || true
 done; }
trap cleanup EXIT
TARGET_COMMAND=("${JB_VLLM_BIN:-vllm}" serve "$TARGET_MODEL" --served-model-name target
    --host 0.0.0.0 --port 8001 --gpu-memory-utilization 0.9)
if [[ -n "$TARGET_PROFILE" ]]; then
    TARGET_COMMAND=(python3 -m src.model_serving serve --model "$TARGET_PROFILE"
        --served-name target --port 8001 --max-len "${JB_TARGET_MAX_MODEL_LEN:-4096}" --tp "$TARGET_TP")
fi
JUDGE_COMMAND=("${JB_VLLM_BIN:-vllm}" serve "$JUDGE_MODEL" --served-model-name judge
    --host 0.0.0.0 --port 8002 --gpu-memory-utilization 0.9)
# Preserve the legacy judge arguments without shell evaluation.
read -ra JUDGE_LEGACY_ARGS <<< "$JUDGE_EXTRA_ARGS"
JUDGE_COMMAND+=("${JUDGE_LEGACY_ARGS[@]}")
if [[ -n "$JUDGE_PROFILE" ]]; then
    JUDGE_COMMAND=(python3 -m src.model_serving serve --model "$JUDGE_PROFILE"
        --served-name judge --port 8002 --max-len "${JUDGE_MAX_LEN:-4096}" --tp "$JUDGE_TP")
fi

# target vLLM (weaker model)
TARGET_PID=$(start_with_retry "target vLLM" sota/Jailbreak/target_vllm.out \
    env CUDA_VISIBLE_DEVICES="$TARGET_DEV" "${TARGET_COMMAND[@]}") || exit 1
echo "$HOST:8001" > sota/Jailbreak/target_host.log
export JB_TARGET_VLLM_URL="http://$HOST:8001/v1"

# judge vLLM (in-loop rubric judge; fitness-relevant)
JUDGE_PID=$(start_with_retry "judge vLLM" sota/Jailbreak/judge_vllm.out \
    env CUDA_VISIBLE_DEVICES="$JUDGE_DEV" "${JUDGE_COMMAND[@]}") || exit 1
echo "$HOST:8002" > sota/Jailbreak/judge_host.log
export JB_JUDGE_VLLM_URL="http://$HOST:8002/v1"
export JB_JUDGE_SERVED_NAME=judge

# reporting vLLM (HarmBench classifier; verification/audit only, never fitness)
REPORTING_PID=$(start_with_retry "reporting vLLM" sota/Jailbreak/reporting_vllm.out \
    env CUDA_VISIBLE_DEVICES="$REPORTING_DEV" "${JB_VLLM_BIN:-vllm}" serve "$REPORTING_MODEL" \
    --served-model-name "$JB_REPORTING_SERVED_NAME" --host 0.0.0.0 --port 8003 \
    --gpu-memory-utilization 0.9) || exit 1
echo "$HOST:8003" > sota/Jailbreak/reporting_host.log
export JB_REPORTING_VLLM_URL="http://$HOST:8003/v1"

export SERVER_HOSTNAME=$HOST
echo "$HOST" > hostname.log
if [[ -n "$MUTATOR_PROFILE" ]]; then
    MUT_VLLM_PID=$(start_with_retry "mutator vLLM" mutator_vllm.log \
        env CUDA_VISIBLE_DEVICES="$MUTATOR_DEV" python3 -m src.model_serving serve \
        --model "$MUTATOR_PROFILE" --port 8004 --max-len "${JB_MUTATOR_MAX_MODEL_LEN:-4096}" --tp "$MUTATOR_TP") || exit 1
    export JB_MUTATOR_VLLM_URL="http://$HOST:8004/v1"
    # Proxy startup validates model identity, so wait for its backend first.
    deadline=$((SECONDS + ${JB_SERVER_READY_TIMEOUT:-1200}))
    until curl -sf "$JB_MUTATOR_VLLM_URL/models" >/dev/null; do
        if ! kill -0 "$MUT_VLLM_PID" 2>/dev/null || (( SECONDS >= deadline )); then
            tail -n 35 mutator_vllm.log >&2
            exit 1
        fi
        sleep 5
    done
fi
MUT_PID=$(start_with_retry "mutator" mutator.log \
    env CUDA_VISIBLE_DEVICES="$MUTATOR_DEV" uv run python -m uvicorn server:app \
    --host "$HOST" --port 8137) || exit 1


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
echo "Mutator ready."

SMOKE_ONLY="${JB_SMOKE_ONLY:-0}"
case "$SMOKE_ONLY" in
    1|true|TRUE|True|yes|YES|Yes)
        echo "Running smoke test only; evolution will not start."
        uv run python sota/Jailbreak/smoke_test.py
        echo "=== smoke test complete ==="
        exit 0
        ;;
    0|false|FALSE|False|no|NO|No|"") ;;
    *)
        echo "ERROR: JB_SMOKE_ONLY must be 0/1, false/true, or no/yes; got '$SMOKE_ONLY'" >&2
        exit 1
        ;;
esac

echo "Starting evolution."

# run's checkpoint/global dirs
CHECKPOINT_ARGS=()
[[ -z "${JB_CHECKPOINT_FILE:-}" ]] || CHECKPOINT_ARGS=(--checkpoint-file "$JB_CHECKPOINT_FILE")
uv run python run_improved.py --checkpoints "${RUN_ID}/ckpt" --global_path "${RUN_ID}/global" \
    --llm_model "${MUTATOR_PROFILE:-llama3}" --prompt_group Jailbreak/traits --run-id "$RUN_ID" \
    --resume-mode "${JB_RESUME_MODE:-auto}" "${CHECKPOINT_ARGS[@]}"

echo "=== run complete ==="
