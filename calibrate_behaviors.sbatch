#!/bin/bash
# Behavior-difficulty calibration: score a spread of already-evolved genes on
# the WHOLE behavior pool (no sampling), so stratify_behaviors.py can tier every
# behavior against the same gene set. Needs target + judge + reporting servers
# only (no mutator). Usage:
#   sbatch --export=ALL,GENES_FILE=calibration_genes.txt calibrate_behaviors.sbatch
#SBATCH --job-name=jb_calib
#SBATCH -G 3
#SBATCH -t 4:00:00
#SBATCH --mem=128G
#SBATCH -c 16
#SBATCH -o jb_calib_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-${SLURM_JOB_ID}"
mkdir -p "$UV_CACHE_DIR"
export VLLM_USE_FLASHINFER_SAMPLER=0

GENES_FILE="${GENES_FILE:?set GENES_FILE to a file of gene ids, one per line (seed allowed)}"
CALIB_RUN_ID="${CALIB_RUN_ID:-calib_${SLURM_JOB_ID}}"
VARIANT_DIR_ABS="$PWD/sota/Jailbreak/models/llmge_models"

TARGET_MODEL="$HOME/scratch/llm_storage/Mistral-7B-Instruct-v0.2"
JUDGE_MODEL="$HOME/scratch/llm_storage/Llama-3.1-8B-Instruct"
REPORTING_MODEL="$HOME/scratch/llm_storage/HarmBench-Llama-2-13b-cls"
# whole pool, no stratification
# original runs' results/<gene>_results.* not overwritten
export LOCAL=true LOCAL_LLM=true JB_JUDGE_MODE=llm JB_FULL_BEHAVIORS=1 JB_STRATIFIED=0
export JB_GENE_RESULTS_DIR="results/$CALIB_RUN_ID/genes"
export JB_TARGET_MODEL_PATH="$TARGET_MODEL" JB_JUDGE_MODEL_PATH="$JUDGE_MODEL"
export JB_REPORTING_MODEL_PATH="$REPORTING_MODEL"
export JB_REPORTING_SERVED_NAME="cais/HarmBench-Llama-2-13b-cls"

HOST=$(hostname)
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [ "${#GPU_IDS[@]}" -lt 3 ]; then
    echo "ERROR: expected 3 GPUs from Slurm, got: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi

start_with_retry() {
    local label="$1" logfile="$2"; shift 2
    local attempt pid
    for attempt in 1 2 3; do
        : > "$logfile"
        ( "$@" ) > "$logfile" 2>&1 &
        pid=$!
        sleep 20
        if kill -0 "$pid" 2>/dev/null; then echo "$pid"; return 0; fi
        echo "$label died on attempt $attempt; retrying..." >&2
        wait "$pid" 2>/dev/null
        sleep 10
    done
    echo "ERROR: $label failed to start after 3 attempts" >&2
    return 1
}

TARGET_PID=$(start_with_retry "target vLLM" sota/Jailbreak/target_vllm.out \
    env CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" vllm serve "$TARGET_MODEL" \
    --served-model-name target --host 0.0.0.0 --port 8001 --gpu-memory-utilization 0.9) || exit 1
echo "$HOST:8001" > sota/Jailbreak/target_host.log
export JB_TARGET_VLLM_URL="http://$HOST:8001/v1"

JUDGE_PID=$(start_with_retry "judge vLLM" sota/Jailbreak/judge_vllm.out \
    env CUDA_VISIBLE_DEVICES="${GPU_IDS[1]}" vllm serve "$JUDGE_MODEL" \
    --served-model-name judge --host 0.0.0.0 --port 8002 --gpu-memory-utilization 0.9) || exit 1
echo "$HOST:8002" > sota/Jailbreak/judge_host.log
export JB_JUDGE_VLLM_URL="http://$HOST:8002/v1" JB_JUDGE_SERVED_NAME=judge

REPORTING_PID=$(start_with_retry "reporting vLLM" sota/Jailbreak/reporting_vllm.out \
    env CUDA_VISIBLE_DEVICES="${GPU_IDS[2]}" vllm serve "$REPORTING_MODEL" \
    --served-model-name "$JB_REPORTING_SERVED_NAME" --host 0.0.0.0 --port 8003 \
    --gpu-memory-utilization 0.9) || exit 1
echo "$HOST:8003" > sota/Jailbreak/reporting_host.log
export JB_REPORTING_VLLM_URL="http://$HOST:8003/v1"

cleanup() { kill "$TARGET_PID" "$JUDGE_PID" "$REPORTING_PID" 2>/dev/null || true; }
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
wait_or_die "http://$HOST:8001/v1/models" "$TARGET_PID" "target vLLM" sota/Jailbreak/target_vllm.out
wait_or_die "http://$HOST:8002/v1/models" "$JUDGE_PID" "judge vLLM" sota/Jailbreak/judge_vllm.out
wait_or_die "http://$HOST:8003/v1/models" "$REPORTING_PID" "reporting vLLM" sota/Jailbreak/reporting_vllm.out
echo "Servers ready. Calibrating run_id=$CALIB_RUN_ID"

done_n=0; skipped=0
while read -r gene; do
    [[ -z "$gene" || "$gene" == \#* ]] && continue
    if [ "$gene" = "seed" ]; then
        model="prompt"; vdir="$PWD/sota/Jailbreak"
    else
        model="prompt_${gene}"; vdir="$VARIANT_DIR_ABS"
        if [ ! -f "$vdir/$model.py" ]; then
            echo "SKIP $gene: $vdir/$model.py missing" >&2; skipped=$((skipped+1)); continue
        fi
    fi
    echo "=== calibrating $gene ==="
    # a failed gene should not sink the whole pass
    uv run python sota/Jailbreak/eval.py --model "$model" --variant_dir "$vdir" \
        --generation 0 --run-id "$CALIB_RUN_ID" || echo "FAILED $gene" >&2
    done_n=$((done_n+1))
done < "$GENES_FILE"

echo "=== calibration complete: $done_n genes attempted, $skipped skipped ==="
echo "next: uv run python sota/Jailbreak/stratify_behaviors.py sota/Jailbreak/results/$CALIB_RUN_ID/candidates.jsonl"
