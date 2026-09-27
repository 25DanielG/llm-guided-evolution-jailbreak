#!/bin/bash
# Standalone target/judge/reporter launcher. Reserve a separate mutator allocation.
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$HERE/.vllm_pids"
if [[ "${1:-}" == "--stop" ]]; then
    if [[ -f "$PIDFILE" ]]; then
        while read -r pid; do [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true; done < "$PIDFILE"
        rm -f "$PIDFILE"
    fi
    exit 0
fi
TARGET="${JB_TARGET_MODEL_PATH:-$HOME/scratch/llm_storage/Mistral-7B-Instruct-v0.2}"
JUDGE="${JB_JUDGE_MODEL_PATH:-$HOME/scratch/llm_storage/Llama-3.1-8B-Instruct}"
REPORTER="${JB_REPORTING_MODEL_PATH:-$HOME/scratch/llm_storage/HarmBench-Llama-2-13b-cls}"
for path in "$TARGET" "$JUDGE" "$REPORTER"; do
    [[ -f "$path/config.json" ]] || { echo "Missing checkpoint: $path" >&2; exit 1; }
done
[[ "$TARGET" != "$JUDGE" && "$TARGET" != "$REPORTER" && "$JUDGE" != "$REPORTER" ]] || { echo "Models must be distinct" >&2; exit 1; }
: > "$PIDFILE"
HOST=$(hostname)
CUDA_VISIBLE_DEVICES="${JB_TARGET_GPU:-0}" "${JB_VLLM_BIN:-vllm}" serve "$TARGET" --served-model-name "${JB_TARGET_SERVED_NAME:-target}" --host 0.0.0.0 --port "${JB_TARGET_VLLM_PORT:-8001}" --max-model-len "${JB_TARGET_MAX_MODEL_LEN:-4096}" > "$HERE/target_vllm.out" 2>&1 &
echo $! >> "$PIDFILE"
echo "$HOST:${JB_TARGET_VLLM_PORT:-8001}" > "$HERE/target_host.log"
CUDA_VISIBLE_DEVICES="${JB_JUDGE_GPU:-1}" "${JB_VLLM_BIN:-vllm}" serve "$JUDGE" --served-model-name "${JB_JUDGE_SERVED_NAME:-judge}" --host 0.0.0.0 --port "${JB_JUDGE_VLLM_PORT:-8002}" --max-model-len "${JB_JUDGE_MAX_MODEL_LEN:-4096}" > "$HERE/judge_vllm.out" 2>&1 &
echo $! >> "$PIDFILE"
echo "$HOST:${JB_JUDGE_VLLM_PORT:-8002}" > "$HERE/judge_host.log"
CUDA_VISIBLE_DEVICES="${JB_REPORTING_GPU:-2}" "${JB_VLLM_BIN:-vllm}" serve "$REPORTER" --served-model-name "${JB_REPORTING_SERVED_NAME:-cais/HarmBench-Llama-2-13b-cls}" --host 0.0.0.0 --port "${JB_REPORTING_VLLM_PORT:-8003}" > "$HERE/reporting_vllm.out" 2>&1 &
echo $! >> "$PIDFILE"
echo "$HOST:${JB_REPORTING_VLLM_PORT:-8003}" > "$HERE/reporting_host.log"
wait
