#!/bin/bash
# Launch the target + judge vLLM servers for the Jailbreak domain.
#
# vLLM is self-hosted and OpenAI-compatible. Run this on a two-GPU node
# BEFORE running eval.py / the evolution loop. It writes the servers' host:port to
# target_host.log and judge_host.log next to this script, which eval.py reads.
#
# Usage:
#   bash serve_vllm.sh              # start both servers, stream logs, block
#   bash serve_vllm.sh --stop       # stop servers started by this script
#
# Override via env: JB_TARGET_MODEL_PATH, JB_JUDGE_MODEL_PATH,
#   JB_TARGET_VLLM_PORT (8001), JB_JUDGE_VLLM_PORT (8002),
#   JB_TARGET_SERVED_NAME (target), JB_JUDGE_SERVED_NAME (judge),
#   JB_TARGET_GPU (0), JB_JUDGE_GPU (1), JB_GPU_MEM_UTIL (0.90),
#   JB_VLLM_API_KEY (required).

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$HERE/.vllm_pids"

if [[ "${1:-}" == "--stop" ]]; then
    if [[ -f "$PIDFILE" ]]; then
        while read -r pid; do
            [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
        done < "$PIDFILE"
        rm -f "$PIDFILE"
        echo "Stopped vLLM servers."
    else
        echo "No $PIDFILE found; nothing to stop."
    fi
    exit 0
fi

VLLM_BIN="${JB_VLLM_BIN:-vllm}"
: "${JB_VLLM_API_KEY:?JB_VLLM_API_KEY must be set}"

TARGET_MODEL_PATH="${JB_TARGET_MODEL_PATH:-/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.1-8B-Instruct/}"
JUDGE_MODEL_PATH="${JB_JUDGE_MODEL_PATH:-/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.1-8B-Instruct/}"
TARGET_PORT="${JB_TARGET_VLLM_PORT:-8001}"
JUDGE_PORT="${JB_JUDGE_VLLM_PORT:-8002}"
TARGET_NAME="${JB_TARGET_SERVED_NAME:-target}"
JUDGE_NAME="${JB_JUDGE_SERVED_NAME:-judge}"

NGPU=1
if command -v nvidia-smi >/dev/null 2>&1; then
    NGPU="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi
TARGET_GPU="${JB_TARGET_GPU:-0}"
JUDGE_GPU="${JB_JUDGE_GPU:-1}"
MEM_UTIL="${JB_GPU_MEM_UTIL:-0.90}"
if [[ "$NGPU" -lt 2 ]]; then
    echo "Expected two visible GPUs; detected $NGPU" >&2
    exit 1
fi
echo "Detected $NGPU GPU(s): target on GPU $TARGET_GPU, judge on GPU $JUDGE_GPU, mem_util=$MEM_UTIL"

HOSTNAME_STR="$(hostname)"
: > "$PIDFILE"

cleanup() {
    while read -r pid; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done < "$PIDFILE"
}
trap cleanup EXIT INT TERM

echo "Starting target vLLM ($TARGET_MODEL_PATH) on port $TARGET_PORT"
CUDA_VISIBLE_DEVICES="$TARGET_GPU" "$VLLM_BIN" serve "$TARGET_MODEL_PATH" \
    --served-model-name "$TARGET_NAME" \
    --host "$HOSTNAME_STR" --port "$TARGET_PORT" \
    --api-key "$JB_VLLM_API_KEY" \
    --gpu-memory-utilization "$MEM_UTIL" \
    > "$HERE/target_vllm.out" 2>&1 &
echo $! >> "$PIDFILE"
echo "$HOSTNAME_STR:$TARGET_PORT" > "$HERE/target_host.log"

echo "Starting judge vLLM ($JUDGE_MODEL_PATH) on port $JUDGE_PORT"
CUDA_VISIBLE_DEVICES="$JUDGE_GPU" "$VLLM_BIN" serve "$JUDGE_MODEL_PATH" \
    --served-model-name "$JUDGE_NAME" \
    --host "$HOSTNAME_STR" --port "$JUDGE_PORT" \
    --api-key "$JB_VLLM_API_KEY" \
    --gpu-memory-utilization "$MEM_UTIL" \
    > "$HERE/judge_vllm.out" 2>&1 &
echo $! >> "$PIDFILE"
echo "$HOSTNAME_STR:$JUDGE_PORT" > "$HERE/judge_host.log"

echo "Wrote host files:"
echo "  target_host.log -> $(cat "$HERE/target_host.log")"
echo "  judge_host.log  -> $(cat "$HERE/judge_host.log")"
echo "Logs: target_vllm.out, judge_vllm.out. Stop with: bash serve_vllm.sh --stop"
echo "Waiting on vLLM servers (Ctrl-C to stop)..."
wait
