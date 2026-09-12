#!/bin/bash
# Launch the target + judge vLLM servers for the Jailbreak domain.
#
# vLLM is self-hosted and OpenAI-compatible; no API key is needed (clients send a
# dummy "EMPTY" token). Run this on a GPU node (interactive srun or its own sbatch)
# BEFORE running eval.py / the evolution loop. It writes the servers' host:port to
# target_host.log and judge_host.log next to this script, which eval.py reads.
#
# Usage:
#   bash serve_vllm.sh              # start both servers, stream logs, block
#   bash serve_vllm.sh --stop       # stop servers started by this script
#
# Override via env: JB_TARGET_MODEL_PATH, JB_JUDGE_MODEL_PATH,
#   JB_TARGET_VLLM_PORT (8001), JB_JUDGE_VLLM_PORT (8002),
#   JB_TARGET_SERVED_NAME (target), JB_JUDGE_SERVED_NAME (guard),
#   JB_TARGET_GPU (0), JB_JUDGE_GPU (1 if present else 0),
#   JB_GPU_MEM_UTIL (0.45 when sharing one GPU, else 0.90).

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

TARGET_MODEL_PATH="${JB_TARGET_MODEL_PATH:-/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.1-8B-Instruct/}"
JUDGE_MODEL_PATH="${JB_JUDGE_MODEL_PATH:-/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-Guard-3-8B/}"
TARGET_PORT="${JB_TARGET_VLLM_PORT:-8001}"
JUDGE_PORT="${JB_JUDGE_VLLM_PORT:-8002}"
TARGET_NAME="${JB_TARGET_SERVED_NAME:-target}"
JUDGE_NAME="${JB_JUDGE_SERVED_NAME:-guard}"

# GPU placement: separate GPUs if at least two are visible, otherwise share GPU 0
# with a reduced memory fraction so both 8B models fit.
NGPU=1
if command -v nvidia-smi >/dev/null 2>&1; then
    NGPU="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi
TARGET_GPU="${JB_TARGET_GPU:-0}"
if [[ "$NGPU" -ge 2 ]]; then
    JUDGE_GPU="${JB_JUDGE_GPU:-1}"
    MEM_UTIL="${JB_GPU_MEM_UTIL:-0.90}"
else
    JUDGE_GPU="${JB_JUDGE_GPU:-0}"
    MEM_UTIL="${JB_GPU_MEM_UTIL:-0.45}"
fi
echo "Detected $NGPU GPU(s): target on GPU $TARGET_GPU, judge on GPU $JUDGE_GPU, mem_util=$MEM_UTIL"

HOSTNAME_STR="$(hostname)"
: > "$PIDFILE"

echo "Starting target vLLM ($TARGET_MODEL_PATH) on port $TARGET_PORT"
CUDA_VISIBLE_DEVICES="$TARGET_GPU" uv run vllm serve "$TARGET_MODEL_PATH" \
    --served-model-name "$TARGET_NAME" \
    --host 0.0.0.0 --port "$TARGET_PORT" \
    --gpu-memory-utilization "$MEM_UTIL" \
    > "$HERE/target_vllm.out" 2>&1 &
echo $! >> "$PIDFILE"
echo "$HOSTNAME_STR:$TARGET_PORT" > "$HERE/target_host.log"

echo "Starting judge vLLM ($JUDGE_MODEL_PATH) on port $JUDGE_PORT"
CUDA_VISIBLE_DEVICES="$JUDGE_GPU" uv run vllm serve "$JUDGE_MODEL_PATH" \
    --served-model-name "$JUDGE_NAME" \
    --host 0.0.0.0 --port "$JUDGE_PORT" \
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
