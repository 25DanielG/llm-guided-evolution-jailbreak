#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "vLLM CUDA bootstrap is supported only on Linux; run this on PACE ICE." >&2
    exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required (on ICE: module load uv)." >&2
    exit 2
fi

VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
VLLM_ENV="${JB_VLLM_ENV:-${HOME}/scratch/llmge-vllm}"

uv venv "$VLLM_ENV" --python 3.12 --seed
uv pip install --python "$VLLM_ENV/bin/python" --torch-backend=auto "vllm==$VLLM_VERSION"
"$VLLM_ENV/bin/python" -c "import vllm; print('vLLM', vllm.__version__)"
echo "Set JB_VLLM_BIN=$VLLM_ENV/bin/vllm before launching."
