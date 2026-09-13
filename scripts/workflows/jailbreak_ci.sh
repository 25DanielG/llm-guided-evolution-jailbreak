#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

uv sync --frozen
export LLMGE_AUTO_START_SERVER=0
export JB_VLLM_API_KEY=test-key
uv run flake8 sota/Jailbreak tests/jailbreak slurm.py \
    --count --select=E9,F63,F7,F82 --show-source --statistics
uv run pytest -q tests/jailbreak
