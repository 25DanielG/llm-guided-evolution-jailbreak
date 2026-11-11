#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if ! command -v uv >/dev/null 2>&1; then
  curl --fail --location --max-time 60 https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
# Separate CPU environment; never sync/replace the PACE project or serving env.
uv venv --python 3.12 .ci-venv
uv pip install --python .ci-venv/bin/python --torch-backend=cpu -r requirements-ci.txt
