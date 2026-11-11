#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
CI_PYTHON="${LLMGE_TEST_PYTHON:-.ci-venv/bin/python}"
"$CI_PYTHON" -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude .venv,.ci-venv
"$CI_PYTHON" -m flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude .venv,.ci-venv
exec bash scripts/workflows/python_app_no_lint.sh "$@"
