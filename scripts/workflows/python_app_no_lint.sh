#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
CI_PYTHON="${LLMGE_TEST_PYTHON:-.ci-venv/bin/python}"
"$CI_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
export LLMGE_AUTO_START_SERVER=0
export LLM_SERVER_READY_TIMEOUT=0 LLM_SERVER_READY_CHECK_INTERVAL=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_TORCH=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
mkdir -p tests/results
# Bound the test command independently of dependency installation.
timeout --kill-after=10s 10m "$CI_PYTHON" -m pytest tests --strict-markers \
  -m 'not live_inference and not gpu and not dataset' \
  --junitxml=tests/results/unit.xml --durations=10 -q "$@"
