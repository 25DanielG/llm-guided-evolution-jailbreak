#!/usr/bin/env bash
# Explicit, separate environment; never changes the experiment environment.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${JB_SERVING_ENV:?Absolute path for the separate serving environment}"
: "${JB_BOOTSTRAP_PYTHON:?Existing Python 3.12 executable; no interpreter downloads}"
[[ "$JB_SERVING_ENV" = /* ]] || { echo 'JB_SERVING_ENV must be absolute' >&2; exit 2; }
"$JB_BOOTSTRAP_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12)'
if [[ -e "$JB_SERVING_ENV" ]]; then
    echo 'Target exists; refusing to modify an existing environment. Choose a fresh directory.' >&2
    exit 2
fi
uv venv --python "$JB_BOOTSTRAP_PYTHON" "$JB_SERVING_ENV"
export UV_HTTP_RETRIES=0 UV_HTTP_TIMEOUT=20
uv pip compile --python "$JB_SERVING_ENV/bin/python" --torch-backend=cu129 --generate-hashes \
    --output-file "$JB_SERVING_ENV/requirements.lock" requirements-serving.txt
uv pip sync --python "$JB_SERVING_ENV/bin/python" --torch-backend=cu129 --require-hashes "$JB_SERVING_ENV/requirements.lock"
uv pip freeze --python "$JB_SERVING_ENV/bin/python" > "$JB_SERVING_ENV/installed-requirements.txt"
"$JB_SERVING_ENV/bin/python" - <<'PY'
import importlib.metadata as m
import json, pathlib, sys
import torch, vllm, transformers, mistral_common, openai_harmony
assert m.version('vllm') == '0.17.0'
assert torch.__version__.split('+')[0] == '2.10.0'
assert torch.version.cuda == '12.9', torch.version.cuda
record = {'python':sys.version, 'torch_cuda':torch.version.cuda,
          'packages':{n:m.version(n) for n in ('torch','vllm','transformers','mistral_common','openai-harmony')},
          'gpu_runtime_verified':False}
pathlib.Path(sys.prefix, 'runtime-installation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY
