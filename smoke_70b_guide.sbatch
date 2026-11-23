#!/bin/bash
# Validate the 70B guide judge's live chat/logprobs API with two GPUs only.
#SBATCH --job-name=jb_guide_smoke
#SBATCH -N 1
#SBATCH -G 2
#SBATCH -t 0:30:00
#SBATCH --mem=96G
#SBATCH -c 8
#SBATCH -o jb_guide_smoke_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${JB_VLLM_BIN:?Set the tested vLLM executable}"
: "${JB_EXPERIMENT_PYTHON:?Set the experiment Python executable}"
module load "${JB_CUDA_MODULE:-cuda/12.9.1}"
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export JB_JUDGE_PROMPT=guide JB_JUDGE_SERVED_NAME=judge
export JB_JUDGE_MODEL_PATH="${JB_RUN_JUDGE_MODEL:-/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct}"
export JB_CACHE_DIR="${JB_CACHE_DIR:-$PWD/cache/guide-smoke-${SLURM_JOB_ID}}"
port=$((20000 + SLURM_JOB_ID % 20000))
export JB_JUDGE_VLLM_URL="http://127.0.0.1:${port}/v1"

"$JB_VLLM_BIN" serve "$JB_JUDGE_MODEL_PATH" --served-model-name judge \
    --host 127.0.0.1 --port "$port" --tensor-parallel-size 2 \
    --max-model-len 4096 --enforce-eager --gpu-memory-utilization 0.9 \
    > "judge_smoke_vllm_${SLURM_JOB_ID}.out" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true' EXIT
deadline=$((SECONDS + 1500))
until curl -fsS "$JB_JUDGE_VLLM_URL/models" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null || (( SECONDS >= deadline )); then
        tail -n 40 "judge_smoke_vllm_${SLURM_JOB_ID}.out" >&2
        exit 1
    fi
    sleep 5
done

"$JB_EXPERIMENT_PYTHON" - <<'PY'
import sys
sys.path.insert(0, "sota/Jailbreak")
import judge
import vllm_client
import jb_config

vllm_client.assert_model(jb_config.judge_base_url(), "judge")
result = judge.score_response(
    "Explain photosynthesis in two sentences.",
    "Plants use sunlight to turn water and carbon dioxide into sugar, releasing oxygen. The sugar stores energy for growth.",
)
print({key: result.get(key) for key in ("score", "parse_ok", "rating_method", "error", "guide")}, flush=True)
if result.get("parse_ok") is not True or result.get("rating_method") != "guide_p_full":
    raise SystemExit("70B guide judge did not return a valid probability")
print("70B guide judge smoke: PASS", flush=True)
PY
