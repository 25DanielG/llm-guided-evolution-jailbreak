#!/bin/bash
# Post-hoc LLM judge: label a blind sheet with an open-weights model using the human labeling guide.
# Script arguments pass straight through to sota/Jailbreak/llm_label_judge.py, e.g.
#   sbatch run_llm_label_judge.sbatch --sheet <dir>/labeler_2.csv --out <dir>/llama70b_labels.csv
# Default model: the team's shared Llama-3.3-70B-Instruct (bf16, needs 2 x 80GB GPUs). Override with
# JB_LABEL_JUDGE_MODEL_PATH (and pass --tp 1 for a model that fits on one GPU). Gemma-2 and Mixtral need --fold-system.
#SBATCH --job-name=jb_llm_label
#SBATCH -G 2
#SBATCH -C "A100-80GB|H100|H200"
#SBATCH -t 1:00:00
#SBATCH --mem=160G
#SBATCH -c 8
#SBATCH -o jb_llm_label_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda 2>/dev/null || true
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
export PATH="$HOME/.local/share/uv/tools/vllm/bin:$PATH"  # ninja for any flashinfer JIT build

MODEL="${JB_LABEL_JUDGE_MODEL_PATH:-/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct}"
"$HOME/.local/share/uv/tools/vllm/bin/python" sota/Jailbreak/llm_label_judge.py --model "$MODEL" --tp 2 "$@"
