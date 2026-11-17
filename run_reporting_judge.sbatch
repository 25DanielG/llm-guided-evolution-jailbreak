#!/bin/bash
# Post-hoc HarmBench scoring of every logged response (reporting only; never in the evolution loop).
# Script arguments pass straight through to sota/Jailbreak/reporting_judge.py, e.g.
#   sbatch run_reporting_judge.sbatch \
#       --candidates sota/Jailbreak/results/<run_id>/candidates.jsonl \
#       --out sota/Jailbreak/results/<run_id>/reporting.csv
# Override the classifier weights with JB_REPORTING_MODEL_PATH=/path/to/HarmBench-Llama-2-13b-cls.
#SBATCH --job-name=jb_reporting
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -C "A100-40GB|A100-80GB|A40|L40S|H100|H200"
#SBATCH -t 1:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -o jb_reporting_%j.out

# Constraint excludes AMD MI210 and GPUs under 40GB.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda 2>/dev/null || true
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1

MODEL="${JB_REPORTING_MODEL_PATH:-$HOME/scratch/llm_storage/HarmBench-Llama-2-13b-cls}"
"$HOME/.local/share/uv/tools/vllm/bin/python" sota/Jailbreak/reporting_judge.py --model "$MODEL" "$@"
