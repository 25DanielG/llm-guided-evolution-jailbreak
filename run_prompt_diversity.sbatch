#!/bin/bash
# Submit from the repository root:
# sbatch run_prompt_diversity.sbatch <run_directory> [target_checkpoint]
#SBATCH --job-name=prompt_diversity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=prompt_diversity_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"

RUN_DIRECTORY="${1:-jb_5956355}"
TARGET_CHECKPOINT="${2:-${JB_TARGET_MODEL_PATH:-$HOME/scratch/llm_storage/Llama-3.1-8B-Instruct}}"
SBERT_REVISION=e8c3b32edf5434bc2275fc9bab85f82640a19130
SBERT_CHECKPOINT="${JB_DIVERSITY_EMBEDDING_MODEL_PATH:-$PWD/results/analysis/embedding_smoke/embedding_cache/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/$SBERT_REVISION}"
ANALYSIS_PYTHON="$PWD/.venv/bin/python"

for checkpoint in "$TARGET_CHECKPOINT" "$SBERT_CHECKPOINT"; do
    if [[ ! -f "$checkpoint/config.json" ]]; then
        echo "Missing local checkpoint: $checkpoint" >&2
        exit 1
    fi
done
if [[ ! -d "$RUN_DIRECTORY" || ! -x "$ANALYSIS_PYTHON" ]]; then
    echo "Run directory or repository Python environment is missing" >&2
    exit 1
fi

# Keep inference on the allocated CPUs and use the already downloaded assets.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
export USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname): run=$RUN_DIRECTORY CPUs=$SLURM_CPUS_PER_TASK"
echo "Target tokenizer: $TARGET_CHECKPOINT"
echo "Sentence-BERT checkpoint: $SBERT_CHECKPOINT"

srun "$ANALYSIS_PYTHON" analysis/scripts/prompt_diversity.py "$RUN_DIRECTORY" \
    --target-tokenizer "$TARGET_CHECKPOINT" \
    --embedding-model "$SBERT_CHECKPOINT" \
    --device cpu --batch-size 8

# Analysis preserves n-gram outputs on embedding errors; make that failure
# visible in the batch job's exit status when both metrics were requested.
"$ANALYSIS_PYTHON" - "$RUN_DIRECTORY" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1]).resolve()
run_id = run.parent.name if run.name in ("ckpt", "checkpoints") else run.name
output = Path("results/analysis") / run_id / "prompt_diversity"
metadata = json.loads((output / "metadata.json").read_text())
if metadata["semantic_error"]:
    raise SystemExit("Semantic analysis failed: " + metadata["semantic_error"])
print("Completed both metrics. Reports:", output.resolve())
PY
