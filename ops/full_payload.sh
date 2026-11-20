#!/bin/bash
set -euo pipefail
bash run_jailbreak.sbatch
"$JB_EXPERIMENT_PYTHON" -m src.checkpointing --checkpoints "$RUN_ID/ckpt" \
    --global-path "$RUN_ID/global" --checkpoint-file checkpoint_gen_16.pkl
echo 'FULL PAYLOAD PASSED: committed generation 16.'
