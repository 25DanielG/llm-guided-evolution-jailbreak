#!/bin/bash
# Serving only; never submits an island controller or evolution.
# Resource conventions copied from server.sh; live partition/QoS verification pending.
#SBATCH --job-name=vip_model
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -C H200
#SBATCH --mem=96G
#SBATCH -c 8
#SBATCH -t 1:00:00
#SBATCH -o vip_model_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${JB_SERVE_MODEL:?Set JB_SERVE_MODEL to a registry key}"
: "${JB_VLLM_BIN:?Set JB_VLLM_BIN to the shared serving environment bin/vllm}"
# Use the same CUDA module as the approved serving environment; do not select latest.
: "${JB_CUDA_MODULE:?Set the CUDA module verified for this serving environment}"
module load "$JB_CUDA_MODULE"
echo "Endpoint: http://$(hostname):${JB_SERVE_PORT:-8001}/v1"
exec python3 -m src.model_serving serve --model "$JB_SERVE_MODEL" \
    --port "${JB_SERVE_PORT:-8001}" --max-len "${JB_SERVE_MAX_LEN:-4096}"
