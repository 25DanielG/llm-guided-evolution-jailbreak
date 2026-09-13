#!/bin/bash
#SBATCH --job-name=Jailbreak_vLLM
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH -C "H200|H100|A100"
#SBATCH --mem 96G
#SBATCH -c 16
#SBATCH --output=run_job_outputs/jailbreak-vllm/slurm-%j.out
echo "launching jailbreak target and judge vLLM services"
hostname
module load cuda
nvidia-smi -L

: "${JB_VLLM_API_KEY:?JB_VLLM_API_KEY must be set}"
export JB_VLLM_BIN="${JB_VLLM_BIN:-${HOME}/scratch/llmge-vllm/bin/vllm}"
export JB_JUDGE_MODE="${JB_JUDGE_MODE:-llm}"
export JB_TARGET_SERVED_NAME="${JB_TARGET_SERVED_NAME:-target}"
export JB_JUDGE_SERVED_NAME="${JB_JUDGE_SERVED_NAME:-judge}"

bash sota/Jailbreak/serve_vllm.sh
