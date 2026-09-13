# Running the Jailbreak Evolution Pipeline on PACE ICE

This runbook covers the prepared `codex/pace-jailbreak-prep` branch. The pipeline uses three persistent services: a 70B mutation model, an 8B target model, and an independent 8B LLM judge. Evaluation jobs are CPU-only HTTP clients.

## Local verification before pushing

```bash
git switch codex/pace-jailbreak-prep
module load uv 2>/dev/null || true
uv sync --frozen
uv run python slurm.py
LLMGE_AUTO_START_SERVER=0 JB_VLLM_API_KEY=test-key \
  uv run pytest -q tests/jailbreak
LLMGE_AUTO_START_SERVER=0 JB_VLLM_API_KEY=test-key \
  uv run python sota/Jailbreak/preflight.py --mode local
```

`src/cfg/constants.py` must resolve to `constants_jailbreak.py`. Generated files should remain unchanged after rerunning `slurm.py`.

## One-time ICE setup

```bash
ssh <GT_USERNAME>@login-ice.pace.gatech.edu
git clone <REPOSITORY_URL>
cd llm-guided-evolution-jailbreak
git switch codex/pace-jailbreak-prep

module load uv
uv sync --frozen

module load cuda
bash scripts/setup_vllm_env.sh
```

The vLLM bootstrap creates a separate Python 3.12 environment at `~/scratch/llmge-vllm` and installs the pinned `vllm==0.29.0`. Set `JB_VLLM_ENV` to override that location.

Verify the scheduler and shared models:

```bash
sinfo -o "%P %N %G %f" | grep -Ei "H200|H100|A100"
test -r /storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct/config.json
test -r /storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.1-8B-Instruct/config.json
```

The mutator defaults to two H200/H100 GPUs. A 70B BF16 model does not fit on two 40 GB A100s; an A100-40GB deployment needs a separately reviewed four-GPU profile.

## Prepare the approved benchmark

The repository commits only a benign test fixture. Downloaded harmful-behavior CSVs are ignored by Git.

```bash
uv run python sota/Jailbreak/pull_behaviors.py \
  --n 30 \
  --seed 0 \
  --out "$PWD/sota/Jailbreak/behaviors/advbench_30.csv"
```

The downloader pins and verifies the source SHA-256. Review the resulting file under the lab's red-team research policy before use.

## Configure and launch the pilot

```bash
export JB_BEHAVIORS_PATH="$PWD/sota/Jailbreak/behaviors/advbench_30.csv"
export JB_VLLM_BIN="$HOME/scratch/llmge-vllm/bin/vllm"
export JB_VLLM_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"

export JB_TARGET_MODEL_PATH=/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.1-8B-Instruct/
export JB_JUDGE_MODEL_PATH=/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.1-8B-Instruct/
export JB_MUTATOR_MODEL_PATH=/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct/

uv run python slurm.py
sbatch --test-only server.sh
sbatch --test-only jailbreak_vllm.sbatch
bash launch_jailbreak.sh
```

The launcher creates output directories, starts both service jobs, waits for their endpoints, performs PACE preflight, runs one seed evaluation, submits evolution, and schedules service cleanup after the controller exits.

Pilot defaults are one generation, four starting individuals, four retained individuals, four behaviors per evaluation, and two concurrent HTTP requests. Override without editing source:

```bash
export LLMGE_NUM_GENERATIONS=3
export LLMGE_START_POPULATION_SIZE=16
export LLMGE_POPULATION_SIZE=16
export JB_N_BEHAVIORS_PER_EVAL=8
export JB_MAX_CONCURRENCY=4
```

Regenerate `run.sh` after changing generation defaults only if the cluster configuration itself changed; evolutionary size settings are read from the environment at runtime.

## Monitoring and outputs

```bash
watch squeue -u "$USER"
tail -f run_job_outputs/server/slurm-<JOBID>.out
tail -f run_job_outputs/jailbreak-vllm/slurm-<JOBID>.out
tail -f sota/Jailbreak/target_vllm.out
tail -f sota/Jailbreak/judge_vllm.out
```

Outputs are:

- `sota/Jailbreak/models/llmge_models/prompt_<gene>.py`: evolved strategies.
- `sota/Jailbreak/results/<gene>_results.csv`: two-objective fitness.
- `sota/Jailbreak/results/<gene>_audit.jsonl`: per-behavior prompt, response, judge rationale, confidence, and cache metadata.
- `jailbreak_test/checkpoint_gen_<n>.pkl`: evolution checkpoints.
- `run_job_outputs/`: scheduler logs.

## Recovery and cleanup

The launcher records job IDs in `.run_state/jailbreak-<RUN_JOB>.env`.

```bash
source .run_state/jailbreak-<RUN_JOB>.env
squeue -j "$RUN_JOB,$VLLM_JOB,$MUTATOR_JOB,$CLEANUP_JOB"
```

To stop a failed or abandoned run:

```bash
scancel "$RUN_JOB" "$VLLM_JOB" "$MUTATOR_JOB" "$CLEANUP_JOB"
```

To resume, rerun `launch_jailbreak.sh`. `run_improved.py` loads the latest checkpoint from `jailbreak_test` when present. Move that directory aside before launching a deliberately fresh experiment.
