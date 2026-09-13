# Jailbreak Domain

An individual is a trait-segmented prompt builder, found in `prompt.py` with `# --OPTION--` blocks.
Fitness is how the evolved prompt jailbreaks a target model, scored by a cached judge cascade
(refusal regex -> LLM judge).

Objectives: `harm_success_rate` (maximize), `avg_prompt_tokens` (minimize).

## One-time setup

```bash
uv sync --cache-dir ~/scratch/.uv # project env (no vllm)
uv tool install vllm --with ninja --python 3.12 # vllm needs its own env + python 3.12 + ninja
ln -sf constants_jailbreak.py src/cfg/constants.py
uv run slurm.py
uv run python sota/Jailbreak/pull_behaviors.py --n 30 # real AdvBench behaviors (needs internet)
```

## Run

```bash
sbatch run_jailbreak.sbatch
squeue --me
tail -f jb_run_<jobid>.out
grep -a "harm_success_rate=" jb_run_<jobid>.out
```

`run_jailbreak.sbatch` requests 2 GPUs, starts one vLLM server (serves both target
and judge) on GPU0, the mutator LLM server on GPU1, waits for both, then runs
`run_improved.py`. Override models/behaviors via env before submitting
(`JB_TARGET_MODEL_PATH`, `JB_MUTATOR_MODEL_PATH`, `JB_BEHAVIORS_PATH`, ...) or edit
`src/cfg/constants_jailbreak.py` for population size / generation count.

Results: `sota/Jailbreak/results/<gene_id>_results.csv` per gene,
`jb_ckpt/checkpoint_gen_*.pkl` + `global_gen_*.pkl` per generation.
