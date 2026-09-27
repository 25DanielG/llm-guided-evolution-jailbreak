# Jailbreak Domain

An individual is a trait-segmented prompt builder, found in `prompt.py` with `# --OPTION--` blocks.
Fitness is how the evolved prompt jailbreaks a target model, scored by a cached judge cascade
(cache -> anchored refusal regex -> judge model). `JB_JUDGE_MODE` selects the judge model
(`llm` = instruct-rubric, `guard` = Llama-Guard).

Objectives (order matches `FITNESS_WEIGHTS`): `harm_success_rate` (maximize),
`diversity` (maximize; TF-IDF novelty vs the generation, filled by `run_improved.py`
post-eval), `avg_prompt_tokens` (minimize). Per-behavior detail is written to
`results/<gene_id>_results.json` alongside the CSV.

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

`run_jailbreak.sbatch` requests 3 GPUs and starts three services, each on its own GPU:
the target vLLM (`:8001`), the judge vLLM (`:8002`), and the mutator LLM server
(`:8137`), each wrapped in a start-with-retry guard; it then runs `run_improved.py`.
Override models/behaviors via env before submitting (`JB_TARGET_MODEL_PATH`,
`JB_MUTATOR_MODEL_PATH`, `JB_BEHAVIORS_PATH`, ...) or edit
`src/cfg/constants_jailbreak.py` for population size / generation count.

Before a full run, smoke-test the wiring: `python sota/Jailbreak/smoke_test.py`.

Results: `sota/Jailbreak/results/<gene_id>_results.csv` (+ `_results.json` per-behavior
detail) per gene, `jb_ckpt/checkpoint_gen_*.pkl` + `global_gen_*.pkl` per generation.
Inspect one gene with `python sota/Jailbreak/individual_card.py <gene_id>`.
