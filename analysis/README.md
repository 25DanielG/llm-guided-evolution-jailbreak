# Analysis

Centralized location for analysis scripts and dashboard-related work for llm-guided-evolution.

## Purpose
This folder is for post-run analysis only. Scripts here should read experiment outputs, logs, and result files without modifying raw artifacts.

## Structure
- `scripts/`: runnable analysis scripts
- `utils/`: shared helper functions
- `dashboard/`: future dashboard-related code
- outputs should be written to `results/analysis/`

## Conventions
- do not modify raw run outputs
- scripts should run end-to-end
- prefer reusable helper functions over duplicated logic
- use descriptive snake_case filenames

## Prompt diversity

`scripts/prompt_diversity.py` measures case-matched token n-gram and Sentence-BERT
diversity for checkpointed Jailbreak populations. It writes generation/case CSVs,
prompt diagnostics, model metadata, cached embeddings, and two trend plots to
`results/analysis/<run_id>/prompt_diversity/` at the repository root. It preserves
raw experiments and makes no target-model calls. See
[the Jailbreak documentation](../sota/Jailbreak/README.md#prompt-diversity) for
setup, metric definitions, chunk pooling, and historical-run limitations.
