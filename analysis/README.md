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

## Semantic projection

After semantic diversity completes, submit the cached-embedding visualization:

```bash
sbatch run_semantic_projection.sbatch results/analysis/jb_5956355/prompt_diversity
```

Or run on an allocated compute node:

```bash
.venv/bin/python analysis/scripts/semantic_projection.py results/analysis/jb_5956355/prompt_diversity
```

The `diversity` optional dependencies include `umap-learn`. Outputs go into
`semantic_projection/` beneath that analysis: a PNG/PDF overlay with earlier
generations faded and the latest opaque, PNG/PDF generation panels, a
self-contained interactive HTML with case selection and playback, coordinates,
projection-quality measurements, and provenance metadata. No inference or
raw-run writes are performed.

One joint densMAP is fit per case across all completed generations, using
unique vectors so identical prompts have identical coordinates. Colors come
from HDBSCAN on original cosine distances. Static plots select the case with
median relative endpoint nearest-neighbor change; `--case-id` overrides it.
Cases have separate coordinate systems. Generic axes have no semantic units.
The original cosine-distance trend is the evidence for local tightening;
2D geometry and exploratory group labels are approximate. See the
[densMAP method documentation](https://umap-learn.readthedocs.io/en/latest/densmap_demo.html).
