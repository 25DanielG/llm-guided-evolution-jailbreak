# Diversity through generation 10

All **ten completed overnight experiments**, covering generations **0–10**.
The analysis completed on October 22, 2026. These figures were regenerated from
its validated numerical data and fixed projection coordinates on October 25.
The [October 19 gallery](../clustering_robustness_20261019_1535/README.md)
contains the earlier generation-0-to-4 comparison.

## Figures for a midpoint update

1. **Main diversity result:** [pairwise_diversity.png](pairwise_diversity.png)
   shows evolution and held-out behavior trajectories. Strong lines are equally
   weighted means across ten runs; faint lines show individual runs.
2. **Population maps:** [generation_opacity_overview.png](generation_opacity_overview.png)
   shows one representative behavior per run. Earlier generations fade;
   generation 10 is opaque. Each panel has independent coordinates and colors.
3. **Initial/middle/final snapshots:** [generation_opacity_atlas.pdf](generation_opacity_atlas.pdf)
   has one page per run, with generations **0, 5, and 10** on fixed coordinates.
4. **Baseline sensitivity:** [baseline_sensitivity.png](baseline_sensitivity.png)
   compares generation **0 → 10** with **1 → 10**, using the same retrospective
   generation-0-to-10 cluster fit. Each endpoint comparison matches eligible cases.

### Pairwise diversity

![Pairwise diversity across generations 0–10](pairwise_diversity.png)

| Behavior set | Generation 0 | Generation 10 | Relative increase |
| --- | ---: | ---: | ---: |
| Evolution | 0.0614 | 0.1683 | 174% |
| Held-out | 0.0625 | 0.1701 | 172% |

Pairwise distance increases in all ten runs on both behavior sets. Removing
duplicate prompt texts still leaves a 141% increase in mean evolution pairwise
distance. Across 74,044 unique observed prompts from both sets and all generations,
the global mean pairwise cosine distance is 0.1889; a change of 0.04 is 21.2% of
that empirical reference.

### Generation-opacity overview

![Representative behavior maps for all ten runs](generation_opacity_overview.png)

Cases are selected numerically: within each run, choose the behavior closest to
the median relative generation-0-to-10 nearest-neighbor change, before inspecting
the projections. Exact duplicates overlap, with marker area reflecting multiplicity.
Colors represent HDBSCAN groups fitted in the original embedding space. The 2D
axes have no cosine-distance units; original-space measurements determine the
quantitative findings.

### Baseline sensitivity

![Generation-0 and generation-1 baseline sensitivity](baseline_sensitivity.png)

Generation 0 contains 41.7% duplicate prompt texts on average within behavior
populations. The generation-1 baseline and exact-prompt deduplication therefore
provide separate sensitivity checks. In the primary population-weighted evolution
comparison, all ten runs have greater between-centroid distance and none has
within-cluster tightening. After deduplication, 3/10 evolution runs tighten within
clusters. Mean held-out between-centroid distance increases 88%; mean held-out
within-cluster distance increases 96%. Expansion is consistent; tightening varies.

## Additional figures

- [All four diversity metrics, both behavior sets](diversity_trajectories.png)
- [Changes normalized by initial mean pairwise distance](normalized_diversity_changes.png)
- [Per-run evolution trajectories](clustering_trajectories.png)
- [Matched generation-0-to-10 clustering changes](clustering_changes.png)
- [Prompt-pair coverage, duplicates, population size, and unclustered fraction](data_quality.png)

Each PNG has a PDF counterpart with the same basename. The atlas is PDF-only.

## Individual runs: generations 0, 5, and 10

| Run | PNG | PDF |
| --- | --- | --- |
| broad_01 | [View](plots/broad_01_repeat_01/generations_0_5_10.png) | [Download](plots/broad_01_repeat_01/generations_0_5_10.pdf) |
| broad_02 | [View](plots/broad_02_repeat_01/generations_0_5_10.png) | [Download](plots/broad_02_repeat_01/generations_0_5_10.pdf) |
| broad_03 | [View](plots/broad_03_repeat_01/generations_0_5_10.png) | [Download](plots/broad_03_repeat_01/generations_0_5_10.pdf) |
| narrow_01 | [View](plots/narrow_01_repeat_01/generations_0_5_10.png) | [Download](plots/narrow_01_repeat_01/generations_0_5_10.pdf) |
| narrow_02 | [View](plots/narrow_02_repeat_01/generations_0_5_10.png) | [Download](plots/narrow_02_repeat_01/generations_0_5_10.pdf) |
| narrow_03 | [View](plots/narrow_03_repeat_01/generations_0_5_10.png) | [Download](plots/narrow_03_repeat_01/generations_0_5_10.pdf) |
| random_01 | [View](plots/random_01_repeat_01/generations_0_5_10.png) | [Download](plots/random_01_repeat_01/generations_0_5_10.pdf) |
| random_02 | [View](plots/random_02_repeat_01/generations_0_5_10.png) | [Download](plots/random_02_repeat_01/generations_0_5_10.pdf) |
| random_03 | [View](plots/random_03_repeat_01/generations_0_5_10.png) | [Download](plots/random_03_repeat_01/generations_0_5_10.pdf) |
| random_04 | [View](plots/random_04_repeat_01/generations_0_5_10.png) | [Download](plots/random_04_repeat_01/generations_0_5_10.pdf) |

## Method and limits

- Four random, three narrow, and three broad 30-behavior subsets; a common
  30-behavior held-out set; one shared initial population of 24 strategies and
  evolution seed 42.
- Sentence-BERT `sentence-transformers/all-mpnet-base-v2` with the analysis's pinned
  model/preprocessing provenance; cosine distances measured in original space.
- Retrospective HDBSCAN, primary minimum cluster size 5, with sensitivity sizes
  3 and 8. Evolution groups are fitted per run/behavior; held-out groups pool
  runs within each held-out behavior.
- Main trajectory plots hold behavior eligibility fixed across all generations
  separately for each run and metric. Cases and prompt pairs are not independent
  evolution replicates. Each map uses one fixed fit over generations 0–10.
- All 110 generation checkpoints passed checksum checks, and held-out evaluation
  is complete. There are 132 missing retained-member evolution observations;
  these reduce prompt-pair coverage rather than being treated as zero distances.
- One seed, shared initialization, and overlapping subsets make these descriptive
  comparisons. Dispersion changes do not establish improved attack performance.

## Reproduce and inspect

Run `python regenerate_figures.py` in this directory with NumPy and Matplotlib
installed. It regenerates all 37 PNG/PDF files from the accompanying summary data
and selected map coordinates. It does not rerun embeddings or fit new maps.

- [Data provenance and map-selection rule](data/provenance.json)
- [Numerical findings and deduplication sensitivity](data/findings.json)
- [Aggregate trajectories](data/aggregate_trajectories.csv)
- [Clustering endpoint changes across sensitivity settings](data/endpoint_changes.csv)
- [Original full-analysis validation](data/validation.json)
- [Regeneration input/output checksums and package versions](figure_manifest.json)

The original full analysis validated 300 run/behavior maps. This figure export
contains the ten selected maps used in the overview and atlas, plus aggregated
distance data. It contains no evaluated prompt text, model responses, or model weights.
