# Paired difficulty-weighted tournament experiment

Base: `MosesTheRedSea-main` at `7349752f1a246730056277ba066bf5cdd2f2950c`.
Branch: `experiment/difficulty-weighted-tournament`.

The two **new parent selectors** are `JB_SELECTION=tournament` and
`JB_SELECTION=difficulty_tournament`. Tournament size defaults to 3 and can be
set with `JB_TOURNAMENT_SIZE` outside this fixed campaign. Both sample with
replacement, with random ties. The plain selector uses mean guide score. The
weighted selector uses `sum((1 - population_mean[b]) * score[i,b]) / sum(weights)`.
Scores are continuous guide P(full), including the existing prefilter's zeros.
All-one populations fall back to the plain mean. All-zero cases have weight one
but provide no ranking signal. Weights refresh before every parent selection.

Raw DEAP fitness, SPEA2 elites, mutation/crossover probabilities, and generational
survival are unchanged. This deliberately isolates **parent selection**; it does
not test difficulty-weighted survival. The original loop selects up to 24 parents,
adds six elites, then deduplicates, so population size is not fixed at exactly 24.

## Inputs and preparation

Use the existing Python 3.12 experiment environment. `prepare` does not submit jobs,
contact models, download weights, or invent tiers. It rejects an existing campaign
path. The default is a **fixed unstratified panel**, sampled once with seed 0 and frozen for all six runs.
The default uses all available behaviors, capped at 50; the checked-in CSV has
30 behaviors, so this campaign uses all 30 in a frozen order. The manifest records
the actual panel size. An explicit `--n-behaviors` fails if that many are unavailable. No tier
labels are required. Overall verified ASR is the primary endpoint; hard-tier metrics
are omitted entirely. This is a fixed search panel, not held-out generalization.

For a separate future stratified experiment, explicitly pass `--panel-mode stratified
--behaviors /path/to/calibrated.csv`. That mode requires unique behavior text and
valid easy/medium/hard tiers; it does not create labels automatically.

If Mistral is missing, explicitly stage it first (downloads approximately 14 GB):

```bash
.venv/bin/python -m analysis.experiments.difficulty_tournament stage-target \
  --destination "$PWD/experiments/difficulty_tournament/models/Mistral-7B-Instruct-v0.2"
```

This resolves `main` to an exact Hugging Face revision and saves that revision.
The optional `--revision COMMIT` selects an already known revision. No silent
fallback to another target occurs.

```bash
CAMPAIGN="$PWD/experiments/difficulty_tournament/v1"
.venv/bin/python -m analysis.experiments.difficulty_tournament prepare \
  --campaign "$CAMPAIGN" --seeds 42 43 44 \
  --target-model "$PWD/experiments/difficulty_tournament/models/Mistral-7B-Instruct-v0.2"
```

Defaults: 24 initial individuals, 16 evolutionary generations, six elites,
mutation 0.8, crossover 0.15; Mistral-7B target, shared Llama-3.3-70B guide judge,
local Llama-3.1-8B HF mutator, and local HarmBench verifier. Set `--judge-model`,
`--mutator-model`, `--reporting-model`, or `--vllm-bin` only when preparing a new
campaign. Changed models constitute a different experiment.

Preparation records resolved model paths, available origin revisions, small config
hashes, and weight-file sizes/mtimes. **It neither copies nor hashes existing weight
files.** It freezes the working source into an archive with per-file hashes, records
its Git commit/base, and fingerprints Python/serving dependencies. Uncommitted
implementation files are included; source hashes identify the exact snapshot.
Commit the changes before preparing a publication run for a convenient code revision.
Runtime checks verify source/input hashes, dependency versions, and unchanged model
metadata. This is a lightweight deployment fingerprint, not a full weight-content
integrity check. Do not modify model files in place.

## PACE execution

`run_difficulty_tournament.sbatch` requests **five H200 GPUs per task**, 48 CPUs,
256 GB RAM, and 16 hours on `coe-gpu`, account `coc`, QoS `coe-ice`. Target uses one
GPU, the 70B judge two (tensor parallelism), verifier one, mutator one. Each task
starts all four services concurrently, with bounded retries, isolated node-local
compiler caches, loopback ports, and independent work/cache/output directories.
The existing HF mutator is retained to preserve its generation behavior.

```bash
bash scripts/submit_difficulty_tournament.sh "$CAMPAIGN"
```

This submits three concurrent initial-population jobs, followed by all six runs
via `afterok`. There is no array concurrency cap: the main experiment requests
30 H200 GPUs concurrently. The driver does not submit jobs itself.

Each initial job benchmarks request concurrency 16, 32, and 64 on identical benign
warm-up requests, chooses the fastest error-free setting, then freezes that setting
for its pair. Services use the same GPU topology in all runs. This is a throughput
check for target/guide batching, not proof of globally optimal hardware utilization.

Array mapping:

| Index | Seed | Parent selection |
|---|---|---|
| 0 | 42 | plain tournament |
| 1 | 42 | difficulty-weighted tournament |
| 2 | 43 | plain tournament |
| 3 | 43 | difficulty-weighted tournament |
| 4 | 44 | plain tournament |
| 5 | 44 | difficulty-weighted tournament |

Generation-zero gene files, evaluation sidecars, and caches are hash-checked and
copied into both arms. Checkpoint import verifies the original generation-zero
commit, relocates runtime paths, and saves a new arm-specific atomic checkpoint.
Python, NumPy, and selection RNG states are restored. Requests receive stable
seeds where supported; HF mutation uses content-derived batch seeds. GPU inference
and concurrent batching can still vary: repeatability is not bitwise determinism.

## Stop and resume

Slurm requests a checkpoint 15 minutes before timeout. The driver finishes the
current generation and exits 75 after committing. Failed service/model calls
are not treated as behavioral failures. Three evaluation attempts are allowed;
unresolved infrastructure or judge parse failures stop the run. Inspect logs,
resolve the infrastructure problem, and resume explicitly:

```bash
sbatch --array=2,5 run_difficulty_tournament.sbatch run "$CAMPAIGN" required
# Initial stages can also resume from a committed generation-zero checkpoint:
sbatch --array=1 run_difficulty_tournament.sbatch initial "$CAMPAIGN" required
```

A hard timeout may interrupt a generation before a checkpoint is committed; resume
starts from the last committed boundary. Artifacts from an incomplete generation
must not influence the resumed experiment (the driver restores a committed artifact
boundary). New mode refuses any existing run directory. Required mode refuses to
reseed a run with no committed checkpoint. If initialization fails before its first
checkpoint, prepare a new campaign rather than overwriting its partial history.

If an initial array fails, dependent jobs stay pending. After recovering the initial
job, cancel the still-pending dependent array and explicitly submit a new run array
once all three initial bundles are complete. Do not submit duplicate run indices.
A per-run file lock prevents simultaneous writers.

## Evaluation and outputs

```bash
.venv/bin/python -m analysis.experiments.difficulty_tournament summarize \
  --campaign "$CAMPAIGN"
```

The report requires all six completion markers. Per-run champion selection uses
highest **unweighted** guide mean among evaluated genes, with gene-ID tie breaks.
Each champion receives three fresh target generations per behavior, verified by
HarmBench majority vote, without a guide threshold. Verification uses the baseline
target temperature (0), so repeated generations can be identical; these are not
independent stochastic trials or confidence estimates.

Primary endpoint: champion overall verified ASR. Also report champion guide score, per-behavior paired changes, evaluation/request counts,
request failures, and allocated GPU-hours. The observed historical verified ceiling
uses the original **guide-gated** verification logs; it is descriptive and is not
an independently audited population-wide maximum. Verification never enters fitness.

Artifacts include `manifest.json`, frozen panel/source, `initial/`, `runs/`, and
per-job `attempts/`. Selection logs contain the full case vectors, solve rates,
weights, both scalar scores, parent IDs, and selected IDs. Request telemetry includes
phase/model, generation, usage, latency, and errors. Reports include JSON/CSV and
PNG/PDF plots; three paired differences and their mean/range are descriptive, not
strong statistical evidence. Search-panel performance is not unseen-target transfer.

For a bounded live smoke test, prepare a **separate** campaign with `--seeds 42
--population 8 --final-generation 1 --n-behaviors 6`. Keep its output separate from
production. Ordinary tests use mocked inference and never allocate GPUs.
