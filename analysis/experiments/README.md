# Behavior coverage experiment

This experiment compares ten fixed 30-behavior AdvBench subsets: four random,
three locally concentrated, and three broad. A common 30-behavior held-out set
is excluded before subset construction. One frozen population of 24 strategies
starts every run. Generation 0 is the evaluated initial population; the default
pilot performs five evolution transitions, producing six snapshots.

The experiment preserves the `MosesTheRedSea-main` fitness objectives, including
TF-IDF diversity. Its measurements are separate from those selection objectives.
Held-out scores never enter selection. No command below submits work implicitly:
only `sbatch` starts a cluster job.

## Prepare inputs

Install the optional analysis dependencies with `uv sync --extra diversity`.
Run preparation on a machine with access to GitHub and the pinned embedding
checkpoint; on cluster login nodes, perform embedding work in a scheduled CPU
allocation. The input is the `goal` column, not AdvBench's target completions.

```bash
uv run --extra diversity python analysis/scripts/coverage_experiment.py prepare \
  --output results/coverage/pilot --seed 42
```

Preparation resolves the upstream commit, downloads its CSV, removes exact
behavior duplicates, and caches pinned Sentence-BERT embeddings. For offline
preparation supply `--source-csv`, `--source-revision` (the upstream commit), and
`--embedding-model` (a local checkpoint). Local checkpoints are content hashed.
Use the same local checkpoint location on analysis nodes.

Coverage distance is the average, over the non-held-out pool, of the cosine
distance to the nearest selected behavior. Lower is broader. Held-out and broad
sets use greedy coverage; narrow sets use the 30 closest behaviors around three
separated centers. Exact subset membership, checksums, child seeds, coverage
scores, and pairwise overlaps are in `experiment.json`. Broad sets are distinct
but may substantially overlap. Inspect actual coverage rather than assuming
sampling labels imply a particular coverage ordering.

## Freeze the initial population

Use absolute paths for the launcher variables. `freeze-initial` invokes existing
strategy creation once, validates that 24 source files exist and compile, and
archives them without evaluating fitness or selecting by score.

```bash
export COVERAGE_MANIFEST="$PWD/results/coverage/pilot/experiment.json"
export COVERAGE_INITIAL="$PWD/results/coverage/pilot/initial"
export COVERAGE_RUN_ROOT="$PWD/results/coverage/pilot/initial_creation"
COVERAGE_ACTION=freeze-initial sbatch run_coverage_experiment.sbatch
```

The launcher uses this base's four-service model topology and local model paths.
It writes service logs and connection files inside `COVERAGE_RUN_ROOT`. Run jobs
serially: fixed service ports require exclusive use of that topology on a node.
The existing general-purpose launcher remains unchanged.

## Run, resume, and extend

For the first pilot, run one subset before estimating the whole campaign:

```bash
export COVERAGE_RUN_ROOT="$PWD/results/coverage/pilot/runs/random_01_repeat_01"
export COVERAGE_SUBSET=random_01
export COVERAGE_SEED=42
sbatch run_coverage_experiment.sbatch
```

The launcher calls `run`, which requires running model services, evaluates the
assigned subset, then evaluates archived retained strategies on the held-out
set. Held-out evaluation may run after evolution: immutable generation archives
preserve which strategies belong to each generation. It has separate outputs
and caches and runs in child processes. Completed held-out observations are
reused on reruns; incomplete evaluations are retried.

Other subset IDs are `random_02` through `random_04`, `narrow_01` through
`narrow_03`, and `broad_01` through `broad_03`. Use a distinct run directory for
each subset/repeat. Reuse the same seed across the ten subsets in a repeat block;
use a different seed and new directories for the next repeat block.

Resubmit the same run to resume it. To extend through generation 10:

```bash
COVERAGE_FINAL_GENERATION=10 sbatch run_coverage_experiment.sbatch
```

Only generations with completion markers are resumed. Checkpoints preserve host
RNG state; remote model inference and GPU operations are not guaranteed bitwise
deterministic. Initial strategies, dataset membership, operator settings, and
model/configuration fingerprints must match on resume. Experiment-mode population
collapse raises an error instead of silently generating a replacement population.

Direct CLI equivalents, for already provisioned services:

```bash
uv run --extra diversity python analysis/scripts/coverage_experiment.py run \
  --manifest "$COVERAGE_MANIFEST" --initial-population "$COVERAGE_INITIAL" \
  --run-root "$COVERAGE_RUN_ROOT" --subset random_01 --seed 42 --final-generation 5

uv run --extra diversity python analysis/scripts/coverage_experiment.py evaluate-heldout \
  --manifest "$COVERAGE_MANIFEST" --run-root "$COVERAGE_RUN_ROOT" --final-generation 5
```

## Estimate and analyze

```bash
uv run --extra diversity python analysis/scripts/coverage_experiment.py estimate \
  --manifest "$COVERAGE_MANIFEST" --pilot "$COVERAGE_RUN_ROOT" --repeats 1

uv run --extra diversity python analysis/scripts/coverage_experiment.py analyze \
  --manifest "$COVERAGE_MANIFEST" --runs results/coverage/pilot/runs/* \
  --output results/coverage/pilot/analysis_g05 --final-generation 5
```

Estimates use recorded execution times, assume four allocated GPUs, and scale
approximately with generations. They are planning estimates, not resource
reservations. Repeat counts remain a scheduling decision after the first pilot.

Analysis fits HDBSCAN in original cosine space on unique vectors, restoring
strategy multiplicity when measuring populations. Groups are pooled across time
within each evolution case/run, and across all runs/time within each held-out
case. Primary minimum cluster size is 5, with sensitivity settings 3 and 8.
Within-cluster distance weights member pairs; between-centroid distance weights
cluster pairs equally. Noise remains in cluster-free metrics. Undefined distances
are blank, not zero. Prompt failures and embedding failures reduce coverage.

Outputs include case and generation CSVs, cluster assignments/occupancy,
trajectory summaries, adjusted/unadjusted coverage associations, missing-data
diagnostics, trajectory plots, and a report. Clustering and input hashes are
saved with the report. Use a new analysis directory when adding generations or
runs: retrospective groups can change. Old analyses are not overwritten.

Primary outcomes are held-out within-cluster and between-centroid distances.
Pairwise and nearest-neighbor distances corroborate them without clustering.
Initial fitness adjustment uses generation-0 evolution fitness. Associations
use subset-level summaries, with leave-one-subset-out sensitivity. If at least
three subsets all have two or more runs, a nested subset/run mixed model is also
reported. Convergence failures are explicit. Single-repeat results cannot
separate coverage effects from evolution randomness; neither behaviors nor
prompt pairs count as independent evolution experiments.

## Full finished-campaign diversity analysis

After all runs and held-out evaluations finish, rebuild the reports in a new
directory outside the raw campaign:

```bash
mkdir -p results/analysis/logs
sbatch --output=results/analysis/logs/diversity-%j.out \
  scripts/analyze_coverage_campaign.sbatch \
  results/coverage/overnight results/analysis/overnight_g10_NEW 10
```

This requests eight CPUs, 48 GB, and no GPUs. It verifies every generation's
completion checksums and held-out evaluations, then reads the existing embedding
cache with its recorded model and preprocessing provenance. Missing or mismatched
cache entries stop the pipeline; the analysis does not regenerate prompts or
modify campaign artifacts.

Outputs include the full evolution and held-out report; matched trajectories and
changes normalized by initial pairwise distance; evolution clustering sensitivity
at minimum sizes 3/5/8; duplicate-prompt and persistent-cluster checks; and
interactive maps, generation-opacity overviews, and a PDF atlas through the
requested final generation. `SUCCESS.json` is written only after every stage
finishes. Raw input hashes, analysis source hashes, and coverage diagnostics are
saved alongside the results. One evolution seed and overlapping subsets still
limit inference to descriptive comparisons.

## Offline verification

```bash
.venv/bin/python -m pytest tests/sota/test_coverage_experiment.py \
  tests/sota/test_prompt_diversity.py tests/sota/test_semantic_projection.py \
  tests/sota/test_jailbreak_phase1.py --confcutdir=tests/sota -q
```

The lifecycle test executes the real evolution-loop control flow with mocked
model work: generation 0, interruption/resume, and extension from 5 to 10. The
report integration test uses benign synthetic prompts and a deterministic fake
embedder. Neither test needs model services or downloads.

## Unattended Slurm campaign

The campaign controller runs preparation, frozen initialization, all ten subsets,
and analysis without a persistent login session. By default it submits **one GPU
worker at a time**. Set `--max-gpu-workers` to run independent subsets concurrently;
pending workers also count toward that bound. Preparation and initialization must
finish before subset runs, and analysis waits for all runs. The default GPU request is **6 GPUs × 2 hours
40 minutes = 16 GPU-hours**. The live `coe-ice` QoS checked on 2026-10-18 reports
`MaxTRESMinsPerJob=gres/gpu=960`: 16 GPU-hours per job, and
`MaxTRESRunMinsPU=gres/gpu=1920`: 32 GPU-hours of remaining **running** work per
user. Other jobs under your
account still count toward Slurm's limit; Slurm may hold this campaign's worker
until capacity is available. The controller is CPU-only.

First install the locked optional dependencies once:

```bash
uv sync --frozen --extra diversity
```

Recommended one-command launch from a login node:

```bash
bash scripts/submit_coverage_campaign.sh results/coverage/overnight --generations 5
```

The wrapper validates local target/judge/reporting/mutator directories, downloads
AdvBench and the embedding assets on the login node, then submits the CPU
controller. Staging downloads does **not** load the embedding model or run
embeddings on the login node. The preparation worker uses those staged local
assets, so it does not need external network access. All ten subsets use evolution
seed 42 by default. To run multiple repeat blocks, use `--seeds 42 43 44`.

For a longer horizon, choose `--generations 10` at initial creation. A campaign's
configuration is fixed; later init calls do not change an existing campaign.
The standalone experiment commands above remain available to extend existing
runs and generate a new analysis horizon.

If inputs are already local, direct `sbatch` is also supported:

```bash
sbatch run_coverage_campaign.sbatch "$PWD/results/coverage/overnight" \
  --source-csv /absolute/path/to/harmful_behaviors.csv \
  --source-revision FULL_UPSTREAM_COMMIT_SHA \
  --embedding-model /absolute/path/to/all-mpnet-base-v2 \
  --generations 5
```

Without local inputs, the direct-sbatch route requires internet access from the
preparation node. Prefer the wrapper to stage inputs before submission.

Useful initialization options:

- `--worker-time 02:40:00`: walltime per GPU attempt. Six times this walltime
  must fit `--gpu-hour-limit` (default 16). CPU stages request four hours.
- `--constraint 'H100|H200'`, `--gpu-partition ice-gpu`,
  `--cpu-partition ice-cpu`, plus optional `--account` and `--qos`.
- `--exclude NODELIST`: keep GPU workers off specific nodes after repeated
  node-specific startup failures; CPU controller and preparation jobs are unaffected.
- `--target-model`, `--judge-model`, `--reporting-model`, `--mutator-model`:
  local paths. Defaults are Llama-3.1-8B-Instruct for the target,
  shared Llama-3.3-70B-Instruct for the judge and mutator, and
  HarmBench-Llama-2-13b-cls for reporting. The target and reporter each use one
  GPU; the judge and mutator each use two.
- `--max-chunks 12`: maximum GPU/CPU allocations per stage.
  `--max-failures 3`: bound repeated failures and attempts without durable progress.
- `--max-gpu-workers 8`: allow up to eight GPU jobs (48 GPUs), including queued
  jobs. Combine with a shorter `--worker-time`, for example `01:15:00`, and
  `--gpu-partition coe-gpu,ice-gpu` where your account permits both partitions.
  Slurm staggers starts to satisfy the 32 GPU-hour remaining-runtime allowance;
  eight workers are a ceiling, not a promise that all eight will start together.
  Each timed-out run resumes from its last committed generation. Changing only
  these scheduler settings does not change model or experiment fingerprints.

If source edits prevent an existing campaign from resuming, use:

```bash
bash scripts/resume_coverage_campaign.sh results/coverage/overnight
```

This recovers a runtime under the campaign's `runtimes/` directory, verifies
its source bytes against the recorded hashes, and checks every saved run's
configuration and latest checkpoint before resubmitting failed stages. Current
working-tree edits, run configurations, and checkpoints are preserved. Recovery
stops if it cannot reconstruct an exact source version; it does not bypass the
configuration guard. The submitted controller and workers run from that runtime
copy, so later edits in the main checkout do not change their source files.

Inspect commands and status without submitting anything:

```bash
.venv/bin/python -m analysis.experiments.campaign init results/coverage/overnight
.venv/bin/python -m analysis.experiments.campaign plan results/coverage/overnight
.venv/bin/python -m analysis.experiments.campaign status results/coverage/overnight
```

`campaign.json` holds the configuration. `campaign_state.json` records the active
Slurm job, attempts, stage outcomes, progress, and submission history. Logs are
under `logs/`; each run also has its service logs and generation artifacts.
The controller has an 18-hour CPU allocation and submits a dependent successor
before its own time expires. It reconciles scheduler state before submitting
more work, including recovering a lost submission receipt by its unique job name.

Workers that time out, are preempted, or lose a node resume from completed
generations and scored held-out observations. Three attempts without durable
progress, three ordinary failures, or the chunk limit mark that stage failed.
A failed evolution run does not prevent other subsets from running. Failed
preparation/initialization blocks dependent runs. Final analysis runs only when
all runs complete; it does not silently present a partial campaign as complete.
This bounds resource use but cannot guarantee all ten runs finish in one night:
queue waits and generation runtimes determine the completion time.

To stop new submissions, create `STOP` in the campaign root. The current worker
is allowed to finish. Cancelling a worker with `scancel JOB_ID` also stops the
controller from automatically replacing that cancelled job. Do not rely on
cancelling only the controller to cancel an already-running GPU worker.

```bash
touch results/coverage/overnight/STOP

# Later, after the old controller exits:
.venv/bin/python -m analysis.experiments.campaign resume results/coverage/overnight

# Explicitly retry exhausted failures after fixing the cause:
.venv/bin/python -m analysis.experiments.campaign resume results/coverage/overnight --retry-failed
```

Configuration/data checksum errors stop scheduling for inspection. Do not edit
evolution code or model settings midway through the campaign: the experiment
checks them on resume. Slurm/controller logs remain the source of diagnosis for
cluster outages; no scripts can guarantee recovery from an unavailable scheduler.
