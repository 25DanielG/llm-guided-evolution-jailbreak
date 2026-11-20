# GPT-OSS ICE preparation — 2026-11-12 ET

**Update:** serving smoke **7001848** completed successfully; its passed result and
inference verification were inspected. The user approved
the prepared HarmBench 50. The current launch package and remaining access are
documented in [PACE-7001848.md](PACE-7001848.md). The report below is the historical
pre-queue preparation snapshot; its dataset/prior-job status is superseded. The
current launcher handles jobs that have aged out of `scontrol` through accounting.

**GPU execution and submission are blocked in this managed session. No new job IDs exist.**
Scheduler/database and external package/model DNS fail; scratch is read-only under this
session's filesystem restrictions. This does not establish a PACE controller outage.
The existing serving installation and 20B checkpoint work on CPU inspection and remain intact.

## Source and prior job

Reviewed merged [PR #3](https://github.com/25DanielG/llm-guided-evolution-jailbreak/pull/3).
Its head is `f88321e14589e1524e4bce66f18700f01d5bae21`, merge
`1c1b7481fdfc006215538cfb0d1c92ca2856acdf`. The current Moses branch is newer:
`7349752f1a246730056277ba066bf5cdd2f2950c`, adding depth coverage/diversity fixes.
The isolated checkout imports all 12 changed files from that exact ref, verifies their Git
blob hashes, then applies local deployment fixes. `records/branch-source.json` and the
archived source retain provenance; local Git commit `d90ce33` records the import.
Terminal Git network access did not work; the connected GitHub app supplied the files.

Preserved the dirty `/storage/ice1/8/3/pablo3/llmge-redsea` checkout and prior
`/storage/ice1/8/3/pablo3/pr3-gpu` worktree. Work is isolated at
`/home/hice1/pablo3/gptoss-ice-20261112`.

Job **6947936 remains unverified**. Bounded `sacct`, `scontrol show job`, and batch-script
retrieval failed to reach scheduler/accounting. Searches of accessible home/scratch
found no job-specific output; `/storage/ice1/8/3/pablo3/pr3-smoke-results` is empty.
No duplicate smoke was submitted. A successful empty `squeue` result does not establish
that a historical job completed successfully.

## Actual environments and models

| Item | Resolved observation |
| --- | --- |
| Experiment Python | `/storage/ice1/8/3/pablo3/llmge-redsea/.venv/bin/python`, Python 3.12.15 |
| Experiment packages | torch metadata 2.6.0, runtime 2.6.0+cu124 / CUDA 12.4; Transformers 4.56.2; NumPy 2.3.3; DEAP 1.4.3; pandas 2.3.2; HF Hub 0.35.1; pytest 8.4.2; no vLLM |
| Serving Python | `/storage/ice1/8/3/pablo3/vllm-0.17-py312-f88321e/bin/python`, Python 3.12.15 |
| Serving packages | vLLM 0.17.0; torch 2.10.0+cu129; Transformers 4.57.6; mistral_common 1.9.1; openai-harmony 0.0.8; HF Hub 0.36.2; Triton 3.6.0; flashinfer-python 0.6.4 |
| Import validation | Serving torch and vLLM import successfully; no GPU visible on login node |
| CUDA module | `cuda/12.9.1` loads, nvcc 12.9.86; allocated-node driver remains unverified |
| 20B checkpoint | `/storage/ice1/8/3/pablo3/pr3-models/openai/gpt-oss-20b`, revision `6cee5e81ee83917806bbde320786a8fb61efebee`, native MXFP4 |
| 20B integrity | Rehashed all 15 files: 27,540,151,074 bytes match cached HF SHA-256/Git-blob hashes and revision; includes Metal weights in addition to vLLM safetensors. No duplicate download. This session did not independently retrieve hashes from HF online. |
| 120B checkpoint | Absent from accessible model roots; registry revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`; staging pending |
| Judge/mutator | Combined launcher resolves both to Llama 3.1 8B, **not** the 70B mutator constant. Expected `$HOME/scratch/llm_storage/Llama-3.1-8B-Instruct` is absent; no alternative found in accessible caches/shared root. Revision unresolved. |
| Reporter | Expected `$HOME/scratch/llm_storage/HarmBench-Llama-2-13b-cls` is absent; revision unresolved. Existing HarmBench reporter code retained. |
| Storage | Scratch quota 93.73 / 300 GiB, about 206.27 GiB remaining at inspection; shared vip-vvk volume is 100% full. This session cannot write scratch. |

Full package freeze, runtime installation record, environment inventory, checkpoint file
hashes, and live preflight observations are retained in `records/`. No installation was
overwritten and no new weights were downloaded.

## Resolved common scientific configuration

`common.env` records these settings explicitly. The prepared sample is a **candidate**;
the benchmark-source discrepancy below remains unresolved.

| Setting | Value |
| --- | --- |
| Targets only | GPT-OSS 20B / GPT-OSS 120B / Mistral 7B Instruct v0.2 / Devstral Small 2 24B Instruct 2512 |
| Judge | Llama 3.1 8B Instruct, `JB_JUDGE_MODE=llm`, `JB_JUDGE_PROMPT=rubric`, rubric `phase1-v1` |
| Selection | `map_elites`, 6 format bins × 3 prior-message depth bins; balanced initial depths and structural depth mutation enabled, as in current Moses |
| Populations | `LLMGE_START_POPULATION_SIZE=24`, `LLMGE_POPULATION_SIZE=24`; elites 6; crossover probability .15; mutation .8; max attempts 5; migration interval 0 |
| Generations | `LLMGE_NUM_GENERATIONS=17`: initialization/checkpoint 0, then `range(1,17)` = **16 evolutionary generations**, ending at 16. Strict continuation starts at committed N+1. |
| Prompts | Existing `sota/Jailbreak/prompt.py`, `Jailbreak/traits`, ConstantRules and crossover templates. File hashes are recorded in each prepared run's experiment identity. |
| Target decoding | Existing budget **512 total generated tokens**, temperature fallback 0.0. Existing per-individual `trait_decoding` can override temperature/top_p; initial values 0.0/1.0. These operators are unchanged across targets. |
| GPT-OSS reasoning | Native Harmony, default medium effort; final content only enters evaluation. The separate serving smoke uses 2048 tokens. Scientific 512-token final-output validation is still required; any budget change must apply to all four targets. |
| Mutator decoding | Existing Llama 3.1 8B via Transformers `/generate`; client budget 850. Driver initial population: temperature uniform .05–.4; mutation: uniform .02–.35; crossover: uniform .05–.1; all rounded to two decimals, top_p .1. CLI fallback .1/.15 is overridden by the driver. Server fallback 1648 tokens is overridden by the client. |
| Judge decoding | Temperature .1, budget 128, logprobs requested with existing fallback |
| Reporter | Existing HarmBench raw-completion classifier, budget 8; never used as fitness |
| Verification/audit | Threshold .5; audit sample size 3 |
| Behavior RNG | `JB_BEHAVIOR_SEED=0`; curriculum disabled; sample frozen once |
| Evolution RNG | New explicit `LLMGE_EVOLUTION_SEED=0` seeds Python/NumPy **driver** RNG for new runs. Strict resume restores saved RNG. Worker RNG and stochastic inference are not claimed bitwise reproducible. |
| Serving limits | Registered target TP=1, context 4096, max 8 sequences, utilization .9. Judge context set 4096. Request timeout 120s; client concurrency 16; readiness timeout 1200s. |

Population parameters are selection/creation targets; the existing loop merges offspring
and elites and removes duplicates, so checkpoint population length can vary.

## Behavior sample: unresolved benchmark discrepancy

Every accessible curated CSV inspected has **30 untiered AdvBench rows**, not 50:
SHA-256 `6b4de2f75683f66b784592b7a8fe05084793340147dea7dbd2ef066749612ea6`.
The repository's sampler silently caps a request at the available row count. The new
freezing tool rejects this short source. No suitable shared tiered CSV was found.

At the user's request this discrepancy is flagged, not presented as resolved.
Prepared an explicitly marked alternative: 50 unstratified behaviors drawn with seed 0
from the official HarmBench **200 standard text** pool. Contextual/copyright subsets
are excluded because this evaluator passes behavior text alone. This changes benchmark
source from the existing AdvBench subset and is not a confirmed common dataset choice.

Official source revision: `centerforaisafety/HarmBench@8e1604d1171fe8a48d8febecd22f600e462bdcdd`;
source Git blob `8a6b423d85e9acdc168c57a6a957755cfa86c4cb` was verified.
Full CSV SHA-256: `8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc`.
Candidate selected CSV SHA-256:
`1787d4db623a90d1374fdfb8415ff577d55b909bca89f165924aec0008205df5`.

`data/shared50/selected.csv` contains exact IDs/text in evaluation order;
`data/shared50/sample.json` contains all 50 IDs, settings, hashes, and candidate status.
Evaluate that frozen file using FULL=1, STRATIFIED=0, N=50, CURRICULUM=0 to avoid
sampling a second time. `freeze_behaviors.py` also preserves tiers and uses the repository's
stratified sampler if a suitable shared tiered source becomes available.

## Prepared validation and isolated runs

| Directory under `/home/hice1/pablo3/gptoss-ice-runs` | Role ports | Smoke/proxy ports |
| --- | --- | --- |
| `20b-pilot` | 21100–21104 | 21600 / 21601 |
| `20b-full` | 21000–21004 | 21500 / 21501 |
| `120b-pilot` | 22100–22104 | 22600 / 22601 |
| `120b-full` | 22000–22004 | 22500 / 22501 |

Each directory has separate source/host files, response/verdict cache, gene results,
checkpoints, global histories and logs, plus per-run Hugging Face, Torch/Inductor,
Triton, vLLM, FlashInfer and CUDA caches. Existing environments are referenced without
syncing dependencies (`UV_NO_SYNC=1`, `UV_OFFLINE=1`). Run IDs are stable across
continuation; scientific identity and driver seed are checked on strict resume.

`smoke_model.sbatch` remains a bounded serving-only GPU acceptance allocation.
`ops/pipeline_probe.py` adds benign target/judge/reporter identity and parsing checks,
response/verdict cache hits, and mutator `/generate` code output. `ops/validation.sbatch`
requests two hours, runs population 8 through initialization and generation 1, starts a
new driver with required resume, and requires committed generation 2. It is prepared,
**not live-validated**. The original 54 targeted CPU tests passed. After synchronization
and the fixture repair, the full current-branch CPU suite passed **214 tests, 22
explicitly gated integration tests deselected, 2 deprecation warnings, 239.81 seconds**.
See `records/current-branch-cpu-validation.xml`. Shell syntax, Python AST and whitespace
checks also passed. No live/GPU/dataset flags were enabled.

The launcher's local fixes expose all service ports, use the existing experiment Python,
allow the reporter checkpoint override, seed the driver separately, and add opt-in
pipeline/continuation checks. No fitness, prompt content, or inference implementation
was changed by those deployment fixes.

The current branch changed mutation to import `src.llm_utils`; its ordinary test fixture
still mocked only `llm_utils`. The full CPU suite initially reported 10 guarded network
attempt failures. Updated the test fixture to patch both module paths while retaining
the external-call guard. Saved the original failure XML. Checkpoint identity now also
checks actual behavior bytes, population sizes, auxiliary model paths and decoding
settings, rather than relying solely on a prepared manifest.

## Required next execution, in order

1. Resume from a PACE session that can resolve/reach Slurm and HF and write the permitted
   scratch model root. Recheck 6947936 and its batch output before submitting a duplicate.
2. Run live `sacctmgr show assoc user=pablo3`, `sacctmgr show qos`, and
   `scontrol show partition`; verify permitted account/QoS/GPU totals and walltime.
   Static configuration lists a 16h inherited maximum, which is **not** live entitlement.
3. Locate or stage the correct Llama 3.1 8B and HarmBench checkpoints, retain immutable
   revisions/checksums, and point both pilot/full environments at them. Resolve the
   common 50-behavior dataset across all four targets.
4. In `20b-pilot`, source `ops/run.env`, set `JB_SERVE_MODEL=gpt-oss-20b`, then submit
   `smoke_model.sbatch` with verified account/QoS/partition flags. Inspect result.json
   for GPU load, identity, final output, separated reasoning, `/generate`, resources,
   driver and cleanup. After a pass, submit `ops/validation.sbatch` and inspect logs,
   pipeline-probe.json and both committed checkpoint generations.
5. After both live acceptance steps pass, source `20b-full/ops/run.env` and submit
   `run_overnight.sbatch`. A 12h override is appropriate only if the verified live
   limits permit it; otherwise use the existing 4h request and explicit required
   continuations. Preserve RUN_ID and never use legacy auto resume here.
6. Immediately prepare/stage 120B after the 20B full submission; do not wait for the
   20B experiment to finish. Use the repository staging tool with freshly measured quota:

   ```bash
   "$JB_EXPERIMENT_PYTHON" -m src.stage_checkpoint --model gpt-oss-120b \
       --root "$JB_SHARED_MODEL_ROOT" --quota-free-gib "$QUOTA_FREE_GIB"
   ```

   Prefer a 4-GPU H200 combined run (target TP=1). A 2-H100 target needs TP=2 and
   **5 GPUs** total with the other roles unchanged; verify live per-user limits.
   Repeat serving/pipeline/continuation validation before its full submission.

Exact per-checkout activation: `cd /home/hice1/pablo3/gptoss-ice-runs/20b-pilot; source ops/run.env`.
Use the equivalent other directory for its run. No submission command has been executed.

## Remaining time estimate

- **Preparation:** serving installation and 20B integrity work are done. After access is
  restored, allow roughly 30–60 minutes to resolve/stage missing auxiliary checkpoints,
  confirm the dataset, and verify live Slurm limits. 120B staging may add 20–60+ minutes,
  depending on actual repository size and transfer rate; not yet measured.
- **Test execution:** 20B serving smoke about 5–20 minutes, bounded to 40 minutes;
  pipeline plus small evolution/continuation about 30–90 minutes, bounded to 110 minutes.
  120B requires its own validation. These are planning estimates, not measured GPU times.
- **Queue/access waits:** unknown. No live queue/account/QoS observation supports a
  numerical estimate; restoring the session's access is a separate prerequisite.
- **Full run:** earlier PR planning estimated about 10h for 16 generations. It is not
  validated for this 50-behavior setup; measure the pilot before relying on that estimate.

**Submitted this session: none. Serving GPU acceptance: not run. Live continuation:
not run. Full 20B and 120B experiments: not launched.**
