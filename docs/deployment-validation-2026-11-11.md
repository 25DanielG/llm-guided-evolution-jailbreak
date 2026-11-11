# Deployment validation — 2026-11-11 UTC

Repository: `25DanielG/llm-guided-evolution-jailbreak`, PR #3,
`feat/shared-gptoss-devstral`. This record distinguishes observed prerequisites,
deterministic validation, and actual inference. No model is promoted to tested
or operational by this report.

## Actual session and blockers

The session identifies as `pablo3` on `login-ice-gnr-2.pace.gatech.edu`.
`SLURM_JOB_ID` and `CUDA_VISIBLE_DEVICES` are unset; `nvidia-smi` is not available
on this login node. No current GPU allocation or driver was verified.

Bounded live checks were attempted: `scontrol ping`, `squeue`, `sinfo`,
`scontrol show partition`, and association/QoS queries. They failed or timed out.
DNS lookups for both configured controller/database names fail, as do PyPI,
Hugging Face and GitHub lookups in the terminal. `scontrol` prints controllers
DOWN, but this cannot establish a controller outage from a session that cannot
resolve/reach them. **Live availability, account/QoS entitlement and maximum
permitted walltime remain unknown.** Static configuration was not substituted
for those checks. PACE service/network policy still needs authenticated access.

The tool session has restricted network access and writable roots under the
user's home and `/tmp`. Its filesystem check reports the user scratch path
non-writable even though the path is user-owned. A connected PACE session with
access to the intended persistent storage is required for staging.

Storage observations:

| Location | Observation | Consequence |
| --- | --- | --- |
| `/storage/ice-shared/vip-vvk` | `df` reports 10 TB used, 100%, zero available | Do not stage new models or delete existing team content |
| `/storage/ice1/8/3/pablo3` | User quota 57.51 GiB used of 300 GiB; 189,665 of 1,000,000 files; mode 0700 owned by pablo3 | Potential permitted user storage; roughly 242 GiB quota headroom, but current session cannot write it |
| `/tmp` | About 151 GB filesystem free at inspection | Temporary only; unsuitable for persistent checkpoints or deployment evidence |

Model repository byte counts could not be retrieved because Hugging Face DNS
fails. Quota headroom is not a claim that all selected checkpoints and runtime
caches fit. The staging tool checks metadata sizes, headroom, hashes and complete
native/tokenizer files before promotion. It never overwrites an existing model.

## Serving environment attempt

Created a new, separate Python 3.12.15 venv at
`/home/hice1/pablo3/moses-serving-runtime-0.17.0-20261111` using the existing
interpreter. The resolver failed while fetching
`https://pypi.org/simple/openai-harmony/` with `failed to lookup address information`.
No successful serving lock, freeze, import verification or CUDA runtime test
was produced. **The environment is created, not installed or usable.**

The production experiment environment remains unchanged: the inspected existing
environment has torch 2.6.0, Transformers 4.56.2 and Hugging Face Hub 0.35.1.
The separate candidate serving requirements still specify vLLM 0.17.0 / torch
2.10.0 via its dependencies. Installed CUDA module listings are not driver tests.

## Model smoke results

| Required order | Model | Weights staged in this session | GPU inference | Reason |
| --- | --- | --- | --- | --- |
| 1 | GPT-OSS 20B | No | Not run | No verified allocation; scheduler/model/PyPI DNS unavailable; staging path non-writable |
| 2 | Devstral Small 2 24B | No | Not run | Same prerequisites; must follow successful 20B validation |
| 3 | GPT-OSS 120B | No | Not run | Same prerequisites; additional quota/memory sizing required |

No GPU jobs, full evolutionary runs, or model weight downloads occurred. The
registry remains `configured` for all three. New `smoke_model.sbatch` and
`src.serving_smoke` provide bounded commands for real acceptance once prerequisites
work; their unit-test fixtures do not count as inference. The acceptance tool
checks loading, model ID, benign chat, reasoning/final separation, `/generate`,
native FP8 evidence, observed VRAM/RSS, latency, runtime metadata and cleanup.

## Overnight design and deterministic validation

The original four-hour batch request does not cover the expected ten-hour
sixteen-generation run. A twelve-hour request is documented **only if verified
live partition and account/QoS limits allow it**. Otherwise explicit continuation
uses strict committed checkpoints; no retries or continuation jobs are submitted
automatically. The four-hour production/default requests were not silently
extended using unverified policy assumptions.

Inspection found non-atomic independent population/global saves, independent
latest-file selection, absent driver RNG state, startup re-evaluation and a seed
fallback. The new supervised path requires explicit `new`/`required` semantics,
matched hash-verified checkpoint commits, identity checks, saved population,
hall of fame, archive, histories and driver RNG, and no startup re-evaluation.
It fails on incompatible/corrupt checkpoints or insufficient population instead
of restarting from seed. Existing legacy `auto` callers retain their behavior;
that mode is forbidden in the new overnight launcher.

The final targeted Python 3.12 CPU suite passed **52 tests in 11.38 seconds**:
real DEAP checkpoint save/load and two-step continuation, missing/corrupt/
mismatched checkpoints, interrupted writes, explicit rollback quarantine,
supervisor success/failure/deadline and cooperative warning handling, overlap
protection, complete/incomplete staging promotion, mock smoke evidence/cleanup,
read-only exports, existing serving contracts and run controls/archive checks.
The complete local ordinary suite passed **192 tests, 22 integration cases
deselected, in 251.35 seconds** before four final guard cases were added; the
targeted suite covers those final changes. Final hosted results for the published
commit are recorded in PR #3.
This is deterministic recovery verification, not an end-to-end live-inference
evolution or GPU resume claim.

The supervisor and versioned JSON status records run within the allocation and
survive agent disconnection. Low-frequency bounded observations distinguish
slow-but-active work from suspected stalls; a stale timestamp alone is not proof
of a hang. Slurm accounting observations reconcile timeout/node-loss cases where
the final application status could not be written.

## Next executable steps

1. Use a connected PACE terminal with writable persistent storage and authenticated
   policy access; re-run preflight and verify live allocation/account/QoS/time limits.
2. Resolve/install the separate runtime in a fresh permitted directory, retaining
   its hashed lock, freeze and import metadata; verify driver compatibility on-node.
3. Check quota, stage pinned complete checkpoints, then run the serving-only smoke
   tests in the required order. Preserve the generated JSON, benign responses,
   logs, resource samples and Slurm accounting before updating deployment status.
4. Review the deterministic checkpoint test results and supervisor. Before any
   overnight scientific run, validate the approved runtime/configuration and run
   a small authorized end-to-end checkpoint/continuation test. Do not substitute
   CPU fixtures for real-inference continuation acceptance.
5. Review PR #3 and the [operations/recovery runbook](overnight-operations.md).
   Do not merge it or launch the sixteen-generation experiment as part of this task.

Raw [read-only prerequisite evidence](deployment-prerequisites-2026-11-11.json)
is retained alongside this report. See the
[Claude handoff](claude-review-handoff.md) for independent review without shared
credentials or concurrent edits.
