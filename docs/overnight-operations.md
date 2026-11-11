# GPU validation and overnight operations

Use `feat/shared-gptoss-devstral` / PR #3. Codex owns changes and deployment;
Claude Code may independently review a source snapshot and selected logs.
These procedures require no continuously connected AI agent. Nothing here
submits a job automatically, resubmits failures, or starts another experiment.

## Current deployment evidence

See [the validation report](deployment-validation-2026-11-11.md). This tool session
cannot resolve the ICE scheduler, PyPI, or Hugging Face. It has no Slurm job or
CUDA devices. The team volume reports 100% full. The user-owned ICE scratch
quota has headroom, but the managed session reports that path non-writable.
No model inference, GPU smoke test, or full experiment has run.

Pablo's authorization covers preparing environments, staging in storage he may
use, and bounded GPU smoke tests. No additional approval from Baris is required.
PACE policy, actual filesystem permissions, quota and allocation limits still
apply. Do not overwrite or delete another person's models to obtain space.

## Prerequisite checks from a connected PACE terminal

```bash
export JB_EXPERIMENT_PYTHON=/YOUR/EXPERIMENT/ENV/bin/python # Python 3.12
"$JB_EXPERIMENT_PYTHON" -m src.deployment_preflight --root /storage/ice1/8/3/pablo3 \
  --output preflight.json
scontrol ping
squeue -h -u "$USER" -o '%i|%T|%b|%M|%l|%N'
sinfo -N -h -o '%N|%P|%G|%f|%t'
scontrol show partition
sacctmgr -n -P show assoc user="$USER" format=Cluster,Account,Partition,QOS
sacctmgr -n -P show qos format=Name,MaxWall
lfs quota -h -u "$USER" /storage/ice1
```

Read the current authenticated ICE Slurm, software, storage and service/network
policy. Select an entitled GPU partition/account/QoS using live results. Check
**both** partition `MaxTime` and all applicable account/QoS/association walltime
limits. Static `slurm.conf` is not evidence of current limits or idle GPUs.
Record the actual allocation's `scontrol show job JOB_ID`, requested time, GPU
identities and `nvidia-smi`; the login node cannot verify GPU drivers.

The prerequisite collector is read-only, bounds each command to 15 seconds,
and records failed/unavailable queries explicitly. `df` free space and user
quota are separate requirements. Validate write permissions in the intended
directory before preparing large files. Do not put checkpoints on node-local
`/tmp`: they must survive the allocation.

## Separate serving environment and pinned staging

Use explicit paths in legitimately writable persistent storage. Retain the
existing experiment environment. Set `JB_BOOTSTRAP_PYTHON` to an installed
Python 3.12 executable; the setup script does not download Python itself.

```bash
export JB_SERVING_ENV=/YOUR/PERMITTED/STORAGE/vllm-0.17-py312
export JB_BOOTSTRAP_PYTHON=/YOUR/EXPERIMENT/ENV/bin/python
bash scripts/serving/setup_runtime.sh
```

The target must be new. Setup resolves a hashed requirements lock in that
environment, installs it, records a freeze and package/CUDA build metadata,
and checks the vLLM 0.17.0 / torch 2.10.0 imports. Resolution and installation
explicitly select the `cu129` PyTorch backend and enforce the generated hashes.
An empty venv or failed resolver
is not an installed serving runtime. Keep the lock, freeze and installation
record with acceptance results. The default vLLM wheel targets CUDA 12.9;
verify the allocated driver and both Python processes, rather than assuming
that an available CUDA module establishes runtime compatibility.

```bash
export JB_SHARED_MODEL_ROOT=/YOUR/PERMITTED/STORAGE/staged-models
# Recheck quota before EACH model; this is remaining user quota, not df capacity.
"$JB_SERVING_ENV/bin/python" -m src.stage_checkpoint --model gpt-oss-20b \
  --root "$JB_SHARED_MODEL_ROOT" --quota-free-gib VERIFIED_FREE_GIB
```

Repeat for `devstral-small-2-24b`, then `gpt-oss-120b`, only if quota permits.
The registry supplies exact model IDs and full revision SHAs. Staging checks
metadata sizes plus disk/quota headroom, downloads to a separate partial
directory, verifies **every selected file's SHA-256 or Git blob hash**, then
writes `deployment.json` and promotes the complete directory. It refuses an
existing destination and excludes duplicate original/HF weights where needed.
Interrupted partial downloads have no success manifest; inspect/retry staging
explicitly. Use your own authorized HF environment credentials if needed;
never read or copy the team's token files or another agent's authentication.

## Real smoke tests, one model at a time

After the prerequisites above succeed, set the verified shared environment,
storage, results and CUDA module paths. Use a separate serving-only allocation:

```bash
export JB_EXPERIMENT_PYTHON=/YOUR/EXPERIMENT/ENV/bin/python
export JB_SMOKE_RESULTS=/YOUR/PERMITTED/STORAGE/smoke-results
export JB_CUDA_MODULE=YOUR_VERIFIED_CUDA_MODULE
export JB_SERVE_MODEL=gpt-oss-20b
sbatch --partition=VERIFIED_PARTITION --account=VERIFIED_ACCOUNT \
  --qos=VERIFIED_QOS --constraint='H100|H200' --export=ALL smoke_model.sbatch
```

Only supply account/QoS flags required by your verified association. Check the
first result before submitting Devstral, then 120B. For 120B prefer one H200
for headroom, or set `JB_SERVE_TP=2` and request `-G 2 -C H100`. Those are
planning options, not measured memory-fit claims.

`smoke_model.sbatch` requests at most 45 minutes, bounds the command to 40,
disables requeue, and never starts evolution. It runs the pinned vLLM executable
against the staged local path with downloads disabled, checks `/v1/models`,
makes a benign arithmetic chat request, and starts the actual `server.py`
proxy with the existing experiment Python to check `/generate`. GPT-OSS must
demonstrate nonempty reasoning separately from a final answer; Devstral must
provide native FP8 config **and runtime load-log evidence**. Any missing proof
fails acceptance rather than becoming an operational claim.

Each new result directory contains `result.json`, `vllm.log` and
`generate-proxy.log`, including model/revision, Slurm ID, package versions,
driver/torch-CUDA details, command, startup/chat/proxy latency, GPU memory
samples and observed aggregate process-tree host RSS. RSS can double-count
shared pages; sampled peaks are observed bounds, not guaranteed absolute peaks.
Collect Slurm's accounting separately after completion:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,MaxRSS,ReqMem,AllocTRES
```

The script terminates owned service process groups on success or failure.
Inspect cancellation/timeouts using accounting and partial logs. A hard node
failure may prevent the final result write; absence of `result.json` is not a pass.
Do not update registry status to tested/operational until real acceptance and
team access have been demonstrated.

## Overnight walltime and checkpoint policy

Sixteen generations are `LLMGE_NUM_GENERATIONS=17` because the loop uses an
exclusive upper bound. Budget roughly ten hours **plus** service startup and
initial population work. If live partition/account/QoS rules permit twelve
hours, override `--time=12:00:00`. The launcher intentionally retains a four-hour
default because this session cannot verify a longer entitlement. If the allowed
limit is shorter, use explicit continuations of the same run from its last
committed generation. No continuation or retry jobs are submitted automatically.

`run_overnight.sbatch` requires an explicit stable `RUN_ID` and either
`JB_RESUME_MODE=new` or `required`. It refuses `auto`; a missing/corrupt checkpoint
cannot create a new population. `new` refuses existing checkpoint/status paths.
For a future authorized experiment (do **not** run these commands as a smoke test):

```bash
export RUN_ID=run_records/night_001
export JB_RESUME_MODE=new
export LLMGE_NUM_GENERATIONS=17
# Set all validated model, environment, storage and allocation parameters first.
sbatch --time=12:00:00 --partition=VERIFIED_PARTITION --export=ALL run_overnight.sbatch
```

The supervisor runs inside the batch allocation, independent of Codex/Claude,
records the job ID immediately and holds an exclusive per-run lock. It starts
one child, samples progress/log activity every 300 seconds, and has a twelve-hour
hard bound. Slurm's fifteen-minute warning creates a cooperative stop marker.
The evolution driver finishes its current generation if time permits, commits
it and exits 75 (`checkpoint_pause`). If the generation exceeds the warning
window, the previous complete checkpoint remains recoverable. At walltime or
cancellation, owned process groups are terminated; Slurm accounting decides the
final job state. No warning promises that an in-flight generation will finish.
Slurm can report `FAILED / NonZeroExitCode` for the intentional exit 75; the
application's `PAUSED` state and verified checkpoint distinguish that controlled
pause from an exception. Inspect both records before continuing.

Strict checkpoints are atomic population/global pairs plus a final JSON commit
marker containing their hashes and run/model/config identity. They include the
DEAP population/fitness, hall of fame, MAP-Elites archive, global histories and
ancestry, and Python/NumPy driver RNG state. The initial evaluated population
is committed as generation zero. A resumed population is not re-evaluated at
startup. Insufficient valid population fails explicitly rather than reseeding
forever. GPU/model sampling itself is not made bitwise reproducible by RNG restore.

Legacy `auto` behavior remains for existing callers and is **not** the supported
overnight recovery path. Legacy pickle files without committed matched metadata
are rejected by strict resume; preserve them for manual inspection rather than
silently importing a potentially mismatched generation. Only deserialize trusted
project checkpoints. CPU tests exercise real save/load and continuation with
DEAP state and RNG, not end-to-end live-model evolutionary resume.

## Status and recovery

`RUN_ID/status.json` is the stable format (`schema_version: 1`,
[JSON Schema](run-status.schema.json)). Application
`state` is distinct from Slurm `job_state`; after exit the latter is unconfirmed
until accounting is queried. Fields include `run_id`, `job_id`,
`latest_completed_generation`, `checkpoint`, `last_progress_at`,
`last_activity_at`, `heartbeat_at`, `exception`, `returncode`, `exit_reason`,
and `console_log`. Each attempt has its own console and terminal status under
`RUN_ID/attempts/`; a previous status is preserved on explicit continuation.

```bash
"$JB_EXPERIMENT_PYTHON" -m src.run_status summary --status "$RUN_ID/status.json" --refresh
# Optional bounded observation, only on a host where PACE permits it:
nohup "$JB_EXPERIMENT_PYTHON" -m src.run_status observe --status "$RUN_ID/status.json" \
  --interval 300 --checks 144 > "$RUN_ID/observer.log" 2>&1 &
```

Observer calls are bounded to fifteen seconds, stop at a terminal Slurm state
or the configured count, and never submit/cancel jobs. Accounting observations
are persisted separately in `scheduler-observation.json`, avoiding races with
the supervisor. A DNS/accounting failure remains unknown, not a failed GPU job.

Morning checklist:

1. Read the summary, accounting observation and latest attempt log. `progressing`
   means recent committed work. After one hour without a generation, recent log
   activity means `slow_but_active`. After three hours with neither generation
   nor log activity, report `suspected_stall_inspect_logs`. Timing alone cannot
   prove a hang; inspect the active phase, inference services and job resources.
   The monitor never kills or restarts a run merely for slow progress.
2. Confirm the earlier allocation is terminal. Do not run two attempts against
   the same run directory. Keep source, model revisions/paths, prompts, cache
   identity and experiment settings unchanged for continuation.
3. Inspect the latest committed pair with `"$JB_EXPERIMENT_PYTHON" -m src.checkpointing
   --checkpoints "$RUN_ID/ckpt" --global-path "$RUN_ID/global"`. An incomplete
   higher generation is ignored. A corrupt **committed** pair fails; do not
   silently fall back. Verify an older pair explicitly with `--checkpoint-file
   checkpoint_gen_N.pkl` if recovery requires rollback. While all attempts are
   stopped, preserve/quarantine every later checkpoint `.pkl`/`.json` and its
   matching global `.pkl` in a separate recovery directory. The writer refuses
   to overwrite an existing committed generation. Then set
   `JB_CHECKPOINT_FILE=checkpoint_gen_N.pkl` for the explicit continuation.
4. Set `JB_RESUME_MODE=required`, keep the exact `RUN_ID`, and submit one explicit
   continuation with a verified walltime. Resume starts at N+1. A missing,
   corrupt or incompatible pair must stop; never switch to `new` to hide failure.
5. Preserve logs and records for both attempts. Evaluate any partial or orphan
   output from the interrupted generation before scientific reporting; filesystem
   and external inference side effects are not transactionally rolled back.

## Claude Code handoff

Codex owns implementation, deployment and corrective edits. Do not have two
agents edit this checkout. Prepare a detached read-only source export:

```bash
"$JB_EXPERIMENT_PYTHON" -m src.review_snapshot --ref HEAD --destination /NEW/PERMITTED/review-snapshot
```

The destination must be new. It contains committed source and an exact commit
record, not `.git`, untracked credentials, environments, checkpoints or caches.
Use filesystem permissions as an accidental-write guard; the owner can still
change them. Give Claude the [short handoff](claude-review-handoff.md), selected
`status.json` / accounting observations, and relevant sanitized logs separately.
Claude should report findings with evidence, not submit jobs or change deployment
state. It uses its own authentication and bounded review session; no tokens or
subscription capacity are shared or assumed unlimited.
