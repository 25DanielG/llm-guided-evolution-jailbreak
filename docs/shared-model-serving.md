# Shared models on PACE ICE

This extends **25DanielG/llm-guided-evolution-jailbreak**, branch
`MosesTheRedSea-main` (synchronized through merged CI repair
`7f9ded9244aa8cbc8a57e7e1b1e671538f267278`).
The repository default branch is `new_main`; submit these changes to the Moses
branch. No CONTRIBUTING.md, AGENTS.md, or PR template was found on that branch.
Use a feature branch and PR; do not change a running team's checkout.

## Architecture and model registry

`config/served_models.json` is the **manually maintained** source for opt-in
serving profiles. `python3 -m src.model_serving list` shows IDs, revisions,
backend, precision, memory notes, and status. It is not a list of live endpoints.

Existing services are allocation-scoped, not an always-on cluster gateway:

| Role | Existing backend/interface | Discovery and selection |
| --- | --- | --- |
| Mutator | `server.py`: Transformers, FastAPI `/generate`, port 8137 | `hostname.log`, `JB_MUTATOR_MODEL_PATH`, `--llm_model` |
| Target | vLLM `/v1/chat/completions`, port 8001 | `sota/Jailbreak/target_host.log`, `JB_TARGET_VLLM_URL`, served name `target` |
| In-loop judge | vLLM chat, port 8002 | `judge_host.log`, `JB_JUDGE_VLLM_URL`, served name `judge` |
| Held-out reporter | vLLM raw completions, port 8003 | `reporting_host.log`, `JB_REPORTING_VLLM_URL`, HarmBench ID |

`run_jailbreak.sbatch` launches these together on one node. Existing path-based
selection and the Transformers mutator remain available. Opt-in mutator profiles
launch vLLM on port 8004 and make the **same `server.py` `/generate` endpoint** a
proxy. vLLM applies the model's chat template to a user message containing the
pipeline prompt. No second experiment client or new inference library is needed.

Other members can use an active allocation's `http://<compute-host>:<port>/v1`
from authorized ICE sessions, or provide the role URL environment variables.
Host files belong to a checkout: use separate working directories for simultaneous
runs to avoid overwriting discovery files. Services end with the allocation;
no public or permanent endpoint has been provisioned.

## Supported profiles and deployment status

All entries below use vLLM for the opt-in profile. Existing Llama can still use
Transformers through `server.py`. New models are **configured, CPU interface
checks tested, GPU inference pending**. None is operational or downloaded by this
change. `existing` records pre-existing support, not a new deployment verification.

| Profile / experiment selector | Exact model identifier / provider | Precision | GPU memory guidance | Status |
| --- | --- | --- | --- | --- |
| `mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.2` / Mistral AI | BF16 | ~14 GB weights plus runtime/KV; plan >=24 GB | existing |
| `llama-3.3-70b` | `meta-llama/Llama-3.3-70B-Instruct` / Meta | BF16 | ~140 GB weights plus runtime/KV; existing profile TP=2 on 80 GB or larger GPUs | existing |
| `gpt-oss-20b` | `openai/gpt-oss-20b` / OpenAI | Native MXFP4 MoE weights, other tensors in checkpoint precision | Official model minimum 16 GB; shared smoke plan: 1 H100 (80 GB) or H200 (141 GB) | configured |
| `gpt-oss-120b` | `openai/gpt-oss-120b` / OpenAI | Native MXFP4 MoE weights, other tensors in checkpoint precision | Official single-GPU guidance: 80 GB; prefer 1 H200 for headroom or test TP=2 on 2 H100s | configured |
| `devstral-small-2-24b` | `mistralai/Devstral-Small-2-24B-Instruct-2512` / Mistral AI | Native FP8 checkpoint | ~24 GB weights plus runtime/KV/vision overhead; plan >=48 GB; shared smoke plan: 1 H100 or H200 | configured |

Weight estimates are capacity planning, **not measured peak VRAM**. New profiles
use TP=1, context 4096, max 8 sequences, GPU utilization 0.9. GPU memory is distinct
from host RAM: `serve_model.sbatch` requests 96 GB host RAM, one H100 **or** H200,
8 CPUs, and one hour. Override the feature with Slurm `--constraint` at submission;
environment variables inside `#SBATCH` lines are not expanded. These choices need acceptance testing; full 128k/256k
contexts need substantially more KV memory. The combined run still requests 192 GB
host RAM by default; adjust after measurement. It needs target TP + judge TP +
reporter 1 + mutator TP GPUs. Default four; using the 70B profile adds one GPU for
each role that uses it.

### Model-specific behavior

- **GPT-OSS:** Harmony is mandatory. Send structured messages to chat completions;
  do not feed the old raw prompt to a Transformers text pipeline or raw completions.
  vLLM handles Harmony and separates reasoning from final content. The shared parser
  retains reasoning separately and passes only final `message.content` to fitness
  and code extraction. Reasoning-only truncation and unparsed Harmony fail explicitly.
  vLLM 0.17 returns `message.reasoning`; older `reasoning_content` is also accepted.
  Empty/whitespace final answers, malformed messages and tool calls (even with text)
  fail rather than enter fitness scoring. The parser does not execute tools.
  `max_tokens` includes reasoning: try a 2048-token budget initially, leaving room for
  the input inside the 4096 context. The mutator accepts
  `JB_MUTATOR_REASONING_EFFORT=low|medium|high` (default medium). Set
  `JB_MUTATOR_MAX_NEW_TOKENS=2048` to increase the pipeline client default from
  850 (explicit per-call budgets take precedence). Target/judge use
  the model's default medium; direct chat callers can specify `reasoning_effort`.
  MXFP4 is read from checkpoint metadata: do not force generic FP4, bitsandbytes,
  FP8, or BF16 conversion. The profile adds no explicit reasoning parser because
  this vLLM GPT-OSS implementation handles Harmony natively.
- **Devstral:** Use the native Mistral tokenizer/config/load format and FP8 metadata.
  `mistral_common[image]==1.9.1` exceeds the model card's >=1.8.6 requirement and
  matches vLLM's >=1.9.1 requirement. The native-format path avoids relying on a
  Transformers 5 Ministral implementation. This integration sends text only;
  vision and tool execution are not part of the evolutionary pipeline. Direct tool
  clients additionally need `--tool-call-parser mistral --enable-auto-tool-choice`
  and their own tool loop. Do not treat a tool call as a final answer. The card's
  24 GB/4090 statement does not establish that the native FP8 vLLM service fits
  that card. Mistral's hosted API model is now deprecated; the requested downloadable
  open weights remain the model integrated here.
- **Judge:** new judges require rubric mode and validation of JSON parsing before
  experiments. `guide` is calibrated on Llama 3.3 70B; the new profile path rejects
  guide with other models. The HarmBench reporter is unchanged. Use a fresh RUN_ID
  and cache directory whenever any model, precision, prompt, or revision changes.
- `--llm_model` accepts registry keys, with a matching `JB_MUTATOR_PROFILE` and
  `JB_MUTATOR_VLLM_URL`. The CLI allowlist includes profiles without expanding the
  automatic island roster. Legacy `llama3`/`mixtral` selection still works.

## Reproducible staging and environments (maintainer action)

**Model downloads and GPU jobs require project approval. Nothing here has been
executed against GPUs.** Keep environments separate: the experiment's
`pyproject.toml` pins torch 2.6.0, while the candidate vLLM 0.17.0 environment needs
torch 2.10.0. Do not install vLLM into the experiment environment.

The existing team checkpoint root is
`/storage/ice-shared/vip-vvk/llm_storage`. It was readable in this session and
contains `meta-llama/Llama-3.3-70B-Instruct`; the three new model directories were
not present. Mistral 7B was not present under the registry shared path either;
its legacy script path is under `$HOME/scratch/llm_storage`. The opt-in shared
Mistral profile therefore also needs maintainer staging. No live service was
verified. Do not overwrite existing checkpoints, change permissions broadly,
or use shared token files. A shared storage maintainer must stage the new models
and environment with the team's normal access controls and sufficient quota.
`JB_SHARED_MODEL_ROOT` may point at another **team-managed** root.

Choose and record a team serving environment location, then create it using the
repository's Python 3.12/uv convention. These are preparation examples, not a new
personal service:

```bash
# Set TEAM_SERVING_ENV to a maintainer-approved shared environment directory.
uv venv --python 3.12 "$TEAM_SERVING_ENV"
uv pip install --python "$TEAM_SERVING_ENV/bin/python" -r requirements-serving.txt
uv pip freeze --python "$TEAM_SERVING_ENV/bin/python" > "$TEAM_SERVING_ENV/installed-requirements.txt"
export JB_VLLM_BIN="$TEAM_SERVING_ENV/bin/vllm"
```

The requirements file pins the inference runtime, tokenizer, Transformers, and
Harmony versions. Transitive resolution has **not** been verified in this session;
retain the resolved freeze with the GPU acceptance record for identical replay.
The pinned vLLM 0.17.0 source declares torch 2.10.0, Transformers >=4.56,<5,
Mistral Common >=1.9.1 and Harmony >=0.0.3; the direct pins satisfy these ranges.
Its native Mistral config parser reads `params.json`, which avoids the HF
Ministral 3 / Transformers 5 path. This is source compatibility review, not a
successful installation or model-loading test.
Select an installed CUDA module and driver compatible with that wheel, record the
exact module name as `JB_CUDA_MODULE`, and verify `nvidia-smi`, vLLM version, driver,
CUDA, and Python on an allocated node. Do not assume unversioned `cuda` is compatible.

Approved staging example for each new registry entry:

```bash
# MODEL_ID and REVISION come from config/served_models.json, not "main".
# CHECKPOINT is the team root / MODEL_ID. This download is approval-gated.
hf download "$MODEL_ID" --revision "$REVISION" --local-dir "$CHECKPOINT" --exclude 'original/*'
# For Devstral, also exclude 'model-*.safetensors' and
# 'model.safetensors.index.json' to avoid duplicating HF and native-format weights.
```

The registry pins full GPT-OSS revisions from the official HF metadata APIs
(20B: `6cee5e81ee83917806bbde320786a8fb61efebee`;
120B: `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`), and Devstral `55c5b41e98c2dbd21b0c8afffc540dcfc9eb5128`. After a complete,
verified download, put `deployment.json` in each checkpoint directory with
`{"model_id":"<exact ID>","revision":"<resolved full commit SHA>"}`. Keep the
download manifest/checksums and environment freeze with the deployment record.
The launcher rejects missing/mismatched manifests or absent safetensors and uses
HF offline mode. It does not verify every weight hash; staging verification belongs
to the maintainer. Include config, tokenizer/template and native `params.json`/
`tekken.json`/`consolidated*.safetensors` for Devstral; the preflight specifically
checks these native files rather than accepting only HF-format shards. Do not create a manifest for an
incomplete download.

## Start, access, and select

First inspect without starting a server or downloading anything:

```bash
python3 -m src.model_serving list
python3 -m src.model_serving command --model gpt-oss-20b
# Prepare .ci-venv using tests/Testing.md before running these Python 3.12 tests.
.ci-venv/bin/python -m pytest tests/test_shared_model_serving.py --strict-markers -v
bash -n run_jailbreak.sbatch serve_model.sbatch
```

After approval, verified PACE allocation access, staged weights and environment:

```bash
export JB_SERVE_MODEL=gpt-oss-20b    # or gpt-oss-120b / devstral-small-2-24b
# JB_VLLM_BIN and JB_CUDA_MODULE must already name the approved shared environment/module.
export JB_GPU_CONSTRAINT='H100|H200' # use H100 or H200 to select just one family
# Also supply partition/account/QoS verified for your team if PACE requires them.
sbatch --constraint="$JB_GPU_CONSTRAINT" --export=ALL serve_model.sbatch
```

`serve_model.sbatch` is serving-only: no island controller or evolution is
submitted. It prints the node URL; `/v1/models` serves the exact HF model ID.
`src.model_serving serve` requires a Slurm allocation and enough visible GPUs.
For the existing 70B profile, request `-G 2`; TP defaults to 2 in the registry.
For GPT-OSS 120B on two H100s, set `JB_SERVE_TP=2` and submit with `-G 2 -C H100`.
`JB_SERVE_TP` changes vLLM sharding; the Slurm GPU count must match. The one-GPU
registry default remains available for H200 or separately measured 80 GB tests.
Do not submit until the current partition/account/QoS rules are verified.

To use a separately served model as target, export its allocation URL and ID:

```bash
export JB_TARGET_VLLM_URL=http://COMPUTE_HOST:8001/v1
export JB_TARGET_SERVED_NAME=openai/gpt-oss-20b
export JB_TARGET_MODEL_PATH=/storage/ice-shared/vip-vvk/llm_storage/openai/gpt-oss-20b
export JB_TARGET_MAX_NEW_TOKENS=2048
# Use these with the existing eval.py/smoke_test.py interface from an authorized ICE session.
```

For the **existing combined run**, profiles are selected with
`JB_RUN_TARGET_PROFILE`, `JB_RUN_JUDGE_PROFILE`, and/or `JB_RUN_MUTATOR_PROFILE`.
Leave a variable unset to retain the legacy role. For example, after GPU approval,
this performs the existing smoke checks without evolution:

```bash
export JB_RUN_TARGET_PROFILE=gpt-oss-20b
export JB_TARGET_MAX_NEW_TOKENS=2048
export JB_SMOKE_ONLY=1
sbatch -N 1 -G 4 --constraint="$JB_GPU_CONSTRAINT" --export=ALL run_jailbreak.sbatch
```

To select a new mutator instead, set `JB_RUN_MUTATOR_PROFILE` to any new registry
key; the script starts its backend at 8004, exports `JB_MUTATOR_PROFILE` and
`JB_MUTATOR_VLLM_URL`, and passes that key as `--llm_model` to `run_improved.py`.
Set `JB_RUN_JUDGE_PROFILE` for a rubric judge. Select one role per acceptance test
first; existing target/judge/reporter identity checks still apply.

For profile roles, `JB_RUN_TARGET_TP`, `JB_RUN_JUDGE_TP`, and `JB_RUN_MUTATOR_TP`
override registry TP and are passed to both GPU grouping and vLLM. Request
`target TP + judge TP + 1 reporter + mutator TP` GPUs on one node. For example,
GPT-OSS 120B target TP=2 with the other default roles needs `-G 5`, not `-G 4`.

Increase `JB_*_MAX_MODEL_LEN`/generation budgets only after memory validation. Mutator
prompts default to the pipeline's original text; Devstral's optional
`CHAT_SYSTEM_PROMPT.txt` is appropriate for standalone chat, while
`VIBE_SYSTEM_PROMPT.txt` assumes a tool scaffold absent from this pipeline.

Once approval covers a full experiment, unset `JB_SMOKE_ONLY`, use a fresh RUN_ID
and cache, and follow [Jailbreak setup](../sota/Jailbreak/README.md). Do not run a
full experiment as a deployment check.

## PACE evidence and remaining acceptance work

Official PACE sources were consulted/attempted **before implementation**:

- [PACE ICE landing page](https://pace.gatech.edu/ice-cluster/)
- [Current ICE Slurm guide, KB0042096](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096)
- [ICE storage guidance, KB0042094](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042094)
- [Legacy official ICE orientation](https://docs.pace.gatech.edu/ice_cluster/img/ICE_orientation_spring2023.pdf)
- [Georgia Tech 2025 resource overview](https://support.cc.gatech.edu/sites/default/files/2025-grad-orientation.pdf)

Follow-up inspection on **2026-11-11 UTC** verified local ICE scheduler
configuration and installed module listings on `login-ice-gnr-2`:

- `/etc/slurm/slurm.conf` identifies `ice-slurm`, H100/H200 features and typed
  GRES `gpu:h100:8` / `gpu:h200:8`. Its static `ice-gpu` partition includes nodes
  from both families. These are configured resources, not live idle capacity or
  proof of this team's partition/account/QoS entitlement.
- `module -t avail cuda` lists `cuda/12.9.1` and `cuda/13.0.1` among other versions.
  The vLLM 0.17 default binary is built for CUDA 12.9. An available module is not
  proof of the allocated node's driver compatibility; confirm that before use.
- Bounded, read-only `sinfo -N -h -o '%N|%P|%G|%f|%t'` and
  `sacctmgr -n -P show assoc user=YOUR_USERNAME format=Cluster,Account,Partition,QOS`
  still cannot reach the controller/database from this session. Do not infer
  current availability or permissions from the static file.
- The shared root and existing Llama config are readable; all three new registry
  checkpoint paths remain absent. Quota and maintainer write access are unverified.

Current PACE ServiceNow Slurm/storage pages and the ICE landing page remain
inaccessible to the documentation tool. Current service/network usage policy
has not been verified. Read those pages through an authenticated GT session
before deployment. The new serving-only script accepts either H100 or H200;
legacy `server.sh` directives and production loading/readiness timeouts are
unchanged. Do not copy Phoenix or historic ICE partitions into this launch.

Acceptance requires an authorized maintainer to:

1. Read the current ICE Slurm, storage, software/modules and service/network usage
   rules through an authenticated GT session; verify entitlement and current H100/H200
   feature/partition/QoS with PACE before submitting anything.
2. Verify shared quota/permissions, stage immutable model revisions, resolve the
   serving environment, record its freeze and compatible CUDA module/driver.
3. With approval, start one model at a time; verify `/v1/models` and a short benign
   chat completion, final/reasoning separation for GPT-OSS, and FP8/template loading
   for Devstral. Measure GPU and host memory; repeat existing Mistral/Llama checks
   on the candidate vLLM runtime before switching any shared service.
4. Exercise `/generate` through the proxy, then the existing smoke-only run with
   one changed role. For judges check rubric parse failures/logprobs; do not reuse
   the guide calibration with new judges.
5. Record model revision, node/GPU, environment/driver/module, commands, memory,
   request/response evidence and date. Change status to `tested` only after real
   inference and to `operational` only after team access is verified. Review the PR
   before replacing any shared service.

No weights, GPU jobs, or evolutionary experiments were run for this integration.

## Synchronization and CPU validation

PR #3 was synchronized with Moses commit `7f9ded9244aa8cbc8a57e7e1b1e671538f267278`
by feature-branch merge `8d52cd94a81038ae507fe429c945cd33250ba30f`, with no conflicts.
PR #4's CI workflow, Python 3.12 CPU dependencies, guards, explicit integration
markers and production timeout preservation are retained unchanged.

After synchronization, [GitHub Python CI passed](https://github.com/25DanielG/llm-guided-evolution-jailbreak/actions/runs/42940600655)
and the documentation workflow passed. The subsequent deployment preparation adds
tests for GPU/TP selection, native staging checks, team storage overrides and
both reasoning fields; **17 serving unit tests pass locally**. Final full-suite
and hosted results are recorded in PR #3 after this follow-up is published.

CPU tests use mocks and do not establish inference, GPU memory fit or operational
status. The code is prepared for approval-gated smoke testing; staged checkpoints,
a resolved shared serving environment, verified driver and PACE entitlement/policy
are still required before any job can be submitted.

## Official inference references

- [OpenAI vLLM guide](https://developers.openai.com/cookbook/articles/gpt-oss/run-vllm)
  (archived: use current vLLM guidance for runtime versions)
- [GPT-OSS 20B model card](https://huggingface.co/openai/gpt-oss-20b),
  [120B model card](https://huggingface.co/openai/gpt-oss-120b),
  [vLLM GPT-OSS recipe](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)
- [Devstral Small 2 model card](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512),
  [Mistral offline deployment](https://docs.mistral.ai/vibe/code/cli/offline-models),
  [model lifecycle](https://docs.mistral.ai/models/devstral-small-2-25-12)
- [vLLM native Ministral-3 format recipe](https://docs.vllm.ai/projects/recipes/en/latest/Mistral/Ministral-3-Instruct.html)
  (Devstral Small 2 shares this architecture per its model card)
- [vLLM 0.17.0 common requirements](https://github.com/vllm-project/vllm/blob/v0.17.0/requirements/common.txt),
  [CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.17.0/requirements/cuda.txt)
- [vLLM 0.17 chat response protocol (`reasoning`)](https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/entrypoints/openai/chat_completion/protocol.py),
  [native Mistral config parser](https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/transformers_utils/config.py)
