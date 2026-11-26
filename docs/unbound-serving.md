# Inkling 2 Mini Unbound v2.1

Select Unbound with registry key `inkling-2-mini-unbound-v21`. That profile pins
checkpoint repository `nokeepai/Inkling-2-Mini-Unbound` to revision
`ef74b44edd2851b8e8d221a7313f36c419590430` and serves it under API model name
`inkling-unbound-v21`. The registry key, checkpoint repository ID and served
model name are separate identifiers; the launcher, `/v1/models` readiness check
and mutator client use the served model name. The persistent approved
snapshot is `${PACE_INFERENCE_STATE}/snapshots/inkling-unbound-v21/ef74b44edd2851b8e8d221a7313f36c419590430`.
Set `JB_UNBOUND_CHECKPOINT` to that directory before using `src.model_serving`.
The committed manifest records SHA-256 and byte counts for all 12 files; `check`
verifies them before launch. It does not download or stage model files.

The checkpoint declares `Inkling2LoopForConditionalGeneration`,
`model_type=inkling2_loop`, BF16 activations and NVFP4 weights. Its tokenizer and
chat template are from the same pinned revision. Checkpoint provenance is
revision-pinned and hash-verified. Serving compatibility was previously
demonstrated using the guarded PACE vLLM/Hopper implementation. This PR's generic
registry integration has CPU tests only; its direct generic-launcher GPU loading,
kernel selection and readiness have not been qualified. Registry status is
`configured`, not a claim of live qualification. Production deployment should
continue to use the guarded PACE launcher/runtime, whose worker checks kernel
paths and readiness.

The Unbound GPU-smoke runtime is pinned by the qualified `inkling.lock`: Python
3.12.15, vLLM `0.34.1rc1.dev144+g94b13b3d8` (source revision
`94b13b3d84562054f089e709ba1e32cf9a153bb7`), PyTorch `2.14.0+cu132`, CUDA
13.2 libraries, and Triton 3.8.0. The recorded successful qualification used
one H200 and `guarded_worker.Worker`, plus the reviewed FlashAttention and
Humming kernel selections. Job records 7350751 and 7351445 show this runtime.
These requirements are recorded for provenance; `src.serving_smoke` deliberately
does not launch Unbound through its generic vLLM command, because that command
does not reproduce the guarded worker and kernel policy. To GPU-validate Unbound,
use the established guarded Unbound qualification workflow in the
`inkling-qualification` checkout (the replay and completion-diagnostic job
wrappers), with the pinned `runtimes/inkling` environment and an authorized H200
allocation. Do not substitute the generic `serving_smoke` route.

Qualification evidence: one H200 serving; replay job 7350751 completed 62/62
requests. Diagnostic job 7351445 completed successfully and reverified 12/12
persistent files. The diagnostic used a 40,960-token serving context and tested
budgets up to 8,192; maximum observed completion among rescued cases was 4,119
tokens. Reasoning tokens are included in the OpenAI `max_tokens` allowance. The
native `inkling` reasoning parser exposes `reasoning` separately from final
`content`; Unbound requests do not send GPT-OSS-specific `reasoning_effort`.
`parse_chat_response` rejects malformed responses, tool calls and empty final
content, but it preserves nonempty partial content when `finish_reason` is
`length`. Callers that require a complete answer must check for
`finish_reason == "stop"`; `serving_smoke` does this explicitly. The model
metadata advertises a larger context, but that context has not been qualified
here. Existing Unbound serving evidence is H200; H100 compatibility is not
verified.

The `/generate` client selects its output ceiling in this order: explicit
`max_new_tokens` argument, `JB_MUTATOR_MAX_NEW_TOKENS`, the selected registry
profile's `default_max_tokens`, then the legacy 850-token fallback. Therefore
selecting `JB_MUTATOR_PROFILE=inkling-2-mini-unbound-v21` defaults mutation
requests to 8,192 generated tokens. The Unbound profile caps this at 32,768,
below its 40,960-token serving context; prompts still consume context, so a
request whose prompt plus output allowance exceeds the configured context may
be rejected by the server.

When the combined `run_jailbreak.sbatch` launcher selects a manifest-backed
checkpoint, it verifies the checkpoint before starting services. The serving
process verifies it again immediately before loading. This intentionally hashes
the checkpoint twice (about 19.5 GB for Unbound): the second check avoids
trusting a caller-controlled skip flag or a stale preflight receipt and narrows
the window for a checkpoint change between validation and model load.

Example after starting the qualified guarded service:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="unused")
completion = client.chat.completions.create(
    model="inkling-unbound-v21",
    messages=[{"role": "user", "content": "What is 17 + 25? Reply briefly."}],
    max_tokens=8192,
    temperature=0.1,
)
message = completion.choices[0].message
print(message.content, completion.choices[0].finish_reason)
```

Choose a smaller `max_tokens` for short tasks. It is an output ceiling, including
reasoning, not a requested minimum length. Stop the Slurm service using its
qualified manager's shutdown path or cancel only the serving job you own.
