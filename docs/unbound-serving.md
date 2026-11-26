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
