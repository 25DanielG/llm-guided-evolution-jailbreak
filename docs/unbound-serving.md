# Inkling 2 Mini Unbound v2.1

The `inkling-2-mini-unbound-v21` profile pins `nokeepai/Inkling-2-Mini-Unbound`
to revision `ef74b44edd2851b8e8d221a7313f36c419590430`. The persistent approved
snapshot is `${PACE_INFERENCE_STATE}/snapshots/inkling-unbound-v21/ef74b44edd2851b8e8d221a7313f36c419590430`.
Set `JB_UNBOUND_CHECKPOINT` to that directory before using `src.model_serving`.
The committed manifest records SHA-256 and byte counts for all 12 files; `check`
verifies them before launch. It does not download or stage model files.

The checkpoint declares `Inkling2LoopForConditionalGeneration`,
`model_type=inkling2_loop`, BF16 activations and NVFP4 weights. Its tokenizer and
chat template are from the same pinned revision. The profile keeps the native
Inkling reasoning parser and uses the qualified vLLM/Hopper runtime. The generic
registry launcher can construct a vLLM command, but production deployment should
use the qualified guarded PACE launcher/runtime, whose worker checks kernel paths
and service readiness. That guarded launcher is maintained in the qualification
checkout and is not copied or mutated by this adapter.

Qualification evidence: one H200 serving; replay job 7350751 completed 62/62
requests. Diagnostic job 7351445 completed successfully and reverified 12/12
persistent files. The diagnostic used a 40,960-token serving context and tested
budgets up to 8,192; maximum observed completion among rescued cases was 4,119
tokens. Reasoning tokens are included in the OpenAI `max_tokens` allowance.
Responses expose `reasoning` separately from final `content`; consumers should
use `parse_chat_response`, which rejects empty final channels and non-`stop`
finish reasons. The model metadata advertises a larger context, but that context
has not been qualified here. Unbound was GPU-qualified on H200; H100 compatibility
is expected from the Hopper kernel family but has not been independently tested.

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
