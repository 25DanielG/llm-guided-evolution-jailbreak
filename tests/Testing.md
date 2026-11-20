# Testing the shared VIP pipeline

Ordinary tests use Python 3.12 and run without PACE services, GPU allocations,
model weights, external datasets, or API credentials. Production Slurm scripts,
model loading and serving timeouts are unchanged.

## Ordinary CI and local runs

```bash
bash scripts/workflows/setup_python_ci.sh
bash scripts/workflows/python_app_no_lint.sh
```

Setup creates `.ci-venv` with CPU PyTorch and the pinned direct dependencies in
`requirements-ci.txt`. It does not change a project or shared serving environment.
The test command has a ten-minute wall-clock limit, reports slow tests, and writes
JUnit to `tests/results/unit.xml`. GitHub uploads this report on success or failure.
Python application CI has a thirty-minute overall limit; setup and tests are
separate steps. No CIFAR download or GPU assignment occurs in this path.

To reuse an existing compatible Python 3.12 environment without installing anything:

```bash
LLMGE_TEST_PYTHON=/path/to/environment/bin/python bash scripts/workflows/python_app_no_lint.sh
```

The ordinary suite includes real CPU evaluation of all nine valid frozen Titanic
models and the three known-invalid cases. Each evaluator runs in a fresh temporary
sandbox with the existing small CSV fixture, writes its normal `results/` files,
and is checked for exit status, CSV schema and bounded FP/FN values. Each CPU
evaluator has a 180-second subprocess budget within the ten-minute suite cap. Production
output locations are unchanged.

`mock_inference` tests exercise real script generation, mutation, crossover,
response extraction, validation, numeric decoding and file assembly with fixed
inference responses. They assert actual changes, preserved segments, invalid-output
fallback and interface payloads. They are **not verification of a model's outputs,
GPU compatibility or deployment status**. Client requests and server endpoint
handlers also have mocked contract tests. Unmocked requests, pretrained model
loading and scheduler commands are rejected in ordinary tests. Readiness is tested
with mocked HTTP and a fake clock rather than real waiting.

Both plain `python -m pytest tests` and the CI command deselect special tests unless
explicitly enabled. Markers apply to individual tests in mixed modules:

| Marker | Explicit option | Meaning |
| --- | --- | --- |
| `mock_inference` | None | Fixed responses; ordinary deterministic CI |
| `live_inference` | `--run-live-inference` | Genuine model requests, real generation/benchmarks |
| `gpu` | `--run-gpu` | GPU model checks using an existing allocation |
| `dataset` | `--run-dataset` | Training with a separately prepared external dataset |

## Explicit integration tests

These commands are examples for an authorized PACE session; running them may use
GPU time or real inference and requires the team's usual approval. Start the
existing service separately using the shared scripts and checkpoint procedures.
Its `hostname.log` and `LLMGE_PORT` must identify the service used by the tests.

```bash
LLMGE_AUTO_START_SERVER=0 uv run --python 3.12 pytest tests/integration tests/test_benchmarks.py \
  --run-live-inference -m live_inference -v
uv run --python 3.12 pytest tests/sota/test_ExquisiteNetV2.py --run-gpu -m gpu -v
# Requires CIFAR prepared separately; no dataset download is performed by pytest setup.
uv run --python 3.12 pytest tests/sota/test_ExquisiteNetV2_Full.py --run-dataset -m dataset -v
```

Live checks fail promptly if the endpoint cannot perform a real generation.
Mutation/crossover integration tests spy on real responses and reject unchanged
parent fallback; benchmarks fail transport/subprocess errors rather than report
an infrastructure failure as a passing inference result. Benchmark scientific
quality comparisons remain informational.

Legacy pytest server startup helpers remain available only when **both**
`--run-live-inference` and `LLMGE_AUTO_START_SERVER=1` are explicitly selected.
That opt-in preserves the existing local/Slurm startup behavior and loading
budgets. Ordinary collection never starts a service. The production
`run_improved.wait_for_llm_server` default remains 3600 seconds; CI overrides its
budget only as a defensive backstop, while its unit tests mock the boundary.

## Baseline repairs

The original PR #3 Python application run had 17 pre-existing failures:

- One evolutionary-loop test waited 3600 seconds for an absent model service.
- Nine Titanic assertions searched `test_results/`, although the evaluator wrote
  `results/` and reported completion.
- Five crossover tests wrote into a nonexistent output directory.
- Two judge tests expected refusal thresholds after the base branch intentionally
  changed smoke checks to validate wiring and treat compliance scores as data.

The repaired tests retain real file and evaluation checks, mock external unit-test
boundaries, and preserve failures for empty responses, malformed judge output,
unavailable endpoints, invalid scores and missing cache entries. They do not
change judge scoring, model-serving code, or either PR's deployment status.

## Repair validation

The complete ordinary runner passed locally on Python 3.12: **146 passed, 22
integration cases deselected**, in 219.95 seconds. All nine valid and three
invalid frozen Titanic cases ran; no inference was performed. The two warnings
were the existing FastAPI startup-hook deprecation. The same runner is executed
on GitHub with the separate CPU dependency environment. Hosted check results are
recorded in the repair PR, rather than being treated as model deployment evidence.
