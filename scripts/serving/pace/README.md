# PACE ICE model serving

The guarded launcher serves Inkling 2 Mini, Nemotron 3.5 Lightning, Muse Glimmer 30B and Nemotron 3.5 Super VL through an OpenAI-compatible endpoint. Each member chooses a private external state directory. The existing LLMGE GPT-OSS, Llama and Mistral launchers retain their own registry and environments. The existing `server.py /generate` proxy also accepts the guarded PACE endpoints.

See the [ICE qualification report](docs/qualification.md) for exact checkpoint revisions, runtime pins and GPU measurements.

The [Inkling performance report](docs/inkling-performance.md) documents the opt-in Hopper decode graph profile and its measurements. The original eager path remains the default.

## Prepare your checkout and account

```bash
git clone --branch MosesTheRedSea-main https://github.com/25DanielG/llm-guided-evolution-jailbreak.git
cd llm-guided-evolution-jailbreak
export PACE_DIR="$PWD/scripts/serving/pace"
export PACE_INFERENCE_STATE="$HOME/scratch/pace-inference-state"
export PACE_PYTHON="$(command -v python3.12)"
mkdir -p -m 700 "$PACE_INFERENCE_STATE"
"$PACE_PYTHON" --version
```

Use Python 3.12.15 and make `uv` available on the host where you install. Set `PACE_INFERENCE_STATE` to your authorized storage location outside the repository, with permissions `0700`. It holds isolated runtimes, the uv and Hugging Face caches, managed snapshots and job results. The launcher uses the configured Python and validates the runtime against its hash lock.

Set Slurm values from your own ICE allocation. For example, an account authorized for `math` and QoS `coe-ice` would use the following. Omit variables to use scheduler defaults or pass `--account`, `--partition`, and `--qos` to the submitter. `--exclude` or `PACE_EXCLUDE` is optional. Use `--dry-run` to inspect the command and `--test-only` for the scheduler's eligibility check.

```bash
export PACE_ACCOUNT=math PACE_QOS=coe-ice PACE_PARTITION=coe-gpu,ice-gpu
"$PACE_PYTHON" -B "$PACE_DIR/submit.py" inkling --dry-run
```

## Install and validate one model

Run the block for the model you want. Installing a runtime uses its dependency lock. It leaves checkpoints to the separate fetch step. Lightning, Muse, and Super VL reuse one stable runtime. The Nemotron jobs load the PACE `cuda/13.0.1` module for FlashInfer Mamba compilation. A site with the same NVCC `13.0.88` and matching CUDA headers can set `PACE_CUDA_TOOLKIT` to the toolkit root.

| Alias | Checkpoint | Qualified GPU | Default RAM / node-local disk | Validation walltime |
| --- | ---: | --- | --- | --- |
| `inkling` | 18.21 GiB | 1× H100 | 64 / 64 GiB | 60 min |
| `lightning` | 20.10 GiB | 1× H100 | 64 / 64 GiB | 60 min |
| `muse` | 32.06 GiB | 1× H100 | 96 / 96 GiB | 60 min |
| `super-vl` | 69.91 GiB | 1× H200 | 128 / 128 GiB | 90 min |

The baseline uses four CPUs, 4,096 tokens, one concurrent sequence, eager execution and no speculation. Both CUDA and blocked-backend preflight run before staging.

**Inkling text and image**

```bash
bash "$PACE_DIR/scripts/install-runtime.sh" inkling
"$PACE_INFERENCE_STATE/runtimes/inkling/bin/python" -EB "$PACE_DIR/pace_inference.py" fetch inkling
"$PACE_PYTHON" -B "$PACE_DIR/submit.py" inkling --mode text
"$PACE_PYTHON" -B "$PACE_DIR/submit.py" inkling --mode image
```

**Nemotron Lightning**

```bash
bash "$PACE_DIR/scripts/install-runtime.sh" lightning
"$PACE_INFERENCE_STATE/runtimes/stable/bin/python" -EB "$PACE_DIR/pace_inference.py" fetch lightning
"$PACE_PYTHON" -B "$PACE_DIR/submit.py" lightning
```

**Muse Glimmer**

```bash
bash "$PACE_DIR/scripts/install-runtime.sh" muse
"$PACE_INFERENCE_STATE/runtimes/stable/bin/python" -EB "$PACE_DIR/pace_inference.py" fetch muse
"$PACE_PYTHON" -B "$PACE_DIR/submit.py" muse
```

**Nemotron Super VL**

```bash
bash "$PACE_DIR/scripts/install-runtime.sh" super-vl
"$PACE_PYTHON" -B "$PACE_DIR/submit.py" super-vl --source node-fetch
```

The Super VL command downloads the selected revision on the allocated node using public Hugging Face HTTPS, then serves offline. Use the same manager to persist the snapshot if your verified quota allows it. Run `fetch super-vl` with the stable interpreter, then submit with `--source shared`. Any of the four profiles can use either checkpoint source. The optional Super VL two-H100 configuration can be checked with `submit.py super-vl --gpu h100 --tp 2 --test-only`. It has not been GPU-qualified.

## Serve and connect an application

Pick an installed and staged model. Super VL uses its configured job-local download unless you select `--source shared`.

```bash
job=$("$PACE_PYTHON" -B "$PACE_DIR/submit.py" muse --operation serve --time 02:00:00)
job=${job%%;*}
squeue -j "$job"
tail -f "$PACE_INFERENCE_STATE/slurm-muse-$job.out"
```

Once the job prints `Endpoint ready`, the private `run-JOBID.*` directory contains `endpoint.json` and a per-job `api-key` file. The kernel assigns a loopback port by default. Set `--port` for a fixed port. Use the URL and model alias from `endpoint.json` in OpenAI-compatible clients, with the contents of `api-key` as the bearer key. The launcher verifies the listening process, authentication and worker policy before publishing this metadata.

A client within the allocation can use the loopback URL directly. From another authorized host, tunnel over SSH where ICE permits access to your assigned compute node. Set `PACE_LOGIN` to the SSH destination you normally use. Run this in a dedicated terminal with access to the run metadata. Keep the client terminal open separately. [PACE ICE documentation](https://docs.pace.gatech.edu/ice_cluster/ice-cluster/)

```bash
run_dir=$(find "$PACE_INFERENCE_STATE" -maxdepth 1 -type d -name "run-$job.*" -print -quit)
node=$(jq -r .node "$run_dir/endpoint.json")
remote_port=$(jq -r '.url | split(":")[2] | split("/")[0]' "$run_dir/endpoint.json")
export CLIENT_PORT=18001
ssh -o ExitOnForwardFailure=yes -N -J "$PACE_LOGIN" \
  -L "127.0.0.1:$CLIENT_PORT:127.0.0.1:$remote_port" "$USER@$node"
```

Use the run directory from a host that can read it, or transfer the key through your authorized SSH session with permissions `0600`. Keep the listener and tunnel bound to loopback. A direct OpenAI-compatible client test looks like this.

```bash
printf 'Authorization: Bearer %s\n' "$(cat "$run_dir/api-key")" | \
  curl --fail -H @- "http://127.0.0.1:$CLIENT_PORT/v1/models"
```

To reuse LLMGE's existing `/generate` client, run its proxy with the existing experiment environment. Replace `muse` with the published alias and choose a free local proxy port.

```bash
export JB_MUTATOR_VLLM_URL="http://127.0.0.1:$CLIENT_PORT/v1"
export JB_MUTATOR_PROFILE=muse
export JB_MUTATOR_API_KEY_FILE="$run_dir/api-key"
python -m uvicorn server:app --host 127.0.0.1 --port 18137 --workers 1
```

Point the existing application at the proxy's local port. Stop the proxy and tunnel, then run `scancel "$job"` to stop the model service. The job terminates its owned processes and removes its temporary files and key. The qualification metrics and logs remain in the private state directory.

## Manage cached checkpoints

```bash
"$PACE_PYTHON" -B "$PACE_DIR/pace_inference.py" status muse
"$PACE_PYTHON" -B "$PACE_DIR/pace_inference.py" verify muse
"$PACE_PYTHON" -B "$PACE_DIR/pace_inference.py" evict muse
"$PACE_PYTHON" -B "$PACE_DIR/pace_inference.py" evict muse --confirm
```

`fetch` checks Lustre user quota, filesystem headroom and inodes, then downloads only the manifest entries. The manager verifies SHA-256 or Git blob digests before publishing a snapshot. `--partial` targets an interrupted download. Eviction is limited to the chosen revision and uses the same lock as shared snapshot staging. Separate state directories isolate users and deployments.

## Backend restrictions

Use the deployment commands below so that the launcher can apply and verify the backend policy. `backend_policy.py` and `guard_bootstrap/sitecustomize.py` enforce the same restrictions across profiles. `guarded_worker.py` verifies Inkling's custom kernels and `stable_worker.py` verifies the stable runner and loaded layers before service readiness. An unreviewed runtime, model, or worker backend fails closed.

| Selection path in the pinned source | Deployment control |
| --- | --- |
| `envs.py`, `utils/deep_gemm.py`, `layers/fused_moe/oracle/fp8.py`, `warmup/kernel_warmup.py` | Set `VLLM_USE_DEEP_GEMM=0`, `VLLM_MOE_USE_DEEP_GEMM=0`, turn off its scaling flags, and set `VLLM_DEEP_GEMM_WARMUP=skip`. The wrapper can still import an external or bundled library through other call paths. These switches therefore work with the native file and import controls. |
| `engine/arg_utils.py`, `config/kernel.py`, `layers/fused_moe/oracle/{fp8,nvfp4}.py`, `model_executor/kernels/linear/__init__.py` | Explicit choices can select a DeepGEMM hybrid independently of the generic availability switches. The launcher sets a reviewed backend for each profile and prevents registry flags from overriding it. Stable vLLM can fall back by linear layer type. The worker checks the actual kernel and rejects any unreviewed choice. |
| `config/model.py`, `platforms/cuda.py`, `v1/attention/selector.py`, `v1/attention/backends/registry.py` | Set `VLLM_MLA_DISABLE=1` and choose `FLASH_ATTN` for decoder and multimodal encoder selection. The pinned CUDA selector raises when an explicit decoder backend is invalid. Omitting the selection would enable registry priority fallback and can expose FlashMLA for MLA models. The reviewed worker also rejects model architectures that require MLA. |
| `models/inkling2/nvidia/{attention,moe,vision}.py`, `models/inkling/nvidia/sconv_swa_attn.py` | Inkling selects its own layer implementations. Verify decoder layers use `Inkling2AttentionBackend`, convolution state uses `InklingSconvBackend`, prepared experts use `HummingExperts`, and Hopper vision takes the FA3 path in image mode. The worker verifies these before startup profiling and fails if any evidence is missing. |
| `v1/attention/ops/flashmla.py`, `utils/deep_gemm.py`, `utils/import_utils.py`, FlashInfer `deep_gemm.py` and `jit/cubin_loader.py` | A private copied installation sets the vLLM vendored code and native objects, and the stable FlashInfer wheel's named DeepGEMM native sources, to mode `000`. The guard verifies those modes, denies direct imports and native loads and checks loaded modules and `/proc/self/maps`. FlashInfer's shared dispatch wrappers are imported by Mamba. The stable launcher supplies a fresh artifact path and uses `FLASHINFER_NO_DOWNLOAD=1` to prevent fetching kernels from FlashInfer's repository. |

The launcher starts fresh Python subprocesses with the guard, clears inherited vLLM overrides, and requires effective configuration and actual-backend evidence from every worker rank. The batch preflight exercises denial of both FlashMLA extensions and DeepGEMM, then runs a CUDA tensor operation on each allocated GPU. Humming, FlashInfer, Triton and Torch extension caches used by the stable lane are job-local. The stable launcher exposes the pinned NVRTC builtins to Humming's compiler and provides a job-local `libcudart.so` link for FlashInfer SSU. `VLLM_USE_FLASHINFER_SAMPLER=0` retains the baseline sampler. The job logs include timings, process-level GPU memory samples and backend evidence. [Inkling source](https://github.com/vllm-project/vllm/tree/94b13b3d84562054f089e709ba1e32cf9a153bb7/vllm) · [Stable source](https://github.com/vllm-project/vllm/tree/v0.34.0/vllm)

The upstream wheels contain the restricted components. The mode, import, native-load, kernel and worker checks enforce their exclusion from these deployments. Follow current Georgia Tech software requirements for your workload. This project records technical controls and GPU evidence. Formal institutional approval would require a separate determination. [Georgia Tech OIT guidance](https://oit.gatech.edu/ai/guidance) · [Prohibited software policy](https://oit.gatech.edu/node/1163)

## Licenses and extensions

The repository adds source, configuration and synthetic fixtures. Locked packages and weights are downloaded from upstream. Retain the installed distributions' license files and review their terms for redistribution, especially NVIDIA's CUDA and driver packages. vLLM uses Apache 2.0 and PyTorch uses a BSD-style license. The model manifests pin download contents separately from the Python locks. [vLLM license](https://github.com/vllm-project/vllm/blob/v0.34.0/LICENSE) · [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE) · [NVIDIA CUDA terms](https://docs.nvidia.com/cuda/eula/index.html)

Inkling 2 Mini and Muse Glimmer FP8 are published under Apache 2.0. Both pinned Nemotron models use OpenMDW 1.1. Review model terms and provider usage policies before use. [Inkling model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [Muse model card](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-FP8-block) · [Lightning model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) · [Super VL model card](https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b/modelcard)

To add a model, pin its commit and manifest in `config/`, specify the validated allocation, and audit its architecture, quantization and fallback paths against the exact vLLM wheel. Extend the reviewed backend policy and worker allowlists, then add fixtures and tests. Qualify the new route on a bounded GPU allocation before documenting it as supported.

```bash
PYTHONPATH="$PACE_DIR" "$PACE_PYTHON" -B -m unittest discover -s "$PACE_DIR/tests"
bash -n "$PACE_DIR/slurm/validate.sbatch"
bash -n "$PACE_DIR/scripts/install-runtime.sh"
```
