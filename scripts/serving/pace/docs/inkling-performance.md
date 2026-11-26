# Inkling 2 Mini on PACE Hopper

This investigation uses the qualified Inkling checkpoint `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`, Python 3.12.15, vLLM `0.34.1rc1.dev144+g94b13b3d8`, PyTorch `2.14.0+cu132`, and NVIDIA driver `615.71.09`. The publication branch remains at the separately qualified baseline commit `414605fe0613b6c732616f1be0b1cd65747bb160`.

## Why the original result was slow

The original 17.82 tok/s measurement divided the server-reported 256 completion tokens by one 14.366-second nonstreaming request. The numerator already counted hidden reasoning and visible output. The interval included request and prefill time, and the preceding arithmetic smoke used a different shape. The original result provided neither TTFT nor a separate decode interval.

The published 157 tok/s is a median of 12 warm single-user decode measurements on an RTX PRO 6000 Blackwell and a different vLLM nightly. The benchmark used native GPU-specific execution paths. PACE's sm90 worker uses Humming expert GEMMs with BF16 activations and mixed NVFP4/MXFP8 weights. It verifies the Inkling attention and state-convolution paths before serving. [Published benchmark](https://huggingface.co/blog/inkling-2-mini-inference-hillclimb) · [Pinned Hopper experts implementation](https://github.com/vllm-project/vllm/blob/94b13b3d84562054f089e709ba1e32cf9a153bb7/vllm/models/inkling2/nvidia/ops/moe_sm90.py)

## Methods

Each model server ran one pinned checkpoint on one Hopper GPU. The comparison fixes the 4,096-token context, four requested host cores, temperatures and request seeds, and disables prefix caching. The graph profile captures decode only with compilation mode `NONE`. The eager switch also skips the startup JIT warmup, which the graph worker measured as 0.00 seconds at log precision. Both paths receive the same application warmup. The compilation comparison holds decode graph capture fixed. The concurrency case captures batches of 1, 2 and 4 and separately reports four-client aggregate throughput.

Each prompt length and concurrency group gets an unmeasured request group followed by three measured groups. Generation forces 256 output tokens for timing. The client checks streamed token IDs against server-reported completion usage and records the server's reasoning-token breakdown. TTFT is time to the first token-bearing event. Decode rate uses the remaining tokens divided by the interval between the first and last token-bearing events. End-to-end rate uses the full response time. Aggregate throughput divides all output tokens in a concurrent group by shared wall time. The short and long inputs use 33 and 1,909 prompt tokens in the observed runs. Reported latency includes local HTTP serving and token streaming.

Server histograms provide independent prefill, queue and decode timing. Startup includes node-local staging and engine initialization. CUDA preflight precedes that interval. The first request per shape exercises any remaining Triton JIT compilation and is excluded from medians. H100 graph logs recorded first-shape JIT for `_k_group_kernel` and long-shape JIT for `_select_kernel`. Acceptance uses separate normal-stop arithmetic, explicit reasoning and a synthetic local red PNG to check the final-answer field and parser separation.

In the fixed-length short runs the server reported 254 reasoning tokens out of 256 completion tokens. The measured throughput describes generated tokens across channels. On the separate reasoning fixture, the accepted final answer was `391` and the server reported 50 reasoning tokens within 58 completion tokens. The remaining completion tokens include framing.

GPU telemetry includes SM and memory clocks, GPU and memory activity, power and peak worker VRAM. NVML memory utilization describes memory activity. The optional read-plus-write GPU copy diagnostic measures synthetic streaming bandwidth with two 256 MiB buffers. It is distinct from actual model decode traffic. NVIDIA lists 3.35 TB/s for H100 SXM and 4.8 TB/s for H200 SXM. [NVIDIA HGX specifications](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html)

## Sampler and backend safeguards

The initial FlashInfer failure was reproducible with a host link probe. The pinned CUDA wheel provides `libcudart.so.13`, and the build requested `-lcudart`. A temporary unversioned symlink and a job-local `LIBRARY_PATH` resolved the symbol against the pinned library in the probe. The existing vLLM sampler was used for inference. The pinned runner excludes greedy or explicitly seeded requests from FlashInfer fused sampling, so sampler linkage cannot explain the tested performance gap. End-to-end stochastic FlashInfer sampling remains unqualified. [Pinned sampler selection](https://github.com/vllm-project/vllm/blob/94b13b3d84562054f089e709ba1e32cf9a153bb7/vllm/v1/worker/gpu/sample/sampler.py)

The first H200 eager measurement failed at a Humming compiler lock on shared storage. The opt-in profiles put Humming's compiler cache and temporary files on node-local storage and expose the pinned NVRTC libraries to its compiler helper. The paired H200 rerun used that isolation in both modes. The first H100 comparison and first H200 graph run could use an already warm shared Humming cache, so their startup times describe those runs. The original eager production entry point remains available.

The performance profiles use the installed Torch and vLLM without adding packages. Workers retain filesystem, import, native-load and backend verification. Each worker logs effective graph and compilation modes after warmup and aborts if the selected mode differs. Speculation is disabled. DeepGEMM and FlashMLA remain blocked.

## Results

| GPU and mode | Slurm job | Warm short decode | Warm short end to end | Warm short TTFT | 1,909-token prompt decode | Startup | Peak worker VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 eager | `7328093` | 17.66 tok/s | 17.57 tok/s | 127.87 ms | 17.87 tok/s | 193.55 s | 77,378 MiB |
| H100 decode graph | `7328093` | 162.79 tok/s | 156.33 tok/s | 70.91 ms | 159.76 tok/s | 136.40 s | 77,372 MiB |
| H200 eager | `7328238` | 16.65 tok/s | 16.57 tok/s | 133.59 ms | 16.45 tok/s | 160.56 s | 139,442 MiB |
| H200 decode graph | `7328238` | 158.87 tok/s | 151.90 tok/s | 80.24 ms | 157.20 tok/s | 156.52 s | 139,452 MiB |

On the same H100, graph capture raised warm decode throughput by 9.22×. The warm eager end-to-end rate reproduced the original 17.82 tok/s. Client timing and first-request warmup therefore account for a small fraction of the original gap. The server's mean short-request decode interval fell from 14.43 to 1.57 seconds. Mean server prefill fell from 117 to 61 ms. First-request TTFT was 530 ms eager and 382 ms with graphs, compared with warm medians of 128 and 71 ms. The log reports the graph capture as less than one second at its logging resolution and 0.06 GiB of capture memory.

During the H100 serving window, median SM and memory clocks were 1,980 and 2,619 MHz in both runs. Median GPU activity rose from 14% to 100% and memory activity from 4% to 34%. The combined API and worker CPU time averaged about one core in steady single-stream runs. With host allocation and clocks held fixed, the result is consistent with eager host dispatch and kernel-launch gaps limiting GPU work. The graph result still includes the Hopper kernels, no speculation and the same checkpoint. No instruction-level profiler was used.

The paired H200 graph run was 9.54× faster than eager, with median GPU activity increasing from 12% to 100% at the same clocks. A second graph run reached 158.96 tok/s in job `7328094`. The copy diagnostic measured 3.002 TB/s on H100 and 4.203 TB/s on H200. H100 was slightly faster on this workload despite H200's greater synthetic memory bandwidth. The exact bottleneck of the graphed kernel mix remains unmeasured. Peak worker VRAM includes vLLM's KV-cache allocation. The text model reported 17.64 GiB loaded on both GPUs. The image configuration reported 18.24 GiB loaded.

Four concurrent short requests on H100 delivered **542.24 aggregate tok/s**, with median per-request decode of 146.08 tok/s and median TTFT of 141.82 ms. In the same server, a single request decoded at 163.15 tok/s with 72.62 ms TTFT. This is a text-only benchmark configuration. The serving profile stays at one concurrent sequence.

The H100 and H200 image graph cases returned `Red`, alongside `391` for both text and explicit-reasoning requests. Their final answers matched the H100 eager image case. The H100 and H200 graph image cases reached readiness in 135.39 and 136.47 seconds respectively, with peak worker VRAM of 77,266 and 139,346 MiB. The selected GPU paths were FlashAttention for Inkling decoder attention, the Inkling state-convolution backend, Hopper Humming experts, and FlashAttention 3 for vision. The worker verified these paths and the policy before readiness. The baseline sampler remained active.

The pinned runtime resolves the default breakable graph plus compilation request to mode `NONE`. The worker rejected the changed setting before readiness in job `7328093`. With breakable graphs explicitly disabled for both cases in job `7328326`, the runtime retained the requested compilation mode and warned that the Inkling model does not implement its `torch.compile` interface. Both cases measured 163.55 tok/s. The runtime advertised the same norm and activation quantization fusion settings in the eager and graph logs. We qualified decode graphs in mode `NONE` and added no kernel fusion overrides. [Pinned compilation selection](https://github.com/vllm-project/vllm/blob/94b13b3d84562054f089e709ba1e32cf9a153bb7/vllm/config/vllm.py)

The normal Slurm launcher also passed image validation on H100 in job `7328373` with a 12-minute time limit. It returned `Red` at 143.5 seconds to readiness and 77,258 MiB peak worker VRAM. The worker verified `FULL_DECODE_ONLY`, compilation `NONE`, the baseline sampler and disabled speculation. The job released its allocation after 2 minutes and 48 seconds.

The full CPU regression run passed 23 PACE tests and 310 LLMGE tests in job `7328333`. That run deselected 22 opt-in cases. The GPU cases listed above cover real Hopper workers, text and image correctness, parsed reasoning and backend enforcement for the new profile.

## Reproduce

Use the existing PACE state containing the verified Inkling snapshot and isolated environment, then choose your authorized Slurm account, QoS and partition. From the repository root, set `PACE_DIR="$PWD/scripts/serving/pace"`.

```bash
sbatch -A "$PACE_ACCOUNT" --qos="$PACE_QOS" -p "$PACE_PARTITION" \
  --gres=gpu:h100:1 --constraint=H100 \
  "$PACE_DIR/slurm/inkling_perf.sbatch" "$PACE_DIR" \
  H100 eager graph compile graph-batch4 graph-image eager-image
```

For the H200 comparison pass `--gres=gpu:h200:1 --constraint=H200` and replace the case list with `H200 eager graph graph-image`. For the isolated H100 compile experiment use `H100 graph-nonbreakable compile-nonbreakable`. The diagnostic job returns nonzero if an experimental profile fails. Individual case results remain under `PACE_INFERENCE_STATE/perf-JOBID.*`. The script releases each server and its local checkpoint copy before starting the next case.

The opt-in serving profile uses the same decode graph configuration. It leaves the vision encoder on its qualified eager path. Set `--mode text` for text-only operation and `--gpu h200` for the H200 allocation profile.

```bash
"$PACE_PYTHON" -B "$PACE_DIR/submit.py" inkling --operation serve --mode image --inkling-graph
```

The exact execution delta is `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","decode_cudagraph_capture_request_sizes":[1]}'` with `--max-model-len 4096 --max-num-seqs 1 --no-enable-prefix-caching`. The launcher supplies the pinned backend flags and reasoning parser automatically. Omit `--inkling-graph` to use the previously qualified eager launch.

For a short validation instead of a persistent endpoint, use `--operation validate --mode image --inkling-graph --time 00:12:00`. First startup may take longer if the node or pinned runtime differ. Both commands use the existing verified 18.21 GiB checkpoint and isolated runtime. The launcher releases the temporary local checkpoint copy along with the server.
