# PACE ICE qualification

## Runtime and snapshot pins

The [registry](../config/models.json) and [manifests](../config/snapshots/) specify the exact checkpoint revision, file list, byte count and digests. The jobs ran with CUDA compute capability 9.0 and NVIDIA driver `615.71.09`. See [deployment commands](../README.md#install-and-validate-one-model).

| Model | Repo and revision | Selected payload |
| --- | --- | ---: |
| Inkling 2 Mini | [thinkingmachines/Inkling-2-Mini @ c9d028302cd0f035fdf8e7e91f089a4ce94d2195](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/c9d028302cd0f035fdf8e7e91f089a4ce94d2195) | 19,555,840,375 B |
| Nemotron 3.5 Lightning | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 @ bee7596271d1495f6992ae224aefde4410e816b8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/tree/bee7596271d1495f6992ae224aefde4410e816b8) | 21,583,411,866 B |
| Muse Glimmer 30B | [RedHatAI/Muse-Glimmer-30B-FP8-block @ 024ba7df2e23285cdb72616eb624c5b514ebceb1](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-FP8-block/tree/024ba7df2e23285cdb72616eb624c5b514ebceb1) | 34,421,249,530 B |
| Nemotron 3.5 Super VL | [nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 @ ae1368329b1fc76b71534c5d04b87b35d9cc6e15](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/tree/ae1368329b1fc76b71534c5d04b87b35d9cc6e15) | 75,061,200,717 B |

| Runtime | vLLM | PyTorch | Transformers | Triton | FlashInfer | Humming | Python | Installed blocks |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| Inkling | `0.34.1rc1.dev144+g94b13b3d8` | `2.14.0+cu132` | `5.24.1` | `3.8.0` | `0.7.4` | `0.1.20` | `3.12.15` | 9,405,124,608 B |
| Shared stable | `0.34.0` | `2.14.0+cu130` | `5.22.2` | `3.8.0` | `0.7.3` | `0.1.19` | `3.12.15` | 9,532,231,168 B |

The shared uv cache occupied 14,937,649,152 B across both lanes. Storage measurements use allocated filesystem blocks. Snapshot sizes use the checksummed payload. Super VL fits on job-local XFS with the batch script's 128 GiB request. Its checksum-verified local download uses public Hugging Face HTTPS before offline serving, avoiding a 69.91 GiB persistent scratch allocation.


## Validation and performance

| Model and fixture | GPUs | Slurm job | Validation |
| --- | --- | --- | --- |
| Inkling text | 1× H100 | `7322241` | Passed |
| Inkling image | 1× H100 | `7322242` | Passed |
| Lightning suite | 1× H100 | `7326308` | Passed text, reasoning, final-answer and tool parsing |
| Muse suite | 1× H100 | `7326226` | Passed text, reasoning, ATEM tool parsing, image and offline video |
| Super VL suite | 1× H200 | `7326309` | Passed native loading, text, reasoning, tool parsing, image and offline video |

| Model and fixture | Local stage, s | Service ready, s | Model load, GiB | KV after warmup, GiB | KV budget or capacity, GiB | Peak GPU, MiB | Peak host, GiB | First token, s | Generation tok/s | End-to-end tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inkling text | 10.81 | 130.05 | 17.64 | — | 60.51 budget | 76,910 | 23.31 | — | — | 17.82 |
| Inkling image | 10.11 | 116.03 | 18.24 | — | 59.89 budget | 77,264 | 23.38 | — | — | — |
| Lightning suite | 13.78 | 328.07 | 19.12 | 0.08 | 57.34 capacity | 79,482 | 31.26 | 0.150 | 22.79 | 22.57 |
| Muse suite | 15.17 | 114.03 | 32.39 | 0.23 | 42.86 capacity | 78,104 | 36.28 | 0.274 | 20.08 | 19.73 |
| Super VL suite | 215.29 | 618.13 | 69.67 | 0.23 | 65.79 capacity | 140,924 | 76.34 | 0.265 | 12.42 | 12.31 |

Inkling's GPU workers reported 48 decoder attention layers, 24 InklingSconvBackend layers and 24 Hopper Humming experts. The image job verified 23 vision FA3 layers. Muse's GPU worker verified 52 decoder FlashAttention layers, 50 encoder FlashAttention layers, 260 CUTLASS block FP8 kernels and one vLLM baseline sampler. The shared stable guard logged 1,030 inaccessible paths, including the vLLM native binaries and 124 FlashInfer DeepGEMM source paths, and confirmed disabled FlashInfer cubin downloads.

Lightning verified 23 MoE layers using Humming indexed NVFP4 W4A16, 47 Humming NVFP4 and 46 Humming FP8 linear kernels, 6 decoder FlashAttention layers and 23 Mamba2 layers. The worker selected FlashInfer SSU for Mamba decode, Triton SSD for prefill and the vLLM baseline sampler.

Super VL loaded the native `NemotronH_Nano_VL_V2` implementation. Its worker verified 40 MoE layers using Humming indexed NVFP4 W4A16, 80 Humming NVFP4 and 97 Humming FP8 linear kernels, 8 decoder FlashAttention layers, 32 encoder FlashAttention layers and 40 Mamba2 layers. It used the same FlashInfer SSU, Triton SSD prefill and baseline sampling paths as Lightning.

## Measurement definitions

Readiness runs from server launch to the first successful health response. It includes checkpoint staging and excludes batch CUDA preflight. Cold startup can include local JIT builds and Lustre metadata reads. Super VL staging via `node-fetch` includes transfer and checksum verification. Shared staging measures the local copy from a checksum-verified snapshot.

Model-load memory is the worker-reported device footprint of loading the model. The GPU peak comes from one-second NVML samples of GPU worker processes and includes allocator reservation, cache and transient work. The host peak is Slurm batch `MaxRSS`. vLLM reports committed KV cache after warmup and estimated KV capacity when using an extensible cache. These describe different metrics.

The text benchmark asks the model to list integers at temperature zero with a 256-token cap. The stable suite measures end-to-end throughput, time to the first streamed token or reasoning delta and generation throughput between the first and last delta. Historical Inkling results include end-to-end throughput. The benchmarks use one concurrent sequence at a 4,096-token context in eager mode with speculation disabled.

## Operational findings

- An initial Inkling H100 job failed CUDA initialization. The launcher checks CUDA on every visible GPU before staging and accepts operator-provided scheduler exclusions. The NVIDIA source describes driver requirements and caveats for newer toolchains. [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- OpenTelemetry dependencies are pinned in each lock. Optional FlashInfer sampling hit a missing `-lcudart` linker name during Inkling qualification. Both lanes set `VLLM_USE_FLASHINFER_SAMPLER=0` and use the vLLM baseline sampler. The stable lane additionally supplies a job-local unversioned link to the pinned CUDA runtime for Mamba SSU compilation.
- Lightning job `7323500` and Super VL job `7323511` failed during Humming preparation because its standalone compiler could not locate `libnvrtc-builtins.so.13.0`. The stable launcher now exposes the library bundled with the pinned NVRTC package. Humming and other compiler caches are scoped to the allocation.
- Lightning job `7326060` resolved Humming weight loading and reached FlashInfer Mamba SSU compilation. NVCC `13.5.37` and the wheel's CUDA 13.0 headers failed the compiler compatibility check. The Mamba job profiles now load PACE's `cuda/13.0.1` toolkit with NVCC `13.0.88` and matching headers for that build step, using the same stable Python environment. Super VL job `7326061` was released during a slow tail download after this compiler issue was identified.
- The stable vLLM linear selector can fall back by layer type even when configured explicitly. The GPU workers check the selected attention, quantization, experts and sampler objects and abort on an unreviewed path. FlashInfer Mamba SSU is checked after the KV layout initializes. Stable FlashInfer DeepGEMM native sources are quarantined and cubin downloads are disabled. Both lanes retain the existing checks on the vLLM vendored DeepGEMM and FlashMLA files and imports. [vLLM stable source](https://github.com/vllm-project/vllm/tree/v0.34.0/vllm)

The `run-JOBID.*` directories under `PACE_INFERENCE_STATE` store `preflight.log`, `server.log`, `result.json` and GPU samples. They remain available independently of snapshot eviction. Slurm reported `COMPLETED` with exit code `0:0` for all passing jobs and the queue was clear of these allocations after shutdown.

```bash
sacct -j 7322241,7322242,7326226,7326308,7326309 \
  --units=G --format=JobID,Account,Partition,QOS,State,Elapsed,MaxRSS,ExitCode -P
```

## Scope and remaining limits

- The Super VL two-H100 configuration remains unqualified. A one-hour Slurm dry run with two GPUs and four CPUs estimated a start about eight hours later. The exact command is in the README.
- The media suites use a synthetic image and an eight-frame inline JPEG clip. Timing-dependent video tasks and container decoding require separate fixtures. Stable runs cover the 4,096-token, one-sequence eager profile.
- Nemotron load logs flag FP8 KV-cache scale defaults. The short fixtures passed basic acceptance for each completed profile. Longer workloads are needed to characterize accuracy and performance under load.

## Group launcher checks

The integrated launcher passed bounded H100 text checks for the Inkling lane in job `7327588` and the stable Lightning lane in job `7327589`. Both used the previously verified snapshots, rejected anonymous API requests, accepted the generated per-job key on the kernel-assigned port, checked the GPU worker policy and shut down after validation.
