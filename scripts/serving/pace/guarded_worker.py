"""Verify actual model backends before vLLM profiles or serves the model."""

from collections import Counter

import torch
from vllm import envs
from vllm.models.inkling.nvidia.sconv_swa_attn import InklingSconvBackend
from vllm.models.inkling2.nvidia.attention import Inkling2AttentionBackend
from vllm.models.inkling2.nvidia.moe import Inkling2Experts
from vllm.models.inkling2.nvidia.ops.moe_sm90 import HummingExperts
from vllm.models.inkling2.nvidia.vision import Inkling2VisionAttention
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.v1.worker.gpu_worker import Worker as UpstreamWorker

from backend_policy import emit, verify_effective, verify_process


class Worker(UpstreamWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        verify_effective(vllm_config, envs)
        if vllm_config.model_config.architectures != ['Inkling2LoopForConditionalGeneration']:
            raise ValueError('This model architecture has no completed backend review')
        super().__init__(vllm_config, *args, **kwargs)

    def load_model(self, *, load_dummy_weights=False):
        if torch.cuda.get_device_capability() != (9, 0) or is_deep_gemm_supported():
            raise ValueError('Expected Hopper with DeepGEMM disabled before model load')
        verify_process()
        super().load_model(load_dummy_weights=load_dummy_weights)
        verify_process()

        # Inkling bypasses the generic attention selector. Inspect its actual
        # layer objects and the prepared Hopper expert implementations.
        counts = Counter()
        for layer in self.get_model().modules():
            get_backend = getattr(layer, 'get_attn_backend', None)
            if callable(get_backend):
                backend = get_backend()
                if backend not in (Inkling2AttentionBackend, InklingSconvBackend):
                    raise ValueError(f'Unexpected decoder attention backend {backend}')
                counts[backend.__name__] += 1
            if isinstance(layer, Inkling2Experts):
                if type(layer.humming) is not HummingExperts or layer.sm100 or layer.sm120:
                    raise ValueError('Inkling did not prepare the Hopper Humming expert path')
                counts['HummingExperts'] += 1
            if isinstance(layer, Inkling2VisionAttention):
                counts['Inkling2VisionAttention_FA3_SM90'] += 1
        if not all(counts[name] for name in ('Inkling2AttentionBackend', 'InklingSconvBackend', 'HummingExperts')):
            raise ValueError(f'Could not verify the model backends: {counts}')
        mm = self.vllm_config.model_config.multimodal_config
        image = counts['Inkling2VisionAttention_FA3_SM90']
        # The language-model-only route may create a nonexecuting tower stub.
        if not mm.language_model_only and not image:
            raise ValueError('Could not verify the Hopper vision layers')
        emit('worker_backends_verified', rank=self.rank, selected=dict(counts),
             vision_enabled=not mm.language_model_only, deep_gemm=False, flashmla=False)
