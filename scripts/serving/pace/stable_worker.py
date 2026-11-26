"""Verifies the pinned stable vLLM GPU runner and its loaded layer objects."""

from collections import Counter
import os

import torch
from vllm import envs
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    FlashInferSSUBackend,
    get_mamba_ssu_backend,
)
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.v1.worker.gpu_worker import Worker as UpstreamWorker

from backend_policy import STABLE_PROFILES, emit, verify_effective, verify_process


MODEL_CLASSES = {
    'lightning': 'NemotronHForCausalLM',
    'muse': 'MuseGlimmerForCausalLM',
    'super-vl': 'NemotronH_Nano_VL_V2',
}
HUMMING_EXPERTS = {'HummingIndexedExperts', 'HummingGroupedExperts', 'BatchedHummingGroupedExperts'}
HUMMING_LINEAR = {'HummingFP8ScaledMMLinearKernel', 'HummingNvFp4LinearKernel'}
FP8_LINEAR = {'CutlassFp8BlockScaledMMKernel'}


class Worker(UpstreamWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        self.profile = os.environ['PACE_STABLE_PROFILE']
        verify_effective(vllm_config, envs, stable=self.profile)
        super().__init__(vllm_config, *args, **kwargs)

    def load_model(self, *, load_dummy_weights=False):
        if torch.cuda.get_device_capability() != (9, 0) or is_deep_gemm_supported():
            raise ValueError('Expected Hopper with DeepGEMM disabled before model load')
        verify_process()
        super().load_model(load_dummy_weights=load_dummy_weights)
        verify_process()

        model = self.get_model()
        if type(model).__name__ != MODEL_CLASSES[self.profile]:
            raise ValueError(f'Unexpected stable model implementation: {type(model)}')
        multimodal = self.vllm_config.model_config.multimodal_config
        self.vision_enabled = bool(multimodal and not multimodal.language_model_only)
        self.backends = self.check_layers(model)
        sampler = self.model_runner.sampler
        if type(sampler).__module__ != 'vllm.v1.worker.gpu.sample.sampler' or type(sampler).__name__ != 'Sampler':
            raise ValueError(f'Unexpected sampler {type(sampler)}')
        self.backends['vLLM baseline Sampler'] = 1

    def check_layers(self, model):
        counts = Counter()
        mamba = STABLE_PROFILES[self.profile][2]
        expected_attn = {'FlashAttentionBackend', 'Mamba2AttentionBackend'} if mamba else {'FlashAttentionBackend'}
        expected_linear = FP8_LINEAR if self.profile == 'muse' else HUMMING_LINEAR
        for layer in model.modules():
            if callable(get_backend := getattr(layer, 'get_attn_backend', None)):
                name = get_backend().__name__
                if name not in expected_attn:
                    raise ValueError(f'Unreviewed decoder attention backend: {name}')
                counts[name] += 1
            if isinstance(layer, MMEncoderAttention):
                if layer.attn_backend.name != 'FLASH_ATTN':
                    raise ValueError(f'Unreviewed encoder attention backend: {layer.attn_backend}')
                counts['vision FLASH_ATTN'] += 1
            method = getattr(layer, 'quant_method', None)
            if isinstance(layer, RoutedExperts):
                self.check_experts(method, counts)
                continue
            method_name = type(method).__name__
            if method_name in ('NoneType', 'UnquantizedLinearMethod', 'UnquantizedEmbeddingMethod',
                               'ModelOptKVCacheMethod', 'CompressedTensorsKVCacheMethod'):
                continue
            if method_name == 'CompressedTensorsLinearMethod' and self.profile == 'muse':
                kernel = layer.scheme.fp8_linear
            elif method_name == 'ModelOptLinearMethod' and self.profile != 'muse':
                kernel = method.kernel
            else:
                raise ValueError(f'Unreviewed quantization method: {method_name}')
            if type(kernel).__name__ not in expected_linear:
                raise ValueError(f'Unreviewed linear kernel: {type(kernel)}')
            counts[type(kernel).__name__] += 1
        if ('FlashAttentionBackend' not in counts or not expected_linear.intersection(counts)
                or (self.vision_enabled and not counts['vision FLASH_ATTN'])
                or (mamba and (not counts['Mamba2AttentionBackend']
                               or not HUMMING_EXPERTS.intersection(counts)))):
            raise ValueError(f'Incomplete model backend evidence: {counts}')
        return dict(counts)

    def check_experts(self, method, counts):
        # The NVFP4 oracle raises on an invalid explicit MoE backend. Verify
        # its installed class as well, including Lightning's W4A16 contract.
        if (self.profile == 'muse' or type(method).__name__ != 'ModelOptNvFp4FusedMoE'
                or method.nvfp4_backend.name != 'HUMMING'
                or method.experts_cls.__name__ not in HUMMING_EXPERTS
                or type(method.moe_kernel.fused_experts) is not method.experts_cls
                or method.uses_input_scales
                or (self.profile == 'lightning' and not method.use_a16)):
            raise ValueError(f'Unreviewed routed expert implementation: {type(method)}')
        counts[method.experts_cls.__name__] += 1
        counts['NVFP4 W4A16'] += 1

    def compile_or_warm_up_model(self):
        if STABLE_PROFILES[self.profile][2]:
            # This dispatcher is initialized with the attention/KV layout.
            if type(get_mamba_ssu_backend()) is not FlashInferSSUBackend:
                raise ValueError('GPU runner did not select FlashInfer Mamba SSU')
            self.backends['FlashInferSSUBackend'] = 1
        verify_process()
        result = super().compile_or_warm_up_model()
        verify_process()
        emit('worker_backends_verified', rank=self.rank, profile=self.profile,
             selected=self.backends, vision_enabled=self.vision_enabled,
             deep_gemm=False, flashmla=False)
        return result
