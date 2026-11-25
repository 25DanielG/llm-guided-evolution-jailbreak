import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backend_policy import (BlockBackendImports, POLICY_ENV, STABLE_PROFILES, backend_flags,
                            guarded_environment, profile_flags, verify_effective)
from pace_inference import model_profile
from validate import checked_backends, process_peak


class BackendPolicyTests(unittest.TestCase):
    def test_profile_cannot_override_backend_or_use_nested_config(self):
        for args in (['--moe-backend', 'deep_gemm_mega_moe'],
                     ['--linear-backend', 'deep_gemm'],
                     ['--attention-backend', 'FLASHMLA'],
                     ['--attention-config', '{"backend": "FLASHMLA"}'],
                     ['--mamba-backend', 'triton'],
                     ['--model-impl', 'transformers'],
                     ['--kernel-config.moe-backend=deep_gemm', 'unused'],
                     ['--worker-cls', 'vllm.Worker'],
                     ['--reasoning-parser', '--attention-backend=FLASHMLA']):
            with self.subTest(args=args), self.assertRaises(ValueError):
                profile_flags(args)
        self.assertEqual(profile_flags(['--reasoning-parser', 'inkling']),
                         ['--reasoning-parser', 'inkling'])

    def test_environment_overrides_inherited_settings(self):
        with patch.dict(os.environ, {'VLLM_USE_DEEP_GEMM': '1', 'VLLM_ATTENTION_BACKEND': 'FLASHMLA',
                                    'LD_PRELOAD': 'libflashmla.so', 'DG_JIT_CACHE_DIR': '/tmp/stale'}):
            env = guarded_environment(Path('/repo'))
        self.assertEqual({key: env[key] for key in POLICY_ENV}, POLICY_ENV)
        for key in ('VLLM_ATTENTION_BACKEND', 'LD_PRELOAD', 'DG_JIT_CACHE_DIR'):
            self.assertNotIn(key, env)

    def test_guard_rejects_nested_extensions_and_allows_wrapper(self):
        guard = BlockBackendImports()
        for name in ('vllm.third_party.deep_gemm._C_extension', 'vllm._flashmla_extension_C',
                     'vllm.third_party.flashmla.flash_mla_interface', 'deep_gemm',
                     'flashinfer.deep_gemm', 'flashinfer.experimental.deepgemm_common.cake_catalog'):
            with self.subTest(name=name), self.assertRaises(ModuleNotFoundError):
                guard.find_spec(name)
        self.assertIsNone(guard.find_spec('vllm.utils.deep_gemm'))
        self.assertIsNone(guard.find_spec('vllm.vllm_flash_attn'))

    def test_worker_rejects_effective_flashmla_and_available_deep_gemm(self):
        config = SimpleNamespace(
            attention_config=SimpleNamespace(backend=SimpleNamespace(name='FLASH_ATTN')),
            kernel_config=SimpleNamespace(moe_backend='humming', linear_backend='humming'),
            parallel_config=SimpleNamespace(worker_cls='guarded_worker.Worker'),
            model_config=SimpleNamespace(use_mla=False, is_deepseek_mla=False, multimodal_config=None),
            speculative_config=None,
        )
        envs = SimpleNamespace(**{key: bool(int(value)) if value in ('0', '1') else value
                                  for key, value in POLICY_ENV.items()})
        with patch('backend_policy.verify_process'), patch('backend_policy.emit'), \
                patch('backend_policy.sys.meta_path', [BlockBackendImports()]):
            verify_effective(config, envs)
            config.attention_config.backend.name = 'FLASHMLA_SPARSE'
            with self.assertRaisesRegex(ValueError, 'unsafe backends'):
                verify_effective(config, envs)
            config.attention_config.backend.name = 'FLASH_ATTN'
            envs.VLLM_MOE_USE_DEEP_GEMM = True
            with self.assertRaisesRegex(ValueError, 'VLLM_MOE_USE_DEEP_GEMM'):
                verify_effective(config, envs)

    def test_stable_profiles_are_explicit_and_reject_backend_drift(self):
        envs = SimpleNamespace(**{key: bool(int(value)) if value in ('0', '1') else value
                                  for key, value in POLICY_ENV.items()})
        for alias, (arch, linear, mamba, tool) in STABLE_PROFILES.items():
            with self.subTest(alias=alias):
                flags = backend_flags(alias)
                for option, expected in (
                        ('--model-impl', 'vllm'), ('--worker-cls', 'stable_worker.Worker'),
                        ('--attention-backend', 'FLASH_ATTN'), ('--moe-backend', 'humming'),
                        ('--linear-backend', linear), ('--tool-call-parser', tool)):
                    self.assertEqual(flags[flags.index(option) + 1], expected)
                self.assertEqual(bool(mamba), '--mamba-backend' in flags)
                if mamba:
                    self.assertEqual(flags[flags.index('--mamba-backend') + 1], mamba)
                args = model_profile(alias)['args']
                self.assertEqual(profile_flags(args), args)

                mm = SimpleNamespace(mm_encoder_attn_backend=SimpleNamespace(name='FLASH_ATTN'))
                config = SimpleNamespace(
                    attention_config=SimpleNamespace(backend=SimpleNamespace(name='FLASH_ATTN')),
                    kernel_config=SimpleNamespace(moe_backend='humming', linear_backend=linear),
                    parallel_config=SimpleNamespace(worker_cls='stable_worker.Worker'),
                    mamba_config=SimpleNamespace(backend=SimpleNamespace(name='FLASHINFER')),
                    model_config=SimpleNamespace(architectures=[arch], trust_remote_code=False,
                                                 model_impl='vllm', use_mla=False, is_deepseek_mla=False,
                                                 multimodal_config=mm if model_profile(alias)['image'] else None),
                    speculative_config=None,
                )
                with patch('backend_policy.verify_process'), patch('backend_policy.emit'), \
                        patch.dict(os.environ, {'FLASHINFER_NO_DOWNLOAD': '1'}), \
                        patch('backend_policy.sys.meta_path', [BlockBackendImports()]):
                    verify_effective(config, envs, stable=alias)
                    cases = [(config.kernel_config, 'linear_backend', 'auto'),
                             (config.kernel_config, 'moe_backend', 'deep_gemm'),
                             (config.model_config, 'use_mla', True),
                             (config.model_config, 'trust_remote_code', True),
                             (config.model_config, 'model_impl', 'transformers'),
                             (config.attention_config.backend, 'name', 'FLASHMLA'),
                             (config, 'speculative_config', object())]
                    if mamba:
                        cases.append((config.mamba_config.backend, 'name', 'TRITON'))
                    if config.model_config.multimodal_config:
                        cases.append((mm.mm_encoder_attn_backend, 'name', 'FLASHMLA'))
                    with patch.dict(os.environ, {'FLASHINFER_NO_DOWNLOAD': '0'}), self.assertRaisesRegex(
                            ValueError, 'offline FlashInfer'):
                        verify_effective(config, envs, stable=alias)
                    for obj, attr, value in cases:
                        with self.subTest(setting=attr), patch.object(obj, attr, value), self.assertRaises(ValueError):
                            verify_effective(config, envs, stable=alias)

    def test_ready_requires_rank_and_process_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'serve.log'
            samples = Path(tmp) / 'gpu.csv'
            worker = dict(event='worker_backends_verified', pid=77, rank=0, vision_enabled=False,
                          deep_gemm=False, flashmla=False)
            guard = dict(event='guard_installed', pid=77)
            log.write_text('PACE_BACKEND ' + json.dumps(worker) + '\n')
            with self.assertRaises(ValueError):
                checked_backends(log, 1, 'text')
            with log.open('a') as handle:
                handle.write('PACE_BACKEND ' + json.dumps(guard) + '\n')
            self.assertEqual(checked_backends(log, 1, 'text'), [worker])
            with self.assertRaises(ValueError):
                checked_backends(log, 1, 'image')
            for alias in STABLE_PROFILES:
                with self.subTest(alias=alias):
                    with self.assertRaises(ValueError):
                        checked_backends(log, 1, 'suite', model=alias, vision=False)
                    event = dict(worker, profile=alias)
                    log.write_text('PACE_BACKEND ' + json.dumps(event) + '\nPACE_BACKEND ' + json.dumps(guard))
                    self.assertEqual(checked_backends(log, 1, 'suite', model=alias, vision=False), [event])
            samples.write_text('GPU-1, 77, 501\nGPU-1, 77, 654\nGPU-1, 99, 999\n')
            self.assertEqual(process_peak(samples, [worker]), {'GPU-1': 654})
