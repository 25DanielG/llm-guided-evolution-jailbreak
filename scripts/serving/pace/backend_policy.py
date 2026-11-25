"""Fail-closed backend policy for the reviewed upstream vLLM wheel."""

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys


VERSION = '0.34.1rc1.dev144+g94b13b3d8'
STABLE_VERSION = '0.34.0'
POLICY_ENV = {
    'VLLM_USE_DEEP_GEMM': '0',
    'VLLM_MOE_USE_DEEP_GEMM': '0',
    'VLLM_USE_DEEP_GEMM_E8M0': '0',
    'VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES': '0',
    'VLLM_DEEP_GEMM_WARMUP': 'skip',
    'VLLM_MLA_DISABLE': '1',
    'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
    'VLLM_ENABLE_V1_MULTIPROCESSING': '1',
    'VLLM_USE_RUST_FRONTEND': '0',
    'VLLM_PLUGINS': '',
    'VLLM_NO_USAGE_STATS': '1',
    # Use the baseline sampler and avoid the optional FlashInfer JIT link.
    'VLLM_USE_FLASHINFER_SAMPLER': '0',
}
BACKEND_FLAGS = [
    '--attention-backend', 'FLASH_ATTN', '--moe-backend', 'humming',
    '--linear-backend', 'humming', '--worker-cls', 'guarded_worker.Worker',
]
# Per-model choices reviewed against the exact wheel. Stable runs cannot
# substitute a generic "auto" policy for one of these worker profiles.
STABLE_PROFILES = {
    'lightning': ('NemotronHForCausalLM', 'humming', 'flashinfer', 'qwen3_coder'),
    'muse': ('MuseGlimmerForConditionalGeneration', 'cutlass', None, 'muse_glimmer'),
    'super-vl': ('NemotronH_Omni_Reasoning_V3', 'humming', 'flashinfer', 'qwen3_coder'),
}
BLOCKED_MODULES = (
    'deep_gemm', 'deepgemm', 'flash_mla', 'flashmla', 'flash_mla_cuda',
    'vllm.third_party.deep_gemm', 'vllm.third_party.flashmla',
    'vllm._flashmla_C', 'vllm._flashmla_extension_C',
    'flashinfer.deep_gemm',
)
BLOCKED_NAME = re.compile(r'deep[_-]?gemm|flash[_-]?mla', re.IGNORECASE)


def emit(event, **data):
    # Spawned helper Python processes may reserve stdout for machine-readable data.
    print('PACE_BACKEND ' + json.dumps(dict(event=event, pid=os.getpid(), **data), sort_keys=True),
          file=sys.stderr, flush=True)


def profile_flags(args):
    if len(args) != 2:
        raise ValueError('Malformed model profile flags')
    option, value = args
    if option != '--reasoning-parser':
        raise ValueError(f'Model profile may not change backend policy: {option}')
    if value.startswith('-') or BLOCKED_NAME.search(value):
        raise ValueError(f'Unsafe model profile argument for {option}: {value}')
    return args


def backend_flags(alias):
    if alias == 'inkling':
        return BACKEND_FLAGS
    _, linear, mamba, tool = STABLE_PROFILES[alias]
    flags = [
        '--attention-backend', 'FLASH_ATTN', '--moe-backend', 'humming',
        '--linear-backend', linear, '--worker-cls', 'stable_worker.Worker',
        '--enable-auto-tool-choice', '--tool-call-parser', tool,
        '--model-impl', 'vllm',
    ]
    if mamba:
        flags += ['--mamba-backend', mamba]
    return flags


def isolate_wheel_files(configure=False):
    distribution = importlib.metadata.distribution('vllm')
    if distribution.version not in (VERSION, STABLE_VERSION):
        raise ValueError(f'Backend policy has no source review for vLLM {distribution.version}')
    root = Path(sys.prefix).resolve()
    if root == Path(sys.base_prefix).resolve():
        raise ValueError('Backend policy requires the isolated runtime')
    files = [p for p in distribution.files or () if p.parts[0] == 'vllm' and (
        p.parts[:3] in {('vllm', 'third_party', 'deep_gemm'), ('vllm', 'third_party', 'flashmla')}
        or p.name.startswith('_flashmla'))]
    native = {p.name for p in files if p.suffix == '.so'}
    required = {'_flashmla_C.abi3.so', '_flashmla_extension_C.abi3.so', '_C_extension.abi3.so'}
    if not required <= native:
        raise ValueError(f'Unexpected layout for the audited wheel: {sorted(native)}')
    paths = [Path(distribution.locate_file(relative)) for relative in files]
    if distribution.version == STABLE_VERSION:
        flashinfer = importlib.metadata.distribution('flashinfer-python')
        if flashinfer.version != '0.7.3':
            raise ValueError('The additional FlashInfer restrictions require 0.7.3')
        # FlashInfer imports shared GEMM dispatch wrappers during package
        # initialization. Quarantine its named native sources while allowing
        # wrappers that Mamba needs to import. Direct DeepGEMM entry points
        # are also excluded by the import guard below.
        sources = [p for p in flashinfer.files or () if p.parts[:2] == ('flashinfer', 'data')
                   and BLOCKED_NAME.search(str(p))]
        if not any('cake_batch_deepgemm_fp8' in p.parts for p in sources):
            raise ValueError('Unexpected FlashInfer DeepGEMM source layout')
        paths += [Path(flashinfer.locate_file(relative)) for relative in sources]
    for path in paths:
        current = path.lstat()
        if (not path.is_relative_to(root) or path.resolve() != path
                or not stat.S_ISREG(current.st_mode) or current.st_uid != os.getuid()
                or current.st_nlink != 1):
            raise ValueError(f'Backend file is outside the private copied install: {path}')
        if configure:
            # Keep upstream bytes intact. The private installation's DAC denies
            # reads and dlopen even along native paths outside Python imports.
            path.chmod(0)
        elif stat.S_IMODE(current.st_mode) or os.access(path, os.R_OK | os.X_OK):
            raise ValueError(f'Backend file is accessible: {path}')
    return len(paths), sorted(native)


def blocked_module(name):
    return (name.startswith('flashinfer.experimental.deepgemm_') or
            any(name == prefix or name.startswith(prefix + '.') for prefix in BLOCKED_MODULES))


class BlockBackendImports:
    def find_spec(self, fullname, path=None, target=None):
        if blocked_module(fullname):
            # The optional upstream availability probes handle ModuleNotFoundError.
            raise ModuleNotFoundError(f'PACE denied backend import {fullname}', name=fullname)


def verify_process():
    loaded = [name for name in sys.modules if blocked_module(name)]
    mapped = [line.strip() for line in Path('/proc/self/maps').read_text().splitlines()
              if BLOCKED_NAME.search(line) and '.so' in line]
    if loaded or mapped:
        raise ValueError(f'Prohibited code is loaded: modules={loaded}, mappings={mapped}')
    for name, value in POLICY_ENV.items():
        if os.environ.get(name) != value:
            raise ValueError(f'Unsafe or missing process configuration: {name}')


def install_process_guard():
    verify_process()
    count, native = isolate_wheel_files()
    offline_flashinfer = importlib.metadata.version('vllm') == STABLE_VERSION
    if offline_flashinfer:
        # The bootstrap runs before FlashInfer can resolve its artifact path.
        # The reviewed Hopper SSU implementation builds from bundled sources.
        cache = Path(os.environ['FLASHINFER_WORKSPACE_BASE'])
        os.environ['FLASHINFER_CUBIN_DIR'] = str(cache / 'flashinfer-cubins')
        os.environ['FLASHINFER_NO_DOWNLOAD'] = '1'
    # Reject externally installed packages too. Do this before installing the
    # finder so that its artificial absence cannot hide an external candidate.
    for name in BLOCKED_MODULES:
        if '.' not in name and importlib.util.find_spec(name) is not None:
            raise ValueError(f'Unexpected external backend distribution: {name}')
    sys.meta_path.insert(0, BlockBackendImports())

    def audit(event, args):
        if event == 'ctypes.dlopen' and args[0] and BLOCKED_NAME.search(os.fsdecode(args[0])):
            raise RuntimeError(f'PACE denied native backend load {args[0]}')

    sys.addaudithook(audit)
    emit('guard_installed', restricted_files=count, native=native, environment=POLICY_ENV,
         **({'flashinfer_downloads': False} if offline_flashinfer else {}))


def verify_effective(config, envs, stable=None):
    verify_process()
    if not any(isinstance(item, BlockBackendImports) for item in sys.meta_path):
        raise ValueError('Import guard missing from vLLM worker')
    for key, value in POLICY_ENV.items():
        if key == 'VLLM_PLUGINS':
            continue
        expected = bool(int(value)) if value in ('0', '1') else value
        if getattr(envs, key) != expected:
            raise ValueError(f'vLLM resolved unexpected setting {key}')
    actual = (getattr(config.attention_config.backend, 'name', None), config.kernel_config.moe_backend,
              config.kernel_config.linear_backend, config.parallel_config.worker_cls)
    expected = ('FLASH_ATTN', 'humming', 'humming', 'guarded_worker.Worker')
    if stable:
        architecture, linear, mamba, _ = STABLE_PROFILES[stable]
        expected = ('FLASH_ATTN', 'humming', linear, 'stable_worker.Worker')
        if config.model_config.architectures != [architecture]:
            raise ValueError(f'Unexpected architecture for {stable}')
        if mamba and config.mamba_config.backend.name.lower() != mamba:
            raise ValueError(f'Unexpected Mamba backend for {stable}')
        if config.model_config.trust_remote_code or config.model_config.model_impl != 'vllm':
            raise ValueError('The stable profile requires native model code')
        if os.environ.get('FLASHINFER_NO_DOWNLOAD') != '1':
            raise ValueError('The stable profile requires offline FlashInfer compilation')
    if actual != expected:
        raise ValueError(f'vLLM selected unsafe backends or worker: {actual}')
    if (config.speculative_config is not None or config.model_config.use_mla
            or config.model_config.is_deepseek_mla):
        raise ValueError('Speculative or MLA execution is outside this backend review')
    mm = config.model_config.multimodal_config
    if mm and (mm.mm_encoder_attn_backend is None or mm.mm_encoder_attn_backend.name != 'FLASH_ATTN'):
        raise ValueError('vLLM did not preserve the explicit vision attention configuration')
    emit('effective_config', attention=actual[0], moe=actual[1], linear=actual[2],
         worker=actual[3], vision='FLASH_ATTN' if mm else None,
         **({'profile': stable, 'mamba': mamba, 'tool_parser': STABLE_PROFILES[stable][3],
             'flashinfer_downloads': False}
            if stable else {}))


def guarded_environment(source, cache=None, runtime=None, cuda_toolkit=None):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('VLLM_', 'DG_JIT_')) and key not in (
               'LD_PRELOAD', 'LD_AUDIT', 'PYTHONHOME', 'HF_TOKEN',
               'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACEHUB_API_TOKEN',
           )}
    env.update(POLICY_ENV)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONNOUSERSITE='1',
               PYTHONPATH=os.pathsep.join((str(source / 'guard_bootstrap'), str(source))))
    if cache is not None:
        env.update(XDG_CACHE_HOME=str(cache), FLASHINFER_WORKSPACE_BASE=str(cache),
                   TRITON_CACHE_DIR=str(cache / 'triton'),
                   TORCH_EXTENSIONS_DIR=str(cache / 'torch-extensions'))
    if runtime is not None:
        toolkit = runtime / 'lib/python3.12/site-packages/nvidia/cu13'
        compiler = cuda_toolkit or toolkit
        nvcc, ptxas = compiler / 'bin/nvcc', toolkit / 'bin/ptxas'
        if not all(os.access(path, os.X_OK) for path in (nvcc, ptxas, runtime / 'bin/ninja')):
            raise ValueError('Pinned CUDA compiler tools or Ninja are missing')
        env.update(CUDA_HOME=str(compiler), CUDA_PATH=str(compiler), TRITON_PTXAS_PATH=str(ptxas),
                   PATH=os.pathsep.join((str(runtime / 'bin'), str(compiler / 'bin'),
                                         env.get('PATH', os.defpath))))
        if cuda_toolkit:
            env['FLASHINFER_NVCC'] = str(nvcc)
            env.pop('FLASHINFER_NVCC_LAUNCHER', None)
    return env


if __name__ == '__main__':
    if sys.argv[1:] != ['configure-private-install']:
        sys.exit('Expected configure-private-install')
    count, native = isolate_wheel_files(configure=True)
    isolate_wheel_files()
    emit('restricted_private_install', restricted_files=count, native=native)
