"""Prove the native and Python blocks and exercise CUDA before staging."""

import ctypes
import importlib
import os
from pathlib import Path
import subprocess
import sys

import torch
import transformers
import triton
import vllm
from vllm import envs
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from backend_policy import STABLE_VERSION, VERSION, emit, isolate_wheel_files, verify_process


def main():
    verify_process()
    count, libraries = isolate_wheel_files()
    for name in ('vllm._flashmla_C', 'vllm._flashmla_extension_C', 'vllm.third_party.deep_gemm'):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            if not str(error).startswith('PACE denied backend import'):
                raise
        else:
            raise ValueError(f'Unexpected successful import of {name}')
    package = Path(vllm.__file__).parent
    for item in ('_flashmla_C.abi3.so', '_flashmla_extension_C.abi3.so',
                 'third_party/deep_gemm/_C_extension.abi3.so'):
        binary = package / item
        try:
            with binary.open('rb'):
                raise ValueError(f'The native backend file is readable: {binary}')
        except PermissionError:
            pass
        try:
            ctypes.CDLL(str(binary))
        except RuntimeError as error:
            if not str(error).startswith('PACE denied native backend load'):
                raise
        else:
            raise ValueError(f'Unexpected successful native backend load: {binary}')
    if envs.VLLM_USE_DEEP_GEMM or envs.VLLM_MOE_USE_DEEP_GEMM or is_deep_gemm_supported():
        raise ValueError('vLLM considers DeepGEMM enabled')
    backend = AttentionBackendEnum.FLASH_ATTN.get_class()
    if backend.__name__ != 'FlashAttentionBackend':
        raise ValueError(f'Unexpected FLASH_ATTN implementation: {backend}')
    nvcc = subprocess.check_output([os.path.join(os.environ['CUDA_HOME'], 'bin/nvcc'), '--version'],
                                   text=True, timeout=8)
    compiled_cuda, compiler_cuda = {VERSION: ('13.2', '13.2'),
                                    STABLE_VERSION: ('13.0', '13.5')}[vllm.__version__]
    if vllm.__version__ == STABLE_VERSION and os.environ.get('FLASHINFER_NVCC'):
        compiler_cuda = '13.0'
    if f'release {compiler_cuda},' not in nvcc or torch.version.cuda != compiled_cuda:
        raise ValueError(f'Unexpected compiler or PyTorch CUDA combination: {nvcc}')
    if '--cpu-only' not in sys.argv:
        expected = int(os.environ.get('PACE_EXPECTED_GPUS', '1'))
        if not torch.cuda.is_available() or torch.cuda.device_count() != expected:
            raise ValueError(f'Expected {expected} visible CUDA GPU(s)')
        devices = [torch.cuda.get_device_name(index) for index in range(expected)]
        family = os.environ.get('PACE_EXPECTED_GPU_FAMILY')
        if any(torch.cuda.get_device_capability(index) != (9, 0) for index in range(expected)):
            raise ValueError('Expected a CUDA-capable Hopper allocation')
        if family and any(family not in device for device in devices):
            raise ValueError(f'Expected {family}, received {devices}')
        for index in range(expected):
            if (torch.ones(1, device=f'cuda:{index}') + 1).item() != 2:
                raise ValueError(f'CUDA tensor preflight failed on device {index}')
        emit('cuda_devices_verified', devices=devices, capability='sm90')
    verify_process()
    emit('preflight_verified', mode='cpu' if '--cpu-only' in sys.argv else 'cuda',
         vllm=vllm.__version__, torch=torch.__version__, cuda=torch.version.cuda,
         compiler=compiler_cuda,
         transformers=transformers.__version__, triton=triton.__version__,
         cuda_home=os.environ['CUDA_HOME'], ptxas=os.environ['TRITON_PTXAS_PATH'],
         attention=backend.__name__, restricted_files=count, native=libraries)


if __name__ == '__main__':
    main()
