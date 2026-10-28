"""Initialize every assigned CUDA context before starting the HF model server.

Keep this operational bootstrap outside server.py so existing experiment source
fingerprints and saved checkpoints remain valid.
"""
from __future__ import annotations

import argparse
import os


def initialize_cuda(torch):
    count = torch.cuda.device_count()
    if not torch.cuda.is_available() or not count:
        raise RuntimeError('The mutator requires accessible CUDA GPUs')
    print(f"Mutator CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    previous = torch.cuda.current_device()
    try:
        for index in range(count):
            # Enumeration alone does not create a usable context on each GPU.
            # Transformers queries all devices during allocator warmup, so
            # establish and validate each context before loading model shards.
            with torch.cuda.device(index):
                probe = torch.empty(1, device=f'cuda:{index}')
                torch.cuda.synchronize(index)
                free, total = torch.cuda.mem_get_info(index)
                print(f'Mutator CUDA device {index}: {free}/{total} bytes free', flush=True)
                del probe
    finally:
        torch.cuda.set_device(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=8137)
    args = parser.parse_args()
    import torch
    initialize_cuda(torch)
    import uvicorn
    uvicorn.run('server:app', host=args.host, port=args.port)


if __name__ == '__main__':
    main()
