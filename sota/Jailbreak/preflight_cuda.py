"""Check the mutator's Python environment against its allocated GPU."""

import torch


def main():
    try:
        torch.cuda.init()
        device = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        probe = torch.ones(1, device="cuda")
        torch.cuda.synchronize()
        del probe
    except Exception as exc:
        raise SystemExit(f"CUDA preflight failed with torch {torch.__version__}: {exc}") from exc
    print(f"CUDA preflight passed: torch {torch.__version__}, {device.name}, "
          f"{free_bytes // (1024 ** 2)}/{total_bytes // (1024 ** 2)} MiB free", flush=True)


if __name__ == "__main__":
    main()
