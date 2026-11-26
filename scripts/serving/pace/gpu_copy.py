"""Measure a small allocation-local device-to-device streaming copy."""

import json
from statistics import median

import torch


def main():
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise ValueError('The copy diagnostic requires exactly one allocated Hopper GPU')
    data = torch.zeros(256 * 1024**2, device='cuda', dtype=torch.uint8)
    target = torch.empty_like(data)
    for _ in range(5):
        target.copy_(data)
    torch.cuda.synchronize()
    rates = []
    for _ in range(5):
        start, end = (torch.cuda.Event(enable_timing=True) for unused in range(2))
        start.record()
        for _ in range(40):
            target.copy_(data)
        end.record()
        end.synchronize()
        rates.append(2 * data.nbytes * 40 / (start.elapsed_time(end) / 1000) / 1e12)
    print(json.dumps({'gpu': torch.cuda.get_device_name(), 'bytes_per_buffer': data.nbytes,
                      'median_read_plus_write_tb_s': round(median(rates), 3), 'samples_tb_s': rates}))


if __name__ == '__main__':
    main()
