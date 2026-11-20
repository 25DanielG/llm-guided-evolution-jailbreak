"""Sample allocated CUDA devices by UUID, never by Slurm's node index.

CUDA ordinals, Slurm GRES IDs and NVML indices are distinct namespaces. Resolve
CUDA runtime visible PCI addresses against NVML before selecting GPUs.
No CUDA tensors or model contexts are created by this monitor.
"""
import argparse
import csv
import ctypes
import datetime
import fcntl
import glob
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

ENV_KEYS = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'SLURM_JOB_GPUS',
            'SLURM_STEP_GPUS', 'SLURM_JOB_ID', 'SLURM_STEP_ID')
QUERY = 'index,uuid,pci.bus_id,name,memory.total,memory.used'


def pci_id(value):
    match = re.fullmatch(r'([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value.strip())
    if not match:
        raise ValueError('Invalid PCI bus ID: ' + value)
    return tuple(int(part, 16) for part in match.groups())


def parse_inventory(output):
    result = []
    for fields in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(fields) != 6:
            raise ValueError('Malformed nvidia-smi record: ' + repr(fields))
        index, uuid, pci, name, total, used = [part.strip() for part in fields]
        if not index.isdecimal() or not uuid.startswith('GPU-'):
            raise ValueError('Invalid GPU identity: ' + repr(fields))
        pci_id(pci)
        total, used = int(total), int(used)
        if total <= 0 or not 0 <= used <= total:
            raise ValueError('Invalid GPU memory: ' + repr(fields))
        result.append(dict(index=index, uuid=uuid, pci_bus_id=pci, name=name,
                           total_mib=total, used_mib=used))
    if not result or len({g['uuid'] for g in result}) != len(result):
        raise ValueError('Empty or duplicate GPU inventory')
    return result


def query_gpus(uuids=None):
    command = ['nvidia-smi', '--query-gpu=' + QUERY, '--format=csv,noheader,nounits']
    if uuids is not None:
        command.append('--id=' + ','.join(uuids))
    proc = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if proc.returncode or proc.stderr.strip():
        raise RuntimeError('GPU query failed: ' + repr((command, proc.returncode, proc.stderr)))
    return parse_inventory(proc.stdout)


def legacy_filter_evidence():
    """Preserve a reproduction of the old filter; never use it for coverage."""
    raw = os.environ.get('SLURM_JOB_GPUS') or os.environ.get('CUDA_VISIBLE_DEVICES', '')
    command = ['nvidia-smi', '--id=' + raw, '--query-gpu=' + QUERY,
               '--format=csv,noheader,nounits']
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=15)
        evidence = dict(command=command, returncode=proc.returncode,
                        stdout=proc.stdout, stderr=proc.stderr)
        if proc.returncode == 0:
            evidence['uuids'] = [g['uuid'] for g in parse_inventory(proc.stdout)]
        return evidence
    except Exception as exc:
        return dict(command=command, error=type(exc).__name__ + ': ' + str(exc))


def cuda_pci_ids():
    serving = os.environ.get('JB_SERVING_ENV', '')
    candidates = sorted(glob.glob(serving + '/lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*')) if serving else []
    runtime = ctypes.CDLL(candidates[0] if candidates else 'libcudart.so.12')
    runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.cudaDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    count = ctypes.c_int()
    code = runtime.cudaGetDeviceCount(ctypes.byref(count))
    if code:
        raise RuntimeError('cudaGetDeviceCount failed: ' + str(code))
    result = []
    for ordinal in range(count.value):
        bus = ctypes.create_string_buffer(64)
        code = runtime.cudaDeviceGetPCIBusId(bus, len(bus), ordinal)
        if code:
            raise RuntimeError('cudaDeviceGetPCIBusId failed: ' + str((ordinal, code)))
        result.append(bus.value.decode('ascii'))
    return result


def resolve_devices(visible_pci, inventory, expected):
    if expected <= 0 or len(visible_pci) != expected:
        raise ValueError('CUDA allocation count differs: ' + str((len(visible_pci), expected)))
    by_pci = {}
    for gpu in inventory:
        key = pci_id(gpu['pci_bus_id'])
        if key in by_pci:
            raise ValueError('Ambiguous NVML PCI identity')
        by_pci[key] = gpu
    result = []
    for ordinal, bus in enumerate(visible_pci):
        if pci_id(bus) not in by_pci:
            raise ValueError('CUDA device missing from NVML: ' + bus)
        gpu = by_pci[pci_id(bus)]
        result.append(dict(cuda_ordinal=ordinal, uuid=gpu['uuid'],
                           nvml_index=gpu['index'], pci_bus_id=gpu['pci_bus_id']))
    if len({g['uuid'] for g in result}) != expected:
        raise ValueError('Duplicate CUDA device identity')
    return result


def validate_sample(row, allocation):
    if row.get('schema_version') != 2 or row.get('returncode') != 0 or row.get('error'):
        raise ValueError('Failed or unsupported GPU telemetry sample')
    if row.get('job_id') != allocation['job_id'] or row.get('hostname') != allocation['hostname']:
        raise ValueError('GPU sample belongs to another allocation')
    expected = {g['uuid']: g for g in allocation['devices']}
    gpus = row['gpus']
    actual = [g['uuid'] for g in gpus]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError('Incomplete GPU coverage: missing=' + str(sorted(set(expected) - set(actual)))
                         + ', unexpected=' + str(sorted(set(actual) - set(expected))))
    for gpu in gpus:
        if pci_id(gpu['pci_bus_id']) != pci_id(expected[gpu['uuid']]['pci_bus_id']):
            raise ValueError('GPU PCI identity changed')
        if 'H100' not in gpu['name'] or type(gpu['total_mib']) is not int or type(gpu['used_mib']) is not int:
            raise ValueError('Invalid GPU hardware or memory record')
        if gpu['total_mib'] <= 0 or not 0 <= gpu['used_mib'] <= gpu['total_mib']:
            raise ValueError('Invalid GPU memory range')


def check_telemetry(path, expected, job_id, require_active=True, now=None):
    path = Path(path)
    allocation = json.loads(path.with_suffix('.allocation.json').read_text())
    devices = allocation['devices']
    if (allocation.get('schema_version') != 2 or allocation['job_id'] != str(job_id)
            or len(devices) != expected or len({g['uuid'] for g in devices}) != expected
            or sorted(g['cuda_ordinal'] for g in devices) != list(range(expected))):
        raise ValueError('GPU allocation identity differs')
    peaks = {g['uuid']: 0 for g in devices}
    count, active, previous = 0, 0, None
    with path.open() as stream:
        # The sampler may still be alive. Locking prevents reading half a record.
        fcntl.flock(stream, fcntl.LOCK_SH)
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                validate_sample(row, allocation)
                stamp = datetime.datetime.fromisoformat(row['time']).timestamp()
                if previous is not None and not 0 < stamp - previous <= 30:
                    raise ValueError('GPU sampling clock reversed or gap exceeds 30 seconds')
                previous = stamp
                for gpu in row['gpus']:
                    peaks[gpu['uuid']] = max(peaks[gpu['uuid']], gpu['used_mib'])
                active += all(g['used_mib'] >= 1024 for g in row['gpus'])
                count += 1
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'Invalid GPU telemetry at line {line_number}: {exc}') from exc
    if count < 2 or (require_active and active < 2):
        raise ValueError('At least two complete samples with all model GPUs active are required')
    if now is not None and not 0 <= now - previous <= 30:
        raise ValueError('GPU telemetry stopped before acceptance checks')
    return dict(peak_gpu_used_mib=peaks, gpu_sample_count=count, all_gpu_active_samples=active,
                gpu_allocation=allocation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--expected', type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('GPU sampling requires Slurm and explicit CUDA visibility')
    args.path.parent.mkdir(parents=True, exist_ok=True)
    with args.path.open('x') as stream:
        try:
            inventory = query_gpus()
            devices = resolve_devices(cuda_pci_ids(), inventory, args.expected)
            allocation = dict(schema_version=2, job_id=os.environ['SLURM_JOB_ID'],
                              hostname=socket.gethostname(), devices=devices,
                              environment={k: os.environ.get(k) for k in ENV_KEYS},
                              nvml_inventory=inventory, legacy_filter=legacy_filter_evidence())
            with args.path.with_suffix('.allocation.json').open('x') as identity:
                identity.write(json.dumps(allocation, indent=2) + '\n')
            while True:
                row = dict(schema_version=2, time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           job_id=allocation['job_id'], hostname=allocation['hostname'],
                           returncode=0, error='', gpus=query_gpus([g['uuid'] for g in devices]))
                validate_sample(row, allocation)
                fcntl.flock(stream, fcntl.LOCK_EX)
                stream.write(json.dumps(row) + '\n'); stream.flush()
                fcntl.flock(stream, fcntl.LOCK_UN)
                time.sleep(5)
        except Exception as exc:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(dict(schema_version=2, returncode=1,
                                         error=type(exc).__name__ + ': ' + str(exc))) + '\n')
            stream.flush()
            raise


if __name__ == '__main__':
    main()
