"""Read-only deployment evidence; static configuration never counts as a live allocation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import importlib.metadata
import os
import re
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from src.run_status import atomic_json, utc


def capture(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {'command': command, 'returncode': result.returncode,
                'stdout': result.stdout[-64000:], 'stderr': result.stderr[-4000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command': command, 'returncode': None, 'error': str(exc)}


def collect(root):
    root = Path(root).expanduser().resolve()
    queries = [
        ['scontrol', 'ping'], ['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%T|%b|%M|%l|%N'],
        ['sinfo', '-N', '-h', '-o', '%N|%P|%G|%f|%t'], ['scontrol', 'show', 'partition'],
        ['sacctmgr', '-n', '-P', 'show', 'assoc', f'user={getpass.getuser()}',
         'format=Cluster,Account,Partition,QOS'],
        ['sacctmgr', '-n', '-P', 'show', 'qos', 'format=Name,MaxWall'],
        ['lfs', 'quota', '-h', '-u', getpass.getuser(), str(root)],
        ['bash', '-lc', 'module -t avail cuda'],
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(capture, queries))
    dns = {}
    for host in ('sched-ice-slurmctl', 'sched-ice-slurmdb', 'pypi.org', 'huggingface.co'):
        try:
            dns[host] = {'address': socket.getaddrinfo(host, None)[0][4][0]}
        except OSError as exc:
            dns[host] = {'error': str(exc)}
    packages = {}
    for name in ('vllm', 'torch', 'transformers', 'mistral_common', 'openai-harmony'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    parent = root
    while not parent.exists():
        parent = parent.parent
    usage = shutil.disk_usage(parent)
    return {'schema_version': 1, 'collected_at': utc(), 'host': socket.gethostname(),
            'python': sys.version, 'python_executable': sys.executable,
            'slurm_job_id': os.getenv('SLURM_JOB_ID'), 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
            'dns': dns, 'queries': results, 'packages': packages,
            'storage': {'requested_root': str(root), 'existing_parent': str(parent),
                        'disk_free_bytes': usage.free, 'os_access_write': os.access(parent, os.W_OK),
                        'quota_separate_from_disk_free': True},
            'gpu_probe': capture(['nvidia-smi']),
            'live_scheduler_verified': results[0].get('returncode') == 0 and bool(re.search(r'\bis UP\b', results[0].get('stdout', '')))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    evidence = collect(args.root)
    atomic_json(args.output, evidence)
    print(f'Preflight saved to {args.output}; live scheduler verified={evidence["live_scheduler_verified"]}')
