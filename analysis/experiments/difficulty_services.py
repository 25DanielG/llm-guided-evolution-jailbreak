"""Isolated Slurm-local services. No allocation or submission at import time."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request


def stop_process(proc):
    """Reap an entire session, including vLLM workers surviving their parent."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


@contextmanager
def port_block():
    # Hold a cooperating lock throughout the run; probe every port before use.
    for base in range(18000, 28000, 10):
        lock = open(f'/tmp/difficulty-ports-{base}.lock', 'a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue
        probes = []
        try:
            for port in range(base, base + 4):
                sock = socket.socket()
                probes.append(sock)
                sock.bind(('127.0.0.1', port))
        except OSError:
            lock.close()
            continue
        finally:
            for sock in probes:
                sock.close()
        try:
            yield base
        finally:
            lock.close()
        return
    raise RuntimeError('No available local service port block')


class Services:
    def __init__(self, work, attempt, env, models, vllm_bin, base):
        self.work, self.attempt, self.env = Path(work), Path(attempt), env
        self.models, self.vllm_bin, self.base = models, vllm_bin, base
        self.processes = []
        self.logs = []
        self.cache = Path(tempfile.mkdtemp(prefix='difficulty-kernels-'))

    def start(self, label, command, gpu, port, route):
        for trial in range(1, 4):
            log = open(self.attempt / f'{label}.{trial}.log', 'w')
            self.logs.append(log)
            env = dict(self.env, CUDA_VISIBLE_DEVICES=gpu,
                       VLLM_CACHE_ROOT=str(self.cache / label / 'vllm'),
                       TRITON_CACHE_DIR=str(self.cache / label / 'triton'),
                       TORCHINDUCTOR_CACHE_DIR=str(self.cache / label / 'torch'),
                       FLASHINFER_WORKSPACE_BASE=str(self.cache / label / 'flashinfer'))
            proc = subprocess.Popen(command, cwd=self.work, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            self.processes.append(proc)
            deadline = time.monotonic() + 1200
            while proc.poll() is None and time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}{route}', timeout=5) as response:
                        if response.status == 200:
                            return
                except (OSError, TimeoutError):
                    pass
                time.sleep(2)
            stop_process(proc)
        raise RuntimeError(f'{label} failed readiness after three attempts; see {self.attempt}')

    def __enter__(self):
        devices = self.env.get('CUDA_VISIBLE_DEVICES', '').split(',')
        if len(devices) < 5 or not devices[0]:
            raise RuntimeError('This experiment requires five allocated GPUs')
        commands = []
        for i, (role, gpu, tp) in enumerate([
            ('target', devices[0], 1), ('judge', ','.join(devices[1:3]), 2),
            ('reporting', devices[3], 1),
        ]):
            port = self.base + i
            command = [self.vllm_bin, 'serve', self.models[role]['path'],
                       '--served-model-name', role, '--host', '127.0.0.1', '--port', str(port),
                       '--dtype', 'bfloat16', '--tensor-parallel-size', str(tp),
                       '--max-model-len', str(self.models[role].get('max_model_len', 4096)), '--max-num-seqs', '64',
                       '--gpu-memory-utilization', '0.9', '--seed', self.env['DT_SEED']]
            commands.append((role, command, gpu, port, '/v1/models'))
        commands.append(('mutator', [self.env['DT_PYTHON'], '-m', 'uvicorn', 'server:app',
                                     '--host', '127.0.0.1', '--port', str(self.base + 3)],
                         devices[4], self.base + 3, '/'))
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(self.start, *args) for args in commands]
                for future in futures:
                    future.result()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        for proc in self.processes:
            stop_process(proc)
        for log in self.logs:
            log.close()
        shutil.rmtree(self.cache, ignore_errors=True)
