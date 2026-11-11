"""Bounded real-inference acceptance on an existing Slurm GPU allocation."""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import requests
from src.chat_response import parse_chat_response
from src.model_serving import command, model_path, profile, validate_checkpoint
from src.run_status import atomic_json, utc


def gpu_sample(device_uuids):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total,memory.used',
                             '--format=csv,noheader,nounits', '--id', ','.join(device_uuids)],
                            capture_output=True, text=True, timeout=10, check=True)
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    return {'at': utc(), 'gpu_csv': result.stdout.strip(),
            'gpus': [{'uuid':row[1], 'name':row[2], 'total_mib':int(row[4]), 'used_mib':int(row[5])}
                     for row in rows]}


def tree_rss(pid):
    """Observed aggregate RSS; shared pages may be counted more than once."""
    total, seen, pending = 0, set(), [pid]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            for line in Path(f'/proc/{current}/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    total += int(line.split()[1])
            pending.extend(map(int, Path(f'/proc/{current}/task/{current}/children').read_text().split()))
        except (OSError, ValueError):
            continue
    return total


def wait_ready(url, process, deadline):
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Service exited before readiness: {process.returncode}')
        try:
            reply = requests.get(url, timeout=5)
            if reply.status_code == 200:
                return reply.json()
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError('Bounded service startup timed out')


def terminate(process):
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        # A parent may exit before all engine workers. Reap its remaining group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run(name, output, experiment_python, tp=None, port=18001, proxy_port=18137, startup_timeout=900):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {'schema_version': 1, 'model': name, 'started_at': utc(), 'status': 'failed',
              'slurm_job_id': os.getenv('SLURM_JOB_ID'), 'inference_verified': False}
    processes, handles, samples = [], [], []
    stop = threading.Event()
    sampler = None
    old_handlers = {}
    def interrupted(sig, _frame):
        raise InterruptedError(f'Smoke interrupted by signal {sig}')
    try:
        if not record['slurm_job_id'] or not os.getenv('CUDA_VISIBLE_DEVICES'):
            raise RuntimeError('Real smoke requires an existing Slurm GPU allocation')
        if sys.version_info[:2] != (3,12) or (tp is not None and tp < 1):
            raise ValueError('Smoke requires Python 3.12 and a positive tensor parallel size')
        item = profile(name)
        checkpoint = model_path(name)
        validate_checkpoint(name, checkpoint)
        import torch
        import vllm
        if (importlib.metadata.version('vllm') != '0.17.0' or torch.__version__.split('+')[0] != '2.10.0'
                or torch.version.cuda != '12.9'):
            raise RuntimeError('Serving runtime differs from requirements-serving.txt; use the separate pinned environment')
        if not torch.cuda.is_available() or torch.cuda.device_count() < (tp or item['tensor_parallel_size']):
            raise RuntimeError('Serving runtime cannot see the requested CUDA GPUs')
        # Resolve logical CUDA devices to physical UUIDs: Slurm can remap indices.
        uuids = [str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())]
        uuids = [u if u.startswith(('GPU-', 'MIG-')) else 'GPU-' + u for u in uuids]
        record.update(model_id=item['model_id'], revision=item['revision'], checkpoint=checkpoint,
                      torch_cuda=torch.version.cuda, python=sys.version,
                      packages={n:importlib.metadata.version(n) for n in
                                ('torch','vllm','transformers','mistral_common','openai-harmony')},
                      cuda_capabilities=[torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())],
                      cuda_device_uuids=uuids,
                      host_memory_method='sum smoke process-tree VmRSS KiB; shared pages may be counted twice')
        record['nvidia_smi'] = subprocess.run(['nvidia-smi', '--id', ','.join(uuids)],
                                             capture_output=True, text=True, timeout=10, check=True).stdout
        samples.append(gpu_sample(uuids))
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, interrupted)
        backend_log = (output / 'vllm.log').open('w')
        handles.append(backend_log)
        cmd = command(name, checkpoint, port=port, host='127.0.0.1', tp=tp)
        record['command'] = cmd
        started = time.monotonic()
        backend = subprocess.Popen(cmd, env=env, stdout=backend_log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(backend)
        def sample():
            while not stop.wait(10):
                try:
                    point = gpu_sample(uuids)
                    point['host_rss_kib'] = tree_rss(os.getpid())
                    samples.append(point)
                except (OSError, subprocess.SubprocessError) as exc:
                    samples.append({'at': utc(), 'error': str(exc)})
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        base = f'http://127.0.0.1:{port}/v1'
        models = wait_ready(base + '/models', backend, started + startup_timeout)
        if item['model_id'] not in {m['id'] for m in models['data']}:
            raise RuntimeError('Served identity mismatch')
        record.update(models=models, startup_seconds=time.monotonic() - started)
        samples.append(dict(gpu_sample(uuids), host_rss_kib=tree_rss(os.getpid())))
        payload = {'model':item['model_id'], 'messages':[{'role':'user','content':'Calculate 17 times 23. Briefly verify your final answer.'}],
                   'temperature':0, 'max_tokens':2048}
        if item.get('reasoning'):
            payload['reasoning_effort'] = 'medium'
        started = time.monotonic()
        reply = requests.post(base + '/chat/completions', json=payload, timeout=120)
        reply.raise_for_status()
        record['chat_latency_seconds'] = time.monotonic() - started
        record['chat_response'] = reply.json()
        parsed = parse_chat_response(record['chat_response'])
        record['parsed_response'] = parsed
        if parsed['finish_reason'] != 'stop' or not re.search(r'\b391\b', parsed['text']):
            raise RuntimeError('Benign arithmetic final answer was incorrect or incomplete')
        if item.get('reasoning') and not parsed['reasoning']:
            raise RuntimeError('GPT-OSS reasoning/final separation not demonstrated')
        proxy_env = dict(env, JB_MUTATOR_PROFILE=name, JB_MUTATOR_VLLM_URL=base)
        proxy_log = (output / 'generate-proxy.log').open('w')
        handles.append(proxy_log)
        proxy = subprocess.Popen([experiment_python, '-m', 'uvicorn', 'server:app', '--host','127.0.0.1',
                                  '--port',str(proxy_port), '--workers','1'], env=proxy_env,
                                 stdout=proxy_log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(proxy)
        wait_ready(f'http://127.0.0.1:{proxy_port}/', proxy, time.monotonic() + 120)
        started = time.monotonic()
        reply = requests.post(f'http://127.0.0.1:{proxy_port}/generate',
                              json={'prompt':'Return the word READY.', 'max_new_tokens':2048,
                                    'temperature':0, 'top_p':1}, timeout=120)
        reply.raise_for_status()
        record['generate_latency_seconds'] = time.monotonic() - started
        record['generate_response'] = reply.json()
        samples.append(dict(gpu_sample(uuids), host_rss_kib=tree_rss(os.getpid())))
        if not isinstance(reply.json().get('generated_text'), str) or not reply.json()['generated_text'].strip():
            raise RuntimeError('Existing /generate contract returned no final text')
        if not re.search(r'\bREADY\b', reply.json()['generated_text'], re.I):
            raise RuntimeError('Existing /generate did not follow the benign fixture instruction')
        if 'mistral' in item['vllm_args']:
            params = json.loads((Path(checkpoint) / 'params.json').read_text())
            record['native_quantization'] = params.get('quantization')
            record['fp8_load_evidence'] = [l for l in (output / 'vllm.log').read_text().splitlines()
                                         if re.search(r"quantization[=:]\s*['\"]?fp8", l, re.I)]
            if 'fp8' not in str(record['native_quantization']).lower() or not record['fp8_load_evidence']:
                raise RuntimeError('Native Mistral FP8 loading is not confirmed by config AND runtime log')
        record.update(status='passed', inference_verified=True)
    except Exception:
        record['error'] = traceback.format_exc()
    finally:
        stop.set()
        if sampler:
            sampler.join(timeout=12)
        for process in reversed(processes):
            terminate(process)
        for handle in handles:
            handle.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        record.update(finished_at=utc(), resource_samples=samples,
                      peak_observed_host_rss_kib=max((p.get('host_rss_kib',0) for p in samples), default=0),
                      peak_observed_gpu_used_mib={g['uuid']:max(h['used_mib'] for p in samples for h in p.get('gpus',[]) if h['uuid']==g['uuid'])
                                                  for p in samples for g in p.get('gpus',[])},
                      services_stopped=True)
        atomic_json(output / 'result.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--experiment-python', required=True)
    parser.add_argument('--tp', type=int)
    args = parser.parse_args()
    result = run(args.model, args.output, args.experiment_python, args.tp)
    print(f'{result["status"]}: {args.output}/result.json')
    raise SystemExit(0 if result['status'] == 'passed' else 1)
