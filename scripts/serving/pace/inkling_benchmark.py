"""Warm, token-accounted Inkling profiling with the guarded PACE launcher."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import secrets
import signal
from statistics import median
import subprocess
import sys
import tempfile
import time
from urllib.request import Request

from managed_job import await_endpoint, child_process, stop
from pace_inference import SOURCE, cache_root, model_profile, reviewed_runtime
from validate import api_open, checked_backends, completion, process_peak, red_png, serving_metrics


PROMPT = 'List the integers from 1 to 300, separated by commas. Give only the list.'
LONG_PROMPT = ('The garden contains eight small trees and several equally spaced rows of flowers. ' * 125
               + '\n\n' + PROMPT)
PROFILES = {
    'eager': ('eager', 1, 'text'),
    'graph': ('graph', 1, 'text'),
    'compile': ('compile', 1, 'text'),
    'graph-nonbreakable': ('graph-nonbreakable', 1, 'text'),
    'compile-nonbreakable': ('compile-nonbreakable', 1, 'text'),
    'graph-batch4': ('graph', 4, 'text'),
    'graph-image': ('graph', 1, 'image'),
    'eager-image': ('eager', 1, 'image'),
}
METRICS = ('request_prefill_time_seconds', 'request_decode_time_seconds', 'time_to_first_token_seconds',
           'request_queue_time_seconds', 'request_prefill_kv_computed_tokens')


def payload(content, effort='none', tokens=256):
    return {'model': 'inkling', 'messages': [{'role': 'user', 'content': content}],
            'temperature': 0, 'max_tokens': tokens, 'reasoning_effort': effort,
            'seed': 142}


def stream_statistics(lines, started, now):
    usage = None
    first = last = None
    first_count = received = 0
    for line in lines:
        if not line.startswith(b'data: '):
            continue
        data = line[6:].strip()
        if data == b'[DONE]':
            break
        event = json.loads(data)
        usage = event.get('usage') or usage
        count = sum(len(choice.get('token_ids') or ()) for choice in event.get('choices', ()))
        if count:
            last = now()
            received += count
            if first is None:
                first, first_count = last, count
    elapsed = now() - started
    if not usage or usage['completion_tokens'] != received or first is None or last <= first:
        raise ValueError(f'Incomplete SSE token accounting: observed {received}, usage {usage}')
    details = usage.get('completion_tokens_details') or {}
    reasoning = details.get('reasoning_tokens')
    return {'elapsed_s': elapsed, 'ttft_s': first - started, 'decode_s': last - first,
            'completion_tokens': received, 'prompt_tokens': usage['prompt_tokens'],
            'reasoning_tokens': reasoning,
            'other_completion_tokens': received - reasoning if reasoning is not None else None,
            'first_chunk_tokens': first_count,
            'decode_tok_s': (received - first_count) / (last - first),
            'e2e_tok_s': received / elapsed}


def stream_request(base, prompt, tokens=256):
    data = dict(payload(prompt, tokens=tokens), ignore_eos=True, stream=True,
                return_token_ids=True, stream_options={'include_usage': True})
    request = Request(base + '/v1/chat/completions', data=json.dumps(data).encode(),
                      headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with api_open(request, timeout=180) as response:
        result = stream_statistics(response, started, time.monotonic)
    if result['completion_tokens'] != tokens:
        raise ValueError('The benchmark did not generate the requested token count')
    return result


def parse_metrics(text):
    matched = {}
    for line in text.splitlines():
        match = re.fullmatch(r'vllm:([a-z_]+)_(sum|count)(?:\{[^}]*\})? (\S+)', line)
        if match and match[1] in METRICS:
            key = (match[1], match[2])
            matched[key] = matched.get(key, 0) + float(match[3])
    return matched


def metrics(base):
    with api_open(base + '/metrics', timeout=15) as response:
        return parse_metrics(response.read().decode())


def metrics_delta(before, after, expected):
    output = {}
    for name in METRICS:
        total, count = (after.get((name, field), 0) - before.get((name, field), 0)
                        for field in ('sum', 'count'))
        if count:
            output[name] = {'count': int(count), 'average': total / count,
                            'complete': count == expected}
    return output


def cpu_stats(pids):
    return sum(sum(map(int, (Path('/proc') / str(pid) / 'stat').read_text()
                       .rsplit(')', 1)[1].split()[11:13])) / os.sysconf('SC_CLK_TCK') for pid in pids)


def measure(base, prompt, concurrency, pids, repeats):
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        def request_group():
            started = time.monotonic()
            requests = list(pool.map(lambda unused: stream_request(base, prompt), range(concurrency)))
            return {'requests': requests,
                    'aggregate_tok_s': sum(request['completion_tokens'] for request in requests)
                    / (time.monotonic() - started)}
        warmup = request_group()
        before = metrics(base)
        cpu_start = cpu_stats(pids)
        started = time.monotonic()
        rounds = [request_group() for _ in range(repeats)]
        cpu_cores = (cpu_stats(pids) - cpu_start) / (time.monotonic() - started)
        after = metrics(base)
    samples = [request for item in rounds for request in item['requests']]
    return {'concurrency': concurrency, 'warmup': warmup, 'repeats': rounds,
            'server_metrics': metrics_delta(before, after, repeats * concurrency),
            'average_api_worker_cpu_cores': round(cpu_cores, 3),
            'median': {key: round(median(item[key] for item in samples), 5)
                       for key in ('prompt_tokens', 'ttft_s', 'decode_tok_s', 'e2e_tok_s')},
            'median_aggregate_tok_s': round(median(item['aggregate_tok_s'] for item in rounds), 3)}


def acceptance(base, image):
    tests = {'text': (payload('What is 17 multiplied by 23? Answer with the number only.', tokens=64),
                      r'^\s*391[.!]?\s*$', False),
             'reasoning': (payload('Work out how many seedlings are in 17 trays of 23. '
                                   'Give the total as your final answer.', 'low', 512),
                           r'\b391\b', True)}
    if image:
        url = 'data:image/png;base64,' + base64.b64encode(red_png()).decode('ascii')
        content = [{'type': 'image_url', 'image_url': {'url': url}},
                   {'type': 'text', 'text': 'What single color fills this image? Answer in one word.'}]
        tests['image'] = (payload(content, tokens=64), r'\bred\b', False)
    results = {}
    for name, (request, answer, check_reasoning) in tests.items():
        response, elapsed = completion(base, request)
        choice = response['choices'][0]
        message = choice['message']
        final = message.get('content') or ''
        reasoning = message.get('reasoning') or message.get('reasoning_content') or ''
        passed = (re.search(answer, final, re.IGNORECASE) is not None and choice['finish_reason'] == 'stop'
                  and (not check_reasoning or bool(reasoning.strip())) and not message.get('tool_calls'))
        results[name] = {'passed': passed, 'final': final, 'parsed_reasoning_chars': len(reasoning),
                         'server_usage': response['usage'], 'e2e_s': elapsed}
    return results


def run_case(name, output, repeats):
    profile, batch, mode = PROFILES[name]
    output.mkdir(mode=0o700)
    log = output / 'server.log'
    key = output / 'api-key'
    with key.open('x') as file:
        file.write(secrets.token_urlsafe(32) + '\n')
    old_key = os.environ.get('PACE_SERVER_KEY_FILE')
    os.environ['PACE_SERVER_KEY_FILE'] = str(key)
    summary = {'case': name, 'passed': False}
    try:
        with tempfile.TemporaryDirectory(prefix='inkling-perf-') as local, ExitStack() as stack:
            telemetry = ['nvidia-smi', '--query-gpu=timestamp,name,driver_version,pstate,clocks.sm,clocks.mem,'
                         'utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu',
                         '--format=csv', '--loop-ms=500']
            stack.enter_context(child_process(telemetry, output / 'gpu-telemetry.csv'))
            gpu_process = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                           '--format=csv,noheader,nounits', '--loop=1']
            stack.enter_context(child_process(gpu_process, output / 'gpu-process.csv'))
            command = [sys.executable, '-B', str(SOURCE / 'pace_inference.py'), 'serve', 'inkling',
                       '--mode', mode, '--inkling-execution', profile, '--inkling-batch', str(batch)]
            started = time.monotonic()
            server = stack.enter_context(child_process(command, log, env=dict(os.environ, TMPDIR=local)))
            base, _ = await_endpoint(server, 'inkling', 0, started + 900)
            summary['ready_s'] = time.monotonic() - started
            workers = checked_backends(log, 1, mode, 'inkling')
            pids = {server.pid, *(event['pid'] for event in workers)}
            summary['workers'] = workers
            if mode == 'text':
                summary['short'] = measure(base, PROMPT, 1, pids, repeats)
                summary['long'] = measure(base, LONG_PROMPT, 1, pids, repeats)
                if batch > 1:
                    summary['aggregate'] = measure(base, PROMPT, batch, pids, repeats)
            summary['acceptance'] = acceptance(base, mode == 'image')
            summary['process_gpu_peak_mib'] = process_peak(output / 'gpu-process.csv', workers)
            summary.update(serving_metrics(log))
            summary['passed'] = all(item['passed'] for item in summary['acceptance'].values())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        summary['error'] = str(error)
    finally:
        key.unlink(missing_ok=True)
        if old_key is None:
            os.environ.pop('PACE_SERVER_KEY_FILE', None)
        else:
            os.environ['PACE_SERVER_KEY_FILE'] = old_key
        (output / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({name: {key: summary.get(key) for key in ('passed', 'ready_s', 'error')}}), flush=True)
    return summary['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('cases', nargs='+', choices=PROFILES)
    parser.add_argument('--repeats', type=int, default=3, choices=range(2, 13))
    args = parser.parse_args()
    os.umask(0o077)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    results = [run_case(case, args.output / case, args.repeats) for case in args.cases]
    runtime = reviewed_runtime(cache_root(), model_profile('inkling'))
    try:
        diagnostic = subprocess.run(
            [str(runtime / 'bin/python'), '-IB', str(SOURCE / 'gpu_copy.py')],
            capture_output=True, text=True, timeout=120, check=True)
        (args.output / 'gpu-copy.json').write_text(diagnostic.stdout)
    except (OSError, subprocess.SubprocessError) as error:
        print('GPU copy diagnostic failed: ' + str(error), file=sys.stderr)
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
