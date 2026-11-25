"""Check the local OpenAI-compatible endpoint against a small offline fixture."""

import base64
import json
import os
from pathlib import Path
import re
import struct
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
import zlib


def api_open(target, timeout, authenticated=True):
    request = Request(target) if isinstance(target, str) else target
    if authenticated and (key_file := os.environ.get('PACE_SERVER_KEY_FILE')):
        key = Path(key_file).read_text().strip()
        request.add_unredirected_header('Authorization', 'Bearer ' + key)
    # The loopback endpoint belongs to the allocation. Ignore external proxies.
    return build_opener(ProxyHandler({})).open(request, timeout=timeout)


def red_png():
    def chunk(name, data):
        return struct.pack('>I', len(data)) + name + data + struct.pack('>I', zlib.crc32(name + data))

    dimensions = struct.pack('>IIBBBBB', 128, 128, 8, 2, 0, 0, 0)
    pixels = (b'\0' + b'\xff\0\0' * 128) * 128
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', dimensions) +
            chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))


def checked_backends(path, tp, mode, model=None, vision=None):
    events = []
    for line in path.read_text().splitlines():
        if 'PACE_BACKEND {' in line:
            events.append(json.loads(line.split('PACE_BACKEND ', 1)[1]))
    workers = {event['rank']: event for event in events if event['event'] == 'worker_backends_verified'}
    guarded = {event['pid'] for event in events if event['event'] == 'guard_installed'}
    if set(workers) != set(range(tp)):
        raise ValueError('Missing backend evidence for one or more worker ranks')
    if vision is None:
        vision = mode == 'image'
    for event in workers.values():
        if (event['pid'] not in guarded or event['deep_gemm'] is not False or
                event['flashmla'] is not False or event['vision_enabled'] != vision
                or (model in ('lightning', 'muse', 'super-vl') and event.get('profile') != model)):
            raise ValueError(f'Worker backend check failed: {event}')
    return list(workers.values())


def completion(base, payload):
    start = time.monotonic()
    request = Request(base + '/v1/chat/completions', data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    with api_open(request, timeout=180) as response:
        reply = json.load(response)
    return reply, time.monotonic() - start


def streaming_benchmark(base, payload):
    payload.update(stream=True, stream_options={'include_usage': True})
    request = Request(base + '/v1/chat/completions', data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    tokens = first = last = None
    start = time.monotonic()
    with api_open(request, timeout=240) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            event = json.loads(data)
            if event.get('usage'):
                tokens = event['usage']['completion_tokens']
            for choice in event.get('choices', []):
                delta = choice['delta']
                if delta.get('content') or delta.get('reasoning') or delta.get('reasoning_content'):
                    last = time.monotonic()
                    if first is None:
                        first = last
    elapsed = time.monotonic() - start
    if not tokens or first is None:
        raise ValueError('Streaming benchmark returned no tokens or usage')
    return {'seconds': round(elapsed, 3), 'completion_tokens': tokens,
            'end_to_end_tokens_per_second': round(tokens / elapsed, 2),
            'first_token_seconds': round(first - start, 3),
            'generation_tokens_per_second': round((tokens - 1) / (last - first), 2) if last > first else None}


def qualification_suite(base, model, video):
    fixtures = Path(__file__).resolve().parent / 'fixtures'
    suite = json.loads((fixtures / 'suite.json').read_text())

    def request(content, **extra):
        return {'model': model, 'messages': [{'role': 'user', 'content': content}],
                'temperature': 0, 'max_tokens': 1024, **extra}

    def template_options(reasoning=False):
        if model == 'muse':
            return {'reasoning_strength': 'high' if reasoning else 'low'}
        return {'enable_thinking': reasoning}

    results = {}
    for name in ('text', 'reasoning', 'tools', 'image', 'video'):
        if name in ('image', 'video') and not video:
            continue
        spec = suite[name]
        content = spec['prompt']
        if name == 'image':
            data = 'data:image/png;base64,' + base64.b64encode(red_png()).decode('ascii')
            content = [{'type': 'image_url', 'image_url': {'url': data}}, {'type': 'text', 'text': content}]
        if name == 'video':
            frames = (fixtures / 'red-frame.jpg.b64').read_text().strip()
            data = 'data:video/jpeg;base64,' + ','.join([frames] * 8)
            content = [{'type': 'video_url', 'video_url': {'url': data}}, {'type': 'text', 'text': content}]
        payload = request(content, chat_template_kwargs=template_options(name == 'reasoning'))
        if name == 'tools':
            payload.update(tools=[spec['tool']], tool_choice='auto')
        try:
            response, elapsed = completion(base, payload)
            choice = response['choices'][0]
            message = choice['message']
            answer = message.get('content') or ''
            reasoning = message.get('reasoning') or message.get('reasoning_content') or ''
            calls = message.get('tool_calls') or []
            if name == 'tools':
                passed = (len(calls) == 1 and calls[0]['function']['name'] == spec['tool']['function']['name']
                          and json.loads(calls[0]['function']['arguments']) == spec['expected'])
            else:
                passed = bool(re.search(spec['expected'], answer, re.IGNORECASE))
                if name == 'reasoning':
                    passed = passed and bool(reasoning.strip())
            results[name] = {'passed': passed, 'answer': answer, 'reasoning_chars': len(reasoning),
                             'tool_calls': calls, 'finish_reason': choice['finish_reason'],
                             'seconds': round(elapsed, 3)}
        except HTTPError as error:
            results[name] = {'passed': False, 'error': str(error),
                             'response': error.read(4096).decode(errors='replace')}
        except (OSError, ValueError, KeyError, IndexError) as error:
            results[name] = {'passed': False, 'error': str(error)}
    benchmark = {'passed': False}
    try:
        benchmark = streaming_benchmark(
            base, request(suite['benchmark'], max_tokens=256, chat_template_kwargs=template_options()),
        )
        benchmark['passed'] = benchmark['completion_tokens'] == 256
    except (OSError, ValueError, KeyError) as error:
        benchmark['error'] = str(error)
    return results, benchmark


def serving_metrics(path):
    log = path.read_text()
    patterns = {
        'stage_seconds': r'staged in ([\d.]+) seconds',
        'reported_model_load_gib': r'Model loading took ([\d.]+) GiB',
        'reported_kv_budget_gib': r'Available KV cache memory: ([\d.]+) GiB',
        'reported_kv_committed_after_warmup_gib': r'([\d.]+) GiB committed to the KV cache',
        'reported_kv_capacity_gib': r'KV cache blocks \(([\d.]+) GiB\) fit',
    }
    return {key: [float(value) for value in re.findall(pattern, log)]
            for key, pattern in patterns.items()}


def process_peak(path, workers):
    pids = {str(worker['pid']) for worker in workers}
    peaks = {}
    for line in path.read_text().splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) == 3 and fields[1] in pids and fields[2].isdigit():
            peaks[fields[0]] = max(peaks.get(fields[0], 0), int(fields[2]))
    return peaks


def main():
    model, mode, port, pid, tp, serving_log, gpu_samples = sys.argv[1:]
    base = f'http://127.0.0.1:{int(port)}'
    start = float(os.environ.get('PACE_SERVER_STARTED', time.monotonic()))
    deadline = start + (3000 if model == 'super-vl' else 1500)
    while True:
        os.kill(int(pid), 0)
        try:
            with api_open(base + '/health', timeout=2):
                break
        except (URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise TimeoutError('The service did not become ready before the validation deadline')
            time.sleep(2)
    ready = time.monotonic()
    from pace_inference import model_profile

    profile = model_profile(model)
    workers = checked_backends(Path(serving_log), int(tp), mode, model,
                               vision=profile['image'] and mode != 'text')
    with api_open(base + '/v1/models', timeout=5) as response:
        if model not in {entry['id'] for entry in json.load(response)['data']}:
            raise ValueError('The served model alias does not match the requested profile')
    if mode == 'suite':
        checks, benchmark = qualification_suite(base, model, profile['video'])
        passed = all(check['passed'] for check in checks.values()) and benchmark['passed']
        print(json.dumps({'passed': passed, 'model': model, 'mode': mode,
                          'ready_seconds': round(ready - start, 2),
                          'checks': checks, 'benchmark': benchmark, 'backends': workers,
                          'sampled_peak_gpu_mib_by_uuid': process_peak(Path(gpu_samples), workers),
                          **serving_metrics(Path(serving_log))}))
        sys.exit(0 if passed else 1)
    fixture_path = Path(__file__).resolve().parent / 'fixtures' / f'{mode}.json'
    fixture = json.loads(fixture_path.read_text())
    content = fixture['prompt']
    if mode == 'image':
        url = 'data:image/png;base64,' + base64.b64encode(red_png()).decode('ascii')
        content = [{'type': 'image_url', 'image_url': {'url': url}}, {'type': 'text', 'text': content}]
    payload = {'model': model, 'messages': [{'role': 'user', 'content': content}],
               'temperature': 0, 'max_tokens': 512, 'stream': False}
    response, elapsed = completion(base, payload)
    reply = response['choices'][0]
    answer = reply['message']['content'] or ''
    passed = re.search(fixture['expected'], answer, re.IGNORECASE) is not None
    result = {'passed': passed, 'model': model, 'mode': mode, 'answer': answer,
              'finish_reason': reply['finish_reason'], 'ready_seconds': round(ready - start, 2),
              'request_seconds': round(elapsed, 2), 'backends': workers}
    if passed and mode == 'text':
        payload['messages'][0]['content'] = 'List the integers from 1 to 200 separated by commas.'
        payload['max_tokens'] = 256
        measurement, bench_seconds = completion(base, payload)
        generated = measurement['usage']['completion_tokens']
        result['benchmark'] = {'seconds': round(bench_seconds, 3), 'completion_tokens': generated,
                               'end_to_end_tokens_per_second': round(generated / bench_seconds, 2)}
    result['sampled_peak_gpu_mib_by_uuid'] = process_peak(Path(gpu_samples), workers)
    print(json.dumps(result))
    if not passed:
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        print(json.dumps({'passed': False, 'model': sys.argv[1], 'mode': sys.argv[2],
                          'error': str(error), **serving_metrics(Path(sys.argv[6]))}))
        sys.exit(1)
