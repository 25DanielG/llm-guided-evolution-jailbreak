"""Own the guard-checked vLLM server and helper processes for one Slurm job."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError

from validate import api_open, checked_backends
from pace_inference import model_profile, SOURCE


def listener_port(pid, requested=0, proc=Path('/proc')):
    """Resolve the kernel-assigned loopback listener owned by this API process."""
    process = proc / str(pid)
    sockets = set()
    for fd in (process / 'fd').iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if match := re.fullmatch(r'socket:\[(\d+)\]', target):
            sockets.add(match[1])
    ports = set()
    for line in (process / 'net/tcp').read_text().splitlines()[1:]:
        fields = line.split()
        address, port = fields[1].split(':')
        if address == '0100007F' and fields[3] == '0A' and fields[9] in sockets:
            ports.add(int(port, 16))
    if requested:
        return requested if requested in ports else None
    if len(ports) > 1:
        raise ValueError('vLLM API process has multiple loopback TCP listeners. Specify --port.')
    return next(iter(ports), None)


@contextmanager
def child_process(command, output, env=None):
    with output.open('w') as stream:
        process = subprocess.Popen(command, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield process
        finally:
            for signum, timeout in ((signal.SIGTERM, 30), (signal.SIGKILL, 5)):
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    continue
                break


def await_endpoint(process, alias, requested, deadline):
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ValueError('Guarded vLLM server exited before readiness. Inspect server.log.')
        try:
            port = listener_port(process.pid, requested)
            if port is None:
                time.sleep(1)
                continue
            base = f'http://127.0.0.1:{port}'
            try:
                with api_open(base + '/v1/models', timeout=2, authenticated=False):
                    raise ValueError('The loopback endpoint did not require a bearer key')
            except HTTPError as error:
                if error.code != 401:
                    raise
            with api_open(base + '/v1/models', timeout=2) as response:
                if alias not in {entry['id'] for entry in json.load(response)['data']}:
                    raise ValueError('The API model alias did not match the deployment')
            return base, port
        except (URLError, FileNotFoundError, TimeoutError):
            time.sleep(1)
    raise TimeoutError('Timed out waiting for the guarded vLLM endpoint')


def run(args):
    result_path = args.output / 'result.json'
    server_log = args.output / 'server.log'
    key_file = args.output / 'api-key'
    with key_file.open('x') as handle:
        handle.write(secrets.token_urlsafe(32) + '\n')
    os.environ['PACE_SERVER_KEY_FILE'] = str(key_file)
    server_command = [sys.executable, '-B', str(SOURCE / 'pace_inference.py'), 'serve',
                      args.model, '--mode', args.mode, '--tp', str(args.tp), '--port', str(args.port)]
    if args.node_fetch:
        server_command.append('--node-fetch')
    gpu_output = args.output / 'gpu-process-samples.csv'
    sampler = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
               '--format=csv,noheader,nounits', '--loop=1']
    try:
        started = time.monotonic()
        with child_process(sampler, gpu_output), child_process(server_command, server_log) as server:
            base, port = await_endpoint(server, args.model, args.port,
                                        started + (3000 if args.model == 'super-vl' else 1500))
            profile = model_profile(args.model)
            checked_backends(server_log, args.tp, args.mode, args.model,
                             vision=profile['image'] and args.mode != 'text')
            endpoint = args.output / '.endpoint.json.tmp'
            endpoint.write_text(json.dumps({
                'url': base + '/v1', 'node': os.uname().nodename,
                'model': args.model, 'job_id': os.environ['SLURM_JOB_ID'],
            }) + '\n')
            endpoint.replace(args.output / 'endpoint.json')
            print(f'Endpoint ready at {base}/v1. Metadata: {args.output / "endpoint.json"}', flush=True)
            if args.operation == 'serve':
                return server.wait()
            command = [sys.executable, '-B', str(SOURCE / 'validate.py'), args.model, args.mode,
                       str(port), str(server.pid), str(args.tp), str(server_log), str(gpu_output)]
            env = dict(os.environ, PACE_SERVER_STARTED=str(started))
            with child_process(command, result_path, env=env) as test:
                status = test.wait()
            print(result_path.read_text(), end='')
            return status
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        if args.operation == 'validate':
            result_path.write_text(json.dumps({'passed': False, 'model': args.model, 'error': str(error)}) + '\n')
        print(str(error), file=sys.stderr)
        return 1
    finally:
        key_file.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('mode', choices=['text', 'image', 'suite'])
    parser.add_argument('tp', type=int)
    parser.add_argument('operation', choices=['validate', 'serve'])
    parser.add_argument('port', type=int)
    parser.add_argument('output', type=Path)
    parser.add_argument('--node-fetch', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    sys.exit(run(args))


def stop(signum, frame):
    raise SystemExit(128 + signum)


if __name__ == '__main__':
    main()
