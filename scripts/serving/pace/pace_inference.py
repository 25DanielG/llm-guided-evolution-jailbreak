"""Download, inspect and stage revision-pinned checkpoints for ICE jobs."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time

from backend_policy import STABLE_PROFILES, backend_flags, guarded_environment, profile_flags


SOURCE = Path(__file__).resolve().parent
CHECKOUT = SOURCE.parents[2]
GIB = 1024**3


def read_json(path):
    return json.loads(path.read_text())


def model_profile(alias):
    registry = read_json(SOURCE / 'config/models.json')
    if alias not in registry or not re.fullmatch(r'[a-z][a-z0-9-]*', alias):
        raise ValueError(f'Unknown model {alias!r}')
    profile = registry[alias]
    if (not re.fullmatch(r'[a-f0-9]{40}', profile['revision'])
            or not re.fullmatch(r'[\w.-]+/[\w.-]+', profile['repo'])
            or profile['runtime'] != ('inkling' if alias == 'inkling' else 'stable')
            or (alias != 'inkling' and alias not in STABLE_PROFILES)):
        raise ValueError(f'Invalid registry entry for {alias}')
    profile_flags(profile['args'])
    profile['files'] = read_json(SOURCE / 'config' / 'snapshots' / f'{alias}.json')
    if 'config.json' not in profile['files'] or not any(
            name.endswith('.safetensors') for name in profile['files']):
        raise ValueError(f'Incomplete manifest for {alias}')
    for name, entry in profile['files'].items():
        path = Path(name)
        if (path.is_absolute() or path.as_posix() != name or '..' in path.parts
                or not path.parts or any(part.startswith('.') or not re.fullmatch(r'[\w.-]+', part)
                                         for part in path.parts)
                or (path.suffix in {'.py', '.so', '.pyc', '.pkl', '.bin'})
                or type(entry.get('bytes')) is not int or entry['bytes'] < 0):
            raise ValueError(f'Unsafe manifest entry: {name}')
        digests = entry.keys() - {'bytes'}
        if not ((digests == {'sha256'} and re.fullmatch(r'[0-9a-f]{64}', entry['sha256']))
                or (digests == {'git_blob'} and re.fullmatch(r'[0-9a-f]{40}', entry['git_blob']))):
            raise ValueError(f'Invalid manifest digest for {name}')
    return profile


def cache_root():
    configured = os.environ.get('PACE_INFERENCE_STATE')
    if not configured:
        raise ValueError('Set PACE_INFERENCE_STATE to a private directory outside this checkout')
    path = Path(configured).expanduser().resolve()
    if path.is_relative_to(CHECKOUT) or CHECKOUT.is_relative_to(path):
        raise ValueError('The state directory must be separate from the checkout')
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
        raise ValueError(f'State root must be owned by you with permissions 0700: {path}')
    return path


def child(path, name):
    result = path / name
    if result.is_symlink() or (result.exists() and not result.is_dir()):
        raise ValueError(f'Expected an ordinary directory at {result}')
    return result


@contextlib.contextmanager
def cache_lock(root, exclusive=True):
    fd = os.open(root / '.cache.lock', os.O_CREAT | os.O_NOFOLLOW | os.O_RDWR, 0o600)
    with os.fdopen(fd, 'w') as lock:
        flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(lock, flags)
        yield


def disk_headroom(path):
    fs, mount = subprocess.check_output(
        ['findmnt', '-n', '-T', str(path), '-o', 'FSTYPE,TARGET'], text=True,
    ).split(maxsplit=1)
    available = shutil.disk_usage(path).free
    inodes = os.statvfs(path).f_favail
    if fs == 'lustre':
        quota = subprocess.check_output(
            ['lfs', 'quota', '-u', str(os.getuid()), mount.strip()], text=True, timeout=20,
        )
        row = next((line.split() for line in quota.splitlines()
                    if line.split() and line.split()[0] == mount.strip()), None)
        if row is None or len(row) < 8:
            raise ValueError(f'Cannot read user quota for {mount.strip()}')
        used, soft, hard = (int(value.rstrip('*')) for value in row[1:4])
        files, isoft, ihard = (int(value.rstrip('*')) for value in row[5:8])
        limits = [limit for limit in (soft, hard) if limit]
        ilimits = [limit for limit in (isoft, ihard) if limit]
        if limits:
            available = min(available, (min(limits) - used) * 1024)
        if ilimits:
            inodes = min(inodes, min(ilimits) - files)
    return fs, max(0, available), max(0, inodes)


def size_of(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.exists() else 0


def verify(snapshot, files, digest=False):
    for name, expected in files.items():
        path = snapshot / name
        if path.resolve() != path or not path.is_file() or path.stat().st_size != expected['bytes']:
            raise ValueError(f'Missing, linked or incorrectly sized checkpoint file: {path}')
        if digest:
            with path.open('rb') as handle:
                if expected.get('sha256'):
                    actual = hashlib.file_digest(handle, 'sha256').hexdigest()
                    wanted = expected['sha256']
                else:
                    checksum = hashlib.sha1(f'blob {expected["bytes"]}\0'.encode(), usedforsecurity=False)
                    actual = hashlib.file_digest(handle, lambda: checksum).hexdigest()
                    wanted = expected['git_blob']
            if actual != wanted:
                raise ValueError(f'Checksum mismatch: {path}')


def assert_managed(snapshot, profile):
    marker = snapshot / '.pace-snapshot.json'
    expected = {key: profile[key] for key in ('repo', 'revision')}
    if marker.is_symlink() or not marker.is_file() or read_json(marker) != expected:
        raise ValueError(f'Unrecognized snapshot directory: {snapshot}')


def fetch(root, final, partial, profile):
    if final.exists():
        assert_managed(final, profile)
        verify(final, profile['files'])
        return
    files = profile['files']
    fs, available, inodes = disk_headroom(root)
    complete_bytes = sum(info['bytes'] for name, info in files.items()
                         if (partial / name).is_file() and
                         (partial / name).stat().st_size == info['bytes'])
    required = sum(info['bytes'] for info in files.values()) - complete_bytes + 4 * GIB
    if available < required or inodes < 512:
        raise ValueError(f'{fs} has {available / GIB:.2f} GiB and {inodes} inodes free; '
                         f'fetch needs {required / GIB:.2f} GiB including headroom')
    partial.mkdir(mode=0o700, exist_ok=True)
    marker = partial / '.pace-snapshot.json'
    if not marker.exists() and not any(partial.iterdir()):
        marker.write_text(json.dumps({key: profile[key] for key in ('repo', 'revision')}))
    assert_managed(partial, profile)

    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HOME'] = str(child(root, 'hf'))
    os.environ['HF_HUB_CACHE'] = str(root / 'hf/hub')
    os.environ['HF_XET_CACHE'] = str(root / 'hf/xet')
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ValueError('Run fetch with the Python interpreter from the model runtime') from error

    snapshot_download(repo_id=profile['repo'], revision=profile['revision'], token=False,
                      local_dir=partial, allow_patterns=list(files), max_workers=2)
    verify(partial, files, digest=True)
    partial.rename(final)


def check_local_disk(job_dir, profile):
    fs, available, inodes = disk_headroom(job_dir)
    required = sum(info['bytes'] for info in profile['files'].values()) + 8 * GIB
    if fs not in {'xfs', 'ext4', 'btrfs'} or available < required or inodes < 512:
        raise ValueError(f'Need a job-local disk with {required / GIB:.2f} GiB available: {job_dir}')


def stage_snapshot(root, snapshot, profile, job_dir):
    check_local_disk(job_dir, profile)
    with cache_lock(root, exclusive=False):
        assert_managed(snapshot, profile)
        verify(snapshot, profile['files'])
        local = job_dir / 'model'
        for name in profile['files']:
            target = local / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot / name, target)
        verify(local, profile['files'])
    return local


def reviewed_runtime(root, profile):
    lane = profile['runtime']
    if lane not in {'inkling', 'stable'}:
        raise ValueError(f'No pinned backend policy for runtime {lane}')
    runtime = child(child(root, 'runtimes'), lane)
    if not (runtime / '.pace-lock-sha256').is_file():
        raise ValueError('Install the reviewed runtime before serving')
    expected_lock = hashlib.sha256((SOURCE / 'locks' / f'{lane}.lock').read_bytes()).hexdigest()
    if (runtime / '.pace-lock-sha256').read_text().strip() != expected_lock:
        raise ValueError('Runtime does not match the current pinned dependency lock')
    return runtime


def compiler_toolkit(model):
    if model in STABLE_PROFILES and STABLE_PROFILES[model][2]:
        toolkit = os.environ.get('PACE_CUDA_TOOLKIT')
        if not toolkit:
            raise ValueError('Load the PACE cuda/13.0.1 module and export PACE_CUDA_TOOLKIT=$CUDA_HOME')
        path = Path(toolkit).resolve()
        nvcc = subprocess.check_output([str(path / 'bin/nvcc'), '--version'], text=True, timeout=8)
        if 'V13.0.88' not in nvcc:
            raise ValueError(f'Mamba SSU requires the PACE CUDA 13.0.1 toolkit: {nvcc}')
        return path
    return None


def preflight(root, profile, model):
    runtime = reviewed_runtime(root, profile)
    command = [str(runtime / 'bin/python'), '-B', str(SOURCE / 'gpu_preflight.py')]
    if not os.environ.get('SLURM_JOB_ID'):
        command.append('--cpu-only')
    cache = Path(tempfile.mkdtemp(prefix='pace-preflight-', dir=os.environ.get('TMPDIR', '/tmp')))
    env = guarded_environment(SOURCE, cache, runtime=runtime, cuda_toolkit=compiler_toolkit(model))
    os.execve(command[0], command, env)


def serve(root, snapshot, profile, args):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Run serving in a Slurm GPU allocation')
    if args.mode == 'image' and not profile.get('image'):
        raise ValueError(f'{args.model} is text-only')
    if args.mode == 'suite' and profile['runtime'] != 'stable':
        raise ValueError('Use separate text and image tests for Inkling')
    if args.inkling_execution and args.model != 'inkling':
        raise ValueError('The execution comparison is specific to Inkling')
    key_file = Path(os.environ['PACE_SERVER_KEY_FILE'])
    key_stat = key_file.lstat()
    if not stat.S_ISREG(key_stat.st_mode) or key_stat.st_uid != os.getuid() or key_stat.st_mode & 0o077:
        raise ValueError('The server key file must be a regular file owned by you with permissions 0600')
    token = key_file.read_text().strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]{43}', token):
        raise ValueError('Invalid managed server key')
    runtime = reviewed_runtime(root, profile)
    toolkit = compiler_toolkit(args.model)
    model_flags = profile_flags(profile['args'])
    job_dir = Path(tempfile.mkdtemp(prefix='pace-inference-', dir=os.environ.get('TMPDIR', '/tmp')))
    stage_started = time.monotonic()
    if args.node_fetch:
        check_local_disk(job_dir, profile)
        download_env = os.environ.copy()
        download_env.update(PACE_INFERENCE_STATE=str(job_dir / 'download'),
                            HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0',
                            HF_HUB_ETAG_TIMEOUT='15', HF_HUB_DOWNLOAD_TIMEOUT='90',
                            PYTHONNOUSERSITE='1')
        subprocess.run([str(runtime / 'bin/python'), '-I', '-B',
                        '-c', 'import socket; socket.getaddrinfo("huggingface.co", 443)',
                        ], env=download_env, check=True, timeout=35)
        subprocess.run([str(runtime / 'bin/python'), '-E', '-B', str(SOURCE / 'pace_inference.py'),
                        'fetch', args.model], env=download_env, check=True, timeout=1800)
        local = job_dir / 'download' / 'snapshots' / args.model / profile['revision']
    else:
        local = stage_snapshot(root, snapshot, profile, job_dir)
    flags = ['--language-model-only'] if args.mode == 'text' and profile.get('image') else []
    if args.mode == 'image':
        limits = {'image': 1}
        if profile.get('video'):
            limits['video'] = 0
        flags += ['--limit-mm-per-prompt', json.dumps(limits)]
    if args.mode == 'suite' and profile.get('video'):
        flags += ['--limit-mm-per-prompt', '{"image":1,"video":1}']
    if profile.get('image'):
        flags += ['--mm-encoder-attn-backend', 'FLASH_ATTN']
    env = guarded_environment(SOURCE, job_dir / 'cache', runtime=runtime, cuda_toolkit=toolkit)
    env.pop('PACE_INKLING_EXECUTION', None)
    execution_flags = ['--max-num-seqs', '1', '--enforce-eager']
    if args.inkling_execution:
        execution_flags = ['--max-num-seqs', str(args.inkling_batch), '--no-enable-prefix-caching']
        env['PACE_INKLING_EXECUTION'] = args.inkling_execution
        # Humming has its own defaults outside XDG_CACHE_HOME. Separate
        # concurrent node experiments and use the wheel's matching NVRTC.
        env['HUMMING_CACHE_DIR'] = str(job_dir / 'cache/humming')
        env['HUMMING_TMP_DIR'] = str(job_dir / 'cache/humming-tmp')
        env['LD_LIBRARY_PATH'] = str(Path(env['CUDA_HOME']) / 'lib')
        if args.inkling_execution.endswith('-nonbreakable'):
            env['VLLM_USE_BREAKABLE_CUDAGRAPH'] = '0'
        if args.inkling_execution == 'eager':
            execution_flags += ['--enforce-eager']
        else:
            compilation = {'mode': 3 if args.inkling_execution.startswith('compile') else 0,
                           'cudagraph_mode': 'FULL_DECODE_ONLY',
                           'decode_cudagraph_capture_request_sizes': (
                               [1] if args.inkling_batch == 1 else [1, 2, 4])}
            execution_flags += ['--compilation-config', json.dumps(compilation)]
    # vLLM reads its key from the environment so the key stays out of
    # generated command lines and non-default argument logs.
    env['VLLM_API_KEY'] = token
    if profile['runtime'] == 'stable':
        env['PACE_STABLE_PROFILE'] = args.model
        # FlashInfer SSU can JIT-link the pinned CUDA wheel's versioned
        # libcudart. Give the linker a job-local name without changing a wheel.
        link_dir = job_dir / 'cuda-link'
        link_dir.mkdir()
        cudart = runtime / 'lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13'
        if not cudart.is_file():
            raise ValueError(f'Missing pinned CUDA runtime library at {cudart}')
        (link_dir / 'libcudart.so').symlink_to(cudart)
        env['LIBRARY_PATH'] = str(link_dir)
        # Humming's standalone NVRTC helper resolves libnvrtc-builtins
        # outside PyTorch's library search path. Use the pinned companion.
        if not (cudart.parent / 'libnvrtc-builtins.so.13.0').is_file():
            raise ValueError('Missing the locked NVRTC builtins library')
        env['LD_LIBRARY_PATH'] = str(cudart.parent)
        env['HUMMING_CACHE_DIR'] = str(job_dir / 'cache/humming')
    command = [str(runtime / 'bin' / 'vllm'), 'serve', str(local), '--host', '127.0.0.1',
               '--port', str(args.port), '--served-model-name', args.model,
               '--max-model-len', '4096', *execution_flags,
               '--tensor-parallel-size', str(args.tp), *flags, *model_flags, *backend_flags(args.model)]
    print(f'Local snapshot {local} staged in {time.monotonic() - stage_started:.2f} seconds'
          f' ({"download and checksum" if args.node_fetch else "shared copy"})', flush=True)
    os.execve(command[0], command, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'fetch', 'verify', 'evict', 'preflight', 'serve'])
    parser.add_argument('model', choices=sorted(read_json(SOURCE / 'config' / 'models.json')))
    parser.add_argument('--confirm', action='store_true', help='Delete only this managed snapshot')
    parser.add_argument('--partial', action='store_true', help='Target the partial snapshot for eviction')
    parser.add_argument('--mode', choices=['text', 'image', 'suite'], default='text')
    parser.add_argument('--node-fetch', action='store_true',
                        help='Fetch a public pinned snapshot directly to allocated node-local disk')
    parser.add_argument('--port', type=int, choices=range(65536), metavar='0-65535', default=0)
    parser.add_argument('--tp', type=int, choices=[1, 2], default=1)
    parser.add_argument('--inkling-execution',
                        choices=['eager', 'graph', 'compile', 'graph-nonbreakable', 'compile-nonbreakable'])
    parser.add_argument('--inkling-batch', type=int, choices=[1, 4], default=1)
    args = parser.parse_args()
    profile = model_profile(args.model)
    root = cache_root()
    if args.action == 'preflight':
        preflight(root, profile, args.model)
        return
    snapshots = child(root, 'snapshots')
    snapshots.mkdir(mode=0o700, exist_ok=True)
    folder = child(snapshots, args.model)
    folder.mkdir(mode=0o700, exist_ok=True)
    final = child(folder, profile['revision'])
    partial = child(folder, f'.{profile["revision"]}.partial')
    if args.action == 'serve':
        serve(root, final, profile, args)
        return
    with cache_lock(root, exclusive=args.action in {'fetch', 'evict'}):
        if args.action == 'status':
            fs, available, inodes = disk_headroom(root)
            print(json.dumps({'repo': profile['repo'], 'revision': profile['revision'],
                              'required_payload_bytes': sum(item['bytes'] for item in profile['files'].values()),
                              'filesystem': fs, 'available_bytes': available, 'available_inodes': inodes,
                              'cached_bytes': size_of(final), 'partial_bytes': size_of(partial)}, indent=2))
        elif args.action == 'fetch':
            fetch(root, final, partial, profile)
            print(final)
        elif args.action == 'verify':
            assert_managed(final, profile)
            verify(final, profile['files'], digest=True)
            print(final)
        elif args.action == 'evict':
            path = partial if args.partial else final
            assert_managed(path, profile)
            print(f'{path} ({size_of(path)} bytes)')
            if args.confirm:
                shutil.rmtree(path)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        sys.exit(str(error))
