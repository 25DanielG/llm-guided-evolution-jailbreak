"""Stage one pinned official checkpoint; promote only after complete hash verification."""
import argparse
import fcntl
import hashlib
from pathlib import Path, PurePosixPath
import shutil
from src.model_serving import profile
from src.run_status import atomic_json, utc


def selected_files(files, native):
    return [f for f in files if not f.rfilename.startswith('original/') and
            not (native and (f.rfilename.startswith('model-') and f.rfilename.endswith('.safetensors')
                             or f.rfilename in ('model.safetensors','model.safetensors.index.json')))]


def verify_files(folder, files):
    for item in files:
        relative = PurePosixPath(item.rfilename)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe checkpoint metadata path')
        path = Path(folder) / item.rfilename
        if not path.is_file() or path.stat().st_size != item.size:
            raise ValueError(f'Missing/truncated checkpoint file: {item.rfilename}')
        lfs = item.lfs
        expected = lfs.sha256 if lfs else item.blob_id
        h = hashlib.sha256() if lfs else hashlib.sha1()
        if not lfs:
            h.update(f'blob {item.size}\0'.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        if not expected or h.hexdigest() != expected:
            raise ValueError(f'Checkpoint hash mismatch: {item.rfilename}')


def stage(name, root, quota_free_gib):
    from huggingface_hub import HfApi, snapshot_download
    item = profile(name)
    if not item.get('revision'):
        raise ValueError('This staging command requires an immutable registry revision')
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.staging.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = root / item['model_id']
        if destination.exists():
            raise ValueError('Destination exists; do not overwrite a team checkpoint')
        info = HfApi().model_info(item['model_id'], revision=item['revision'], files_metadata=True)
        if info.sha != item['revision']:
            raise ValueError('Resolved revision differs from the registry')
        files = selected_files(info.siblings, 'mistral' in item['vllm_args'])
        if not files or any(f.size is None for f in files):
            raise ValueError('Incomplete repository size metadata; cannot budget safely')
        required = int(sum(f.size for f in files) * 1.15)
        if min(shutil.disk_usage(root).free, quota_free_gib * 1024**3) < required:
            raise ValueError(f'Insufficient verified quota/disk space; need at least {required} bytes')
        partial = root / f'.staging-{name}-{item["revision"]}'
        snapshot_download(item['model_id'], revision=item['revision'], local_dir=str(partial),
                          allow_patterns=[f.rfilename for f in files], max_workers=4)
        verify_files(partial, files)
        native = 'mistral' in item['vllm_args']
        required_metadata = ('params.json', 'tekken.json') if native else ('config.json', 'tokenizer_config.json')
        if not all((partial / p).is_file() for p in required_metadata):
            raise ValueError('Required native/config/tokenizer metadata missing')
        atomic_json(partial / 'deployment.json', {'model_id':item['model_id'], 'revision':item['revision'],
                    'verified_at':utc(), 'verified_files':len(files), 'verified_bytes':sum(f.size for f in files)})
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError('Destination appeared during staging; refusing promotion')
        partial.rename(destination)
        return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--quota-free-gib', type=float, required=True,
                        help='Free quota verified immediately beforehand; df alone is insufficient')
    args = parser.parse_args()
    print(stage(args.model, args.root, args.quota_free_gib))
