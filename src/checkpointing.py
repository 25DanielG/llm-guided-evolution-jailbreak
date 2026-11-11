"""Atomic matched checkpoint commits. Only load trusted project pickle files."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import tempfile
from src.run_status import atomic_json, progress

PATTERN = re.compile(r'^checkpoint_gen_(\d+)\.pkl$')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_pickle(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def commit(gen, folder, global_folder, population_data, global_data, identity):
    folder, global_folder = Path(folder), Path(global_folder)
    checkpoint = folder / f'checkpoint_gen_{gen}.pkl'
    global_file = global_folder / f'global_gen_{gen}.pkl'
    if checkpoint.with_suffix('.json').exists():
        raise ValueError(f'Checkpoint generation {gen} already committed; refusing overwrite')
    atomic_pickle(global_file, global_data)
    atomic_pickle(checkpoint, population_data)
    marker = {'schema_version': 1, 'generation': gen, 'identity': identity,
              'checkpoint_sha256': digest(checkpoint), 'global_sha256': digest(global_file)}
    atomic_json(checkpoint.with_suffix('.json'), marker)
    progress(gen, checkpoint)
    return checkpoint


def inspect(folder, global_folder, checkpoint_file=None):
    folder, global_folder = Path(folder), Path(global_folder)
    if checkpoint_file is None:
        markers = [p for p in folder.glob('checkpoint_gen_*.json')
                   if PATTERN.fullmatch(p.with_suffix('.pkl').name)]
        if not markers:
            raise ValueError('No committed checkpoint; refusing to restart from seed')
        marker = max(markers, key=lambda p: int(PATTERN.fullmatch(p.with_suffix('.pkl').name)[1]))
    else:
        if Path(checkpoint_file).name != checkpoint_file or not PATTERN.fullmatch(checkpoint_file):
            raise ValueError('Expected checkpoint_gen_N.pkl basename')
        marker = folder / Path(checkpoint_file).with_suffix('.json')
    data = json.loads(marker.read_text())
    generation = int(PATTERN.fullmatch(marker.with_suffix('.pkl').name)[1])
    if data.get('schema_version') != 1 or data['generation'] != generation:
        raise ValueError('Invalid checkpoint commit metadata')
    checkpoint = marker.with_suffix('.pkl')
    global_file = global_folder / f'global_gen_{generation}.pkl'
    if digest(checkpoint) != data['checkpoint_sha256'] or digest(global_file) != data['global_sha256']:
        raise ValueError('Checkpoint/global checksum mismatch; select a verified older commit explicitly')
    return checkpoint, global_file, data


def restore(folder, global_folder, identity, checkpoint_file=None):
    checkpoint, global_file, data = inspect(folder, global_folder, checkpoint_file)
    if data['identity'] != identity:
        raise ValueError('Checkpoint run/model/config identity differs; refusing unsafe resume')
    with checkpoint.open('rb') as stream:
        population = pickle.load(stream)
    with global_file.open('rb') as stream:
        globals_data = pickle.load(stream)
    return population, data['generation'] + 1, globals_data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('--global-path', required=True)
    parser.add_argument('--checkpoint-file')
    args = parser.parse_args()
    _, _, metadata = inspect(args.checkpoints, args.global_path, args.checkpoint_file)
    print(json.dumps(metadata, indent=2))
