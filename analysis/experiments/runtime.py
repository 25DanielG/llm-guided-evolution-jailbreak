"""Experiment isolation, frozen strategy artifacts, and generation snapshots."""
from __future__ import annotations
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import time
import numpy as np
from analysis.experiments.coverage import digest, load_experiment, write_json


def freeze_population(destination, population, variant_dir, ancestry):
    destination = Path(destination)
    if (destination / 'population.json').exists():
        raise ValueError('Frozen population already exists')
    destination.mkdir(parents=True, exist_ok=True)
    members = []
    for ind in population:
        gene = ind[0]
        source = Path(variant_dir) / f'prompt_{gene}.py'
        if not source.is_file():
            raise ValueError(f'Missing initial strategy: {gene}')
        compile(source.read_text(), str(source), 'exec')
        shutil.copy2(source, destination / source.name)
        members.append({'id': gene, 'file': source.name, 'sha256': digest(source.read_bytes())})
    if len(members) != 24 or len({m['id'] for m in members}) != 24:
        raise ValueError('Expected 24 distinct initial strategy IDs')
    write_json(destination / 'population.json', {'members': members, 'ancestry': ancestry})


def load_initial(path, variant_dir):
    path = Path(path)
    data = json.loads((path / 'population.json').read_text())
    if len(data['members']) != 24:
        raise ValueError('Initial population must contain 24 strategies')
    Path(variant_dir).mkdir(parents=True, exist_ok=True)
    for m in data['members']:
        source = path / m['file']
        if digest(source.read_bytes()) != m['sha256']:
            raise ValueError(f'Frozen strategy changed: {source}')
        shutil.copy2(source, Path(variant_dir) / m['file'])
    return data


def environment_fingerprint(environment):
    # Service locations may change between jobs without changing the experiment.
    ignored = ('_URL', '_DIR', '_HOST', '_PORT')
    return {k: v for k, v in sorted(environment.items()) if k.startswith(('JB_', 'LLMGE_', 'LLM_'))
            and not k.endswith(ignored) and k not in {'LLMGE_RUN_ID', 'JB_GEN', 'JB_NUM_GENS',
                'JB_BEHAVIORS_PATH', 'JB_EXPERIMENT_ROOT', 'LLMGE_HOSTNAME_PATH'}}


def seed_runtime(seed):
    import sys
    random.seed(seed)
    np.random.seed(seed)
    torch = sys.modules.get('torch')
    if torch is not None:
        torch.manual_seed(seed)


def configuration(manifest, initial, subset, seed):
    experiment = load_experiment(manifest)
    selection = next(s for s in experiment['subsets'] if s['id'] == subset)
    expected_csv = (Path(manifest).resolve().parent / selection['csv']).resolve()
    if Path(os.environ.get('JB_BEHAVIORS_PATH', '')).resolve() != expected_csv:
        raise ValueError('Experiment evaluation must use the assigned subset CSV')
    for key, expected in {'JB_FULL_BEHAVIORS': '1', 'JB_STRATIFIED': '0', 'JB_CURRICULUM': '0'}.items():
        if os.environ.get(key) != expected:
            raise ValueError(f'Experiment requires {key}={expected}')
    initial = Path(initial).resolve()
    frozen = json.loads((initial / 'population.json').read_text())
    for member in frozen['members']:
        if digest((initial / member['file']).read_bytes()) != member['sha256']:
            raise ValueError('Frozen strategy checksum mismatch')
    root = Path(__file__).resolve().parents[2]
    config_files = ['run_improved.py', 'src/cfg/constants_jailbreak.py', 'sota/Jailbreak/eval.py',
                    'sota/Jailbreak/jb_config.py', 'server.py', 'slurm-config/slurm_config.yaml',
                    'analysis/experiments/runtime.py', 'analysis/experiments/runner.py',
                    'src/llm_utils.py', 'src/llm_mutation.py', 'sota/Jailbreak/vllm_client.py',
                    'sota/Jailbreak/judge.py', 'sota/Jailbreak/cache.py']
    hashes = {p: digest((root / p).read_bytes()) for p in config_files}
    for p in sorted((root / 'templates/Jailbreak').rglob('*.txt')):
        hashes[str(p.relative_to(root))] = digest(p.read_bytes())
    from sota.Jailbreak.prompt_records import checkpoint_fingerprint
    model_keys = ['JB_TARGET_MODEL_PATH', 'JB_JUDGE_MODEL_PATH', 'JB_REPORTING_MODEL_PATH', 'JB_MUTATOR_MODEL_PATH']
    models = {k: {'path': os.environ.get(k), 'fingerprint': checkpoint_fingerprint(os.environ.get(k))} for k in model_keys}
    env = environment_fingerprint(os.environ)
    return {'experiment_hash': digest(Path(manifest).read_bytes()), 'subset': selection,
            'initial_hash': digest((initial / 'population.json').read_bytes()), 'seed': seed,
            'source_hashes': hashes, 'models': models, 'environment': env}


def bind_run(root, config):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'run.json'
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError('Run inputs/configuration changed; create a new run')
    else:
        write_json(path, config)
    return digest(path.read_bytes())


def rng_state():
    import sys
    state = {'python': random.getstate(), 'numpy': np.random.get_state()}
    torch = sys.modules.get('torch')
    if torch is not None:
        state['torch'] = torch.get_rng_state()
        if torch.cuda.is_available():
            state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    if 'torch' in state:
        import torch
        torch.set_rng_state(state['torch'])
        if 'cuda' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda'])


def record_evaluation(root, generation, gene, split, records, source, decoding, complete=False):
    """One atomic observation bundle; a later failed attempt cannot reuse success."""
    from sota.Jailbreak.prompt_records import serialize_messages
    rows = []
    for rec in records:
        messages = rec.get('messages')
        text = serialize_messages(messages) if messages else None
        rows.append({'behavior_id': digest(rec['behavior'].strip()), 'behavior': rec['behavior'],
                     'messages': messages, 'prompt_text': text, 'prompt_hash': digest(text) if text else None,
                     'status': 'ok' if text else 'build_failed',
                     'score': rec.get('in_loop', {}).get('score'), 'tokens': rec.get('prompt_tokens')})
    payload = {'generation': generation, 'candidate_id': gene, 'split': split,
               'strategy_hash': digest(Path(source).read_bytes()), 'decoding': decoding,
               'model_paths': {k: os.environ.get(k) for k in ('JB_TARGET_MODEL_PATH', 'JB_JUDGE_MODEL_PATH')},
               'records': rows, 'evaluation_complete': complete}
    write_json(Path(root) / 'observations' / split / f'{generation:04d}' / f'{gene}.json', payload)


def snapshot(root, generation, population, variant_dir, ancestry, run_hash):
    root = Path(root)
    path = root / 'snapshots' / f'{generation:04d}.json'
    if path.exists():
        raise ValueError(f'Completed snapshot already exists: {path}')
    archive = root / 'strategies'
    archive.mkdir(exist_ok=True)
    members = []
    for ind in population:
        gene = ind[0]
        source = Path(variant_dir) / f'prompt_{gene}.py'
        source_hash = digest(source.read_bytes()) if source.exists() else None
        if source_hash:
            target = archive / f'{source_hash}.py'
            if not target.exists():
                shutil.copy2(source, target)
        candidates = sorted((root / 'observations/evolution').glob(f'*/{gene}.json'))
        candidates = [p for p in candidates if int(p.parent.name) <= generation]
        observation = None
        if candidates:
            latest = candidates[-1]
            data = json.loads(latest.read_text())
            if data['strategy_hash'] == source_hash:
                observation = str(latest.relative_to(root))
        values = [float(v) if np.isfinite(v) else None for v in ind.fitness.values]
        if any(v is None for v in values):
            observation = None
        members.append({'id': gene, 'strategy_hash': source_hash, 'observation': observation, 'fitness': values})
    write_json(path, {'generation': generation, 'run_hash': run_hash, 'members': members,
                      'ancestry': ancestry})
    return path


def save_complete(root, generation, checkpoint, global_file):
    """Commit marker written last; incomplete checkpoints are ignored on resume."""
    root = Path(root)
    snap = root / 'snapshots' / f'{generation:04d}.json'
    paths = [Path(checkpoint), Path(global_file), snap]
    write_json(root / 'completed' / f'{generation:04d}.json',
               {'generation': generation, 'files': {str(p.resolve()): digest(p.read_bytes()) for p in paths}})


def latest_complete(root):
    markers = sorted((Path(root) / 'completed').glob('*.json'))
    if not markers:
        return None
    marker = json.loads(markers[-1].read_text())
    for name, checksum in marker['files'].items():
        if digest(Path(name).read_bytes()) != checksum:
            raise ValueError(f'Completed generation artifact changed: {name}')
    return marker['generation']
