"""Reproducible behavior-space sampling. No model-service or evolution imports."""
from __future__ import annotations
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import numpy as np


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def read_behaviors(raw):
    rows = csv.DictReader(io.StringIO(raw))
    column = next((c for c in ('goal', 'behavior') if c in (rows.fieldnames or [])), None)
    if column is None:
        raise ValueError('CSV requires goal or behavior column')
    texts = {(r[column] or '').strip() for r in rows} - {''}
    return sorted(({'id': digest(t), 'behavior': t, 'source': 'advbench'} for t in texts), key=lambda r: r['id'])


def distances(vectors):
    v = np.asarray(vectors, dtype=float)
    if v.ndim != 2 or not np.isfinite(v).all() or np.any(np.linalg.norm(v, axis=1) == 0):
        raise ValueError('Expected finite nonzero embedding rows')
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    d = np.clip(1 - v @ v.T, 0, 2)
    np.fill_diagonal(d, 0)
    return d


def coverage_distance(d, selected):
    return float(d[:, selected].min(axis=1).mean())


def greedy_cover(d, count, first=None):
    if not 0 < count <= len(d):
        raise ValueError('Invalid subset size')
    selected = [int(np.argmin(d.mean(axis=0))) if first is None else int(first)]
    nearest = d[:, selected[0]].copy()
    while len(selected) < count:
        costs = np.minimum(nearest[:, None], d).mean(axis=0)
        costs[selected] = np.inf
        best = int(np.argmin(costs))
        selected.append(best)
        nearest = np.minimum(nearest, d[:, best])
    return selected


def design(vectors, seed=42, size=30, holdout_size=30):
    d = distances(vectors)
    if len(d) < holdout_size + size + 3:
        raise ValueError('Insufficient distinct behaviors for the experiment')
    heldout = greedy_cover(d, holdout_size)
    pool = np.array(sorted(set(range(len(d))) - set(heldout)))
    pd = d[np.ix_(pool, pool)]
    centers = [int(np.argmin(pd.mean(axis=0)))]
    while len(centers) < 3:
        nearest = pd[:, centers].min(axis=1)
        nearest[centers] = -np.inf
        centers.append(int(np.argmax(nearest)))
    children = np.random.SeedSequence(seed).spawn(7)
    subsets, memberships = [], set()

    def add(kind, indices, child_seed=None):
        key = tuple(sorted(map(int, indices)))
        if key in memberships:
            raise ValueError('Duplicate subset membership; choose another master seed')
        memberships.add(key)
        subsets.append({'id': f'{kind}_{sum(s["kind"] == kind for s in subsets) + 1:02d}',
                        'kind': kind, 'seed': child_seed,
                        'indices': pool[list(key)].tolist(),
                        'coverage_distance': coverage_distance(pd, list(key))})

    for child in children[:4]:
        child_seed = int(child.generate_state(1)[0])
        add('random', np.random.default_rng(child_seed).choice(len(pool), size, replace=False), child_seed)
    for center in centers:
        neighbors = [center] + [int(i) for i in np.argsort(pd[center], kind='stable') if i != center][:size-1]
        add('narrow', neighbors)
    for child in children[4:]:
        child_seed = int(child.generate_state(1)[0])
        rng = np.random.default_rng(child_seed)
        for first in rng.permutation(len(pool)):
            selected = greedy_cover(pd, size, int(first))
            if tuple(sorted(selected)) not in memberships:
                add('broad', selected, child_seed)
                break
        else:
            raise ValueError('Could not construct three distinct broad subsets')
    return heldout, pool.tolist(), subsets


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'source', 'behavior'])
        w.writeheader()
        w.writerows(rows)


def prepare(output, embedder, source_csv=None, source_revision=None, seed=42):
    output = Path(output).resolve()
    if (output / 'experiment.json').exists():
        raise ValueError('Experiment already exists; use another output directory')
    if source_csv:
        if not source_revision:
            raise ValueError('Local CSV requires --source-revision for provenance')
        raw = Path(source_csv).read_bytes()
    else:
        request = urllib.request.Request('https://api.github.com/repos/llm-attacks/llm-attacks/commits/main',
                                         headers={'User-Agent': 'behavior-coverage-experiment'})
        if not source_revision:
            with urllib.request.urlopen(request, timeout=30) as response:
                source_revision = json.load(response)['sha']
        if len(source_revision) != 40 or any(c not in '0123456789abcdef' for c in source_revision.lower()):
            raise ValueError('Use an immutable 40-character upstream commit')
        url = f'https://raw.githubusercontent.com/llm-attacks/llm-attacks/{source_revision}/data/advbench/harmful_behaviors.csv'
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
    behaviors = read_behaviors(raw.decode('utf-8-sig'))
    vectors = np.stack([embedder.embed(r['behavior'])[0] for r in behaviors])
    heldout, pool, subsets = design(vectors, seed)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'source.csv').write_bytes(raw)
    write_csv(output / 'pool.csv', behaviors)
    np.save(output / 'behavior_embeddings.npy', vectors)
    write_csv(output / 'heldout.csv', [behaviors[i] for i in heldout])
    for s in subsets:
        s['behavior_ids'] = [behaviors[i]['id'] for i in s.pop('indices')]
        s['csv'] = s['id'] + '.csv'
        lookup = {r['id']: r for r in behaviors}
        write_csv(output / s['csv'], [lookup[i] for i in s['behavior_ids']])
    files = ['source.csv', 'pool.csv', 'heldout.csv', 'behavior_embeddings.npy'] + [s['csv'] for s in subsets]
    manifest = {'version': 1, 'seed': seed, 'source_revision': source_revision,
                'embedding': embedder.metadata, 'heldout_ids': [behaviors[i]['id'] for i in heldout],
                'pool_ids': [behaviors[i]['id'] for i in pool], 'subsets': subsets,
                'files': {f: digest((output / f).read_bytes()) for f in files},
                'overlap': [[len(set(a['behavior_ids']) & set(b['behavior_ids'])) for b in subsets] for a in subsets]}
    write_json(output / 'experiment.json', manifest)
    return manifest


def load_experiment(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    for name, checksum in manifest['files'].items():
        if digest((path.parent / name).read_bytes()) != checksum:
            raise ValueError(f'Experiment input changed: {name}')
    return manifest
