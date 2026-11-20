"""Freeze an exact, reproducible sample using the repository's sampler."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sota/Jailbreak'))
import eval as evaluator


def freeze(source, output, n=50, seed=0):
    source, output = Path(source).resolve(), Path(output).resolve()
    rows = list(csv.DictReader(source.open(newline='')))
    # This evaluator supplies only behavior text, not contextual payloads.
    if rows and 'FunctionalCategory' in rows[0]:
        rows = [dict(id=r['BehaviorID'], source='HarmBench-standard', behavior=r['Behavior'])
                for r in rows if r['FunctionalCategory'] == 'standard']
    for i, row in enumerate(rows):
        row.setdefault('id', f'row_{i+1}')
        row['behavior'] = next((row[k].strip() for k in ('behavior', 'goal', 'prompt', 'text')
                                if row.get(k, '').strip()), '')
    rows = [r for r in rows if r['behavior']]
    if len(rows) < n or len({r['behavior'] for r in rows}) != len(rows):
        raise ValueError(f'Need {n} distinct behaviors; found {len(rows)} rows, '
                         f'{len({r["behavior"] for r in rows})} distinct texts')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Behavior IDs must be unique')
    output.mkdir(parents=True, exist_ok=False)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    canonical = output / 'pool.csv'
    with canonical.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(rows)
    old = dict(os.environ)
    try:
        os.environ['JB_BEHAVIORS_PATH'] = str(canonical)
        ordered = evaluator.load_behavior_rows()
        rated = sum(t in evaluator.TIERS for _, _, t in ordered)
        stratified = rated >= n
        if stratified:
            selected = evaluator.sample_stratified(ordered, n, seed)
        else:
            import random
            selected = random.Random(seed).sample([b for _, b, _ in ordered], n)
    finally:
        os.environ.clear(); os.environ.update(old)
    by_text = {r['behavior']: r for r in rows}
    chosen = [by_text[b] for b in selected]
    with (output / 'selected.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(chosen)
    manifest = dict(count=len(chosen), seed=seed, pool_count=len(rows),
                    sampling='stratified' if stratified else 'unstratified',
                    source_path=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    selected_sha256=hashlib.sha256((output/'selected.csv').read_bytes()).hexdigest(),
                    ids=[r['id'] for r in chosen],
                    evaluation_settings=dict(JB_FULL_BEHAVIORS='1', JB_STRATIFIED='0',
                                             JB_CURRICULUM='0', JB_N_BEHAVIORS_PER_EVAL=str(n),
                                             JB_BEHAVIOR_SEED=str(seed)),
                    note='Evaluate frozen CSV in its stored order; do not sample it again. Original tier fields are retained.')
    (output/'sample.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'ids'}, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    freeze(args.source, args.output, args.count, args.seed)
