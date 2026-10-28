"""Retrospective, case-matched diversity measurements and pilot associations."""
from __future__ import annotations
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from analysis.experiments.coverage import digest, distances, load_experiment, write_json
from analysis.experiments.runtime import latest_complete

METRICS = ('pairwise', 'nearest_neighbor', 'within_cluster', 'between_centroids')


def cluster_labels(vectors, minimum=5):
    from sklearn.cluster import HDBSCAN
    unique, inverse = np.unique(np.asarray(vectors), axis=0, return_inverse=True)
    labels = np.full(len(unique), -1, dtype=int)
    if len(unique) >= minimum:
        labels = HDBSCAN(min_cluster_size=minimum, min_samples=3, metric='precomputed').fit_predict(distances(unique))
    return labels[inverse]


def measure(vectors, labels, population_size, hashes):
    n = len(vectors)
    result = {k: None for k in METRICS}
    result.update(valid_prompts=n, population_size=population_size,
                  pair_coverage=n * (n-1) / (population_size * (population_size-1)) if population_size > 1 else None,
                  duplicate_fraction=1 - len(set(hashes)) / n if n else None,
                  unclustered_fraction=float(np.mean(np.asarray(labels) < 0)) if n else None,
                  active_clusters=len(set(labels) - {-1}))
    if not n:
        return result
    v = np.asarray(vectors, dtype=float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    d = distances(v)
    if n > 1:
        result['pairwise'] = float(d[np.triu_indices(n, 1)].mean())
        nn = d.copy()
        np.fill_diagonal(nn, np.inf)
        result['nearest_neighbor'] = float(nn.min(axis=1).mean())
    within, centroids = [], []
    for label in sorted(set(labels) - {-1}):
        members = np.flatnonzero(np.asarray(labels) == label)
        if len(members) > 1:
            block = d[np.ix_(members, members)]
            within.extend(block[np.triu_indices(len(members), 1)])
        center = v[members].mean(axis=0)
        if np.linalg.norm(center) > 1e-12:
            centroids.append(center / np.linalg.norm(center))
    result['within_cluster'] = float(np.mean(within)) if within else None
    if len(centroids) > 1:
        cd = distances(centroids)
        result['between_centroids'] = float(cd[np.triu_indices(len(centroids), 1)].mean())
    return result


def csv_output(path, rows):
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ['status']
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_present(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def association(rows, outcome, adjusted=False):
    keys = ['coverage_distance'] + (['initial_fitness'] if adjusted else [])
    usable = [r for r in rows if all(r.get(k) is not None for k in keys + [outcome])]
    if len(usable) < len(keys) + 2:
        return {'status': 'insufficient_runs', 'n': len(usable)}
    x = np.array([[1.] + [r[k] for k in keys] for r in usable])
    if np.linalg.matrix_rank(x) < x.shape[1]:
        return {'status': 'rank_deficient', 'n': len(usable)}
    y = np.array([r[outcome] for r in usable])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    loo = []
    for subset in {r['subset'] for r in usable}:
        keep = [i for i, r in enumerate(usable) if r['subset'] != subset]
        if len(keep) > x.shape[1] and np.linalg.matrix_rank(x[keep]) == x.shape[1]:
            loo.append(float(np.linalg.lstsq(x[keep], y[keep], rcond=None)[0][1]))
    return {'status': 'descriptive', 'n': len(usable), 'coverage_coefficient': float(beta[1]),
            'leave_one_subset_out_min': min(loo) if loo else None,
            'leave_one_subset_out_max': max(loo) if loo else None}


def analyze(manifest_path, runs, output, final_generation=5, device='cpu', embedder=None):
    from analysis.utils.prompt_diversity import SentenceBertEmbedder
    manifest = load_experiment(manifest_path)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new analysis directory; completed analyses are immutable')
    output.mkdir(parents=True, exist_ok=True)
    em = manifest['embedding']
    if embedder is None:
        embedder = SentenceBertEmbedder(Path(manifest_path).resolve().parent / 'embedding_cache',
                                       em['model'], em['revision'], device=device)
    for key in ('model', 'revision', 'pooling', 'serialization'):
        if embedder.metadata.get(key) != em.get(key):
            raise ValueError(f'Embedding configuration differs: {key}')
    configs, cohorts, groups, feature_cache, diagnostics = {}, {}, defaultdict(list), {}, []
    input_hashes = {}
    compatibility = None
    initial_fitness = {}
    for run in runs:
        root = Path(run).resolve()
        rid = str(root)
        if rid in configs:
            raise ValueError('Duplicate run argument')
        config = json.loads((root / 'run.json').read_text())
        if config['experiment_hash'] != digest(Path(manifest_path).read_bytes()):
            raise ValueError('Run belongs to another experiment')
        signature = {k: config.get(k) for k in ('initial_hash', 'source_hashes', 'models', 'environment', 'operator_settings')}
        if compatibility is None:
            compatibility = signature
        elif signature != compatibility:
            raise ValueError('Runs differ in initial population or model/evolution settings')
        configs[rid] = config
        complete = latest_complete(root)
        if complete is None or complete < final_generation:
            raise ValueError(f'Run has not completed generation {final_generation}: {root}')
        for gen in range(final_generation + 1):
            snap_path = root / 'snapshots' / f'{gen:04d}.json'
            input_hashes[str(snap_path)] = digest(snap_path.read_bytes())
            snap = json.loads(snap_path.read_text())
            if snap['run_hash'] != digest((root / 'run.json').read_bytes()):
                raise ValueError('Snapshot does not match run configuration')
            if gen == 0:
                initial_fitness[rid] = mean_present([m['fitness'][0] for m in snap['members']])
            for split in ('evolution', 'heldout'):
                cases = config['subset']['behavior_ids'] if split == 'evolution' else manifest['heldout_ids']
                for case in cases:
                    cohorts[(rid, gen, split, case)] = {'rows': [], 'population_size': len(snap['members'])}
                for member in snap['members']:
                    ref = member['observation'] if split == 'evolution' else f'observations/heldout/{gen:04d}/{member["id"]}.json'
                    if not ref or not (root / ref).exists():
                        diagnostics.append({'run': rid, 'generation': gen, 'split': split, 'candidate_id': member['id'], 'status': 'missing_observation'})
                        continue
                    obs_path = root / ref
                    input_hashes[str(obs_path)] = digest(obs_path.read_bytes())
                    obs = json.loads(obs_path.read_text())
                    if obs['strategy_hash'] != member['strategy_hash']:
                        raise ValueError('Observation strategy mismatch')
                    seen = set()
                    for row in obs['records']:
                        case = row['behavior_id']
                        if case not in cases or case in seen:
                            raise ValueError('Unexpected or duplicate behavior in observation')
                        seen.add(case)
                        if row['status'] != 'ok':
                            diagnostics.append({'run': rid, 'generation': gen, 'split': split, 'candidate_id': member['id'], 'case': case, 'status': row['status']})
                            continue
                        prompt = row['prompt_text']
                        if digest(prompt) != row['prompt_hash']:
                            raise ValueError('Prompt hash mismatch')
                        key = row['prompt_hash']
                        if key not in feature_cache:
                            try:
                                feature_cache[key] = embedder.embed(prompt)[0]
                            except Exception as exc:
                                feature_cache[key] = None
                                diagnostics.append({'run': rid, 'generation': gen, 'split': split, 'case': case, 'status': 'embedding_error', 'error': str(exc)})
                        vector = feature_cache[key]
                        if vector is None:
                            continue
                        entry = {'vector': vector, 'hash': key, 'score': row.get('score'), 'tokens': row.get('tokens'), 'candidate_id': member['id']}
                        cohorts[(rid, gen, split, case)]['rows'].append(entry)
                        group = (split, case, rid if split == 'evolution' else 'shared')
                        groups[group].append(entry)
    label_maps = {}
    for group, rows in groups.items():
        unique = {r['hash']: r['vector'] for r in rows}
        keys = sorted(unique)
        for minimum in (3, 5, 8):
            labels = cluster_labels([unique[k] for k in keys], minimum)
            label_maps[(group, minimum)] = dict(zip(keys, map(int, labels)))
    case_rows, occupancy = [], []
    for (rid, gen, split, case), cohort in cohorts.items():
        entries = cohort['rows']
        group = (split, case, rid if split == 'evolution' else 'shared')
        for minimum in (3, 5, 8):
            labels = [label_maps[(group, minimum)][e['hash']] for e in entries]
            stats = measure([e['vector'] for e in entries], labels, cohort['population_size'], [e['hash'] for e in entries])
            row = {'run': rid, 'generation': gen, 'split': split, 'behavior_id': case, 'min_cluster_size': minimum,
                   'mean_score': mean_present([e['score'] for e in entries]), 'mean_tokens': mean_present([e['tokens'] for e in entries]), **stats}
            case_rows.append(row)
            for label in sorted(set(labels)):
                occupancy.append({'run': rid, 'generation': gen, 'split': split, 'behavior_id': case,
                                  'min_cluster_size': minimum, 'cluster': label, 'members': labels.count(label)})
    grouped = defaultdict(list)
    for r in case_rows:
        grouped[(r['run'], r['generation'], r['split'], r['min_cluster_size'])].append(r)
    generation_rows = []
    measures = (*METRICS, 'pair_coverage', 'duplicate_fraction', 'unclustered_fraction', 'active_clusters', 'mean_score', 'mean_tokens', 'population_size')
    for (rid, gen, split, minimum), rows in grouped.items():
        config = configs[rid]
        row = {'run': rid, 'subset': config['subset']['id'], 'kind': config['subset']['kind'],
               'coverage_distance': config['subset']['coverage_distance'], 'generation': gen, 'split': split,
               'min_cluster_size': minimum, 'expected_cases': len(rows)}
        for metric in measures:
            row[metric] = mean_present([r[metric] for r in rows])
            row[metric + '_eligible_cases'] = sum(r[metric] is not None for r in rows)
        generation_rows.append(row)
    summaries = []
    for rid in configs:
        for split in ('evolution', 'heldout'):
            for minimum in (3, 5, 8):
                trajectory = sorted([r for r in generation_rows if r['run'] == rid and r['split'] == split and r['min_cluster_size'] == minimum], key=lambda r: r['generation'])
                for metric in METRICS:
                    valid = [r for r in trajectory if r[metric] is not None]
                    first, last = trajectory[0][metric], trajectory[-1][metric]
                    row = {k: trajectory[0][k] for k in ('run', 'subset', 'kind', 'coverage_distance', 'split', 'min_cluster_size')}
                    row.update(metric=metric, initial_fitness=initial_fitness[rid],
                               endpoint_change=last-first if first is not None and last is not None else None,
                               slope=float(np.polyfit([r['generation'] for r in valid], [r[metric] for r in valid], 1)[0]) if len(valid) >= 2 else None,
                               eligible_generations=len(valid))
                    summaries.append(row)
    associations = []
    subset_summaries = []
    # Average repeated-run summaries within subset: ten subsets remain ten units.
    for split in ('evolution', 'heldout'):
        for minimum in (3, 5, 8):
            for metric in METRICS:
                by_subset = defaultdict(list)
                for r in summaries:
                    if (r['split'], r['min_cluster_size'], r['metric']) == (split, minimum, metric):
                        by_subset[r['subset']].append(r)
                for subset_id, values in by_subset.items():
                    item = {'subset': subset_id, 'split': split, 'metric': metric, 'min_cluster_size': minimum, 'runs': len(values)}
                    for outcome in ('slope', 'endpoint_change'):
                        available = [r[outcome] for r in values if r[outcome] is not None]
                        item[outcome + '_mean'] = mean_present(available)
                        item[outcome + '_sd'] = float(np.std(available, ddof=1)) if len(available) > 1 else None
                    subset_summaries.append(item)
                aggregated = [{'subset': k, **{field: mean_present([r[field] for r in rows]) for field in ('coverage_distance', 'initial_fitness', 'slope', 'endpoint_change')}} for k, rows in by_subset.items()]
                for outcome in ('slope', 'endpoint_change'):
                    for adjusted in (False, True):
                        associations.append({'split': split, 'min_cluster_size': minimum, 'metric': metric,
                                             'outcome': outcome, 'adjusted_initial_fitness': adjusted,
                                             **association(aggregated, outcome, adjusted)})
    csv_output(output / 'case_metrics.csv', case_rows)
    csv_output(output / 'generation_metrics.csv', generation_rows)
    csv_output(output / 'cluster_occupancy.csv', occupancy)
    csv_output(output / 'trajectory_summaries.csv', summaries)
    csv_output(output / 'subset_summaries.csv', subset_summaries)
    csv_output(output / 'coverage_associations.csv', associations)
    csv_output(output / 'diagnostics.csv', diagnostics)
    assignments = [{'split': group[0], 'behavior_id': group[1], 'scope': group[2], 'min_cluster_size': minimum,
                    'prompt_hash': h, 'cluster': label} for (group, minimum), labels in label_maps.items() for h, label in labels.items()]
    csv_output(output / 'cluster_assignments.csv', assignments)
    from importlib.metadata import version, PackageNotFoundError
    packages = {}
    for name in ('numpy', 'scikit-learn', 'sentence-transformers', 'statsmodels', 'matplotlib'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = 'unavailable'
    write_json(output / 'metadata.json', {'final_generation': final_generation, 'inputs': input_hashes, 'packages': packages,
                                        'analysis_source_hash': digest(Path(__file__).read_bytes()),
                                        'experiment_hash': digest(Path(manifest_path).read_bytes()), 'embedding': embedder.metadata,
                                        'clustering': 'retrospective pooled HDBSCAN cosine; unique vectors fitted; strategy weights restored'})
    plot(output, generation_rows, summaries)
    repeated_models(output, generation_rows, configs, initial_fitness)
    (output / 'report.txt').write_text(
        f'Behavior coverage pilot: {len(configs)} runs, generations 0–{final_generation}.\n'
        'Primary outcomes: held-out within-cluster contraction and between-centroid separation.\n'
        'Lower coverage distance means broader initial coverage. Associations are descriptive.\n'
        'Subset means are the units in coverage_associations.csv; behaviors are not independent runs.\n'
        'Initial fitness adjustment uses generation-0 evolution mean judge score.\n'
        'Selection retains the base branch TF-IDF diversity objective.\n'
        'Clustering pools this analysis horizon retrospectively; extending the horizon changes group definitions.\n'
        'Inspect eligible-case counts, missing observations, population size, and clustering sensitivity before interpretation.\n'
        'One run per subset cannot separate subset effects from stochastic evolution.\n')
    return generation_rows


def plot(output, generations, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'narrow': '#c44e52', 'random': '#4c72b0', 'broad': '#55a868'}
    for split in ('evolution', 'heldout'):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for metric, ax in zip(METRICS, axes.flat):
            rows = [r for r in generations if r['split'] == split and r['min_cluster_size'] == 5]
            for run in sorted({r['run'] for r in rows}):
                points = sorted([r for r in rows if r['run'] == run], key=lambda r: r['generation'])
                ax.plot([r['generation'] for r in points], [r[metric] if r[metric] is not None else np.nan for r in points],
                        color=colors[points[0]['kind']], alpha=.65, marker='.')
            ax.set(title=metric.replace('_', ' '), xlabel='Generation', ylabel='Cosine distance')
        fig.suptitle(split + ' trajectories; red=narrow, blue=random, green=broad')
        fig.tight_layout()
        fig.savefig(output / f'{split}_trajectories.png', dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for metric, ax in zip(METRICS, axes.flat):
        for r in summaries:
            if r['split'] == 'heldout' and r['min_cluster_size'] == 5 and r['endpoint_change'] is not None and r['metric'] == metric:
                ax.scatter(r['coverage_distance'], r['endpoint_change'], color=colors[r['kind']])
        ax.set(title=metric.replace('_', ' '), xlabel='Coverage distance (lower = broader)', ylabel='Change from generation 0')
    fig.tight_layout()
    fig.savefig(output / 'coverage_changes.png', dpi=150)
    plt.close(fig)


def repeated_models(output, rows, configs, initial):
    counts = defaultdict(int)
    for c in configs.values():
        counts[c['subset']['id']] += 1
    if len(counts) < 3 or not all(n >= 2 for n in counts.values()):
        write_json(output / 'repeated_models.json', {'status': 'requires_at_least_three_subsets_with_two_runs_each'})
        return
    import pandas as pd
    import statsmodels.formula.api as smf
    results = {}
    for metric in METRICS:
        usable = [dict(r) for r in rows if r['split'] == 'heldout' and r['min_cluster_size'] == 5 and r[metric] is not None]
        for r in usable:
            r['initial_fitness'] = initial[r['run']]
        try:
            frame = pd.DataFrame(usable).dropna(subset=['initial_fitness', metric])
            fit = smf.mixedlm(f'{metric} ~ coverage_distance * generation + initial_fitness', frame,
                             groups=frame['subset'], re_formula='1', vc_formula={'run': '0 + C(run)'}).fit()
            results[metric] = {'converged': bool(fit.converged), 'coefficients': {k: float(v) if np.isfinite(v) else None for k, v in fit.params.items()}}
        except Exception as exc:
            results[metric] = {'status': 'unavailable', 'error': str(exc)}
    write_json(output / 'repeated_models.json', results)
