#!/usr/bin/env python3
"""Audit a finished campaign and rebuild its diversity reports from cached embeddings."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')

import numpy as np

from analysis.experiments.coverage import digest, load_experiment, write_json
from analysis.experiments.report import METRICS, analyze, csv_output, mean_present
from analysis.scripts.clustering_robustness import (
    analyze_cohorts, inventory, load_prompts, plots,
)
from analysis.utils.prompt_diversity import normalize


class CachedOnlyEmbedder:
    """Read the existing cache with its full provenance; never load or run a model."""

    def __init__(self, cache, metadata):
        self.cache = Path(cache)
        self.metadata = {**metadata, 'device': 'cached-only'}
        self.config = {k: v for k, v in self.metadata.items() if k != 'device'}
        self.vectors, self.errors = {}, []

    def embed(self, text):
        key = digest(text)
        if key in self.vectors:
            return self.vectors[key], 0
        path = self.cache / (digest(json.dumps(self.config, sort_keys=True) + '\n' + text) + '.json')
        try:
            cached = json.loads(path.read_text())
            if cached['config'] != self.config or cached['prompt_hash'] != key:
                raise ValueError(f'Embedding cache provenance mismatch: {path}')
            vector = normalize(cached['vector'])
            if vector.shape != (768,):
                raise ValueError(f'Unexpected embedding dimension: {path}')
        except Exception as exc:
            self.errors.append(str(exc))
            raise
        self.vectors[key] = vector
        if len(self.vectors) % 10000 == 0:
            print(f'Validated {len(self.vectors)} cached prompt embeddings', flush=True)
        return vector, cached['chunk_count']


def audit_campaign(campaign, final_generation):
    """Check every committed generation and every retained held-out evaluation."""
    manifest_path = campaign / 'experiment.json'
    manifest = load_experiment(manifest_path)
    configuration = json.loads((campaign / 'campaign.json').read_text())
    state = json.loads((campaign / 'campaign_state.json').read_text())
    expected_runs = {f"{subset['id']}_repeat_{i:02d}" for subset in manifest['subsets']
                     for i, _ in enumerate(configuration['seeds'], 1)}
    runs = sorted(p for p in (campaign / 'runs').iterdir() if (p / 'run.json').exists())
    if {r.name for r in runs} != expected_runs:
        raise ValueError('Campaign run membership is incomplete or unexpected')
    if state.get('controller_status') != 'complete':
        raise ValueError('Campaign controller is not complete')
    if not all(state['stages'][r.name]['status'] == 'complete' for r in runs):
        raise ValueError('Not every experiment stage completed')
    initial = campaign / 'initial/population.json'
    frozen = json.loads(initial.read_text())
    for member in frozen['members']:
        if digest((initial.parent / member['file']).read_bytes()) != member['sha256']:
            raise ValueError('Frozen initial strategy changed')
    results, hashes = [], {str(manifest_path): digest(manifest_path.read_bytes())}
    compatibility = None
    for run in runs:
        config_path = run / 'run.json'
        config_hash = digest(config_path.read_bytes())
        config = json.loads(config_path.read_text())
        hashes[str(config_path)] = config_hash
        signature = {k: config.get(k) for k in ('initial_hash', 'source_hashes', 'models', 'environment', 'operator_settings')}
        if compatibility is None:
            compatibility = signature
        if signature != compatibility or config['experiment_hash'] != hashes[str(manifest_path)]:
            raise ValueError(f'Incompatible run configuration: {run}')
        if config['initial_hash'] != digest(initial.read_bytes()):
            raise ValueError(f'Frozen initial population differs: {run}')
        heldout, missing, status = 0, 0, Counter()
        for gen in range(final_generation + 1):
            marker_path = run / 'completed' / f'{gen:04d}.json'
            marker = json.loads(marker_path.read_text())
            if marker['generation'] != gen:
                raise ValueError(f'Invalid generation marker: {marker_path}')
            for name, checksum in marker['files'].items():
                if digest(Path(name).read_bytes()) != checksum:
                    raise ValueError(f'Completed artifact changed: {name}')
                hashes[name] = checksum
            snap = json.loads((run / 'snapshots' / f'{gen:04d}.json').read_text())
            if snap['run_hash'] != config_hash:
                raise ValueError(f'Snapshot configuration mismatch: {run}, {gen}')
            for member in snap['members']:
                missing += not bool(member.get('observation'))
                if not member['strategy_hash']:
                    continue
                path = run / 'observations/heldout' / f'{gen:04d}' / f'{member["id"]}.json'
                raw = path.read_bytes()
                obs = json.loads(raw)
                records = obs['records']
                if (not obs.get('evaluation_complete') or obs['strategy_hash'] != member['strategy_hash']
                        or len(records) != len(manifest['heldout_ids'])
                        or {r['behavior_id'] for r in records} != set(manifest['heldout_ids'])):
                    raise ValueError(f'Incomplete held-out evaluation: {path}')
                for row in records:
                    if row['status'] == 'ok' and digest(row['prompt_text']) != row['prompt_hash']:
                        raise ValueError(f'Held-out prompt hash mismatch: {path}')
                hashes[str(path)] = digest(raw)
                status.update(r['status'] for r in records)
                heldout += 1
        results.append({'run': run.name, 'completed_generations': final_generation + 1,
                        'heldout_strategy_evaluations': heldout,
                        'heldout_record_statuses': dict(status),
                        'missing_evolution_observations': missing})
        print('Audited', results[-1], flush=True)
    return {'created_utc': datetime.now(timezone.utc).isoformat(),
            'campaign_state': state['controller_status'], 'campaign_finished_utc': state['updated'],
            'final_generation': final_generation, 'runs': results, 'input_hashes': hashes}


def matched_trajectories(case_rows, horizon):
    """Hold behavior eligibility fixed across the entire trajectory of each metric."""
    groups = defaultdict(dict)
    for row in case_rows:
        if row['min_cluster_size'] == '5':
            groups[(row['run'], row['split'])][(int(row['generation']), row['behavior_id'])] = row
    output = []
    for (run, split), by in sorted(groups.items()):
        for metric in METRICS:
            eligible = [case for gen, case in by if gen == 0
                        and all((g, case) in by and by[(g, case)][metric] != '' for g in range(horizon + 1))]
            initial_pairwise = mean_present([float(by[(0, c)]['pairwise']) for c in eligible])
            baseline = mean_present([float(by[(0, c)][metric]) for c in eligible])
            for g in range(horizon + 1):
                value = mean_present([float(by[(g, c)][metric]) for c in eligible])
                delta = value - baseline if value is not None else None
                output.append({'run': Path(run).name, 'split': split, 'metric': metric,
                               'generation': g, 'eligible_cases': len(eligible), 'value': value,
                               'baseline_value': baseline, 'delta': delta,
                               'initial_pairwise_distance': initial_pairwise,
                               'delta_fraction_of_initial_pairwise': delta / initial_pairwise if initial_pairwise else None})
    return output


def summarize(output, full, robust_rows, vectors, horizon, audit):
    import csv
    with (full / 'case_metrics.csv').open() as f:
        matched = matched_trajectories(list(csv.DictReader(f)), horizon)
    csv_output(output / 'matched_trajectories.csv', matched)
    summary, endpoints = [], []
    for split in ('evolution', 'heldout'):
        for metric in METRICS:
            for g in range(horizon + 1):
                rows = [r for r in matched if (r['split'], r['metric'], r['generation']) == (split, metric, g)
                        and r['value'] is not None]
                initial = mean_present([r['initial_pairwise_distance'] for r in rows])
                value = mean_present([r['value'] for r in rows])
                baseline = mean_present([r['baseline_value'] for r in rows])
                delta = value - baseline if value is not None else None
                summary.append({'split': split, 'metric': metric, 'generation': g,
                                'runs': len(rows), 'eligible_behavior_run_groups': sum(r['eligible_cases'] for r in rows),
                                'value': value, 'baseline_value': baseline, 'delta': delta,
                                'initial_pairwise_distance': initial,
                                'delta_fraction_of_initial_pairwise': delta / initial if initial else None,
                                'delta_fraction_of_own_initial': delta / baseline if baseline else None,
                                'increasing_runs': sum(r['delta'] > 1e-8 for r in rows),
                                'decreasing_runs': sum(r['delta'] < -1e-8 for r in rows)})
                if g == horizon:
                    endpoints.append(summary[-1])
    csv_output(output / 'aggregate_trajectories.csv', summary)
    # A fixed empirical scale over all unique cached prompts used by this analysis.
    matrix = np.asarray(list(vectors.values()), dtype=np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    mean = matrix.mean(axis=0)
    pairwise = float(len(matrix) / (len(matrix) - 1) * (1 - mean @ mean))
    findings = {'final_generation': horizon, 'endpoints': endpoints,
                'global_reference': {'unique_prompts': len(matrix), 'mean_pairwise_cosine': pairwise,
                                     'example_0_04_fraction': .04 / pairwise,
                                     'scope': 'Unique successful retained prompts across both splits, all runs and generations.'},
                'deduplicated_evolution': []}
    for weighting in ('population', 'unique_prompts'):
        selected = [r for r in robust_rows if r['horizon'] == horizon and r['min_cluster_size'] == 5 and r['weighting'] == weighting]
        findings['deduplicated_evolution'].append({'weighting': weighting,
            **{metric: {'initial': mean_present([r[metric + '_initial'] for r in selected]),
                        'final': mean_present([r[metric + '_final'] for r in selected]),
                        'delta': mean_present([r[metric + '_delta'] for r in selected])} for metric in METRICS}})
    write_json(output / 'findings.json', findings)
    os.environ['MPLCONFIGDIR'] = str(output / 'matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    titles = {'pairwise': 'Mean pairwise distance', 'nearest_neighbor': 'Nearest-neighbor distance',
              'within_cluster': 'Within-cluster distance', 'between_centroids': 'Between-centroid distance'}
    with PdfPages(output / 'diversity_report.pdf') as pdf:
        for normalized, filename in [(False, 'diversity_trajectories'), (True, 'normalized_diversity_changes')]:
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            for ax, metric in zip(axes.flat, METRICS):
                for split, color in [('evolution', '#2c6eaa'), ('heldout', '#c66c25')]:
                    for run in sorted({r['run'] for r in matched}):
                        rows = [r for r in matched if (r['split'], r['metric'], r['run']) == (split, metric, run)]
                        ax.plot([r['generation'] for r in rows],
                                [r['delta_fraction_of_initial_pairwise'] if normalized else r['value'] for r in rows],
                                color=color, alpha=.12, lw=.8)
                    rows = [r for r in summary if (r['split'], r['metric']) == (split, metric)]
                    ax.plot([r['generation'] for r in rows],
                            [r['delta_fraction_of_initial_pairwise'] if normalized else r['value'] for r in rows],
                            color=color, marker='o', lw=2, label=split)
                ax.set(title=titles[metric], xlabel='Generation', xticks=range(horizon + 1),
                       ylabel='Change / generation-0 mean pairwise distance' if normalized else 'Cosine distance')
                ax.grid(alpha=.2)
                if normalized:
                    ax.axhline(0, color='gray', lw=.7)
                ax.legend()
            fig.suptitle('Diversity over generations · means across ten runs; faint lines show individual runs')
            fig.text(.05, .015, 'Each metric holds behavior eligibility fixed across all generations. Retrospective clusters; shared seed and initial population. Descriptive, not independent replicates.', fontsize=9)
            fig.tight_layout(rect=(0, .04, 1, .95))
            for suffix in ('png', 'pdf'):
                fig.savefig(output / f'{filename}.{suffix}', dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
    lines = [f'Overnight campaign: generations 0–{horizon}',
             'All ten experiments completed, including held-out evaluation.',
             'Means use the same eligible behaviors throughout each metric trajectory.', '',
             'split | metric | initial | final | change | change / initial mean pairwise distance']
    for r in endpoints:
        lines.append(f"{r['split']} | {r['metric']} | {r['baseline_value']:.6f} | {r['value']:.6f} | {r['delta']:+.6f} | {r['delta_fraction_of_initial_pairwise']:+.1%}")
    lines += ['', f'Global unique-prompt distance reference: {pairwise:.6f}, {len(matrix)} prompts.',
              f'A cosine-distance change of 0.04 is {.04 / pairwise:.1%} of this reference.', '',
              'Generation 0 has repeated prompts; see evolution_robustness for exact-prompt deduplication,',
              'minimum cluster sizes 3/5/8, persistent clusters, and generation-1 baseline sensitivity.',
              f"{sum(r['missing_evolution_observations'] for r in audit['runs'])} missing evolution observations; see full/diagnostics.csv.",
              'Held-out observations are complete. Invalid evolution observations reduce prompt pair coverage.',
              'One seed, overlapping subsets, and a shared initial population prevent independent-replicate significance claims.',
              'These are dispersion changes, not parent-child displacement or evidence of improved performance.',
              'The 2D maps are descriptive; original-space cosine distances determine the quantitative results.']
    (output / 'report.txt').write_text('\n'.join(lines) + '\n')
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--final-generation', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    campaign, output = args.campaign.resolve(), args.output.resolve()
    if output == campaign or campaign in output.parents:
        parser.error('Output must be outside the raw campaign')
    if args.final_generation < 2 or args.workers < 1:
        parser.error('final-generation must be >=2; workers must be positive')
    if output.exists() and any(output.iterdir()):
        parser.error('Use a new analysis output directory')
    output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / 'analysis/experiments/report.py',
               ROOT / 'analysis/scripts/clustering_robustness.py', ROOT / 'analysis/scripts/robustness_projection.py',
               ROOT / 'analysis/scripts/summarize_clustering_robustness.py', ROOT / 'analysis/utils/semantic_projection.py']
    write_json(output / 'analysis_sources.json', {str(p): digest(p.read_bytes()) for p in sources})
    audit = audit_campaign(campaign, args.final_generation)
    write_json(output / 'audit.json', audit)
    print('Campaign audit complete. Rebuilding full evolution and held-out analysis.', flush=True)
    manifest = load_experiment(campaign / 'experiment.json')
    embedder = CachedOnlyEmbedder(campaign / 'embedding_cache', manifest['embedding'])
    runs = [campaign / 'runs' / r['run'] for r in audit['runs']]
    full = output / 'full'
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.workers, user_api='blas'):
        analyze(campaign / 'experiment.json', runs, full, args.final_generation, embedder=embedder)
    if embedder.errors:
        raise ValueError(f'Cached embeddings unavailable: {embedder.errors[:3]}')
    print('Full evolution and held-out analysis complete.', flush=True)
    robust = output / 'evolution_robustness'
    robust.mkdir()
    plan = inventory(campaign, args.final_generation)
    if plan['excluded'] or len(plan['runs']) != len(runs):
        raise ValueError('Full-horizon robustness analysis excluded a run')
    write_json(robust / 'plan.json', plan)
    cohorts, prompts, diagnostics, inputs = load_prompts(plan)
    vectors = {k: embedder.vectors[k] for k in prompts}
    summaries, rows = analyze_cohorts(plan, cohorts, vectors, robust)
    csv_output(robust / 'diagnostics.csv', diagnostics)
    cache = robust / 'embedding_cache'
    cache.mkdir()
    np.savez_compressed(cache / 'vectors.npz', **vectors)
    write_json(cache / 'vectors_metadata.json', embedder.config)
    write_json(robust / 'metadata.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
               'embedding': embedder.metadata, 'input_hashes': inputs,
               'method': 'Per-run/per-behavior HDBSCAN in original cosine space; minimum sizes 3,5,8; population and unique-prompt weighting.',
               'common_final_generation': args.final_generation})
    plots(robust, summaries, rows, args.final_generation)
    findings = summarize(output, full, summaries, embedder.vectors, args.final_generation, audit)
    print('Distance summaries and robustness measurements complete. Building maps.', flush=True)
    from analysis.scripts.robustness_projection import main as projections
    projections([str(robust), '--workers', str(args.workers)])
    # Source identity is checked again before marking the analysis complete.
    for name, expected in json.loads((output / 'analysis_sources.json').read_text()).items():
        if digest(Path(name).read_bytes()) != expected:
            raise ValueError(f'Analysis source changed while running: {name}')
    write_json(output / 'SUCCESS.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
               'final_generation': args.final_generation, 'runs': len(runs),
               'unique_prompt_embeddings': len(embedder.vectors),
               'global_distance_reference': findings['global_reference']})
    print('FULL ANALYSIS COMPLETE:', output, flush=True)


if __name__ == '__main__':
    main()
