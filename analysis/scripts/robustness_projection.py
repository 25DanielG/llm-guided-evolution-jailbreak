#!/usr/bin/env python3
"""Reuse analysis-scratch generation-opacity maps for the cross-subset analysis."""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from analysis.scripts.clustering_robustness import load_prompts
from analysis.scripts.semantic_projection import plot_outputs, draw_map, write_interactive
from analysis.utils.semantic_projection import project_case, choose_representative_case
from analysis.experiments.coverage import write_json


def project_cohorts(plan, cohorts, vectors, horizon, workers):
    """Project independent behavior maps concurrently, without fitting across cases."""
    results = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        tasks = {}
        for spec in plan['runs']:
            rid = spec['run']
            for case in sorted({key[2] for key in cohorts if key[0] == rid}):
                rows = [{'generation': g, 'candidate_id': e['candidate'], 'prompt_hash': e['hash']}
                        for g in range(horizon + 1) for e in cohorts[(rid, g, case)]['rows']]
                if rows:
                    selected = {r['prompt_hash']: vectors[r['prompt_hash']] for r in rows}
                    tasks[pool.submit(project_case, rows, selected)] = (rid, case)
        for i, future in enumerate(as_completed(tasks), 1):
            results[tasks[future]] = future.result()
            if i % 10 == 0 or i == len(tasks):
                print(f'Projected {i}/{len(tasks)} behavior maps', flush=True)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('workers must be positive')
    directory = args.directory.resolve()
    from analysis.scripts.summarize_clustering_robustness import main as summarize
    summarize([str(directory)])
    plan = json.loads((directory / 'plan.json').read_text())
    horizon = plan.get('common_final_generation', 4)
    cohorts, _, _, _ = load_prompts(plan)
    with np.load(directory / 'embedding_cache/vectors.npz') as data:
        vectors = {key: data[key] for key in data.files}
    with (directory / 'case_metrics.csv').open() as stream:
        metrics = [r for r in csv.DictReader(stream)
                   if int(r['horizon']) == horizon and r['min_cluster_size'] == '5' and r['weighting'] == 'population']
    metric_lookup = {(r['run'], int(r['generation']), r['case']): r for r in metrics}
    os.environ['MPLCONFIGDIR'] = str(directory / 'matplotlib-cache')
    os.environ['NUMBA_CACHE_DIR'] = str(directory / 'numba-cache')
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    projected_cases = project_cohorts(plan, cohorts, vectors, horizon, args.workers)
    combined = {'run_id': f'Starting behavior subsets: generation 0–{horizon}', 'cohort': 'retained',
                'cases': {}, 'summary': []}
    selected_cases = {}
    quality_rows = []
    for spec in plan['runs']:
        rid = spec['run']
        output = directory / 'plots' / rid
        output.mkdir(parents=True, exist_ok=True)
        payload = {'run_id': rid, 'cohort': 'retained', 'cases': {}, 'summary': []}
        for generation in range(horizon + 1):
            rows = [r for r in metrics if r['run'] == rid and int(r['generation']) == generation]
            payload['summary'].append({'generation': generation,
                'semantic_mean': float(np.mean([float(r['pairwise']) for r in rows if r['pairwise']])),
                'semantic_nearest_neighbor': float(np.mean([float(r['nearest_neighbor']) for r in rows if r['nearest_neighbor']])),
                'semantic_pair_coverage': float(np.mean([float(r['pair_coverage']) for r in rows]))})
        cases = sorted({key[2] for key in cohorts if key[0] == rid})
        for number, case in enumerate(cases, 1):
            rows = [{'generation': g, 'candidate_id': e['candidate'], 'prompt_hash': e['hash']}
                    for g in range(horizon + 1) for e in cohorts[(rid, g, case)]['rows']]
            if not rows:
                continue
            points, stats, quality = projected_cases[(rid, case)]
            for stat in stats:
                expected = metric_lookup[(rid, stat['generation'], case)]
                for plotted, measured in (('semantic_mean', 'pairwise'), ('semantic_nearest_neighbor', 'nearest_neighbor')):
                    if stat[plotted] is not None:
                        assert np.isclose(stat[plotted], float(expected[measured]), atol=5e-7), 'Map and analysis disagree'
            payload['cases'][case] = {'case_number': number, 'points': points, 'observations': rows,
                                      'stats': stats, 'quality': quality,
                                      'label': f"{rid.replace('_repeat_01', '')} · case {number:02d}",
                                      'summary': payload['summary']}
            quality_rows.append({'run': rid, 'case': case, **quality})
        selected = choose_representative_case({key: c['stats'] for key, c in payload['cases'].items()})
        selected_cases[rid] = selected
        payload['selected'] = selected
        write_interactive(output, payload, selected)
        plot_outputs(output, payload, selected)
        write_json(output / 'projection_data.json', payload)
        for key, case in payload['cases'].items():
            combined['cases'][rid + ':' + key] = case
        print('Saved maps:', rid, flush=True)
    first = plan['runs'][0]['run']
    combined['selected'] = first + ':' + selected_cases[first]
    combined['summary'] = combined['cases'][combined['selected']]['summary']
    template = (ROOT / 'analysis/utils/semantic_projection.html').read_text()
    template = template.replace('o.textContent=`Case ${String(c.case_number).padStart(2,\'0\')}`',
                                'o.textContent=c.label')
    template = template.replace("function render(){const c=data.cases[el('case').value],index=", 
                                "function render(){data.summary=data.cases[el('case').value].summary;const c=data.cases[el('case').value],index=")
    template = template.replace('Evaluation case <select', 'Run and evaluation case <select')
    template = template.replace('Semantic diversity over time', 'Clustering across starting behavior subsets')
    serialized = json.dumps(combined, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    (directory / 'clustering_explorer.html').write_text(template.replace('__PROJECTION_DATA__', serialized))
    # Overview panels: representative cases chosen numerically before inspecting maps.
    nrows = int(np.ceil(len(plan['runs']) / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(15, nrows * 4.7), squeeze=False)
    for ax, spec in zip(axes.flat, plan['runs']):
        rid = spec['run']
        case = combined['cases'][rid + ':' + selected_cases[rid]]
        draw_map(ax, case, horizon)
        ax.set_title(f"{rid.replace('_repeat_01', '')} · case {case['case_number']:02d}", fontsize=12)
    for ax in list(axes.flat)[len(plan['runs']):]:
        ax.set_visible(False)
    fig.suptitle(f'Shared maps across generations: earlier populations fade; generation {horizon} is opaque', fontsize=17)
    fig.text(.04, .02, 'Each panel has separate coordinates and cluster colors. Cases are selected by median relative nearest-neighbor change.\n'
             'Groups are inferred in original embedding space. Use the distance plots to assess contraction and separation; 2D geometry is approximate.', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .96))
    for suffix in ('png', 'pdf'):
        fig.savefig(directory / f'generation_opacity_overview.{suffix}', dpi=180)
    plt.close(fig)
    # Show how much the starting behavior subsets overlap.
    manifest = json.loads((Path(plan['campaign']) / 'experiment.json').read_text())
    membership = {s['id']: set(s['behavior_ids']) for s in manifest['subsets']}
    ids = [s['subset'] for s in plan['runs']]
    overlap = np.array([[len(membership[a] & membership[b]) for b in ids] for a in ids])
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(overlap, vmin=0, vmax=30, cmap='Blues')
    for i in range(len(ids)):
        for j in range(len(ids)):
            ax.text(j, i, str(overlap[i, j]), ha='center', va='center', color='white' if overlap[i, j] > 16 else '#203040')
    ax.set(xticks=range(len(ids)), yticks=range(len(ids)), xticklabels=ids, yticklabels=ids,
           title='Starting behavior subsets: shared behaviors out of 30')
    ax.tick_params(axis='x', rotation=45)
    fig.colorbar(image, ax=ax, label='Shared behaviors')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(directory / f'subset_overlap.{suffix}', dpi=180)
    plt.close(fig)
    # A case-level view makes heterogeneous or contradictory changes visible.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    palette = plt.get_cmap('tab10')
    for ax, kind in zip(axes, ('random', 'narrow', 'broad')):
        for index, spec in enumerate(plan['runs']):
            if spec['kind'] != kind:
                continue
            by = {(int(r['generation']), r['case']): r for r in metrics if r['run'] == spec['run']}
            changes = []
            for gen, case in by:
                if gen != 0 or (horizon, case) not in by:
                    continue
                if not all(by[(g, case)][m] != '' for g in (0, horizon) for m in ('within_cluster', 'between_centroids')):
                    continue
                changes.append([float(by[(horizon, case)][m])-float(by[(0, case)][m])
                                for m in ('within_cluster', 'between_centroids')])
            if not changes:
                continue
            changes = np.array(changes)
            ax.scatter(changes[:, 0], changes[:, 1], color=palette(index), alpha=.32, s=22)
            center = changes.mean(axis=0)
            ax.scatter(*center, color=palette(index), s=160, marker='D', edgecolor='white', linewidth=1.2,
                       label=spec['subset'])
        ax.axhline(0, color='#596574', linewidth=.8)
        ax.axvline(0, color='#596574', linewidth=.8)
        ax.set(title=kind.capitalize(), xlabel='Change in within-cluster distance')
        ax.grid(alpha=.15)
        ax.legend(fontsize=9)
    axes[0].set_ylabel('Change in between-centroid distance')
    fig.suptitle(f'Generation 0 → {horizon}: the upper-left quadrant is tighter within clusters and farther apart between clusters')
    fig.text(.05, .015, 'Faint dots: behaviors with both metrics available at both endpoints. Diamonds: run means on those same behaviors. Behaviors are not independent evolution runs.', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    for suffix in ('png', 'pdf'):
        fig.savefig(directory / f'case_change_landscape.{suffix}', dpi=180)
    plt.close(fig)
    # One page per run, showing its initial and evolved populations on identical axes.
    with PdfPages(directory / 'generation_opacity_atlas.pdf') as pdf:
        for spec in plan['runs']:
            rid = spec['run']
            case = combined['cases'][rid + ':' + selected_cases[rid]]
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, gen in zip(axes, (0, horizon // 2, horizon)):
                draw_map(ax, case, gen)
                ax.set_title(f'Generation {gen}', fontsize=13)
            fig.suptitle(rid.replace('_repeat_01', '') + ' · shared coordinates; earlier generations faded', fontsize=15)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    with (directory / 'projection_quality.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(quality_rows[0]))
        writer.writeheader()
        writer.writerows(quality_rows)
    write_json(directory / 'projection_metadata.json', {
        'reference_branch': 'analysis-scratch', 'reference_commit': '7921b0a77',
        'selection': 'Median relative endpoint nearest-neighbor change within each run, before looking at projections.',
        'fit': f'One fixed densMAP per run/behavior on unique vectors pooled across generations 0–{horizon}; duplicates restored.',
        'opacity': 'analysis-scratch generation_alpha; oldest 0.10, newest 1.0',
        'clustering': 'HDBSCAN in original cosine space; colors are local to each map.',
        'selected_cases': selected_cases,
        'limitations': 'Maps are descriptive projections; original-space metrics determine contraction and separation.'})
    print('All projection outputs complete:', directory, flush=True)


if __name__ == '__main__':
    main()
