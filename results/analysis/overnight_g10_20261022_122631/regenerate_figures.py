#!/usr/bin/env python3
"""Regenerate the published generation-0-to-10 figures from validated summary data.

Requires numpy and matplotlib. No model services, embeddings, or raw prompts are
needed: the selected maps retain the audited fixed coordinates from the analysis.
"""
from collections import Counter
import colorsys
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.render-cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(HERE / '.render-cache/system'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
COLORS = {'evolution': '#2469a4', 'heldout': '#d47724'}
KIND_COLORS = {'random': '#3875b6', 'narrow': '#cc5656', 'broad': '#35946a'}
METRICS = ('pairwise', 'nearest_neighbor', 'within_cluster', 'between_centroids')
TITLES = {'pairwise': 'Mean pairwise distance', 'nearest_neighbor': 'Nearest-neighbor distance',
          'within_cluster': 'Within-cluster distance', 'between_centroids': 'Between-centroid distance'}


def read(name):
    with (HERE / 'data' / name).open() as stream:
        return list(csv.DictReader(stream))


def number(value):
    return float(value) if value not in ('', None) else np.nan


def save(fig, name):
    path = HERE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(path.with_suffix('.' + suffix), dpi=200)
    print('Saved', name, flush=True)


def trend(ax, summary, matched, metric, normalized=False):
    field = 'delta_fraction_of_initial_pairwise' if normalized else 'value'
    for split, color in COLORS.items():
        for run in sorted({r['run'] for r in matched}):
            rows = sorted([r for r in matched if (r['split'], r['metric'], r['run']) == (split, metric, run)],
                          key=lambda r: int(r['generation']))
            assert [int(r['generation']) for r in rows] == list(range(11))
            ax.plot(range(11), [number(r[field]) for r in rows], color=color, alpha=.12, lw=.9)
        rows = sorted([r for r in summary if (r['split'], r['metric']) == (split, metric)],
                      key=lambda r: int(r['generation']))
        assert [int(r['generation']) for r in rows] == list(range(11))
        assert all(int(r['runs']) == 10 for r in rows)
        ax.plot(range(11), [number(r[field]) for r in rows], color=color,
                marker='o' if split == 'evolution' else 's', ms=5, lw=2.4,
                label='Evolution behaviors' if split == 'evolution' else 'Held-out behaviors')
    ax.set(title=TITLES[metric], xlabel='Generation', xticks=range(11),
           ylabel='Change / initial mean pairwise distance' if normalized else 'Cosine distance')
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    if normalized:
        ax.axhline(0, color='gray', lw=.7)


def distance_figures(summary, matched):
    for normalized, name in [(False, 'diversity_trajectories'), (True, 'normalized_diversity_changes')]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, metric in zip(axes.flat, METRICS):
            trend(ax, summary, matched, metric, normalized)
        fig.suptitle('Diversity over generations 0–10 · ten completed runs', fontsize=17)
        fig.text(.04, .02, 'Strong lines: equally weighted run means. Faint lines: individual runs. Each metric holds behavior eligibility fixed across generations.\n'
                 'One seed and shared initialization; overlapping behavior subsets. Original-space cosine distances; descriptive comparisons.', fontsize=9)
        fig.tight_layout(rect=(0, .065, 1, .95))
        save(fig, name)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(11.5, 6.3))
    trend(ax, summary, matched, 'pairwise')
    ax.set_title('Semantic diversity increases on evolution and held-out behaviors', fontsize=16, pad=15)
    ax.text(.03, .94, 'Generation 0 → 10\nEvolution: 0.0614 → 0.1683 (+174%)\nHeld-out: 0.0625 → 0.1701 (+172%)',
            transform=ax.transAxes, va='top', fontsize=11,
            bbox={'boxstyle': 'round,pad=.5', 'facecolor': 'white', 'edgecolor': '#d8dce2'})
    ax.legend(loc='lower right', fontsize=10)
    fig.text(.07, .02, 'Means across ten runs; faint lines show individual runs. The same behaviors contribute throughout each trajectory. One evolution seed.', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    save(fig, 'pairwise_diversity')
    plt.close(fig)


def cluster_figures(matched):
    primary = [r for r in read('endpoint_changes.csv') if r['horizon'] == '10'
               and r['min_cluster_size'] == '5' and r['weighting'] == 'population']
    later = {r['run']: r for r in read('generation_1_to_10_changes.csv') if r['weighting'] == 'population'}
    assert len(primary) == len(later) == 10
    for baseline, name in [(False, 'clustering_changes'), (True, 'baseline_sensitivity')]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
        for ax, metric, title in zip(axes, ('within_cluster', 'between_centroids'),
                                   ('Within-cluster change: negative = tighter', 'Between-centroid change: positive = farther apart')):
            for i, row in enumerate(primary):
                if baseline:
                    ax.scatter(number(row[metric + '_delta']), i-.12, color=COLORS['evolution'], s=48)
                    ax.scatter(number(later[row['run']][metric + '_delta']), i+.12,
                               color=COLORS['heldout'], marker='s', s=48)
                else:
                    ax.scatter(number(row[metric + '_delta']), i, color=KIND_COLORS[row['kind']], s=55)
            ax.axvline(0, color='#596574', lw=.9)
            ax.set(title=title, xlabel='Change in cosine distance')
            ax.grid(axis='x', alpha=.2)
        axes[0].set_yticks(range(10), [r['run'].replace('_repeat_01', '') for r in primary])
        if baseline:
            fig.legend(handles=[Line2D([], [], marker='o', ls='none', color=COLORS['evolution'], label='Generation 0 → 10'),
                                Line2D([], [], marker='s', ls='none', color=COLORS['heldout'], label='Generation 1 → 10')],
                       loc='lower center', ncols=2)
            title = 'Baseline sensitivity · evolution behaviors · same generation-0-to-10 cluster fit'
        else:
            title = 'Generation 0 → 10 · evolution behaviors · matched cases at both endpoints'
        fig.suptitle(title, fontsize=15)
        fig.tight_layout(rect=(0, .065 if baseline else .02, 1, .95))
        save(fig, name)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, metric in zip(axes.flat, METRICS):
        for run in sorted({r['run'] for r in matched}):
            rows = sorted([r for r in matched if (r['run'], r['split'], r['metric']) == (run, 'evolution', metric)],
                          key=lambda r: int(r['generation']))
            ax.plot(range(11), [number(r['value']) for r in rows], color=KIND_COLORS[run.split('_')[0]],
                    marker='.', lw=1.3, alpha=.75, label=run.replace('_repeat_01', ''))
        ax.set(title=TITLES[metric], xlabel='Generation', ylabel='Cosine distance', xticks=range(11))
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, ncols=2)
    fig.suptitle('Evolution trajectories across all ten subsets · generations 0–10', fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, 'clustering_trajectories')
    plt.close(fig)


def cluster_color(label):
    if label < 0:
        return '#737b88'
    rgb = colorsys.hls_to_rgb((.58 + label*.61803398875) % 1, .43, .68)
    return '#' + ''.join(f'{round(v*255):02x}' for v in rgb)


def draw_map(ax, case, generation):
    points, observations = case['points'], case['observations']
    generations = list(range(generation + 1))
    for g in generations:
        counts = Counter(r['prompt_hash'] for r in observations if r['generation'] == g)
        keys = sorted(counts)
        alpha = 1 if g == generation else .10 + .35*g/max(1, generation-1)
        ax.scatter([points[k]['x'] for k in keys], [points[k]['y'] for k in keys],
                   c=[cluster_color(points[k]['cluster']) for k in keys],
                   s=[(64 if g == generation else 16)*(1+.35*(counts[k]-1)) for k in keys],
                   alpha=alpha, edgecolors='white' if g == generation else 'none',
                   linewidths=.6, zorder=3 if g == generation else 2)
    xy = np.array([[p['x'], p['y']] for p in points.values()])
    assert np.isfinite(xy).all()
    low, high = xy.min(axis=0), xy.max(axis=0)
    pad = np.maximum(high-low, 1)*.10
    ax.set(xlim=(low[0]-pad[0], high[0]+pad[0]), ylim=(low[1]-pad[1], high[1]+pad[1]))
    ax.set_aspect('equal', adjustable='box', anchor='N')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top', 'right']].set_visible(False)


def map_figures():
    selected = json.loads((HERE / 'data/selected_maps.json').read_text())
    assert len(selected) == 10
    for case in selected.values():
        assert [r['generation'] for r in case['stats']] == list(range(11))
    fig, axes = plt.subplots(2, 5, figsize=(20, 9.5), squeeze=False)
    for ax, (run, case) in zip(axes.flat, selected.items()):
        draw_map(ax, case, 10)
        ax.set_title(f"{run.replace('_repeat_01', '')} · case {case['case_number']:02d}", fontsize=12)
    fig.suptitle('All ten runs · earlier generations fade; generation 10 is opaque', fontsize=20)
    fig.text(.025, .025, 'One representative behavior per run, selected by median relative nearest-neighbor change before inspecting maps.\n'
             'Each panel has independent coordinates and cluster colors. HDBSCAN groups use original embeddings; 2D geometry is approximate.', fontsize=11)
    # Reserve a fixed gap between rows; equal-aspect panels have different heights.
    fig.subplots_adjust(left=.025, right=.99, top=.86, bottom=.13, hspace=.35, wspace=.15)
    save(fig, 'generation_opacity_overview')
    plt.close(fig)
    with PdfPages(HERE / 'generation_opacity_atlas.pdf') as pdf:
        for run, case in selected.items():
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, g in zip(axes, (0, 5, 10)):
                draw_map(ax, case, g)
                ax.set_title(f'Generation {g}', fontsize=14)
            fig.suptitle(f"{run.replace('_repeat_01', '')} · representative behavior {case['case_number']:02d}", fontsize=17)
            fig.text(.04, .025, 'Fixed coordinates across all three panels; previous generations fade. Original-space HDBSCAN colors; projected distances have no cosine units.', fontsize=9)
            fig.tight_layout(rect=(0, .055, 1, .93))
            save(fig, f'plots/{run}/generations_0_5_10')
            pdf.savefig(fig)
            plt.close(fig)


def quality_figures():
    rows = [r for r in read('generation_metrics.csv') if r['min_cluster_size'] == '5']
    assert len(rows) == 220
    fields = {'pair_coverage': 'Available prompt pairs / expected pairs',
              'duplicate_fraction': 'Exact duplicate prompt fraction',
              'population_size': 'Retained population size',
              'unclustered_fraction': 'Fraction outside inferred clusters'}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (field, title) in zip(axes.flat, fields.items()):
        for split, color in COLORS.items():
            means = []
            for run in sorted({r['run'] for r in rows}):
                group = sorted([r for r in rows if (r['split'], r['run']) == (split, run)], key=lambda r: int(r['generation']))
                ax.plot(range(11), [number(r[field]) for r in group], color=color, alpha=.12, lw=.9)
            for g in range(11):
                means.append(np.mean([number(r[field]) for r in rows if r['split'] == split and int(r['generation']) == g]))
            ax.plot(range(11), means, color=color, marker='o', lw=2.3, label=split)
        ax.set(title=title, xlabel='Generation', xticks=range(11))
        if field != 'population_size':
            ax.set_ylim(-.025, 1.025)
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle('Coverage and population changes · generations 0–10', fontsize=17)
    fig.text(.04, .02, '132 missing evolution observations reduce pair coverage. Held-out observations are complete. Faint lines: individual runs; strong lines: run means.', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    save(fig, 'data_quality')
    plt.close(fig)


def main():
    summary, matched = read('aggregate_trajectories.csv'), read('matched_trajectories.csv')
    findings = json.loads((HERE / 'data/findings.json').read_text())
    assert findings['final_generation'] == 10
    for split, expected in [('evolution', .16834661565845857), ('heldout', .17005262286826234)]:
        row = next(r for r in summary if r['split'] == split and r['metric'] == 'pairwise' and r['generation'] == '10')
        assert np.isclose(float(row['value']), expected, atol=1e-10)
    distance_figures(summary, matched)
    cluster_figures(matched)
    map_figures()
    quality_figures()
    inputs = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((HERE / 'data').iterdir())}
    outputs = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.rglob('*'))
               if p.is_file() and p.suffix in ('.png', '.pdf') and '.render-cache' not in p.parts}
    assert len(outputs) == 37
    manifest = {'regenerated_utc': datetime.now(timezone.utc).isoformat(), 'runs': 10, 'first_generation': 0,
                'last_generation': 10, 'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'packages': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
                'inputs_sha256': inputs, 'outputs_sha256': outputs}
    (HERE / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Regenerated and checked all 37 figure files.', flush=True)


if __name__ == '__main__':
    main()
