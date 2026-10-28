#!/usr/bin/env python3
"""Summarize the interim cross-subset analysis, with explicit coverage limits."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.experiments.report import csv_output
from analysis.scripts.clustering_robustness import paired_changes


def baseline_plot(root, primary, later, horizon=4):
    os.environ['MPLCONFIGDIR'] = str(root / 'matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    later_by_run = {r['run']: r for r in later if r['weighting'] == 'population'}
    for ax, metric, title in zip(axes, ('within_cluster', 'between_centroids'),
                               ('Within-cluster distance: negative = tighter', 'Between-centroid distance: positive = farther apart')):
        for index, row in enumerate(primary):
            for offset, value, marker, color in [(-.12, number(row[metric+'_delta']), 'o', '#3475b4'),
                                                   (.12, later_by_run[row['run']][metric+'_delta'], 's', '#d17a34')]:
                if value is not None:
                    ax.scatter(value, index+offset, marker=marker, color=color, s=50)
        ax.axvline(0, color='#555', linewidth=1)
        ax.set(title=title, xlabel='Endpoint change in cosine distance')
        ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(range(len(primary)), [r['run'].replace('_repeat_01', '') for r in primary])
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker='o', linestyle='none', color='#3475b4', label=f'Generation 0 → {horizon}'),
                        Line2D([], [], marker='s', linestyle='none', color='#d17a34', label=f'Generation 1 → {horizon}')],
               loc='lower center', ncols=2)
    fig.suptitle('Baseline sensitivity: the same clusters and matched behaviors, with two starting generations')
    fig.tight_layout(rect=(0, .06, 1, .95))
    for suffix in ('png', 'pdf'):
        fig.savefig(root / f'baseline_sensitivity.{suffix}', dpi=180)
    plt.close(fig)


def number(value):
    return float(value) if value not in ('', None) else None


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args(argv)
    root = args.directory
    plan = json.loads((root / 'plan.json').read_text())
    horizon = plan.get('common_final_generation', 4)
    rows = read(root / 'endpoint_changes.csv')
    primary = [r for r in rows if int(r['horizon']) == horizon and r['min_cluster_size'] == '5' and r['weighting'] == 'population']

    def count(selected, within='within_cluster_delta', between='between_centroids_delta'):
        eligible = [r for r in selected if number(r.get(within)) is not None and number(r.get(between)) is not None]
        within_rows = [r for r in selected if number(r.get(within)) is not None]
        between_rows = [r for r in selected if number(r.get(between)) is not None]
        return {'eligible_runs': len(eligible),
                'within_eligible_runs': len(within_rows), 'between_eligible_runs': len(between_rows),
                'tightening': sum(number(r[within]) < -1e-8 for r in within_rows),
                'separation': sum(number(r[between]) > 1e-8 for r in between_rows),
                'both': sum(number(r[within]) < -1e-8 and number(r[between]) > 1e-8 for r in eligible)}

    counts = {'primary': count(primary), 'sensitivity': []}
    counts['by_subset_kind'] = {kind: count([r for r in primary if r['kind'] == kind])
                                for kind in ('random', 'narrow', 'broad')}
    latest_rows = [r for spec in plan['runs'] for r in rows
                   if r['run'] == spec['run'] and int(r['horizon']) == spec['latest_generation']
                   and r['min_cluster_size'] == '5' and r['weighting'] == 'population']
    counts['latest_available'] = count(latest_rows)
    for weighting in ('population', 'unique_prompts'):
        for minimum in ('3', '5', '8'):
            selected = [r for r in rows if int(r['horizon']) == horizon and r['min_cluster_size'] == minimum and r['weighting'] == weighting]
            counts['sensitivity'].append({'weighting': weighting, 'min_cluster_size': int(minimum), **count(selected),
                'persistent_clusters': count(selected, 'matched_within_delta', 'matched_between_delta')})
    # A stricter joint test uses exactly the same behaviors for both outcomes.
    all_cases = read(root / 'case_metrics.csv')
    cases = [r for r in all_cases if int(r['horizon']) == horizon and r['min_cluster_size'] == '5' and r['weighting'] == 'population']
    # Exclude generation 0 to check whether duplicate initial prompts mask a later trend.
    later = []
    metrics = ('pairwise', 'nearest_neighbor', 'within_cluster', 'between_centroids',
               'duplicate_fraction', 'unclustered_fraction', 'active_clusters')
    for weighting in ('population', 'unique_prompts'):
        for spec in plan['runs']:
            selected = [dict(r) for r in all_cases if r['run'] == spec['run'] and int(r['horizon']) == horizon
                        and r['min_cluster_size'] == '5' and r['weighting'] == weighting]
            for row in selected:
                row['generation'] = int(row['generation'])
                for metric in metrics:
                    row[metric] = number(row[metric])
            later.append({'run': spec['run'], 'weighting': weighting, **paired_changes(selected, 1, horizon)})
    csv_output(root / f'generation_1_to_{horizon}_changes.csv', later)
    counts[f'generation_1_to_{horizon}'] = {weighting: count([r for r in later if r['weighting'] == weighting])
                                     for weighting in ('population', 'unique_prompts')}
    joint = []
    for spec in plan['runs']:
        by = {(int(r['generation']), r['case']): r for r in cases if r['run'] == spec['run']}
        eligible = [case for gen, case in by if gen == 0 and (horizon, case) in by and
                    all(number(by[(g, case)][metric]) is not None
                        for g in (0, horizon) for metric in ('within_cluster', 'between_centroids'))]
        item = {'run': spec['run'], 'joint_eligible_cases': len(eligible)}
        for metric in ('within_cluster', 'between_centroids'):
            item[metric + '_delta'] = (sum(number(by[(horizon, c)][metric])-number(by[(0, c)][metric])
                                                  for c in eligible)/len(eligible) if eligible else None)
        item['both_direction_cases'] = sum(number(by[(horizon, c)]['within_cluster']) < number(by[(0, c)]['within_cluster']) - 1e-8
                                          and number(by[(horizon, c)]['between_centroids']) > number(by[(0, c)]['between_centroids']) + 1e-8
                                          for c in eligible)
        joint.append(item)
    csv_output(root / 'joint_case_changes.csv', joint)
    counts['joint_cases'] = count(joint)
    manifest = json.loads((Path(plan['campaign']) / 'experiment.json').read_text())
    memberships = {r['id']: set(r['behavior_ids']) for r in manifest['subsets']}
    ids = [s['subset'] for s in plan['runs']]
    overlap = [{'subset_a': a, 'subset_b': b, 'shared_behaviors': len(memberships[a] & memberships[b])}
               for a in ids for b in ids]
    csv_output(root / 'subset_overlap.csv', overlap)
    off_diagonal = [r for r in overlap if r['subset_a'] < r['subset_b']]
    closest = max(off_diagonal, key=lambda r: r['shared_behaviors'])
    counts['run_count'] = len(plan['runs'])
    (root / 'findings.json').write_text(json.dumps(counts, indent=2) + '\n')
    n = len(primary)
    lines = ['Clustering across starting behavior subsets',
             f"Data cutoff: {plan['created_utc']}",
             f'{n} eligible runs; primary comparison is generation 0 to generation {horizon}.',
             'All runs share one frozen initial population of 24 strategies and evolution seed 42.',
             'The varying initial condition is the 30-behavior selection/evaluation subset.', '',
             'PRIMARY FINDINGS',
             f"Within-cluster distance decreases in {counts['primary']['tightening']}/{counts['primary']['within_eligible_runs']} eligible runs.",
             f"Between-centroid distance increases in {counts['primary']['separation']}/{counts['primary']['between_eligible_runs']} eligible runs.",
             f"Both directions occur in {counts['primary']['both']}/{counts['primary']['eligible_runs']} eligible runs.",
             f"Requiring the same eligible cases for both outcomes gives both directions in {counts['joint_cases']['both']}/{counts['joint_cases']['eligible_runs']} runs.", '',
             'PER-RUN CHANGES (cosine distance; final minus initial)',
             'run | within-cluster change (eligible behaviors) | between-centroid change (eligible behaviors)']
    for row in primary:
        def show(metric):
            v = number(row[metric+'_delta'])
            return 'unavailable' if v is None else f"{v:+.5f} ({row[metric+'_cases']}/30)"
        lines.append(f"{row['run'].replace('_repeat_01', '')} | {show('within_cluster')} | {show('between_centroids')}")
    lines += ['', 'SENSITIVITY (both directions / eligible runs)']
    for item in counts['sensitivity']:
        p = item['persistent_clusters']
        lines.append(f"Minimum cluster size {item['min_cluster_size']}, {item['weighting']}: {item['both']}/{item['eligible_runs']}; persistent cluster identities: {p['both']}/{p['eligible_runs']}.")
    lines += ['', 'BY STARTING BEHAVIOR DISTRIBUTION']
    for kind, item in counts['by_subset_kind'].items():
        lines.append(f"{kind}: tightening {item['tightening']}/{item['within_eligible_runs']}; separation {item['separation']}/{item['between_eligible_runs']}; both {item['both']}/{item['eligible_runs']}.")
    initial_duplicates = [number(r['duplicate_fraction']) for r in cases if r['generation'] == '0']
    initial_duplicate_mean = sum(initial_duplicates)/len(initial_duplicates)
    lines += ['', 'GENERATION-0 AND LATER-TRAJECTORY CHECK',
              f'Generation 0 has {initial_duplicate_mean:.1%} duplicate prompts on average within behavior cases.',
              'Exact-prompt deduplication above checks whether this alone explains the measured changes.',
              f'The following comparison uses generation 1 to generation {horizon}, keeping the original generation 0–{horizon} cluster definitions fixed.']
    for weighting, item in counts[f'generation_1_to_{horizon}'].items():
        lines.append(f"{weighting}: tightening {item['tightening']}/{item['within_eligible_runs']}; separation {item['separation']}/{item['between_eligible_runs']}; both {item['both']}/{item['eligible_runs']}.")
    item = counts['latest_available']
    lines += ['', 'LATEST AVAILABLE GENERATIONS',
              f'Use the common generation {horizon} horizon for matched comparisons if latest endpoints differ.',
              f"Generation 0 to latest: tightening {item['tightening']}/{item['within_eligible_runs']}; separation {item['separation']}/{item['between_eligible_runs']}; both {item['both']}/{item['eligible_runs']}."]
    for row in latest_rows:
        lines.append(f"{row['run'].replace('_repeat_01', '')}, generation {row['horizon']}: within {number(row['within_cluster_delta']):+.5f}; between {number(row['between_centroids_delta']):+.5f}.")
    eligible = counts['primary']['eligible_runs']
    interpretation = ('The joint pattern repeats across every included subset in this analysis.'
                      if counts['primary']['both'] == n else
                      'The joint pattern repeats across assessable subsets, but some included subsets have undefined cluster metrics.'
                      if eligible and counts['primary']['both'] == eligible else
                      f'The joint pattern does not repeat uniformly across assessable behavior subsets through generation {horizon}.'
                      if eligible else
                      'Cluster metrics are undefined for this comparison; the joint pattern cannot be assessed.')
    lines += ['', 'INTERPRETATION', interpretation,
              'Use the sensitivity checks and original-space distances to interpret the opacity maps.',
              'Subsets overlap and there is only one evolution run per subset, so this is descriptive evidence, not an independent-replicate significance test.',
              f"The most overlapping included pair is {closest['subset_a']} / {closest['subset_b']}: {closest['shared_behaviors']}/30 behaviors are shared.",
              'This analysis does not test different initial strategy populations or evolution seeds.',
              'These tables describe evolution/evaluation behaviors. Analyze held-out observations separately to assess transfer.',
              'Cluster assignments are fit retrospectively within each run and behavior. Fixed-horizon and latest-horizon assignments may differ.',
              'Endpoint comparisons use the same eligible behaviors at both endpoints for each metric; the stricter joint comparison requires both metrics for the same behaviors.',
              'Prompt duplication, changing population size, cluster births/deaths, and undefined metrics are reported rather than treated as zero.', '',
              'EXCLUSIONS']
    lines += [f"{r['run']}: {r['reason']} (latest generation {r['latest_generation']})." for r in plan['excluded']]
    lines += ['', 'PLOTS AND TABLES',
              'clustering_changes.png/pdf: endpoint changes for every eligible run.',
              f'clustering_trajectories.png/pdf: comparable generation 0–{horizon} trends.',
              f'baseline_sensitivity.png/pdf: matched generation 0–{horizon} versus generation 1–{horizon} changes.',
              'generation_opacity_overview.png/pdf: one representative case per run; earlier generations fade.',
              f'generation_opacity_atlas.pdf: initial, middle, and generation-{horizon} views on shared axes.',
              'clustering_explorer.html: standalone interactive maps for every run/behavior, with playback and a generation slider.',
              'endpoint_changes.csv: primary and latest-generation changes, clustering sensitivity, deduplication, and persistent-cluster checks.',
              'case_metrics.csv, joint_case_changes.csv, subset_overlap.csv, diagnostics.csv: coverage and case-level audit tables.']
    (root / 'report.txt').write_text('\n'.join(lines) + '\n')
    baseline_plot(root, primary, later, horizon)
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    main()
