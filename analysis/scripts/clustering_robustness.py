#!/usr/bin/env python3
"""Read-only, case-matched clustering comparison of completed run snapshots."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from analysis.experiments.coverage import digest, distances, write_json
from analysis.experiments.runtime import latest_complete
from analysis.experiments.report import cluster_labels, csv_output, mean_present, measure


def inventory(campaign, common_final_generation=4):
    """Freeze eligibility and generation horizons before the analysis queues."""
    rows, excluded = [], []
    for root in sorted((campaign / 'runs').iterdir()):
        complete = latest_complete(root)
        if complete is None or complete < common_final_generation:
            excluded.append({'run': root.name, 'reason': f'fewer_than_{common_final_generation}_completed_transitions',
                             'latest_generation': complete})
            continue
        config = json.loads((root / 'run.json').read_text())
        rows.append({'run': root.name, 'path': str(root.resolve()), 'kind': config['subset']['kind'],
                     'subset': config['subset']['id'], 'latest_generation': complete,
                     'initial_hash': config['initial_hash'], 'evolution_seed': config['seed'],
                     'config_hash': digest((root / 'run.json').read_bytes()),
                     'snapshot_hashes': {str(g): digest((root / 'snapshots' / f'{g:04d}.json').read_bytes())
                                         for g in range(complete + 1)}})
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'campaign': str(campaign.resolve()),
            'runs': rows, 'excluded': excluded, 'common_final_generation': common_final_generation}


def load_prompts(plan):
    """Use retained members and their recorded prompts, never execute strategies."""
    cohorts, prompts, diagnostics, inputs = {}, {}, [], {}
    for spec in plan['runs']:
        root = Path(spec['path'])
        config_path = root / 'run.json'
        assert digest(config_path.read_bytes()) == spec['config_hash'], 'Run config changed'
        config = json.loads(config_path.read_text())
        observations = {}
        for generation in range(spec['latest_generation'] + 1):
            path = root / 'snapshots' / f'{generation:04d}.json'
            assert digest(path.read_bytes()) == spec['snapshot_hashes'][str(generation)], 'Snapshot changed'
            inputs[str(path)] = spec['snapshot_hashes'][str(generation)]
            snap = json.loads(path.read_text())
            assert snap['run_hash'] == spec['config_hash'], 'Snapshot/config mismatch'
            for case in config['subset']['behavior_ids']:
                cohorts[(spec['run'], generation, case)] = {'population_size': len(snap['members']), 'rows': []}
            for member in snap['members']:
                ref = member.get('observation')
                if not ref:
                    diagnostics.append({'run': spec['run'], 'generation': generation,
                                        'candidate': member['id'], 'status': 'missing_observation'})
                    continue
                if ref not in observations:
                    raw = (root / ref).read_bytes()
                    inputs[str(root / ref)] = digest(raw)
                    observations[ref] = json.loads(raw)
                obs = observations[ref]
                assert obs['strategy_hash'] == member['strategy_hash'], 'Strategy hash mismatch'
                seen = set()
                for record in obs['records']:
                    case = record['behavior_id']
                    assert case not in seen and case in config['subset']['behavior_ids'], 'Case mismatch'
                    seen.add(case)
                    if record['status'] != 'ok':
                        diagnostics.append({'run': spec['run'], 'generation': generation,
                                            'candidate': member['id'], 'case': case, 'status': record['status']})
                        continue
                    text, key = record['prompt_text'], record['prompt_hash']
                    assert digest(text) == key, 'Prompt hash mismatch'
                    prompts[key] = text
                    cohorts[(spec['run'], generation, case)]['rows'].append(
                        {'hash': key, 'candidate': member['id'], 'strategy_hash': member['strategy_hash']})
    return cohorts, prompts, diagnostics, inputs


def embed_prompts(prompts, cache, model, device='cpu', batch_size=32):
    """Batch across prompts while preserving the established chunk-pooling method."""
    from analysis.utils.prompt_diversity import SentenceBertEmbedder, token_chunks, pooled_embedding
    import torch
    torch.set_num_threads(min(8, int(os.getenv('SLURM_CPUS_PER_TASK', '8'))))
    embedder = SentenceBertEmbedder(cache, model, device=device, batch_size=batch_size)
    archive = Path(cache) / 'vectors.npz'
    metadata_path = Path(cache) / 'vectors_metadata.json'
    vectors = {}
    if archive.exists():
        assert json.loads(metadata_path.read_text()) == embedder.cache_config, 'Embedding cache configuration changed'
        with np.load(archive) as saved:
            vectors = {key: saved[key] for key in saved.files}
    missing = [key for key in prompts if key not in vectors]
    print(f'Embedding {len(missing)} new unique prompts; {len(vectors)} cached', flush=True)
    # Bounded chunks avoid holding the complete tokenized corpus in memory.
    for start in range(0, len(missing), 256):
        keys = missing[start:start+256]
        chunks, owners, lengths = [], [], []
        for key in keys:
            for chunk in token_chunks(embedder.tokenizer, prompts[key], embedder.max_length):
                chunks.append(chunk)
                owners.append(key)
                lengths.append(len(chunk))
        outputs = defaultdict(list)
        weights = defaultdict(list)
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset:offset+batch_size]
            features = [embedder.tokenizer.prepare_for_model(ids, add_special_tokens=True, truncation=False,
                                                            return_attention_mask=True) for ids in batch]
            data = embedder.tokenizer.pad(features, padding=True, return_tensors='pt')
            data = {k: v.to(embedder.model.device) for k, v in data.items()}
            with torch.inference_mode():
                result = embedder.model(data)['sentence_embedding'].detach().cpu().float().numpy()
            for index, vector in enumerate(result, offset):
                outputs[owners[index]].append(vector)
                weights[owners[index]].append(lengths[index])
        for key in keys:
            vectors[key] = pooled_embedding(outputs[key], weights[key])
        print(f'Embedded {min(start+256, len(missing))}/{len(missing)} prompts', flush=True)
        # Save periodically so interruption never requires another full embedding pass.
        if (start // 256) % 8 == 0 or start+256 >= len(missing):
            temporary = archive.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **vectors)
            temporary.replace(archive)
            write_json(metadata_path, embedder.cache_config)
    return vectors, embedder.metadata


def paired_changes(case_rows, first, last):
    """Use the same eligible cases at both endpoints, separately for each metric."""
    by = {(r['generation'], r['case']): r for r in case_rows}
    result = {}
    for metric in ('pairwise', 'nearest_neighbor', 'within_cluster', 'between_centroids',
                   'duplicate_fraction', 'unclustered_fraction', 'active_clusters'):
        pairs = [(by[(first, case)][metric], by[(last, case)][metric])
                 for case in sorted({r['case'] for r in case_rows})
                 if (first, case) in by and (last, case) in by
                 and by[(first, case)][metric] is not None and by[(last, case)][metric] is not None]
        a, b = (float(np.mean([p[i] for p in pairs])) if pairs else None for i in (0, 1))
        result[metric + '_initial'] = a
        result[metric + '_final'] = b
        result[metric + '_delta'] = b-a if pairs else None
        result[metric + '_cases'] = len(pairs)
        result[metric + '_decreasing_cases'] = sum(y < x - 1e-8 for x, y in pairs)
        result[metric + '_increasing_cases'] = sum(y > x + 1e-8 for x, y in pairs)
    return result


def common_cluster_change(first, last, vectors, labels):
    """Hold cluster identities fixed, excluding births/deaths from endpoint changes."""
    buckets = []
    for entries in (first, last):
        groups = defaultdict(list)
        for entry in entries:
            label = labels[entry['hash']]
            if label >= 0:
                groups[label].append(vectors[entry['hash']])
        buckets.append(groups)
    common = sorted(set(buckets[0]) & set(buckets[1]))
    output = {'common_clusters': len(common), 'matched_within_delta': None, 'matched_between_delta': None}
    # Each persistent cluster receives equal weight; both endpoints need >=2 members.
    within = []
    centers = [[], []]
    for label in common:
        groups = [np.asarray(b[label]) for b in buckets]
        for i, group in enumerate(groups):
            center = group.mean(axis=0)
            centers[i].append(center / np.linalg.norm(center))
        if min(map(len, groups)) >= 2:
            values = [distances(g)[np.triu_indices(len(g), 1)].mean() for g in groups]
            within.append(values[1]-values[0])
    if within:
        output['matched_within_delta'] = float(np.mean(within))
    if len(common) >= 2:
        means = [distances(c)[np.triu_indices(len(common), 1)].mean() for c in centers]
        output['matched_between_delta'] = float(means[1]-means[0])
    return output


def analyze_cohorts(plan, cohorts, vectors, output):
    summaries, all_cases, assignments = [], [], []
    for spec in plan['runs']:
        rid = spec['run']
        for horizon in sorted({plan.get('common_final_generation', 4), spec['latest_generation']}):
            cases = sorted({key[2] for key in cohorts if key[0] == rid})
            for minimum in (3, 5, 8):
                for weighting in ('population', 'unique_prompts'):
                    rows, matched = [], []
                    for case in cases:
                        by_gen = {}
                        for gen in range(horizon+1):
                            entries = cohorts[(rid, gen, case)]['rows']
                            if weighting == 'unique_prompts':
                                entries = list({e['hash']: e for e in entries}.values())
                            by_gen[gen] = entries
                        keys = sorted({e['hash'] for entries in by_gen.values() for e in entries})
                        labels = dict(zip(keys, map(int, cluster_labels([vectors[k] for k in keys], minimum)))) if keys else {}
                        if weighting == 'population':
                            assignments.extend({'run': rid, 'horizon': horizon, 'case': case,
                                                'min_cluster_size': minimum, 'prompt_hash': key, 'cluster': label}
                                               for key, label in labels.items())
                        for gen, entries in by_gen.items():
                            stats = measure([vectors[e['hash']] for e in entries],
                                            [labels[e['hash']] for e in entries],
                                            cohorts[(rid, gen, case)]['population_size'], [e['hash'] for e in entries])
                            rows.append({'run': rid, 'kind': spec['kind'], 'horizon': horizon,
                                         'min_cluster_size': minimum, 'weighting': weighting,
                                         'case': case, 'generation': gen, **stats})
                        matched.append(common_cluster_change(by_gen[0], by_gen[horizon], vectors, labels))
                    summary = {'run': rid, 'kind': spec['kind'], 'horizon': horizon,
                               'min_cluster_size': minimum, 'weighting': weighting,
                               **paired_changes(rows, 0, horizon)}
                    for metric in ('matched_within_delta', 'matched_between_delta'):
                        summary[metric] = mean_present([m[metric] for m in matched])
                        summary[metric + '_cases'] = sum(m[metric] is not None for m in matched)
                    summary['both_directions'] = (summary['within_cluster_delta'] < 0 and summary['between_centroids_delta'] > 0
                                                  if summary['within_cluster_delta'] is not None and summary['between_centroids_delta'] is not None else None)
                    summaries.append(summary)
                    all_cases.extend(rows)
            print('Measured', rid, 'through generation', horizon, flush=True)
    csv_output(output / 'case_metrics.csv', all_cases)
    csv_output(output / 'endpoint_changes.csv', summaries)
    csv_output(output / 'cluster_assignments.csv', assignments)
    return summaries, all_cases


def plots(output, summaries, rows, horizon=4):
    os.environ['MPLCONFIGDIR'] = str(output / 'matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'random': '#3875b6', 'narrow': '#cc5656', 'broad': '#35946a'}
    primary = [r for r in summaries if r['horizon'] == horizon and r['min_cluster_size'] == 5 and r['weighting'] == 'population']
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, metric, title in zip(axes, ('within_cluster', 'between_centroids'),
                               ('Within-cluster distance: negative = tighter', 'Between-centroid distance: positive = farther apart')):
        for i, row in enumerate(primary):
            value = row[metric+'_delta']
            if value is not None:
                ax.scatter(value, i, color=colors[row['kind']], s=65)
                ax.annotate(f"{row[metric+'_cases']} cases", (value, i), xytext=(5, 6), textcoords='offset points', fontsize=8)
        ax.axvline(0, color='#555', linewidth=1)
        ax.set(title=title, xlabel=f'Generation {horizon} minus generation 0; cosine distance')
        ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(range(len(primary)), [r['run'].replace('_repeat_01', '') for r in primary])
    fig.suptitle(f'Matched generation 0–{horizon} comparison across starting behavior subsets')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'clustering_changes.{suffix}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, metric in zip(axes.flat, ('within_cluster', 'between_centroids', 'pairwise', 'nearest_neighbor')):
        for spec in primary:
            values = [r for r in rows if r['run'] == spec['run'] and r['horizon'] == horizon
                      and r['min_cluster_size'] == 5 and r['weighting'] == 'population']
            # Keep case eligibility constant for the whole trajectory.
            cases = [case for case in {r['case'] for r in values}
                     if sum(r['case'] == case and r[metric] is not None for r in values) == horizon + 1]
            trend = [mean_present([r[metric] for r in values if r['generation'] == g and r['case'] in cases]) for g in range(horizon + 1)]
            ax.plot(range(horizon + 1), [np.nan if v is None else v for v in trend], marker='.',
                    color=colors[spec['kind']], alpha=.7, label=spec['run'].replace('_repeat_01', ''))
        ax.set(title=metric.replace('_', ' '), xlabel='Generation', ylabel='Cosine distance', xticks=range(horizon + 1))
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, ncols=2)
    fig.suptitle('Case-matched trajectories; clustering measured in original embedding space')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'clustering_trajectories.{suffix}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inventory', 'analyze'))
    parser.add_argument('--campaign', type=Path, default=ROOT / 'results/coverage/overnight')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--common-final-generation', type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.campaign.resolve() == output or args.campaign.resolve() in output.parents:
        raise ValueError('Analysis output must be outside raw experiment directories')
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / 'plan.json'
    if args.command == 'inventory':
        if plan_path.exists():
            raise ValueError('Use a new output directory for a new analysis cutoff')
        if args.common_final_generation < 1:
            parser.error('common-final-generation must be positive')
        plan = inventory(args.campaign.resolve(), args.common_final_generation)
        write_json(plan_path, plan)
        print(json.dumps(plan, indent=2))
        return
    plan = json.loads(plan_path.read_text())
    cohorts, prompts, diagnostics, inputs = load_prompts(plan)
    print(f"Loaded {len(cohorts)} cohorts and {len(prompts)} unique prompts", flush=True)
    model = json.loads((Path(plan['campaign']) / 'experiment.json').read_text())['embedding']['model']
    vectors, embedding_metadata = embed_prompts(prompts, output / 'embedding_cache', model, args.device, args.batch_size)
    summaries, rows = analyze_cohorts(plan, cohorts, vectors, output)
    csv_output(output / 'diagnostics.csv', diagnostics)
    write_json(output / 'metadata.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
               'analysis_source_hash': digest(Path(__file__).read_bytes()), 'input_hashes': inputs,
               'embedding': embedding_metadata,
               'method': 'Per-run, per-behavior retrospective HDBSCAN in original cosine space; unique vectors fitted; population multiplicity restored; same eligible cases at endpoints.',
               'sensitivity': f"Minimum cluster sizes 3,5,8; exact-prompt deduplication; persistent cluster identities; common horizon 0–{plan['common_final_generation']} and each run latest completed generation.",
               'limitations': ['All campaign runs share the same initial strategy population and evolution seed.',
                               'One evolution run per subset; behaviors and prompt pairs are not independent experiment replicates.',
                               'Evolution cases differ across subsets; most shared held-out evaluations are not yet available.',
                               'Retrospective cluster definitions depend on analysis horizon.',
                               'Partial-run, descriptive analysis; no causal claim or significance test.']})
    plots(output, summaries, rows, plan['common_final_generation'])
    print('Analysis complete:', output, flush=True)


if __name__ == '__main__':
    main()
