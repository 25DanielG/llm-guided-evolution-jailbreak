"""Inspectable per-generation diagnostics; no inference and no score changes."""
import collections
import hashlib
import json


def summarize_generation(generation, population, archive, result_root, candidates_path):
    all_rows = [json.loads(line) for line in candidates_path.open() if line.strip()]
    rows = [r for r in all_rows if r['generation'] == generation]
    ids = {r['candidate_id'] for r in all_rows}
    strategies = {}
    for gene_id in ids:
        payload = json.loads((result_root / f'{gene_id}_results.json').read_text())
        strategies[gene_id] = payload['rendered_strategy_sha256']
    new_ids = {r['candidate_id'] for r in rows}
    return {
        'generation': generation, 'population_size': len(population),
        'evaluated_genes': len(new_ids), 'cumulative_evaluated_genes': len(ids),
        'behavior_records': len(rows), 'cumulative_behavior_records': len(all_rows),
        'unique_rendered_strategies': len({strategies[g] for g in new_ids}),
        'cumulative_unique_rendered_strategies': len(set(strategies.values())),
        'target_errors': sum(bool(r.get('target_error_type')) for r in rows),
        'target_error_types': dict(collections.Counter(r['target_error_type'] for r in rows
                                                      if r.get('target_error_type'))),
        'target_cache_hits': sum(r.get('target_cached', False) for r in rows),
        'target_finish_reasons': dict(collections.Counter(str(r.get('target_finish_reason')) for r in rows)),
        'target_completion_tokens': sum(r.get('target_usage', {}).get('completion_tokens', 0) for r in rows),
        'judge_errors': sum(not r['in_loop'].get('parse_ok', False) and not r.get('target_error_type') for r in rows),
        'threshold_successes': sum(r['in_loop']['score'] >= 0.5 for r in rows),
        'verified_successes': sum(bool(r.get('verification') and r['verification']['verified_success']) for r in rows),
        'invalid_fitness_count': 0,  # caller must validate all state before writing
        'occupied_cells': len(archive), 'capacity': archive.capacity,
        'archive': [{'gene_id': e.gene_id, 'fitness': list(e.fitness), 'quality': list(e.quality),
                     'descriptor': list(e.descriptor)} for e in archive.elites()],
    }
