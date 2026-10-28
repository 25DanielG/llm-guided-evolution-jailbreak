"""Provenance and matched-case checks for the finished-campaign pipeline."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.experiments.coverage import digest
from analysis.scripts.full_campaign_analysis import CachedOnlyEmbedder, matched_trajectories


def test_cached_embedder_requires_exact_provenance_and_never_recomputes(tmp_path):
    metadata = {'model': 'frozen-model', 'revision': 'sha256:123', 'pooling': 'chunks', 'device': 'cpu'}
    config = {k: v for k, v in metadata.items() if k != 'device'}
    text = 'a harmless synthetic prompt'
    path = tmp_path / (digest(json.dumps(config, sort_keys=True) + '\n' + text) + '.json')
    vector = np.zeros(768); vector[0] = 2
    entry = {'config': config, 'prompt_hash': digest(text), 'vector': vector.tolist(), 'chunk_count': 1}
    path.write_text(json.dumps(entry))
    embedder = CachedOnlyEmbedder(tmp_path, metadata)
    result, chunks = embedder.embed(text)
    assert chunks == 1 and np.linalg.norm(result) == pytest.approx(1)
    assert len(embedder.vectors) == 1
    entry['config']['revision'] = 'wrong'
    path.write_text(json.dumps(entry))
    with pytest.raises(ValueError, match='provenance mismatch'):
        CachedOnlyEmbedder(tmp_path, metadata).embed(text)
    with pytest.raises(FileNotFoundError):
        CachedOnlyEmbedder(tmp_path, metadata).embed('a missing prompt')


def test_matched_trajectories_exclude_a_behavior_missing_in_the_middle():
    rows = []
    for g in range(3):
        for case in ['complete', 'missing']:
            row = {'min_cluster_size': '5', 'run': '/runs/example', 'split': 'heldout',
                   'generation': str(g), 'behavior_id': case}
            for metric in ['pairwise', 'nearest_neighbor', 'within_cluster', 'between_centroids']:
                row[metric] = str(.1 + .02*g if case == 'complete' else .8 - .2*g)
            if g == 1 and case == 'missing':
                row['within_cluster'] = ''
            rows.append(row)
    result = matched_trajectories(rows, 2)
    within = [r for r in result if r['metric'] == 'within_cluster']
    assert [r['eligible_cases'] for r in within] == [1, 1, 1]
    assert within[-1]['delta'] == pytest.approx(.04)
    assert within[-1]['initial_pairwise_distance'] == pytest.approx(.1)
    assert within[-1]['delta_fraction_of_initial_pairwise'] == pytest.approx(.4)
    assert next(r for r in result if r['metric'] == 'pairwise')['eligible_cases'] == 2
