"""Synthetic checks of matched-case and persistent-cluster comparisons."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.scripts.clustering_robustness import paired_changes, common_cluster_change, analyze_cohorts


def test_endpoint_changes_do_not_compare_different_case_sets():
    metrics = ('pairwise', 'nearest_neighbor', 'within_cluster', 'between_centroids',
               'duplicate_fraction', 'unclustered_fraction', 'active_clusters')
    rows = []
    for generation, case, value in [(0, 'shared', .4), (4, 'shared', .3),
                                     (0, 'initial_only', .01), (4, 'final_only', .99)]:
        rows.append({'generation': generation, 'case': case, **{m: value for m in metrics}})
    result = paired_changes(rows, 0, 4)
    assert result['within_cluster_delta'] == pytest.approx(-.1)
    assert result['within_cluster_cases'] == 1
    assert result['within_cluster_decreasing_cases'] == 1


def test_persistent_cluster_comparison_excludes_new_cluster_births():
    vectors = {'a': np.array([1., -.3]), 'b': np.array([1., .3]),
               'c': np.array([-1., -.3]), 'd': np.array([-1., .3]),
               'e': np.array([1., -.01]), 'f': np.array([1., .01]),
               'g': np.array([-1., -.01]), 'h': np.array([-1., .01]),
               'new': np.array([0., 1.])}
    vectors = {k: v / np.linalg.norm(v) for k, v in vectors.items()}
    labels = {k: i for i, keys in enumerate(('abef', 'cdgh', ['new'])) for k in keys}
    first = [{'hash': k} for k in 'abcd']
    last = [{'hash': k} for k in ['e', 'f', 'g', 'h', 'new']]
    result = common_cluster_change(first, last, vectors, labels)
    assert result['common_clusters'] == 2
    assert result['matched_within_delta'] < 0
    assert result['matched_between_delta'] == pytest.approx(0)


def test_undefined_persistent_cluster_metrics_remain_missing():
    result = common_cluster_change([{'hash': 'a'}], [{'hash': 'b'}],
                                   {'a': np.array([1., 0.]), 'b': np.array([0., 1.])},
                                   {'a': 0, 'b': 1})
    assert result['matched_within_delta'] is None
    assert result['matched_between_delta'] is None


@pytest.mark.parametrize('horizon', [4, 10])
def test_full_measurement_detects_tightening_and_separation(tmp_path, horizon):
    vectors, cohorts = {}, {}
    for generation in range(horizon + 1):
        # Five distinct states, held for longer when testing a ten-generation horizon.
        # Adding intermediate points would change the retrospective density clusters.
        step = round(4 * generation / horizon)
        center, width = .6 + step*.125, .15 - step*.03
        entries = []
        for index, angle in enumerate([sign*center+delta for sign in (-1, 1) for delta in (-width, 0, width)]):
            key = f'{generation}:{index}'
            vectors[key] = np.array([np.cos(angle), np.sin(angle)])
            entries.append({'hash': key, 'candidate': key})
        cohorts[('synthetic', generation, 'case')] = {'population_size': 6, 'rows': entries}
    plan = {'common_final_generation': horizon,
            'runs': [{'run': 'synthetic', 'kind': 'random', 'latest_generation': horizon}]}
    summaries, _ = analyze_cohorts(plan, cohorts, vectors, tmp_path)
    primary = next(r for r in summaries if r['weighting'] == 'population' and r['min_cluster_size'] == 5)
    assert {r['horizon'] for r in summaries} == {horizon}
    assert primary['within_cluster_delta'] < 0
    assert primary['between_centroids_delta'] > 0
    assert primary['matched_within_delta'] < 0
    assert primary['matched_between_delta'] > 0
    assert primary['both_directions']
