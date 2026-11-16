import random
import pytest
from src.difficulty_selection import population_scores, tournament, validated_vector


def test_hand_calculation_and_rank_reversal():
    rows = {'common': {'easy': 1., 'hard': 0.}, 'specialist': {'easy': 0., 'hard': .8},
            'other1': {'easy': 1., 'hard': 0.}, 'other2': {'easy': 1., 'hard': 0.}}
    result = population_scores(rows)
    assert result['weights'] == pytest.approx({'easy': .25, 'hard': .8})
    assert result['plain']['common'] > result['plain']['specialist']
    assert result['weighted']['specialist'] == pytest.approx(.64 / 1.05)
    assert result['weighted']['specialist'] > result['weighted']['common']


@pytest.mark.parametrize('value', [0., .5, 1.])
def test_uniform_scores_and_fallback(value):
    result = population_scores({'a': {'x': value, 'y': value}, 'b': {'x': value, 'y': value}})
    assert result['weighted'] == result['plain']
    assert result['uniform_fallback'] == (value == 1.)


def test_zero_behavior_does_not_rank_parents():
    result = population_scores({'a': {'zero': 0., 'other': .2}, 'b': {'zero': 0., 'other': .8}})
    assert result['weights']['zero'] == 1
    assert result['weighted']['b'] > result['weighted']['a']


@pytest.mark.parametrize('rows', [{}, {'a': {}}, {'a': {'x': 0}, 'b': {'y': 0}},
                                  {'a': {'x': float('nan')}}, {'a': {'x': -1}}, {'a': {'x': 1.1}}])
def test_rejects_invalid_vectors(rows):
    with pytest.raises(ValueError):
        population_scores(rows)


def test_weights_refresh_without_mutating_input():
    rows = {'a': {'x': .2}, 'b': {'x': .4}}
    first = population_scores(rows)
    rows['b']['x'] = .8
    second = population_scores(rows)
    assert first['weights'] != second['weights']
    assert rows == {'a': {'x': .2}, 'b': {'x': .8}}


def test_rng_roundtrip_replays_selection():
    rng = random.Random(42)
    scores = {'a': .5, 'b': .5, 'c': .8}
    tournament(list(scores), scores, 7, rng=rng)
    state = rng.getstate()
    expected = tournament(list(scores), scores, 20, rng=rng)
    resumed = random.Random()
    resumed.setstate(state)
    assert tournament(list(scores), scores, 20, rng=resumed) == expected


def test_ties_are_random_and_all_counts_supported():
    scores = {'a': .5, 'b': .5}
    picked = tournament(list(scores), scores, 100, rng=random.Random(1))
    assert set(picked) == set(scores)
    assert len(picked) == 100


def test_sidecar_rejects_failures_and_missing_cases():
    row = {'behavior': 'x', 'score': .4, 'parse_ok': True}
    assert validated_vector({'behaviors': [row]}, ['x']) == {'x': .4}
    for data, panel in [({'behaviors': [dict(row, error='timeout')]}, ['x']),
                        ({'behaviors': [row, row]}, ['x']), ({'behaviors': [row]}, ['x', 'y'])]:
        with pytest.raises(ValueError):
            validated_vector(data, panel)
