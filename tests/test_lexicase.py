import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lexicase import case_ids, lexicase_select, mad_epsilons


def _select(pop, k, scores, **kw):
    return lexicase_select(pop, k, key_fn=lambda i: i, case_scores=scores,
                           rng=random.Random(0), **kw)


def test_specialist_survives_but_dominated_never_selected():
    scores = {
        "generalist": {"a": 0.6, "b": 0.6, "c": 0.6},
        "specialist": {"a": 1.0, "b": 0.0, "c": 0.0},
        "dominated": {"a": 0.5, "b": 0.5, "c": 0.5},
    }
    picks = _select(list(scores), 300, scores, epsilon=0.0)
    assert "dominated" not in picks
    assert {"generalist", "specialist"} <= set(picks)


def test_epsilon_merges_near_ties():
    scores = {"x": {"a": 1.0, "b": 0.0}, "y": {"a": 0.99, "b": 1.0}}
    # exact: x wins whenever case a comes first; eps=0.05 makes y tie on a
    assert set(_select(list(scores), 200, scores, epsilon=0.0)) == {"x", "y"}
    assert set(_select(list(scores), 200, scores, epsilon=0.05)) == {"y"}


def test_mad_epsilon_and_case_intersection():
    scores = {"p": {"a": 0.0, "b": 1.0}, "q": {"a": 0.2, "z": 5}, "r": {"a": 0.4, "b": 0.0}}
    assert case_ids(scores) == ["a"]
    assert mad_epsilons(list(scores), scores, ["a"]) == {"a": 0.2}


def test_missing_scores_treated_as_zero():
    scores = {"good": {"a": 0.9}, "bad": None}
    assert set(_select(["good", "bad"], 20, scores, epsilon=0.0)) == {"good"}
