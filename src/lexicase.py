"""Epsilon-lexicase parent selection over per-case (per-behavior) scores.

Pure Python (no DEAP/numpy) so it can be tested without cluster deps.
`case_scores` maps an individual's key (gene id) to {case_id: score}; all
scores are maximized.
"""

import random
import statistics


def case_ids(case_scores):
    """Cases present for every individual (intersection), in a stable order."""
    sets = [set(s) for s in case_scores.values()]
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def mad_epsilons(keys, case_scores, cases):
    """Per-case epsilon = median absolute deviation of that case's scores over
    the pool (La Cava et al. 2016). Zero MAD -> exact-match lexicase on that case."""
    eps = {}
    for c in cases:
        vals = [case_scores[k][c] for k in keys]
        med = statistics.median(vals)
        eps[c] = statistics.median([abs(v - med) for v in vals])
    return eps


def lexicase_pick(keys, case_scores, cases, eps, rng):
    """Select one key: shuffle cases, keep those within eps of the best on each."""
    pool = list(keys)
    order = list(cases)
    rng.shuffle(order)
    for c in order:
        best = max(case_scores[k][c] for k in pool)
        pool = [k for k in pool if case_scores[k][c] >= best - eps[c]]
        if len(pool) == 1:
            break
    return rng.choice(pool)


def lexicase_select(pop, k, key_fn, case_scores, epsilon="mad", rng=None):
    """Return k individuals (with repeats) chosen by epsilon-lexicase.

    pop: individuals; key_fn(ind) -> key into case_scores.
    epsilon: "mad" (per-case MAD over pop), or a float used for every case.
    Individuals missing from case_scores score 0 on every case.
    """
    rng = rng or random
    scores = {key_fn(i): case_scores.get(key_fn(i)) for i in pop}
    known = {key: s for key, s in scores.items() if s}
    cases = case_ids(known)
    if not cases:
        raise ValueError("lexicase: no per-case scores shared by the population")
    scores = {key: (s if s else {c: 0.0 for c in cases}) for key, s in scores.items()}
    keys = list(scores)
    if epsilon == "mad":
        eps = mad_epsilons(keys, scores, cases)
    else:
        eps = {c: float(epsilon) for c in cases}
    by_key = {key_fn(i): i for i in pop}
    return [by_key[lexicase_pick(keys, scores, cases, eps, rng)] for _ in range(k)]
