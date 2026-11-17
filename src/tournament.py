"""Tournament parent selection on a single score: the control for lexicase."""

import random


def tournament_select(pop, k, score_fn, size=3, rng=None):
    """Return k individuals (with repeats), each the best of `size` distinct random
    contestants by score_fn (maximized). Ties go to the contestant drawn first."""
    rng = rng or random
    pop = list(pop)
    if not pop:
        raise ValueError("tournament: empty population")
    size = max(1, min(int(size), len(pop)))
    return [max(rng.sample(pop, size), key=score_fn) for _ in range(k)]
