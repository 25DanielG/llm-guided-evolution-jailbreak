"""Survivor selection helpers for the lexicase-parents + NSGA-II hybrid."""


def merge_pool(parents, offspring, key_fn, is_valid):
    """Parents + offspring with duplicate keys dropped (first seen wins, so a
    parent kept as an elite is not counted twice) and invalid fitness removed."""
    seen = set()
    pool = []
    for ind in list(parents) + list(offspring):
        key = key_fn(ind)
        if key in seen or not is_valid(ind):
            continue
        seen.add(key)
        pool.append(ind)
    return pool


def nsga2_survival(parents, offspring, k, key_fn, is_valid, select_fn, refresh_fn=None):
    """(mu + lambda) survival: NSGA-II over the merged pool, keep k.

    refresh_fn(pool) runs before ranking so population-relative objectives
    (TF-IDF diversity) are comparable across old and new genes.
    """
    pool = merge_pool(parents, offspring, key_fn, is_valid)
    if refresh_fn is not None:
        refresh_fn(pool)
    return select_fn(pool, min(k, len(pool)))
