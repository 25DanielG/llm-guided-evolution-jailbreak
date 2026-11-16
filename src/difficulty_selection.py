"""Population-relative tournament scores, independent of DEAP and inference."""
import math
import random


def population_scores(case_scores):
    """Return full precision weights and both scores; never silently drop cases."""
    if not case_scores:
        raise ValueError('Empty selection population')
    cases = sorted(next(iter(case_scores.values())))
    if not cases:
        raise ValueError('Empty behavior panel')
    for key, row in case_scores.items():
        if set(row) != set(cases):
            raise ValueError(f'Incomplete behavior vector for {key}')
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in row.values()):
            raise ValueError(f'Invalid probability for {key}')
    rates = {b: math.fsum(row[b] for row in case_scores.values()) / len(case_scores)
             for b in cases}
    weights = {b: 1 - rates[b] for b in cases}
    denominator = math.fsum(weights.values())
    plain = {key: math.fsum(row.values()) / len(cases) for key, row in case_scores.items()}
    weighted = {key: math.fsum(weights[b] * row[b] for b in cases) / denominator
                if denominator else plain[key] for key, row in case_scores.items()}
    return dict(cases=cases, solve_rates=rates, weights=weights, plain=plain,
                weighted=weighted, uniform_fallback=not denominator)


def tournament(keys, scores, k, size=3, rng=None):
    if not keys or k < 0 or size < 1:
        raise ValueError('Invalid tournament population, count, or size')
    rng = rng or random
    selected = []
    for _ in range(k):
        contestants = [rng.choice(keys) for _ in range(size)]
        best = max(scores[key] for key in contestants)
        selected.append(rng.choice([key for key in contestants if scores[key] == best]))
    return selected


def validated_vector(data, expected=None):
    rows = data.get('behaviors', [])
    if not rows or any(r.get('error') or not r.get('parse_ok') for r in rows):
        raise ValueError('Missing or failed per-behavior evaluation')
    result = {r['behavior']: float(r['score']) for r in rows}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in result.values()):
        raise ValueError('Invalid probability in sidecar')
    if len(result) != len(rows) or (expected is not None and set(result) != set(expected)):
        raise ValueError('Duplicate or mismatched behavior panel')
    return result
