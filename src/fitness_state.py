"""Gene IDs own evaluation records; individuals are disposable copies.

An absent or nonfinite completed evaluation is an invariant violation, never
an instruction to manufacture a failed evaluation. History contains independent
snapshots so a mutation cannot erase or mutate a parent's evaluation.
"""
import copy
import math


class FitnessStateError(RuntimeError):
    pass


def finite_fitness(values, gene_id, dimensions=3, placeholders=()):
    try:
        values = tuple(float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise FitnessStateError(f"{gene_id}: missing/invalid completed fitness {values!r}") from exc
    if (len(values) != dimensions or not all(math.isfinite(v) for v in values)
            or values in placeholders):
        raise FitnessStateError(f"{gene_id}: invalid completed fitness {values!r}")
    return values


def record_for(current, history, gene_id):
    if gene_id not in current:
        if gene_id not in history:
            raise FitnessStateError(f"{gene_id}: gene metadata unexpectedly missing")
        current[gene_id] = copy.deepcopy(history[gene_id])
    return current[gene_id]


def snapshot_history(current, history):
    for gene_id, record in current.items():
        history[gene_id] = copy.deepcopy(record)


def assert_records(current, history, dimensions=3, placeholders=()):
    for label, records in (("current", current), ("history", history)):
        for gene_id, record in records.items():
            values = record.get("fitness")
            if record.get("status") == "completed":
                finite_fitness(values, f"{label}/{gene_id}", dimensions, placeholders)
            elif values is not None:
                # Pending/failed generation operations may have no evaluation.
                finite_fitness(values, f"{label}/{gene_id}", dimensions, placeholders)
    for gene_id in current.keys() & history.keys():
        a, b = current[gene_id], history[gene_id]
        if a.get("status") == b.get("status") == "completed" and a.get("fitness") != b.get("fitness"):
            raise FitnessStateError(f"{gene_id}: current/history fitness disagree")


def assert_population(population, current, history, dimensions=3, placeholders=()):
    for individual in population:
        gene_id = individual[0]
        record = record_for(current, history, gene_id)
        if record.get("status") != "completed":
            raise FitnessStateError(f"{gene_id}: selected before evaluation completed")
        expected = finite_fitness(record.get("fitness"), gene_id, dimensions, placeholders)
        actual = finite_fitness(individual.fitness.values, gene_id, dimensions, placeholders)
        if actual != expected:
            raise FitnessStateError(f"{gene_id}: individual/record fitness disagree: {actual} != {expected}")
