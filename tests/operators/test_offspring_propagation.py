import run_improved


def _individual(gene_id, fitness=(1.0, 1.0, 1.0)):
    individual = run_improved.creator.Individual([gene_id])
    individual.fitness.values = fitness
    return individual


def test_apply_mutation_replaces_parent_with_returned_child(monkeypatch):
    parent = _individual("parent")
    child = _individual("mutated")
    offspring = [parent]

    monkeypatch.setattr(run_improved.random, "random", lambda: 0.0)
    monkeypatch.setattr(run_improved.toolbox, "mutate", lambda **kwargs: child)

    result = run_improved.apply_mutation(offspring, "test-model")

    assert result[0] is child
    assert result[0][0] == "mutated"
    assert not result[0].fitness.valid


def test_apply_crossover_replaces_parents_with_returned_children(monkeypatch):
    parents = [_individual("parent-a"), _individual("parent-b")]
    children = (_individual("child-a"), _individual("child-b"))

    monkeypatch.setattr(run_improved.random, "random", lambda: 0.0)
    monkeypatch.setattr(run_improved.toolbox, "mate", lambda *args, **kwargs: children)

    result = run_improved.apply_crossover(parents, "test-model")

    assert result[0] is children[0]
    assert result[1] is children[1]
    assert [individual[0] for individual in result] == ["child-a", "child-b"]
    assert all(not individual.fitness.valid for individual in result)


def test_delayed_mutation_failure_restores_parent(monkeypatch):
    child = _individual("failed-child")
    run_improved.GLOBAL_DATA.clear()
    run_improved.LINKED_GENES.clear()
    run_improved.GLOBAL_DATA["failed-child"] = {
        "status": "DELAYED_CHECK",
        "sub_flag": False,
        "job_id": None,
    }
    run_improved.LINKED_GENES["failed-child"] = "parent"
    monkeypatch.setattr(run_improved, "DELAYED_CHECK", True)

    result = run_improved.delayed_mutate_check([child])

    assert result[0][0] == "parent"
    assert "failed-child" not in run_improved.GLOBAL_DATA


def test_delayed_crossover_failure_restores_parent(monkeypatch):
    child = _individual("failed-child")
    run_improved.GLOBAL_DATA.clear()
    run_improved.LINKED_GENES.clear()
    run_improved.GLOBAL_DATA["failed-child"] = {
        "status": "DELAYED_CHECK",
        "sub_flag": True,
        "job_id": "123",
    }
    run_improved.LINKED_GENES["failed-child"] = "parent"
    monkeypatch.setattr(run_improved, "DELAYED_CHECK", True)
    monkeypatch.setattr(run_improved, "check4job_completion", lambda *args, **kwargs: False)

    result = run_improved.delayed_mate_check([child])

    assert result[0][0] == "parent"
    assert "failed-child" not in run_improved.GLOBAL_DATA
