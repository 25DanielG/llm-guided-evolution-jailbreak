"""Regression for evaluated parents lost when a selected copy was mutated."""
import copy
import json
import pickle

import pytest
import run_improved as evo
from src.fitness_state import FitnessStateError, snapshot_history
from src.map_elites import MapElitesArchive, PromptDescriptor

pytestmark = pytest.mark.mock_inference


def individual(gene, fitness=(0.125, 0.4, 200.0)):
    ind = evo.creator.Individual([gene])
    ind.fitness.values = fitness
    return ind


@pytest.fixture
def state(tmp_path, monkeypatch):
    current, history = {}, {}
    for name, value in [('GLOBAL_DATA', current), ('GLOBAL_DATA_HIST', history),
                        ('MAP_ELITES_ARCHIVE', MapElitesArchive()),
                        ('hof', evo.tools.ParetoFront()), ('population', [])]:
        monkeypatch.setattr(evo, name, value, raising=False)
    monkeypatch.setattr(evo, 'box_print', lambda *a, **k: None)
    monkeypatch.setattr(evo, 'VARIANT_DIR', str(tmp_path))
    monkeypatch.setenv('JB_GENE_RESULTS_DIR', str(tmp_path))
    monkeypatch.setenv('JB_RESUME_MODE', 'new')
    monkeypatch.delenv('JB_STATUS_FILE', raising=False)
    return tmp_path, current, history


def evaluated(state, gene='parent', fitness=(0.125, 0.4, 200.0)):
    folder, current, history = state
    current[gene] = dict(sub_flag=True, status='completed', fitness=fitness,
                         nested={'value': [gene]})
    snapshot_history(current, history)
    (folder / f'prompt_{gene}.py').write_text(
        'trait_format = \'<format type="freeform">{ASK}</format>\'\n'
        'trait_turns = \'<turns></turns>\'\n')
    (folder / f'{gene}_results.json').write_text(json.dumps({'tier_scores': {}, 'behaviors': []}))
    return individual(gene, fitness)


@pytest.mark.parametrize('operation', ['Mutation', 'Mating'])
def test_successful_child_does_not_delete_evaluated_parent(state, operation):
    parent = evaluated(state)
    _, current, history = state
    original = copy.deepcopy(current['parent'])
    current['child'] = dict(sub_flag=True, status='subbed file', fitness=None)
    child = evo.update_individual(copy.deepcopy(parent), 'child', 'parent', process_type=operation)
    assert current['parent'] == history['parent'] == original
    assert not child.fitness.valid
    current['parent']['nested']['value'].append('changed')
    assert history['parent']['nested']['value'] == ['parent']


def test_failed_child_recovers_parent_without_stale_child_fitness(state):
    parent = evaluated(state)
    _, current, _ = state
    current['bad'] = dict(status='subbed file', sub_flag=False, fitness=None)
    child = evo.update_individual(copy.deepcopy(parent), 'bad', 'parent', process_success=False)
    assert child[0] == 'parent'
    assert child.fitness.values == parent.fitness.values
    assert current['bad']['status'] == 'FAILED: generation'


def test_mutation_and_crossover_same_parent_keep_other_selected_copies(state, monkeypatch):
    parent = evaluated(state)
    _, current, _ = state
    ids = iter(['cross1', 'cross2', 'mutant'])
    monkeypatch.setattr(evo, 'generate_random_string', lambda **kw: next(ids))
    monkeypatch.setattr(evo, 'load_yaml', lambda: {'gpu_selection': 'unused'})
    monkeypatch.setattr(evo, 'submit_bash', lambda *a, **kw: (True, None, 'job done'))
    monkeypatch.setattr(evo, 'check4job_completion', lambda *a, **kw: True)
    monkeypatch.setattr(evo, 'DELAYED_CHECK', False)
    untouched = copy.deepcopy(parent)
    a, b = evo.customCrossover(copy.deepcopy(parent), copy.deepcopy(parent), 'llama3')
    mutant = evo.customMutation(a, 'llama3', 1.0)
    assert [b[0], mutant[0]] == ['cross2', 'mutant']
    assert not b.fitness.valid and not mutant.fitness.valid
    assert current['parent']['fitness'] == untouched.fitness.values
    # Resetting temporary DEAP fitness does not reset gene-owned evaluations.
    untouched.fitness.values = evo.PLACEHOLDER_FITNESS
    evo.check_and_update_fitness([untouched])
    assert untouched.fitness.values == current['parent']['fitness']


def test_missing_evaluated_metadata_fails_instead_of_manufacturing_infinity(state):
    parent = evaluated(state)
    _, current, history = state
    current.clear(); history.clear()
    parent.fitness.values = evo.PLACEHOLDER_FITNESS
    with pytest.raises(FitnessStateError, match='metadata unexpectedly missing'):
        evo.check_and_update_fitness([parent])


def test_corrupted_completed_fitness_fails(state):
    parent = evaluated(state)
    _, current, history = state
    current['parent']['fitness'] = (-float('inf'), -float('inf'), float('inf'))
    parent.fitness.values = evo.PLACEHOLDER_FITNESS
    with pytest.raises(FitnessStateError, match='invalid completed fitness'):
        evo.check_and_update_fitness([parent])


def test_deduplication_rejects_disagreeing_fitness(state):
    a = evaluated(state)
    b = copy.deepcopy(a); b.fitness.values = (0.9, 0.8, 5)
    with pytest.raises(FitnessStateError, match='inconsistent fitness'):
        evo.remove_duplicates([a, b])
    assert len(evo.remove_duplicates([a, copy.deepcopy(a)])) == 1


def test_archive_refresh_updates_decreased_novelty_before_challenger(state):
    a = evaluated(state, 'a', (0.125, 0.9, 200))
    b = evaluated(state, 'b', (0.125, 0.5, 200))
    evo.update_map_elites_archive([a])
    a.fitness.values = (0.125, 0.2, 200)
    # Challenger occurs first: refresh must be a separate first pass.
    evo.update_map_elites_archive([b, a])
    assert evo.MAP_ELITES_ARCHIVE.elites()[0].gene_id == 'b'


def test_selection_copies_do_not_mutate_archive(state, monkeypatch):
    a = evaluated(state)
    evo.update_map_elites_archive([a])
    monkeypatch.setattr(evo.random, 'choice', lambda values: values[0])
    selected = evo.map_elites_select([a], 2)
    selected[0][0] = 'child'
    del selected[0].fitness.values
    assert selected[1][0] == a[0]
    elite = evo.MAP_ELITES_ARCHIVE.elites()[0]
    assert elite.individual.fitness.values == elite.fitness == a.fitness.values


def test_checkpoint_restore_preserves_all_fitness_and_rejects_corruption(state, monkeypatch):
    folder, current, history = state
    a = evaluated(state)
    monkeypatch.setattr(evo, 'population', [a])
    evo.update_map_elites_archive([a])
    evo.refresh_hall_of_fame(evo.hof, [a])
    evo.save_checkpoint(0, folder / 'ckpt', folder / 'global')
    monkeypatch.setenv('JB_RESUME_MODE', 'required')
    restored, generation, records = evo.load_checkpoint(folder / 'ckpt', global_path=folder / 'global')
    assert generation == 1
    evo.validate_fitness_state(restored['population'], restored['map_elites_archive'],
                              records['GLOBAL_DATA'], records['GLOBAL_DATA_HIST'])
    assert restored['population'][0].fitness.values == a.fitness.values
    current['parent']['fitness'] = (-float('inf'), -float('inf'), float('inf'))
    with pytest.raises(FitnessStateError):
        evo.save_checkpoint(1, folder / 'ckpt', folder / 'global')
    assert not (folder / 'ckpt/checkpoint_gen_1.json').exists()


def test_archive_rejects_nonfinite_fitness_even_with_finite_quality(state):
    a = individual('bad', (-float('inf'), 0.2, 200))
    with pytest.raises(ValueError, match='fitness values must be finite'):
        MapElitesArchive().add('bad', PromptDescriptor('freeform', 'zero'), (0.1,),
                             fitness=a.fitness.values, individual=a)


def test_repeated_selection_and_parent_reuse_across_seventeen_checkpoints(state, monkeypatch):
    folder, current, history = state
    initial = [evaluated(state, f'initial{i}', (0.0, 0.1 + i / 1000, 200.0)) for i in range(24)]
    monkeypatch.setattr(evo, 'population', initial)
    evo.update_map_elites_archive(initial)
    evo.refresh_hall_of_fame(evo.hof, initial)
    evo.save_checkpoint(0, folder / 'ckpt', folder / 'global')
    for generation in range(1, 17):
        parents = evo.map_elites_select(evo.population, 24)
        retained = copy.deepcopy(parents[:6])
        children = []
        for i, parent in enumerate(parents):
            gene = f'g{generation}_{i}'
            evaluated(state, gene, (0.0, 0.1 + i / 1000, 200.0))
            child = evo.update_individual(parent, gene, parent[0])
            child.fitness.values = evo.PLACEHOLDER_FITNESS
            children.append(child)
        population = evo.remove_duplicates(children + retained)
        for ind in population:
            ind.fitness.values = evo.PLACEHOLDER_FITNESS
        evo.check_and_update_fitness(population)
        evo.update_map_elites_archive(population)
        evo.refresh_hall_of_fame(evo.hof, population)
        monkeypatch.setattr(evo, 'population', population)
        evo.save_checkpoint(generation, folder / 'ckpt', folder / 'global')
        evo.assert_fitness_state()
    assert len(list((folder / 'ckpt').glob('*.json'))) == 17


def test_missing_temporary_fitness_on_duplicate_parent_restores_canonical_record(state):
    a = evaluated(state); b = copy.deepcopy(a)
    del b.fitness.values
    output = evo.remove_duplicates([b, a])
    assert len(output) == 1 and output[0].fitness.values == state[1]['parent']['fitness']
