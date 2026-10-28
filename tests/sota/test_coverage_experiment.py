"""Offline tests for experiment controls, geometry, and full report generation."""
import csv
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.experiments import coverage, runtime, runner, report


class Embedder:
    metadata = {'model': 'synthetic', 'revision': 'fixed', 'pooling': 'test', 'serialization': 'role-content-v1'}
    def embed(self, text):
        rng = np.random.default_rng(int(coverage.digest(text)[:8], 16))
        vector = rng.normal(size=8)
        return vector / np.linalg.norm(vector), 1


@pytest.fixture
def experiment(tmp_path):
    source = tmp_path / 'source.csv'
    with source.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['goal', 'target'])
        for i in range(110):
            w.writerow([f'Benign synthetic behavior {i}', 'unused'])
    destination = tmp_path / 'experiment'
    coverage.prepare(destination, Embedder(), source, 'a' * 40)
    return destination / 'experiment.json'


def test_subset_design_and_coverage(experiment):
    manifest = coverage.load_experiment(experiment)
    assert len(manifest['subsets']) == 10
    assert len(manifest['heldout_ids']) == 30
    memberships = set()
    for subset in manifest['subsets']:
        ids = set(subset['behavior_ids'])
        assert len(ids) == 30
        assert not ids.intersection(manifest['heldout_ids'])
        memberships.add(tuple(sorted(ids)))
    assert len(memberships) == 10
    vectors = np.load(experiment.parent / 'behavior_embeddings.npy')
    assert coverage.design(vectors) == coverage.design(vectors)
    d = coverage.distances([[1, 0], [0, 1], [-1, 0]])
    assert coverage.coverage_distance(d, [0]) == pytest.approx(1)
    assert coverage.coverage_distance(d, [0, 1, 2]) == 0
    (experiment.parent / 'heldout.csv').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        coverage.load_experiment(experiment)


def test_metrics_duplicates_noise_and_contraction():
    wide = [[1, -.3], [1, .3], [-1, -.3], [-1, .3]]
    tight = [[1, -.01], [1, .01], [-1, -.01], [-1, .01]]
    a = report.measure(wide, [0, 0, 1, 1], 4, ['a', 'b', 'c', 'd'])
    b = report.measure(tight, [0, 0, 1, 1], 4, ['a', 'b', 'c', 'd'])
    assert b['within_cluster'] < a['within_cluster']
    assert b['between_centroids'] == pytest.approx(2)
    dup = report.measure([[1, 0], [1, 0], [0, 1]], [0, 0, -1], 4, ['a', 'a', 'b'])
    assert dup['duplicate_fraction'] == pytest.approx(1/3)
    assert dup['pair_coverage'] == .5
    assert dup['nearest_neighbor'] == pytest.approx(1/3)
    assert dup['within_cluster'] == 0
    assert dup['between_centroids'] is None
    noise = report.measure(tight, [-1]*4, 4, ['a', 'b', 'c', 'd'])
    assert noise['within_cluster'] is noise['between_centroids'] is None
    assert report.measure([], [], 24, [])['pair_coverage'] == 0


def test_initial_population_integrity_and_snapshot(tmp_path):
    models = tmp_path / 'models'
    models.mkdir()
    population = []
    for i in range(24):
        (models / f'prompt_g{i}.py').write_text('value = 1\n')
        population.append([f'g{i}'])
    frozen = tmp_path / 'frozen'
    runtime.freeze_population(frozen, population, models, {})
    data = runtime.load_initial(frozen, tmp_path / 'other_models')
    assert len(data['members']) == 24
    with pytest.raises(ValueError):
        runtime.freeze_population(frozen, population, models, {})
    (frozen / 'prompt_g0.py').write_text('value = 2')
    with pytest.raises(ValueError, match='changed'):
        runtime.load_initial(frozen, tmp_path / 'other_models')


def test_rng_restore_and_isolated_environment(tmp_path):
    random.seed(8)
    np.random.seed(9)
    state = runtime.rng_state()
    expected = (random.random(), np.random.random())
    runtime.restore_rng(state)
    assert (random.random(), np.random.random()) == expected
    env = runner.environment(tmp_path, tmp_path / 'subset.csv')
    assert env['JB_FULL_BEHAVIORS'] == '1'
    assert env['JB_CURRICULUM'] == env['JB_STRATIFIED'] == '0'
    assert Path(env['JB_GENE_RESULTS_DIR']).is_relative_to(tmp_path)


def test_completion_marker_ignores_partial_generation(tmp_path):
    for generation in (0, 1):
        snap = tmp_path / 'snapshots' / f'{generation:04d}.json'
        coverage.write_json(snap, {'generation': generation})
        ckpt = tmp_path / f'checkpoint{generation}'
        glob = tmp_path / f'global{generation}'
        ckpt.write_text('checkpoint')
        glob.write_text('global')
        if generation == 0:
            runtime.save_complete(tmp_path, generation, ckpt, glob)
    assert runtime.latest_complete(tmp_path) == 0
    (tmp_path / 'checkpoint0').write_text('corrupted')
    with pytest.raises(ValueError, match='changed'):
        runtime.latest_complete(tmp_path)


def test_end_to_end_report_and_no_model_calls(experiment, tmp_path, monkeypatch):
    manifest = coverage.load_experiment(experiment)
    run = tmp_path / 'run'
    subset = manifest['subsets'][0]
    config = {'experiment_hash': coverage.digest(experiment.read_bytes()), 'subset': subset,
              'initial_hash': 'fixed', 'source_hashes': {}, 'models': {}, 'environment': {}, 'seed': 42}
    run_hash = runtime.bind_run(run, config)
    members = []
    for i in range(6):
        source = run / 'strategies' / f'g{i}.py'
        source.parent.mkdir(exist_ok=True)
        source.write_text(f'value={i}')
        members.append({'id': f'g{i}', 'strategy_hash': coverage.digest(source.read_bytes()), 'fitness': [.5, .2, 30]})
    for generation in range(6):
        snap_members = []
        for member in members:
            for split, ids in [('evolution', subset['behavior_ids']), ('heldout', manifest['heldout_ids'])]:
                records = []
                for case in ids:
                    text = f"[user]\nsynthetic prompt {member['id']} generation {generation}"
                    records.append({'behavior_id': case, 'prompt_text': text, 'prompt_hash': coverage.digest(text),
                                    'status': 'ok', 'score': .5, 'tokens': 30})
                coverage.write_json(run / 'observations' / split / f'{generation:04d}' / f'{member["id"]}.json',
                                    {'strategy_hash': member['strategy_hash'], 'records': records})
            snap_members.append({**member, 'observation': f'observations/evolution/{generation:04d}/{member["id"]}.json'})
        coverage.write_json(run / 'snapshots' / f'{generation:04d}.json', {'generation': generation, 'run_hash': run_hash, 'members': snap_members})
        cp, gp = run / f'cp{generation}', run / f'gp{generation}'
        cp.write_text('test')
        gp.write_text('test')
        runtime.save_complete(run, generation, cp, gp)
    before = {str(p): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    out = tmp_path / 'report'
    rows = report.analyze(experiment, [run], out, embedder=Embedder())
    assert len(rows) == 6 * 2 * 3
    assert (out / 'heldout_trajectories.png').exists()
    assert (out / 'coverage_associations.csv').exists()
    assert {str(p): p.read_bytes() for p in run.rglob('*') if p.is_file()} == before
    with pytest.raises(ValueError, match='immutable'):
        report.analyze(experiment, [run], out, embedder=Embedder())
    assert runner.estimate(experiment)['runs'] == 10


class FakeFitness:
    def __init__(self):
        self.values = ()
    @property
    def valid(self):
        return bool(self.values)


class FakeIndividual(list):
    def __init__(self, values):
        super().__init__(values)
        self.fitness = FakeFitness()


class FakeFront:
    def update(self, population):
        pass


def execute_mock_loop(experiment, initial, root, last, monkeypatch, fail_generation=None):
    """Execute the real main loop and checkpoint functions with model work mocked."""
    import argparse
    import ast
    import copy
    import os
    import pickle
    import re
    import time
    source = Path(__file__).resolve().parents[2] / 'run_improved.py'
    tree = ast.parse(source.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'save_checkpoint', 'load_checkpoint', 'extract_generation'}]
    main = next(n for n in tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and ast.unparse(n.test).startswith('__name__'))
    for key, value in runner.environment(root, experiment.parent / 'random_01.csv').items():
        if key.startswith(('JB_', 'LLMGE_')):
            monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, 'argv', [str(source), '--experiment-manifest', str(experiment),
        '--initial-population', str(initial), '--subset', 'random_01', '--checkpoints', str(root / 'ckpt'),
        '--global_path', str(root / 'global'), '--final-generation', str(last)])
    noop = lambda *a, **kw: None
    ns = dict(__name__='__main__', os=os, argparse=argparse, np=np, random=random, pickle=pickle, re=re, time=time,
              OUTPUT_DIR='jailbreak_test', SEED_NETWORK='/test/Jailbreak/prompt.py', DEFAULT_LLM_MODEL='mock',
              ROOT_DIR=str(source.parent), RUN_ID='mock', AVAILABLE_LLM_MODELS=[],
              resolve_prompt_glob=lambda x: x, DNA_TXT='', num_generations=17, migration_gen=0,
              start_population_size=24, population_size=24, num_elites=6, NUM_EOT_ELITES=10,
              FITNESS_WEIGHTS=(1., 1., -1.), PLACEHOLDER_FITNESS=(-999., -999., 999.),
              INVALID_FITNESS_MAX=(-np.inf, -np.inf, np.inf), crossover_probability=.15, mutation_probability=.8,
              MODEL='prompt', TRAIN_FILE='Jailbreak/eval.py', SOTA_ROOT=str(source.parent / 'sota/Jailbreak'),
              CURRENT_GENERATION=0, GENERATION=0, GEN_COUNT=-1, LINKED_GENES={}, GLOBAL_DATA={}, GLOBAL_DATA_HIST={}, GLOBAL_DATA_ANCESTRY={},
              GEN_PATTERNS={'checkpoint': re.compile(r'^checkpoint_gen_(\d+)\.pkl$'), 'global': re.compile(r'^global_gen_(\d+)\.pkl$')},
              creator=SimpleNamespace(Individual=FakeIndividual),
              tools=SimpleNamespace(ParetoFront=FakeFront, selSPEA2=lambda p,n:p[:n], selBest=lambda p,n:p[:n]),
              subprocess=SimpleNamespace(run=noop), box_print=noop, print_population=noop, print_scores=noop,
              print_ancestry=noop, delayed_creation_check=noop, delayed_mate_check=lambda p:p,
              delayed_mutate_check=lambda p:p, apply_crossover=lambda p,m:p,
              remove_duplicates=lambda p:list({i[0]: i for i in p}.values()))
    evaluated = []
    def evaluate(population):
        generation = ns['CURRENT_GENERATION']
        if generation == fail_generation:
            raise RuntimeError('synthetic interruption')
        evaluated.append(generation)
        for ind in population:
            ind.fitness.values = (.5, .2, 20.)
            ns['GLOBAL_DATA'][ind[0]] = {'sub_flag': True, 'status': 'completed', 'fitness': ind.fitness.values}
            runtime.record_evaluation(root, generation, ind[0], 'evolution', [],
                                      Path(ns['VARIANT_DIR']) / f'prompt_{ind[0]}.py', {})
    def mutate(population, model):
        gene = f'g{random.randrange(10**12)}'
        (Path(ns['VARIANT_DIR']) / f'prompt_{gene}.py').write_text('value = 1\n')
        population[0] = FakeIndividual([gene])
        return population
    ns.update(check_and_update_fitness=evaluate, apply_mutation=mutate,
              toolbox=SimpleNamespace(select=lambda p,n:p[:n], clone=copy.deepcopy, evaluate=noop))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), ns)
    exec(compile(ast.Module(body=main.body, type_ignores=[]), str(source), 'exec'), ns)
    return evaluated


def test_real_loop_initial_resume_interruption_and_extension(experiment, tmp_path, monkeypatch):
    initial = tmp_path / 'initial'
    models = tmp_path / 'seed_models'
    models.mkdir()
    for i in range(24):
        (models / f'prompt_seed{i}.py').write_text('value = 1\n')
    runtime.freeze_population(initial, [[f'seed{i}'] for i in range(24)], models, {})
    uninterrupted = tmp_path / 'full'
    assert execute_mock_loop(experiment, initial, uninterrupted, 10, monkeypatch) == list(range(11))
    interrupted = tmp_path / 'resumed'
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        execute_mock_loop(experiment, initial, interrupted, 5, monkeypatch, fail_generation=3)
    assert runtime.latest_complete(interrupted) == 2
    assert execute_mock_loop(experiment, initial, interrupted, 5, monkeypatch) == [3, 4, 5]
    before = {p.name: p.read_bytes() for p in (interrupted / 'snapshots').glob('*.json')}
    assert len(before) == 6
    assert execute_mock_loop(experiment, initial, interrupted, 10, monkeypatch) == [6, 7, 8, 9, 10]
    assert {p.name: p.read_bytes() for p in (interrupted / 'snapshots').glob('*.json') if p.name in before} == before
    for gen in range(11):
        a = json.loads((uninterrupted / 'snapshots' / f'{gen:04d}.json').read_text())
        b = json.loads((interrupted / 'snapshots' / f'{gen:04d}.json').read_text())
        assert a['members'] == b['members']


def test_heldout_evaluator_never_writes_selection_outputs(tmp_path, monkeypatch):
    import ast
    import importlib
    import os
    import types
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / 'sota/Jailbreak'))
    evaluation = importlib.import_module('eval')
    source = tmp_path / 'prompt_mock.py'
    source.write_text('class PromptStrategy:\n    pass\n')
    module = types.ModuleType('prompt_mock')
    module.PromptStrategy = lambda: object()
    monkeypatch.setitem(sys.modules, 'prompt_mock', module)
    behavior_csv = tmp_path / 'subset.csv'
    behavior_csv.write_text('behavior\nA benign request\n')
    for key, value in runner.environment(tmp_path, behavior_csv).items():
        if key.startswith('JB_'):
            monkeypatch.setenv(key, value)
    # Execute only evaluator entrypoint with fake service work, preserving its exit path.
    args = SimpleNamespace(variant_dir=str(tmp_path), model='prompt_mock', generation=0,
                           random_seed=42, split='heldout', run_id='heldout')
    captured = [{'behavior': 'A benign request', 'messages': [{'role': 'user', 'content': 'hello'}],
                 'response': 'hello', 'prompt_tokens': 5, 'in_loop': {'score': .5}}]
    forbidden = lambda *a, **kw: pytest.fail('Held-out evaluation touched selection/reporting output')
    fake_os = SimpleNamespace(**{k: getattr(os, k) for k in dir(os)})
    fake_os.chdir = lambda *a: None
    ns = dict(evaluation.__dict__)
    ns.update(os=fake_os, get_args=lambda: args, load_behaviors=lambda: ['A benign request'],
              resolve_decoding=lambda strategy: {},
              get_target_responses=lambda *a, **kw: captured,
              score=lambda records: (.5, 5, []), write_results=forbidden,
              write_details=forbidden, append_records=forbidden, export_elites=forbidden,
              vllm_client=SimpleNamespace(wait_ready=lambda *a, **kw: True, assert_model=lambda *a: None))
    tree = ast.parse((root / 'sota/Jailbreak/eval.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and ast.unparse(n.test).startswith('__name__'))
    for key, value in [('JB_TARGET_VLLM_URL', 'http://target'), ('JB_JUDGE_VLLM_URL', 'http://judge'), ('JB_REPORTING_VLLM_URL', 'http://report')]:
        monkeypatch.setenv(key, value)
    with pytest.raises(SystemExit) as exit_info:
        exec(compile(ast.Module(body=main.body, type_ignores=[]), 'eval.py', 'exec'), ns)
    assert exit_info.value.code == 0
    saved = json.loads((tmp_path / 'observations/heldout/0000/mock.json').read_text())
    assert saved['records'][0]['score'] == .5
    assert not (tmp_path / 'gene_results').exists()


def test_failed_latest_attempt_never_falls_back(tmp_path):
    model_dir = tmp_path / 'models'
    model_dir.mkdir()
    source = model_dir / 'prompt_a.py'
    source.write_text('value = 1')
    individual = FakeIndividual(['a'])
    individual.fitness.values = (.5, .2, 20.)
    runtime.record_evaluation(tmp_path, 0, 'a', 'evolution',
                              [{'behavior': 'hello', 'messages': [{'role': 'user', 'content': 'hello'}]}], source, {})
    runtime.record_evaluation(tmp_path, 1, 'a', 'evolution', [], source, {})
    path = runtime.snapshot(tmp_path, 1, [individual], model_dir, {}, 'run')
    member = json.loads(path.read_text())['members'][0]
    assert member['observation'] == 'observations/evolution/0001/a.json'
    assert json.loads((tmp_path / member['observation']).read_text())['records'] == []
