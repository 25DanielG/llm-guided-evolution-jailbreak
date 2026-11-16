import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from analysis.experiments.difficulty_tournament import (
    freeze_panel, import_bundle, json_hash, load_manifest, slot, environment, extract_source,
)
from analysis.experiments.difficulty_worker import relocate, restore_initial, champion
from analysis.experiments.difficulty_services import port_block, stop_process
from src import checkpointing
from src.run_status import atomic_json


def test_freeze_panel_is_paired_and_stratified(tmp_path):
    path = tmp_path / 'input.csv'
    path.write_text('behavior,tier\n' + ''.join(f'{tier}{i},{tier}\n' for tier in ('easy', 'medium', 'hard') for i in range(30)))
    first = freeze_panel(path, stratified=True)
    assert first == freeze_panel(path, stratified=True)
    assert len(first) == 50
    assert {tier: sum(r['tier'] == tier for r in first) for tier in ('easy', 'medium', 'hard')} == {'easy': 16, 'medium': 17, 'hard': 17}


def test_missing_tiers_fail_before_submission(tmp_path):
    path = tmp_path / 'input.csv'
    path.write_text('behavior\nexample\n')
    with pytest.raises(ValueError, match='Calibrated'):
        freeze_panel(path, stratified=True)


def test_slot_pairs():
    manifest = {'seeds': [42, 43, 44]}
    assert slot(manifest, 'run', 0) == (42, 'tournament')
    assert slot(manifest, 'run', 1) == (42, 'difficulty_tournament')
    assert slot(manifest, 'run', 5) == (44, 'difficulty_tournament')
    with pytest.raises(ValueError):
        slot(manifest, 'run', 6)


def test_manifest_tamper_rejected(tmp_path):
    manifest = {'seeds': [42]}
    manifest['identity'] = json_hash(manifest)
    manifest['seeds'] = [43]
    atomic_json(tmp_path / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='changed'):
        load_manifest(tmp_path)


def test_initial_bundle_hash_checked_and_copies_identically(tmp_path):
    initial = tmp_path / 'initial'
    initial.mkdir()
    source = initial / 'work/results/gene.json'
    source.parent.mkdir(parents=True)
    source.write_text('{"score": 0.5}')
    atomic_json(initial / 'complete.json', {'stage': 'initial', 'bundle_hashes': {'results/gene.json': checkpointing.digest(source)}})
    for arm in ('plain', 'weighted'):
        import_bundle(initial, tmp_path / arm)
        assert (tmp_path / arm / 'results/gene.json').read_bytes() == source.read_bytes()
    source.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        import_bundle(initial, tmp_path / 'third')


def test_initial_checkpoint_relocation_and_identity(tmp_path):
    import random
    initial = tmp_path / 'initial'
    identity = {'difficulty_experiment': 'campaign'}
    checkpointing.commit(0, initial / 'evolution/ckpt', initial / 'evolution/global',
        {'population': [['gene']], 'python_random_state': random.Random(42).getstate()},
        {'path': str(initial / 'results/gene.json')}, identity)
    state, global_data = restore_initial(initial, tmp_path / 'arm', 'campaign')
    assert state['population'] == [['gene']]
    assert global_data['path'] == str(tmp_path / 'arm/results/gene.json')
    with pytest.raises(ValueError, match='identity'):
        restore_initial(initial, tmp_path / 'arm', 'different')


def test_port_blocks_are_isolated(monkeypatch):
    # Exercise real cooperating locks without requiring sandbox network sockets.
    monkeypatch.setattr('analysis.experiments.difficulty_services.socket.socket',
                        lambda: SimpleNamespace(bind=lambda address: None, close=lambda: None))
    with port_block() as first:
        with port_block() as second:
            assert first != second


def test_process_group_cleanup():
    import subprocess
    import sys
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    stop_process(proc)
    assert proc.poll() is not None


def test_champion_is_unweighted_and_ties_stable(tmp_path):
    directory = tmp_path / 'sota/Jailbreak/results'
    directory.mkdir(parents=True)
    for gene in ['z', 'a']:
        atomic_json(directory / f'{gene}_results.json', {'gene_id': gene, 'mean_judge_score': 999,
            'behaviors': [{'behavior': 'b', 'score': .5, 'parse_ok': True}]})
    score, gene, _ = champion(tmp_path, ['b'])
    assert score == .5 and gene == 'a'


def test_artifact_resume_removes_uncommitted_offspring_and_restores_sidecars(tmp_path):
    from src.difficulty_artifacts import snapshot, restore
    results = tmp_path / 'sota/Jailbreak/results'
    results.mkdir(parents=True)
    parent = results / 'parent_results.json'
    parent.write_text('{"score":0.5}')
    checkpoint_dir = tmp_path / 'evolution/ckpt'
    metadata = snapshot(tmp_path, checkpoint_dir, 0)
    parent.write_text('{"score":0.9}')
    (results / 'uncommitted_results.json').write_text('{}')
    restore(tmp_path, metadata)
    assert parent.read_text() == '{"score":0.5}'
    assert not (results / 'uncommitted_results.json').exists()


def test_selector_integration_recomputes_weights_preserves_raw_fitness(tmp_path, monkeypatch):
    import ast
    import random
    from src.difficulty_selection import population_scores, tournament, validated_vector
    tree = ast.parse(Path('run_improved.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'tournament_select')
    class Individual(list):
        fitness = (0.5, 0.2, 100)
    population = [Individual(['a']), Individual(['b'])]
    scores = {'a': .2, 'b': .8}
    monkeypatch.setenv('DT_SELECTION_DIR', str(tmp_path))
    monkeypatch.delenv('DT_PANEL_JSON', raising=False)
    namespace = dict(os=__import__('os'), json=json, Path=Path, checkpointing=checkpointing,
                     validated_vector=validated_vector, population_scores=population_scores,
                     tournament=tournament, SELECTION_RNG=random.Random(42),
                     CURRENT_GENERATION=1, _selection_method='difficulty_tournament',
                     _gene_result_data=lambda key: {'behaviors': [{'behavior': 'x', 'score': scores[key], 'parse_ok': True}]})
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'run_improved.py', 'exec'), namespace)
    namespace['tournament_select'](population, 5)
    first = json.loads((tmp_path / 'gen_001.json').read_text())
    scores['b'] = .4
    namespace['CURRENT_GENERATION'] = 2
    namespace['tournament_select'](population, 5)
    second = json.loads((tmp_path / 'gen_002.json').read_text())
    assert first['weights'] != second['weights']
    assert all(ind.fitness == (.5, .2, 100) for ind in population)
    assert len(second['selected']) == 5


def test_prepare_and_runtime_are_separate_and_snapshot_is_portable(tmp_path, monkeypatch):
    import argparse
    import tarfile
    from analysis.experiments import difficulty_tournament as campaign
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'run_improved.py').write_text('# frozen\n')
    (root / 'config.json').write_text('{}')
    (root / 'alias.py').symlink_to('run_improved.py')
    monkeypatch.setattr(campaign, 'ROOT', root)
    monkeypatch.setattr(campaign.subprocess, 'check_output',
                        lambda command, **kw: b'run_improved.py\0config.json\0alias.py\0' if '--others' not in command else b'')
    archive = tmp_path / 'source.tar.gz'
    hashes = campaign.snapshot_source(archive)
    work = tmp_path / 'work'
    campaign.extract_source(tmp_path, work)
    assert all(checkpointing.digest(work / path) == value for path, value in hashes.items())
    assert (work / 'alias.py').read_text() == '# frozen\n'


def test_service_start_failure_reaps_all_started_processes(tmp_path, monkeypatch):
    from analysis.experiments import difficulty_services as module
    stopped = []
    monkeypatch.setattr(module, 'stop_process', lambda process: stopped.append(process))
    env = {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4', 'DT_SEED': '42', 'DT_PYTHON': '/python'}
    models = {role: {'path': '/models/' + role} for role in ('target', 'judge', 'reporting')}
    services = module.Services(tmp_path, tmp_path, env, models, '/vllm', 18000)
    def start(label, *args):
        services.processes.append(label)
        if label == 'judge':
            raise RuntimeError('mock startup failure')
    monkeypatch.setattr(services, 'start', start)
    with pytest.raises(RuntimeError, match='startup'):
        with services:
            pytest.fail('Should not enter healthy state')
    assert set(stopped) == {'target', 'judge', 'reporting', 'mutator'}
    assert not services.cache.exists()


def test_unstratified_panel_requires_no_tiers_and_omits_tier_claims(tmp_path):
    path = tmp_path / 'input.csv'
    path.write_text('behavior,tier\n' + ''.join(f'behavior{i},hard\n' for i in range(60)))
    panel = freeze_panel(path)
    assert len(panel) == 50
    assert panel == freeze_panel(path)
    assert all(row['tier'] == '' for row in panel)
    path.write_text('behavior\n' + ''.join(f'behavior{i}\n' for i in range(60)))
    assert freeze_panel(path) == panel


def test_model_inventory_never_reads_weight_shards(tmp_path, monkeypatch):
    from analysis.experiments import difficulty_tournament as campaign
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'model.safetensors').write_bytes(b'placeholder')
    hashed = []
    def small_digest(path):
        assert path.suffix not in ('.safetensors', '.bin')
        hashed.append(path.name)
        return 'config-hash'
    monkeypatch.setattr(campaign, 'digest', small_digest)
    inventory = campaign.model_inventory(tmp_path)
    assert hashed == ['config.json']
    assert 'sha256' not in inventory['files']['model.safetensors']
    assert inventory['files']['model.safetensors']['size'] == 11


@pytest.mark.parametrize('cuda,libraries,version', [(None, False, '2.13.0+cpu'), ('13.0', False, '2.13.0')])
def test_serving_preflight_rejects_cpu_or_incomplete_torch(tmp_path, monkeypatch, cuda, libraries, version):
    from analysis.experiments import difficulty_tournament as campaign
    binary = tmp_path / 'bin/vllm'
    binary.parent.mkdir()
    binary.write_text('#!/bin/sh\n')
    binary.chmod(0o700)
    payload = {'torch': version, 'torch_cuda': cuda, 'cuda_libraries_present': libraries}
    monkeypatch.setattr(campaign.subprocess, 'check_output', lambda *a, **kw: json.dumps(payload))
    with pytest.raises(ValueError, match='CPU-only or incomplete'):
        campaign.serving_fingerprint(binary)


def test_serving_preflight_resolves_cli_symlink(tmp_path, monkeypatch):
    from analysis.experiments import difficulty_tournament as campaign
    binary = tmp_path / 'tools/vllm/bin/vllm'
    binary.parent.mkdir(parents=True)
    binary.write_text('#!/bin/sh\n')
    binary.chmod(0o700)
    link = tmp_path / 'vllm'
    link.symlink_to(binary)
    calls = []
    def probe(command, **kwargs):
        calls.append(command)
        return json.dumps({'torch': '2.13.0', 'torch_cuda': '13.0', 'cuda_libraries_present': True})
    monkeypatch.setattr(campaign.subprocess, 'check_output', probe)
    resolved, info = campaign.serving_fingerprint(link)
    assert resolved == str(binary)
    assert calls[0][0] == str(binary.parent / 'python')
    assert info['torch_cuda'] == '13.0'


def test_mutator_initialization_can_seed_before_loading_weights(monkeypatch, tmp_path):
    import ast
    import os
    tree = ast.parse(Path('server.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LLMModel')
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_initialize')
    seeded = []
    class StopBeforeModelLoad(Exception):
        pass
    def stop():
        raise StopBeforeModelLoad
    namespace = {'os': os, 'MODEL_PATH': str(tmp_path),
                 'torch': SimpleNamespace(manual_seed=seeded.append,
                    cuda=SimpleNamespace(manual_seed_all=seeded.append, device_count=stop))}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'server.py', 'exec'), namespace)
    monkeypatch.setenv('DT_SEED', '42')
    with pytest.raises(StopBeforeModelLoad):
        namespace['_initialize'](SimpleNamespace(_validate_cuda_available=lambda: None))
    assert seeded == [42, 42]


def test_classifier_context_is_capped_at_checkpoint_limit(tmp_path):
    from analysis.experiments.difficulty_tournament import model_inventory
    (tmp_path / 'config.json').write_text('{"max_position_embeddings": 2048}')
    (tmp_path / 'model.safetensors').write_bytes(b'placeholder')
    assert model_inventory(tmp_path)['max_model_len'] == 2048
    (tmp_path / 'config.json').write_text('{"max_position_embeddings": 131072}')
    assert model_inventory(tmp_path)['max_model_len'] == 4096


def test_service_command_uses_per_model_context_limit(tmp_path, monkeypatch):
    from analysis.experiments import difficulty_services as module
    env = {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4', 'DT_SEED': '42', 'DT_PYTHON': '/python'}
    models = {role: {'path': '/models/' + role, 'max_model_len': 2048 if role == 'reporting' else 4096}
              for role in ('target', 'judge', 'reporting')}
    services = module.Services(tmp_path, tmp_path, env, models, '/vllm', 18000)
    commands = {}
    monkeypatch.setattr(services, 'start', lambda label, command, *args: commands.update({label: command}))
    with services:
        for role, limit in [('target', '4096'), ('judge', '4096'), ('reporting', '2048')]:
            command = commands[role]
            assert command[command.index('--max-model-len') + 1] == limit
