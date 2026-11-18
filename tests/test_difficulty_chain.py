import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('difficulty_chain', ROOT / 'scripts/difficulty_chain.py')
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)


@pytest.fixture
def setup_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, 'load_manifest', lambda _: {'identity': 'test', 'seeds': [42]})
    monkeypatch.setenv('SLURM_JOB_ID', '100')
    submissions = []
    monkeypatch.setattr(chain, 'submit', lambda *args: submissions.append(args) or str(200 + len(submissions)))
    return tmp_path, submissions


def test_checkpoint_exit_submits_only_same_run_continuation(setup_chain, monkeypatch):
    root, submissions = setup_chain
    directory = root / 'runs/42-tournament'
    checkpoint = directory / 'work/evolution/ckpt/checkpoint_gen_13.json'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text('{}')
    calls = []
    monkeypatch.setattr(chain.subprocess, 'Popen', lambda args, **kw: calls.append(args) or SimpleNamespace(wait=lambda: 75))
    assert chain.run('run', root, 0) == 75
    assert submissions == [('run', root, 0, '100')]
    assert calls[0][-1] == 'required'
    assert json.loads((directory / 'chain.json').read_text())['attempts'][0]['continuation'] == '201'


def test_completed_initial_dispatches_two_arms_once(setup_chain):
    root, submissions = setup_chain
    directory = root / 'initial/42-initial'
    directory.mkdir(parents=True)
    (directory / 'complete.json').write_text('{"identity":"test"}')
    assert chain.run('initial', root, 0) == 0
    assert chain.run('initial', root, 0) == 0
    assert submissions == [('run', root, 0, '100'), ('run', root, 1, '100')]


def test_completed_run_does_not_restart(setup_chain, monkeypatch):
    root, submissions = setup_chain
    directory = root / 'runs/42-tournament'
    directory.mkdir(parents=True)
    (directory / 'complete.json').write_text('{"identity":"test"}')
    monkeypatch.setattr(chain.subprocess, 'Popen', lambda *a, **k: pytest.fail('completed run restarted'))
    assert chain.run('run', root, 0) == 0
    assert not submissions


def test_failure_does_not_loop(setup_chain, monkeypatch):
    root, submissions = setup_chain
    monkeypatch.setattr(chain.subprocess, 'Popen', lambda *a, **kw: SimpleNamespace(wait=lambda: 1))
    assert chain.run('run', root, 0) == 1
    assert not submissions


def test_missing_checkpoint_requires_investigation(setup_chain):
    root, submissions = setup_chain
    (root / 'runs/42-tournament/work').mkdir(parents=True)
    with pytest.raises(RuntimeError, match='no committed checkpoint'):
        chain.run('run', root, 0)
    assert not submissions


def test_submission_uses_array_index_and_dependency(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(chain.subprocess, 'check_output', lambda args, **kw: calls.append(args) or '123;cluster\n')
    assert chain.submit('run', tmp_path, 7, '99') == '123'
    assert '--array=7' in calls[0]
    assert '--dependency=afterany:99' in calls[0]


def test_extension_preserves_frozen_inputs_and_original(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('extend_campaign', ROOT / 'scripts/extend_difficulty_campaign.py')
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    from src.checkpointing import digest
    source, destination = tmp_path / 'original', tmp_path / 'additional'
    source.mkdir()
    manifest = {'seeds': [42, 43, 44], 'final_generation': 16}
    for name, key in [('source.tar.gz', 'source_sha256'), ('behaviors.csv', 'behaviors_sha256'), ('panel.json', 'panel_sha256')]:
        (source / name).write_text(name)
        manifest[key] = digest(source / name)
    manifest['identity'] = extension.json_hash(manifest)
    original = json.dumps(manifest)
    (source / 'manifest.json').write_text(original)
    monkeypatch.setattr(extension, 'validate_runtime', lambda m: None)
    extension.extend(source, destination, list(range(45, 52)))
    updated = extension.load_manifest(destination)
    assert updated['seeds'] == list(range(45, 52))
    assert updated['final_generation'] == 16
    assert updated['identity'] != manifest['identity']
    for name in ('source.tar.gz', 'behaviors.csv', 'panel.json'):
        assert (source / name).read_bytes() == (destination / name).read_bytes()
    assert (source / 'manifest.json').read_text() == original
    assert json.loads((destination / 'cohort.json').read_text())['all_seeds'] == list(range(42, 52))
    with pytest.raises(ValueError, match='overlap'):
        extension.extend(source, tmp_path / 'bad', [42])
