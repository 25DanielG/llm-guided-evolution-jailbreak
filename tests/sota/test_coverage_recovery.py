"""Recover exact recorded sources without altering current edits or checkpoints."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.experiments import recovery as r


def test_recovers_head_version_without_overwriting_current_file(tmp_path, monkeypatch):
    source = tmp_path / 'server.py'
    source.write_bytes(b'current edited source\n')
    old = b'recorded old source\n'
    monkeypatch.setattr(r, 'head_version', lambda repo, name: old)
    assert r.recover_source(tmp_path, 'server.py', r.sha(old)) == old
    assert source.read_bytes() == b'current edited source\n'


def test_recovers_verification_function_preserving_experiment_instrumentation(tmp_path, monkeypatch):
    source = tmp_path / 'sota/Jailbreak/eval.py'
    source.parent.mkdir(parents=True)
    head = b'base = True\ndef verify_record(x):\n    return 0\n\ndef append_records(x):\n    pass\n'
    current = b'experiment = True\ndef verify_record(x):\n    return .7\n\ndef append_records(x):\n    pass\n'
    expected = current.replace(b'return .7', b'return 0')
    source.write_bytes(current)
    monkeypatch.setattr(r, 'head_version', lambda repo, name: head)
    assert r.recover_source(tmp_path, 'sota/Jailbreak/eval.py', r.sha(expected)) == expected
    assert source.read_bytes() == current
    with pytest.raises(ValueError, match='Cannot recover'):
        r.recover_source(tmp_path, 'sota/Jailbreak/eval.py', r.sha(b'unknown version'))


def test_snapshot_preserves_original_run_metadata_and_rejects_tampering(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'server.py').write_bytes(b'new source\n')
    python = repo / '.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.touch()
    root = tmp_path / 'campaign'
    run = root / 'runs/run1'
    run.mkdir(parents=True)
    recorded = b'old source\n'
    config = run / 'run.json'
    config.write_text(json.dumps({'source_hashes': {'server.py': r.sha(recorded)}}))
    before = config.read_bytes()
    monkeypatch.setattr(r, 'head_version', lambda repo, name: recorded)
    monkeypatch.setattr(r, 'source_files', lambda repo, required: ['server.py'])
    snapshot = r.prepare_runtime(root, repo)
    assert (snapshot / 'server.py').read_bytes() == recorded
    assert (repo / 'server.py').read_bytes() == b'new source\n'
    assert config.read_bytes() == before
    assert (snapshot / '.venv').is_symlink()
    # Further unrelated source edits and a different HEAD must not lose the
    # already recovered version or require rewriting existing run metadata.
    (repo / 'server.py').write_bytes(b'another edit\n')
    monkeypatch.setattr(r, 'head_version', lambda repo, name: b'new HEAD\n')
    assert r.prepare_runtime(root, repo) == snapshot
    (snapshot / 'server.py').write_bytes(b'tampered\n')
    monkeypatch.setattr(r, 'head_version', lambda repo, name: recorded)
    with pytest.raises(ValueError, match='Frozen runtime was modified'):
        r.prepare_runtime(root, repo)


def test_mixed_source_cohorts_are_not_silently_combined(tmp_path):
    for name, version in [('run1', 'one'), ('run2', 'two')]:
        run = tmp_path / 'runs' / name
        run.mkdir(parents=True)
        (run / 'run.json').write_text(json.dumps({'source_hashes': {'server.py': version}}))
    with pytest.raises(ValueError, match='different source versions'):
        r.prepare_runtime(tmp_path, tmp_path)
    assert not (tmp_path / 'runtimes').exists()
