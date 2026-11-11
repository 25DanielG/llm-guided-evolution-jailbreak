"""Ensure ordinary collection cannot accidentally become a live deployment."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
import transformers
import conftest as policy


def test_live_startup_requires_both_explicit_options(monkeypatch):
    start = Mock()
    monkeypatch.setattr(policy, '_start_slurm_server', start)
    monkeypatch.setattr(policy, '_start_local_server', start)
    for opt_in, auto_start in [(False, '1'), (True, '0'), (False, '0')]:
        monkeypatch.setenv('LLMGE_AUTO_START_SERVER', auto_start)
        policy.pytest_sessionstart(SimpleNamespace(config=SimpleNamespace(getoption=lambda *a: opt_in)))
    start.assert_not_called()


@pytest.mark.parametrize('enabled', [False, True])
def test_special_markers_require_explicit_opt_in(enabled):
    items = [SimpleNamespace(name=m, get_closest_marker=lambda marker, own=m: marker == own)
             for m in ('ordinary', 'live_inference', 'gpu', 'dataset')]
    deselect = Mock()
    config = SimpleNamespace(getoption=lambda option: enabled,
                             hook=SimpleNamespace(pytest_deselected=deselect))
    policy.pytest_collection_modifyitems(config, items)
    assert [i.name for i in items] == (['ordinary', 'live_inference', 'gpu', 'dataset'] if enabled else ['ordinary'])
    assert deselect.call_count == (0 if enabled else 1)


def test_unit_test_guard_blocks_real_http_and_weights():
    with pytest.raises(pytest.fail.Exception, match='Ordinary test attempted'):
        requests.get('http://unavailable-model.invalid')
    with pytest.raises(pytest.fail.Exception, match='Ordinary test attempted'):
        transformers.AutoModelForCausalLM.from_pretrained('never-download-this')
