"""Mocked submission tests: real script creation, no inference or scheduler."""
import urllib.error
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import run_improved


@pytest.mark.mock_inference
def test_individual(tmp_path, monkeypatch):
    output = tmp_path / "evolution"
    variants = tmp_path / "variants"
    for name, value in {"OUTPUT_DIR": str(output), "VARIANT_DIR": str(variants), "GENERATION": 0,
                        "GLOBAL_DATA": {}, "GLOBAL_DATA_ANCESTRY": {}, "GEN_COUNT": -1,
                        "DELAYED_CHECK": True, "LOCAL": True}.items():
        monkeypatch.setattr(run_improved, name, value)
    monkeypatch.setattr(run_improved, "wait_for_llm_server", lambda: True)
    monkeypatch.setattr(run_improved, "generate_random_string", lambda **kwargs: "unit-gene")
    monkeypatch.setattr(run_improved, "load_yaml", lambda: {"llm_bash_script": "#!/bin/bash\n{}\n", "gpu_selection": ""})
    submit = Mock(return_value=SimpleNamespace(returncode=0, stdout="Job done", stderr=""))
    monkeypatch.setattr(run_improved.subprocess, "run", submit)

    individual = run_improved.toolbox.individual(llm_model="llama3")

    script = output / "0/unit-gene.sh"
    assert individual[0] == "unit-gene"
    assert script.is_file()
    text = script.read_text()
    assert "src/llm_mutation.py" in text and "--llm_model llama3" in text
    assert str(variants / "prompt_unit-gene.py") in text
    assert (output / "0/unit-gene_model.txt").is_file()
    assert run_improved.GLOBAL_DATA["unit-gene"]["sub_flag"] is True
    submit.assert_called_once()
    assert submit.call_args.args[0][1] == str(script)


@pytest.mark.parametrize("ready", [True, False])
def test_readiness_uses_fake_clock(ready, tmp_path, monkeypatch):
    host = tmp_path / "hostname.log"
    host.write_text("test-compute\n")
    monkeypatch.setattr(run_improved, "HOSTNAME_DIR", str(host))
    monkeypatch.setattr(run_improved, "LOCAL_LLM", True)
    monkeypatch.setattr(run_improved, "LLM_SERVER_READY", False)
    monkeypatch.setenv("LLM_SERVER_READY_TIMEOUT", "2")
    monkeypatch.setenv("LLM_SERVER_READY_CHECK_INTERVAL", "1")
    now = [0]
    monkeypatch.setattr(run_improved.time, "time", lambda: now[0])
    monkeypatch.setattr(run_improved.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    from unittest.mock import MagicMock
    response = MagicMock()
    response.__enter__.return_value.status = 200
    urlopen = Mock(return_value=response) if ready else Mock(side_effect=urllib.error.URLError("offline"))
    monkeypatch.setattr(run_improved.urllib.request, "urlopen", urlopen)
    assert run_improved.wait_for_llm_server() is ready
    assert run_improved.LLM_SERVER_READY is ready
    assert urlopen.call_count == (1 if ready else 3)


def test_cached_readiness_never_contacts_service(monkeypatch):
    monkeypatch.setattr(run_improved, "LOCAL_LLM", True)
    monkeypatch.setattr(run_improved, "LLM_SERVER_READY", True)
    assert run_improved.wait_for_llm_server()


def test_failed_readiness_does_not_submit(monkeypatch, tmp_path):
    monkeypatch.setattr(run_improved, "wait_for_llm_server", lambda: False)
    submit = Mock()
    monkeypatch.setattr(run_improved.subprocess, "run", submit)
    assert run_improved.submit_bash(str(tmp_path / "absent.sh")) == (False, None, None)
    submit.assert_not_called()
