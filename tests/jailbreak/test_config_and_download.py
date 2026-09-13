import csv
import hashlib
import sys

import pytest

import jb_config
import pull_behaviors
import vllm_client


def test_url_environment_takes_precedence_over_host_file(tmp_path, monkeypatch):
    host_file = tmp_path / "target_host.log"
    host_file.write_text("node:8001\n")
    monkeypatch.setattr(jb_config, "TARGET_HOST_FILE", str(host_file))
    assert jb_config.target_base_url() == "http://node:8001/v1"
    monkeypatch.setenv("JB_TARGET_VLLM_URL", "http://override:9000/v1/")
    assert jb_config.target_base_url() == "http://override:9000/v1"


def test_downloader_verifies_hash_and_preserves_stable_source_ids(tmp_path, monkeypatch):
    payload = b"goal,target\nfirst,response\nsecond,response\n"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return payload

    monkeypatch.setattr(pull_behaviors.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    output = tmp_path / "sample.csv"
    monkeypatch.setattr(sys, "argv", [
        "pull_behaviors.py",
        "--url", "https://example.invalid/data.csv",
        "--sha256", hashlib.sha256(payload).hexdigest(),
        "--n", "2",
        "--out", str(output),
        "--source", "fixture",
    ])
    pull_behaviors.main()
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["id"] for row in rows] == ["fixture_001", "fixture_002"]
    assert all(row["source"] == "fixture" for row in rows)


def test_downloader_refuses_checksum_mismatch(monkeypatch, tmp_path):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"goal\nvalue\n"

    monkeypatch.setattr(pull_behaviors.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    monkeypatch.setattr(sys, "argv", [
        "pull_behaviors.py", "--sha256", "0" * 64, "--out", str(tmp_path / "bad.csv"),
    ])
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        pull_behaviors.main()


def test_partial_batch_failure_is_not_silently_ignored(monkeypatch):
    def fake_chat(base_url, model, messages, **kwargs):
        if messages[0]["content"] == "fail":
            raise RuntimeError("HTTP failure")
        return {"text": "ok", "prompt_tokens": 1}

    monkeypatch.setattr(vllm_client, "chat", fake_chat)
    with pytest.raises(RuntimeError, match="HTTP failure"):
        vllm_client.chat_batch(
            "http://localhost/v1",
            "target",
            [[{"role": "user", "content": "ok"}], [{"role": "user", "content": "fail"}]],
            max_concurrency=2,
        )


def test_http_failure_retries_then_raises(monkeypatch):
    attempts = []

    def fail(*args, **kwargs):
        attempts.append(1)
        raise vllm_client.requests.exceptions.Timeout("timed out")

    monkeypatch.setattr(vllm_client.requests, "post", fail)
    monkeypatch.setattr(vllm_client.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError, match="failed after 3 tries"):
        vllm_client.chat("http://localhost/v1", "target", [], retries=2)
    assert len(attempts) == 3
