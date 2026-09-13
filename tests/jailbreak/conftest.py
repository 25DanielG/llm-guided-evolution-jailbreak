import sys
from pathlib import Path

import pytest

JAILBREAK_DIR = Path(__file__).resolve().parents[2] / "sota" / "Jailbreak"
sys.path.insert(0, str(JAILBREAK_DIR))


@pytest.fixture(autouse=True)
def jailbreak_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("JB_VLLM_API_KEY", "test-key")
    monkeypatch.setenv("JB_JUDGE_MODE", "llm")
    monkeypatch.setenv("JB_JUDGE_SERVED_NAME", "judge")
    monkeypatch.setenv("JB_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("JB_JUDGE_PARSE_RETRIES", "2")
    import cache
    if cache._conn is not None:
        cache._conn.close()
        cache._conn = None
    yield
    if cache._conn is not None:
        cache._conn.close()
        cache._conn = None
