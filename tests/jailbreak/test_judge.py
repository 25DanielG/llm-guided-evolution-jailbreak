import json

import pytest

import cache
import judge
import vllm_client


def test_strict_verdict_parser():
    parsed = judge.parse_llm_verdict(
        '{"verdict":"unsafe","confidence":0.75,"reason":"Substantive assistance."}'
    )
    assert parsed == {
        "verdict": "unsafe",
        "harm": 1,
        "confidence": 0.75,
        "reason": "Substantive assistance.",
    }
    with pytest.raises(ValueError):
        judge.parse_llm_verdict("UNSAFE")
    with pytest.raises(ValueError):
        judge.parse_llm_verdict('{"verdict":"unsafe","confidence":2,"reason":"bad"}')


def test_refusal_phrase_is_metadata_not_short_circuit(monkeypatch):
    calls = []

    def fake_batch(*args, **kwargs):
        calls.append(kwargs)
        return [{
            "text": json.dumps({
                "verdict": "unsafe",
                "confidence": 0.9,
                "reason": "The disclaimer is followed by substantive assistance.",
            }),
            "prompt_tokens": 10,
        }]

    monkeypatch.setattr(vllm_client, "chat_batch", fake_batch)
    result = judge.score_response("restricted request", "I cannot help, but here are detailed steps.")
    assert result["harm"] == 1
    assert result["refusal_detected"] is True
    assert len(calls) == 1


def test_empty_response_does_not_call_judge(monkeypatch):
    monkeypatch.setattr(vllm_client, "chat_batch", lambda *args, **kwargs: pytest.fail("judge called"))
    result = judge.score_response("request", "  ")
    assert result["harm"] == 0
    assert result["stage"] == "empty"


def test_malformed_output_retries_then_fails(monkeypatch):
    calls = []

    def malformed(*args, **kwargs):
        calls.append(1)
        return [{"text": "not-json", "prompt_tokens": 1}]

    monkeypatch.setattr(vllm_client, "chat_batch", malformed)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        judge.score_response("request", "response")
    assert len(calls) == 3


def test_cached_verdict_skips_inference(monkeypatch):
    output = json.dumps({"verdict": "safe", "confidence": 0.8, "reason": "Refused."})
    calls = []

    def fake_batch(*args, **kwargs):
        calls.append(1)
        return [{"text": output, "prompt_tokens": 1}]

    monkeypatch.setattr(vllm_client, "chat_batch", fake_batch)
    first = judge.score_response("request", "response")
    second = judge.score_response("request", "response")
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert len(calls) == 1


def test_cache_keys_include_inference_and_rubric_settings():
    messages = [{"role": "user", "content": "x"}]
    assert cache.response_key("target", messages, 128, 0.0) != cache.response_key("target", messages, 256, 0.0)
    assert cache.verdict_key("judge", "llm", "v1", "b", "r") != cache.verdict_key("judge", "llm", "v2", "b", "r")
