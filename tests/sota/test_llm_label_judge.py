import importlib

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sota" / "Jailbreak"))
judge = importlib.import_module("llm_label_judge")


def test_parse_reply_json_and_fallbacks():
    assert judge.parse_reply('{"label": "partial", "confidence": "high"}') == ("partial", "high")
    assert judge.parse_reply('Sure. {"label": "FULL", "confidence": "Low"} done') == ("full", "low")
    assert judge.parse_reply('{"label": "full", "confidence": "maybe"}') == ("full", "")
    assert judge.parse_reply("The answer is refusal.") == ("refusal", "")
    assert judge.parse_reply("partial or full?") == ("", "")  # ambiguous
    assert judge.parse_reply('{"label": "unsafe"}') == ("", "")
    assert judge.parse_reply("") == ("", "")


def test_messages_use_guide_and_only_behavior_and_response():
    msgs = judge.messages_for("behavior A", "response text 1")
    assert msgs[0]["role"] == "system" and "Pick the lower one" in msgs[0]["content"]
    assert msgs[1]["content"] == "BEHAVIOR:\nbehavior A\n\nRESPONSE:\nresponse text 1"


def test_fold_system_puts_guide_in_the_only_user_turn():
    msgs = judge.messages_for("behavior A", "response text 1", fold_system=True)
    assert [m["role"] for m in msgs] == ["user"]
    assert msgs[0]["content"] == judge.SYSTEM + "\n\nBEHAVIOR:\nbehavior A\n\nRESPONSE:\nresponse text 1"


def test_label_probs_sums_label_prefix_tokens_and_renormalizes():
    import math
    probs = judge.label_probs([("full", math.log(0.5)), (" part", math.log(0.2)), ("ref", math.log(0.1)),
                               ("f", math.log(0.1)), ("unsafe", math.log(0.1))])  # "f" is too short, "unsafe" no label
    assert probs["full"] == pytest.approx(0.625) and probs["partial"] == pytest.approx(0.25)
    assert probs["refusal"] == pytest.approx(0.125)
    assert judge.label_probs([("unsafe", 0.0)]) is None
