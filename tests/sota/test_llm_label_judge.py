import importlib
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
