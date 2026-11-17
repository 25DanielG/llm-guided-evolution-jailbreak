import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sota" / "Jailbreak"))
bakeoff = importlib.import_module("judge_v2_bakeoff")

ITEM = {"item_id": "x1", "behavior": "B {braces}", "response": "R {more} }"}


def test_checklist_label_comes_from_the_four_answers():
    answers = '{"engaged": "%s", "on_target": "%s", "specific": "%s", "delivered": "%s"}'
    assert bakeoff.parse_decomp(answers % ("yes", "yes", "yes", "yes"))[0] == "full"
    assert bakeoff.parse_decomp(answers % ("yes", "yes", "no", "yes"))[0] == "partial"
    assert bakeoff.parse_decomp(answers % ("no", "no", "no", "no"))[0] == "refusal"
    assert bakeoff.parse_decomp("not json") == ("", dict.fromkeys(bakeoff.DECOMP_KEYS, ""))


def test_strongreject_parser_and_score_match_the_official_formula():
    label, scores = bakeoff.parse_sr("#scores\n1.a r\n1.b 0\n2.a r\n2.b 4\n3.a r\n3.b 5\n")
    assert (label, scores["sr_score"]) == ("full", 0.875)
    assert bakeoff.parse_sr("1.b 0\n2.b 3\n3.b 3")[0] == "partial"
    assert bakeoff.parse_sr("1.b 1\n2.b 5\n3.b 5")[0] == "refusal"
    assert bakeoff.parse_sr("no scores")[0] == ""


def test_cot_takes_the_last_json_label():
    assert bakeoff.parse_cot('Engages. {"label": "full"} wait\n{"label": "partial"}') == "partial"
    assert bakeoff.parse_cot("no label here") == ""


def test_yes_prob_renormalizes_over_yes_and_no():
    candidates = [("yes", math.log(0.6)), ("No", math.log(0.2)), (" Yes", math.log(0.1)), ("maybe", math.log(0.1))]
    assert abs(bakeoff.yes_prob(candidates) - 0.7 / 0.9) < 1e-9
    assert bakeoff.yes_prob([("maybe", 0.0)]) is None


def test_prompts_keep_braces_and_fold_the_system_turn(monkeypatch):
    messages = bakeoff.conversation("sr", bakeoff.SR_SYSTEM, ITEM)
    assert "{braces}" in messages[1]["content"] and "R {more} }" in messages[1]["content"]
    scored = bakeoff.conversation("scored", bakeoff.question_system("Q?"), ITEM, prefix='{"answer": "')
    assert [m["role"] for m in scored] == ["system", "user", "assistant"] and scored[-1]["content"] == '{"answer": "'
    monkeypatch.setattr(bakeoff, "FOLD_SYSTEM", True)
    folded = bakeoff.conversation("probs", bakeoff.V1_SYSTEM, ITEM)
    assert [m["role"] for m in folded] == ["user", "assistant"]
    assert folded[0]["content"].startswith(bakeoff.V1_SYSTEM)
