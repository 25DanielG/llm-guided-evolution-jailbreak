import csv
import fcntl
import hashlib
import importlib
import json
import math
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sota" / "Jailbreak"))
reporting = importlib.import_module("reporting_judge")
judge = importlib.import_module("judge")
cache = importlib.import_module("cache")
SCRIPT = REPO / "sota" / "Jailbreak" / "reporting_judge.py"
OFFICIAL_SHA256 = "788f4f6aa1491c433c4da76c9140cfc30966cea3ff3875c4d0fcb336d92f60e0"
RUBRIC = {"refused": 0, "on_topic": 1, "specificity": 4, "convincingness": 4}


def write_behaviors(tmp_path):
    path = tmp_path / "behaviors.csv"
    path.write_text("id,source,behavior\nbeh_a,test,behavior A\nbeh_b,test,behavior B\n", encoding="utf-8")
    return reporting.load_behaviors(path), path


def ok(score, rubric=RUBRIC, filter_name=None):
    return {"score": score, "parse_ok": True, "filter": filter_name, "rubric": rubric,
            "rating_method": "parsed_integer" if rubric else None, "error": None, "usage": {}, "latency_seconds": 0}


def row(run_id, generation, candidate_id, index, behavior, response, in_loop, verification=None, usage=None):
    return {"run_id": run_id, "generation": generation, "candidate_id": candidate_id, "behavior_index": index,
            "behavior": behavior, "evolved_prompt": [{"role": "user", "content": "prompt"}],
            "target_response": response, "target_model": "target", "judge_model": "judge-x",
            "reporting_model": "cls", "target_usage": {"completion_tokens": 5} if usage is None else usage,
            "target_latency_seconds": 0, "target_cached": False, "in_loop": in_loop, "verification": verification}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def logprob(token, probability):
    return SimpleNamespace(decoded_token=token, logprob=math.log(probability))


class FakeTokenizer:
    """Whitespace tokenizer with a BOS token; decode re-joins with single spaces."""
    def encode(self, text, max_length=None, truncation=False):
        ids = ["<s>"] + text.split()
        return ids[:max_length] if truncation else ids

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(t for t in ids if not (skip_special_tokens and t == "<s>"))


def test_official_prompt_hash_and_model_card_parity(monkeypatch):
    assert hashlib.sha256(reporting.OFFICIAL_PROMPT.encode("utf-8")).hexdigest() == OFFICIAL_SHA256
    seen = []
    monkeypatch.setattr(judge.vllm_client, "completion",
                        lambda url, model, prompt, **kwargs: seen.append(prompt) or {"text": "No"})
    judge.reporting_score("behavior A", "response text 1")
    assert seen == [reporting.MODEL_CARD_PROMPT.format(behavior="behavior A", generation="response text 1")]


def test_parse_label():
    texts = ("Yes", " yes", "NO", "no\n", "Maybe", "", "yes.")
    assert [reporting.parse_label(t) for t in texts] == [1, 1, 0, 0, -1, -1, -1]


def test_p_yes_from_logprobs():
    top = {1: logprob("Yes", 0.6), 2: logprob(" yes", 0.1), 3: logprob("No", 0.2), 4: logprob("The", 0.1)}
    assert reporting.p_yes_from_logprobs(top) == pytest.approx(0.7 / 0.9)
    assert reporting.p_yes_from_logprobs({1: logprob("The", 1.0)}) == ""
    assert reporting.p_yes_from_logprobs(None) == ""
    raw_pieces = {1: logprob("▁Yes", 0.3), 2: logprob("▁No", 0.6)}
    assert reporting.p_yes_from_logprobs(raw_pieces) == pytest.approx(0.3 / 0.9)


def test_clip_generation_truncates_and_always_decodes():
    tokenizer = FakeTokenizer()
    clipped, truncated = reporting.clip_generation(tokenizer, " ".join(f"w{i}" for i in range(600)))
    assert truncated == 1 and clipped.split() == [f"w{i}" for i in range(511)]  # 512 incl. BOS
    assert reporting.clip_generation(tokenizer, " ".join(["w"] * 511)) == (" ".join(["w"] * 511), 0)
    assert reporting.clip_generation(tokenizer, "response   text 1") == ("response text 1", 0)


def test_harmbench_scores_with_fake_llm():
    class FakeLLM:
        def __init__(self):
            self.calls = []

        def generate(self, prompts, params):
            self.calls.append(prompts)
            return [SimpleNamespace(outputs=[SimpleNamespace(
                text="Maybe" if "response text 2" in p else "Yes",
                logprobs=[{1: logprob("Yes", 0.8), 2: logprob("No", 0.2)}])]) for p in prompts]

    llm = FakeLLM()
    pairs = [("behavior A", "response text 1"), ("behavior B", "response text 2"), ("behavior A", "word " * 2000)]
    counts = Counter()
    results = reporting.harmbench_scores(pairs, FakeTokenizer(), llm, None, 600, counts)
    assert results[pairs[0]] == {"harmbench_score": 1, "harmbench_p_yes": pytest.approx(0.8), "harmbench_raw": "Yes",
                                 "harmbench_truncated": 0, "harmbench_card_score": 1,
                                 "harmbench_card_p_yes": pytest.approx(0.8)}
    assert results[pairs[1]]["harmbench_score"] == -1 and results[pairs[1]]["harmbench_card_score"] == -1
    assert results[pairs[2]] == {"harmbench_score": -1, "harmbench_p_yes": "", "harmbench_raw": "",
                                 "harmbench_truncated": 1, "harmbench_card_score": -1, "harmbench_card_p_yes": ""}
    assert len(llm.calls) == 2 and all(len(batch) == 2 for batch in llm.calls)
    assert "[BEHAVIOR]:\nbehavior A" in llm.calls[0][0] and "Behavior:\nbehavior A" in llm.calls[1][0]
    assert counts == Counter(unique_pairs=3, truncated=1, too_long_official=1, too_long_card=1,
                             unparseable_official=1, unparseable_card=1)


def test_load_candidates(tmp_path):
    behavior_ids, _ = write_behaviors(tmp_path)
    run_dir = tmp_path / "results" / "run1"
    path = write_jsonl(run_dir / "candidates.jsonl", [
        row("run1", 0, "c1", 0, "behavior A", "response text 1", ok(0.75),
            verification={"trials": [], "successes": 2, "verified_success": True}),
        row("run1", 1, "c2", 0, "behavior A", "response text 1", ok(0.5)),  # conflicting duplicate
        row("run1", 0, "c1", 1, "behavior B", "response text 2", judge._zero(parse_ok=False, error="malformed_rubric")),
        row("run1", 1, "c3", 1, "behavior B", "response text 2", ok(0.0, rubric=None, filter_name="refusal_only")),
        row("run1", 0, "c1", 2, "behavior A", "response text 3", judge._zero(parse_ok=False, error="malformed_rubric")),
        row("run1", 0, "c1", 3, "behavior A", None, judge._zero(filter_name="empty")),
        row("run1", 0, "c1", 4, "behavior B", "", judge._zero(filter_name="empty"), usage={}),
        row("run1", 0, "c1", 5, "behavior C", "", judge._zero(filter_name="empty")),
        row("run2", 0, "c1", 0, "behavior A", "response text 1", ok(0.5)),
    ])
    write_jsonl(run_dir / "audits.jsonl", [
        {"run_id": "run1", "generation": 0, "candidate_id": "c1", "behavior_index": 0, "in_loop_score": 0.75,
         "reporting": {"success": True, "parse_ok": True, "raw": "Yes"}, "score_gap": -0.25}])
    items, counts = {}, Counter()
    reporting.load_candidates([path], behavior_ids, 0.5, items, counts)
    assert counts == Counter(candidates_rows=9, null_response=1, infra_empty=1, duplicate_rows=2, inloop_conflict=1)
    assert len(items) == 5

    a = items[("run1", "behavior A", "response text 1")]
    assert a["item_id"] == hashlib.sha256(b"run1\x00behavior A\x00response text 1").hexdigest()[:12]
    assert (a["behavior_id"], a["n_rows"], a["inloop_score"], a["inloop_yes"], a["inloop_gate"]) == ("beh_a", 2, 0.75, 1, 1)
    assert (a["generation"], a["candidate_id"], a["behavior_index"], a["judge_model"]) == (0, "c1", 0, "judge-x")
    assert (a["team_verify_successes"], a["team_verified_success"], a["team_audit_success"]) == (2, 1, 1)
    assert (a["rubric_specificity"], a["rating_method"], a["source"]) == (4, "parsed_integer", "candidates")

    b = items[("run1", "behavior B", "response text 2")]  # parse_ok verdict replaces the earlier failure
    assert (b["n_rows"], b["inloop_score"], b["inloop_yes"], b["inloop_gate"], b["inloop_parse_ok"]) == (2, 0.0, 0, 0, 1)
    assert (b["inloop_filter"], b["rubric_refused"], b["generation"], b["candidate_id"]) == ("refusal_only", "", 0, "c1")
    assert (b["team_verify_successes"], b["team_verified_success"], b["team_audit_success"]) == ("", "", "")

    failed = items[("run1", "behavior A", "response text 3")]
    assert [failed[k] for k in ("inloop_score", "inloop_yes", "inloop_gate", "inloop_parse_ok")] == ["", "", "", 0]

    empty = items[("run1", "behavior C", "")]
    assert (empty["behavior_id"], empty["inloop_score"], empty["inloop_gate"], empty["inloop_filter"]) == ("", 0.0, 0, "empty")
    assert items[("run2", "behavior A", "response text 1")]["inloop_yes"] == 1
    assert items[("run2", "behavior A", "response text 1")]["item_id"] != a["item_id"]


def test_team_fields_filled_from_later_duplicate_rows(tmp_path):
    behavior_ids, _ = write_behaviors(tmp_path)
    run_dir = tmp_path / "results" / "run1"
    path = write_jsonl(run_dir / "candidates.jsonl", [
        row("run1", 0, "c1", 0, "behavior A", "response text 1", ok(0.75)),
        row("run1", 1, "c2", 0, "behavior A", "response text 1", ok(0.75)),  # only this row was audited
        row("run1", 0, "c1", 1, "behavior B", "response text 2", judge._zero(parse_ok=False, error="timeout")),
        row("run1", 1, "c3", 1, "behavior B", "response text 2", ok(0.9),
            verification={"trials": [], "successes": 3, "verified_success": True}),
    ])
    audit = {"run_id": "run1", "generation": 1, "candidate_id": "c2", "behavior_index": 0, "in_loop_score": 0.75,
             "score_gap": -0.25}
    write_jsonl(run_dir / "audits.jsonl", [
        {**audit, "reporting": {"success": False, "parse_ok": False, "error": "timeout"}},
        {**audit, "reporting": {"success": True, "parse_ok": True, "raw": "Yes"}},  # re-audit of the same key
        {**audit, "reporting": {"success": False, "parse_ok": False, "error": "timeout"}},
    ])
    items, counts = {}, Counter()
    reporting.load_candidates([path], behavior_ids, 0.5, items, counts)
    a = items[("run1", "behavior A", "response text 1")]
    assert (a["generation"], a["candidate_id"], a["team_audit_success"], a["team_verify_successes"]) == (0, "c1", 1, "")
    b = items[("run1", "behavior B", "response text 2")]
    assert (b["inloop_yes"], b["team_verify_successes"], b["team_verified_success"], b["candidate_id"]) == (1, 3, 1, "c1")


def test_load_audits_waits_for_writer_lock(tmp_path):
    path = tmp_path / "audits.jsonl"
    line = json.dumps({"run_id": "run1", "generation": 0, "candidate_id": "c1", "behavior_index": 0,
                       "reporting": {"success": True, "parse_ok": True, "raw": "Yes"}}) + "\n"
    result = {}
    with open(path, "a", encoding="utf-8") as writer:
        fcntl.flock(writer, fcntl.LOCK_EX)
        writer.write(line[:30])
        writer.flush()
        reader = threading.Thread(target=lambda: result.update(audits=reporting.load_audits(path)))
        reader.start()
        reader.join(0.3)
        assert reader.is_alive()  # waiting on the shared lock, not parsing the partial line
        writer.write(line[30:])
        writer.flush()
        fcntl.flock(writer, fcntl.LOCK_UN)
    reader.join(5)
    assert result["audits"] == {("run1", 0, "c1", 0): {"success": True, "parse_ok": True, "raw": "Yes"}}

def test_load_cache(tmp_path, monkeypatch):
    behavior_ids, _ = write_behaviors(tmp_path)
    monkeypatch.setattr(cache, "_conn", None)
    cache_dir = str(tmp_path / "cache")
    for prompt, text in (("prompt 1", "response text 1"), ("prompt 2", "response text 2"), ("prompt 3", "response text 1")):
        key = cache.response_key("target", [{"role": "user", "content": prompt}])
        cache.put(cache_dir, key, json.dumps({"text": text, "prompt_tokens": 3, "usage": {}, "latency_seconds": 0}))
    cache.put(cache_dir, cache.verdict_key("judge-x:phase1-v1", "behavior A", "response text 1"), json.dumps(ok(1.0)))
    cache.put(cache_dir, cache.verdict_key("judge-y:phase1-v1", "behavior B", "response text 2"), json.dumps(ok(1.0)))
    cache._conn.close()  # checkpoints the WAL into the main file
    db = Path(cache_dir) / "jb_cache.sqlite"

    items, counts = {}, Counter()
    reporting.load_cache(db, "run-cache", "judge-x", "phase1-v1", behavior_ids, 0.5, items, counts)
    assert counts == Counter(cache_kv_rows=5, cache_response_texts=2, cache_no_verdict=1)
    item = items[("run-cache", "behavior A", "response text 1")]
    assert len(items) == 1 and item["source"] == "cache" and item["judge_model"] == "judge-x"
    assert (item["behavior_id"], item["inloop_score"], item["inloop_yes"], item["inloop_gate"], item["n_rows"]) == ("beh_a", 1.0, 1, 1, 1)
    assert (item["generation"], item["candidate_id"], item["behavior_index"], item["team_audit_success"]) == ("", "", "", "")
    reporting.load_cache(db, "run-cache", "judge-x", "phase1-v1", behavior_ids, 0.5, items, counts)
    assert len(items) == 1 and counts["cache_dup_of_candidates"] == 1


def test_dry_run_end_to_end(tmp_path):
    _, behaviors = write_behaviors(tmp_path)
    path = write_jsonl(tmp_path / "results" / "run1" / "candidates.jsonl", [
        row("run1", 0, "c1", 0, "behavior A", "response text 1,\nline 2", ok(0.75)),
        row("run1", 0, "c1", 1, "behavior B", None, judge._zero(filter_name="empty")),
    ])
    out = tmp_path / "out" / "reporting.csv"
    proc = subprocess.run([sys.executable, str(SCRIPT), "--candidates", str(path), "--behaviors", str(behaviors),
                           "--out", str(out), "--dry-run"], capture_output=True, text=True, check=True, cwd=REPO)
    assert "response text" not in proc.stdout + proc.stderr
    with open(out, newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    assert rows[0] == [
        "item_id", "behavior_id", "behavior", "response", "inloop_score", "harmbench_score", "run_id", "generation",
        "candidate_id", "behavior_index", "n_rows", "inloop_yes", "inloop_gate", "inloop_parse_ok", "inloop_filter",
        "rubric_refused", "rubric_on_topic", "rubric_specificity", "rubric_convincingness", "rating_method",
        "judge_model", "harmbench_p_yes", "harmbench_raw", "harmbench_truncated", "harmbench_card_score",
        "harmbench_card_p_yes", "team_verify_successes", "team_verified_success", "team_audit_success", "source"]
    record = dict(zip(rows[0], rows[1]))
    assert len(rows) == 2 and record["response"] == "response text 1,\nline 2" and record["inloop_score"] == "0.75"
    assert all(record[k] == "" for k in rows[0] if k.startswith("harmbench_"))
    summary = json.loads((out.parent / "reporting_summary.json").read_text(encoding="utf-8"))
    assert summary["items"] == 1 and summary["skipped"]["null_response"] == 1 and summary["harmbench"] is None
    assert summary["per_run"]["run1"]["items"] == 1
