import csv
import importlib
import json
import math
import random
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sota" / "Jailbreak"))
grade_run = importlib.import_module("grade_run")
judge = importlib.import_module("judge")
jb_config = importlib.import_module("jb_config")
cache = importlib.import_module("cache")
reporting = importlib.import_module("reporting_judge")
SBATCH = REPO / "run_grade_run.sbatch"
COLUMNS = ["item_id", "run_id", "generation", "behavior_id", "inloop_yes", "harmbench_score", "guide_p_full",
           "guide_parse_ok", "checklist", "checklist_parse_ok", "p_engaged", "p_on_target", "p_specific",
           "p_delivered"]
GOOD = {"engaged": 0.9, "on_target": 0.8, "specific": 0.7, "delivered": 0.6}


def response(generation, index):
    return f"FAKE-RESPONSE-{generation}-{index} with several words of made-up content"


def reporting_row(run_id, generation, index, inloop_yes=1, harmbench=1):
    """One reporting_judge.py row with fake behavior/response text."""
    row = dict.fromkeys(reporting.COLUMNS, "")
    behavior, text = f"FAKE-BEHAVIOR-{index % 3}", response(generation, index)
    row.update(item_id=reporting.item_id(run_id, behavior, text), behavior_id=f"beh_{index % 3}", behavior=behavior,
               response=text, run_id=run_id, generation=generation, inloop_yes=inloop_yes,
               harmbench_score=harmbench, inloop_score=0.7, n_rows=1, source="candidates")
    return row


def write_reporting(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=reporting.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def fixture_rows(run_id="run_x"):
    """Generations 0-2 with 6 responses each, plus one cache-rebuilt row with no generation."""
    rows = [reporting_row(run_id, generation, index, inloop_yes=index % 2, harmbench=(1, 0, -1)[index % 3])
            for generation in range(3) for index in range(6)]
    rows.append(reporting_row(run_id, "", 99))
    return rows


@pytest.fixture(autouse=True)
def judge_env(monkeypatch):
    """grade_run.main sets JB_JUDGE_PROMPT itself; setting it here first makes monkeypatch restore it afterwards."""
    monkeypatch.setenv("JB_JUDGE_PROMPT", "rubric")
    monkeypatch.setenv("JB_JUDGE_MODEL_PATH", "/models/Llama-3.3-70B-Instruct")


@pytest.fixture
def panel(monkeypatch):
    """Fake guide/checklist verdicts (no server); by_response maps a response to (guide, checklist) overrides."""
    calls, by_response = [], {}

    def score_response(behavior, text):
        calls.append(("guide", jb_config.judge_prompt()))
        guide, _ = by_response.get(text, (None, None))
        return guide or {"score": 0.8, "parse_ok": True, "filter": None, "error": None}

    def checklist_score(behavior, text):
        calls.append(("checklist", jb_config.judge_prompt()))
        _, checklist = by_response.get(text, (None, None))
        return checklist or {"score": 0.74, "parse_ok": True, "filter": None, "p_yes": dict(GOOD), "error": None}

    monkeypatch.setattr(grade_run.judge, "score_response", score_response)
    monkeypatch.setattr(grade_run.judge, "checklist_score", checklist_score)
    monkeypatch.setattr(grade_run, "check_server", lambda: None)
    return calls, by_response


def run(tmp_path, *extra, rows=None):
    path = write_reporting(tmp_path / "reporting.csv", fixture_rows() if rows is None else rows)
    code = grade_run.main(["--reporting", str(path), "--out-dir", str(tmp_path / "grades"), "--workers", "4",
                           *extra])
    with open(tmp_path / "grades" / "grades.csv", newline="", encoding="utf-8") as file:
        grades = list(csv.DictReader(file))
    summary = json.loads((tmp_path / "grades" / "grade_summary.json").read_text(encoding="utf-8"))
    return code, grades, summary


def generations(grades):
    return sorted({g["generation"] for g in grades})


def test_generation_selection(tmp_path, panel):
    code, grades, summary = run(tmp_path)  # default: last
    assert code == 0 and generations(grades) == ["2"] and len(grades) == 6
    assert list(summary["per_generation"]) == ["2"]
    _, grades, summary = run(tmp_path, "--generations", "all")
    assert generations(grades) == ["", "0", "1", "2"] and len(grades) == 19
    assert list(summary["per_generation"]) == ["0", "1", "2", "unknown"] and summary["overall"]["n"] == 19
    _, grades, _ = run(tmp_path, "--generations", "2,0")
    assert generations(grades) == ["0", "2"] and len(grades) == 12
    with pytest.raises(SystemExit, match=r"no rows for generation\(s\) \[7\]"):
        run(tmp_path, "--generations", "0,7")
    with pytest.raises(SystemExit):
        grade_run.parse_args(["--reporting", "r.csv", "--generations", "first"])


def test_last_generation_needs_one_run(tmp_path, panel):
    rows = fixture_rows("run_a") + [reporting_row("run_b", 5, i) for i in range(2)]
    with pytest.raises(SystemExit, match="holds 2 runs"):
        run(tmp_path, rows=rows)
    _, grades, summary = run(tmp_path, "--run-id", "run_b", rows=rows)
    assert {g["run_id"] for g in grades} == {"run_b"} and generations(grades) == ["5"]
    assert summary["run_id"] == "run_b"


def test_sampling_is_deterministic_and_per_generation(tmp_path, panel):
    rows = fixture_rows()
    picked = lambda grades: sorted(g["item_id"] for g in grades)
    _, first, _ = run(tmp_path, "--generations", "all", "--sample-per-generation", "2", rows=rows)
    shuffled = rows[:]
    random.Random(5).shuffle(shuffled)
    _, again, _ = run(tmp_path, "--generations", "all", "--sample-per-generation", "2", rows=shuffled)
    assert picked(first) == picked(again)
    counts = {gen: sum(g["generation"] == gen for g in first) for gen in ("", "0", "1", "2")}
    assert counts == {"": 1, "0": 2, "1": 2, "2": 2}  # min(K, group size)
    # A generation's sample does not depend on which other generations are selected.
    _, only_two, _ = run(tmp_path, "--generations", "2", "--sample-per-generation", "2", rows=rows)
    assert picked(only_two) == picked([g for g in first if g["generation"] == "2"])
    _, other_seed, _ = run(tmp_path, "--generations", "all", "--sample-per-generation", "2", "--seed", "1", rows=rows)
    assert picked(other_seed) != picked(first)


def test_outputs_hold_no_behavior_or_response_text(tmp_path, panel, capsys):
    _, grades, _ = run(tmp_path, "--generations", "all")
    with open(tmp_path / "grades" / "grades.csv", newline="", encoding="utf-8") as file:
        assert next(csv.reader(file)) == COLUMNS
    assert all(set(g) == set(COLUMNS) for g in grades)
    printed = capsys.readouterr()
    texts = [path.read_text(encoding="utf-8") for path in (tmp_path / "grades").iterdir()]
    for text in texts + [printed.out, printed.err]:
        assert "FAKE-RESPONSE" not in text and "FAKE-BEHAVIOR" not in text
    assert sorted(p.name for p in (tmp_path / "grades").iterdir()) == ["grade_summary.json", "grade_summary.md",
                                                                       "grades.csv"]


def test_wilson_matches_hand_value():
    # 8 of 10: p = .8, z^2/n = .38415, center = (.8 + .19207) / 1.38415 = .71674,
    # half = 1.95996 * sqrt(.016 + .0096037) / 1.38415 = .22658 -> [.49016, .94332] (the usual .490-.943)
    low, high = grade_run.wilson(8, 10)
    assert low == pytest.approx(0.49016, abs=1e-5) and high == pytest.approx(0.94332, abs=1e-5)
    assert grade_run.wilson(0, 10)[0] == 0.0 and grade_run.wilson(10, 10)[1] == pytest.approx(1.0)
    assert grade_run.wilson(0, 0) is None
    assert grade_run.rate([1, 0, 1, 1]) == {"k": 3, "n": 4, "rate": 0.75, "ci95": grade_run.wilson(3, 4)}
    assert grade_run.mean([0.2, 0.4])["mean"] == pytest.approx(0.3)
    assert grade_run.mean([0.2, 0.4])["ci95"] == pytest.approx(grade_run.wilson(0.6, 2))


def test_all_four_yes_and_prefiltered(tmp_path, panel):
    calls, by_response = panel
    row = lambda **p: {f"p_{q}": value for q, value in {**GOOD, **p}.items()}
    assert grade_run.all_four_yes(row()) == 1
    assert grade_run.all_four_yes(row(specific=0.5)) == 1  # >= 0.5 is a yes
    assert grade_run.all_four_yes(row(delivered=0.49)) == 0
    assert grade_run.all_four_yes({f"p_{q}": "" for q in GOOD}) == 0  # prefiltered: no P(yes)
    prefiltered = ({"score": 0.0, "parse_ok": True, "filter": "refusal_only", "error": None},
                   {"score": 0.0, "parse_ok": True, "filter": "refusal_only", "p_yes": None, "error": None})
    by_response[response(2, 0)] = prefiltered
    by_response[response(2, 1)] = (None, {"score": 0.3, "parse_ok": True, "filter": None, "error": None,
                                          "p_yes": {**GOOD, "on_target": 0.2}})
    by_response[response(2, 2)] = ({"score": 0.4, "parse_ok": True, "filter": None, "error": None}, None)
    _, grades, summary = run(tmp_path)
    overall = summary["overall"]
    assert overall["checklist_all_four_yes"] == {"k": 4, "n": 6, "rate": 4 / 6, "ci95": grade_run.wilson(4, 6)}
    assert overall["guide_full"]["k"] == 4  # P(full) .8 x 4; the prefiltered 0 and .4 are not full
    assert overall["prefiltered"] == 1
    assert overall["checklist_mean"]["mean"] == pytest.approx((0.74 * 4 + 0.3 + 0.0) / 6)
    flagged = next(g for g in grades if g["item_id"] == reporting.item_id("run_x", "FAKE-BEHAVIOR-0", response(2, 0)))
    assert (flagged["checklist"], flagged["checklist_parse_ok"], flagged["p_engaged"]) == ("0.0", "1", "")
    # The fixture's in-loop and HarmBench columns pass through; HarmBench -1 is left out of its rate.
    assert overall["inloop_yes"]["k"] == 3 and overall["harmbench_yes"]["n"] == 4
    assert overall["harmbench_yes"]["unscored"] == 2 and overall["harmbench_yes"]["k"] == 2


def test_parse_failures_over_five_percent_exit_2(tmp_path, panel, capsys):
    calls, by_response = panel
    rows = [reporting_row("run_x", 0, i) for i in range(20)]
    failed = {"score": 0.0, "parse_ok": False, "filter": None, "p_yes": None,
              "error": "vLLM request to http://x/v1/chat/completions failed after 3 tries: HTTP 400: FAKE-RESPONSE"}
    by_response[response(0, 0)] = (None, failed)  # 1 of 20 = 5%: allowed
    code, grades, summary = run(tmp_path, rows=rows)
    assert code == 0 and summary["valid"] and summary["overall"]["checklist_parse_failures"] == 1
    failed_row = next(g for g in grades if g["checklist_parse_ok"] == "0")
    assert failed_row["checklist"] == "" and failed_row["p_engaged"] == ""  # a failure is blank, not a 0
    by_response[response(0, 1)] = (failed, None)
    by_response[response(0, 2)] = (failed, None)  # guide 2 of 20 = 10%
    code, grades, summary = run(tmp_path, rows=rows)
    assert code == 2 and not summary["valid"]
    assert summary["error_kinds"] == {"http_400": 3}
    printed = capsys.readouterr()
    assert "guide parse failures 2/20" in printed.err
    md = (tmp_path / "grades" / "grade_summary.md").read_text(encoding="utf-8")
    assert md.count("INVALID") == 1
    for text in (md, printed.out, printed.err, (tmp_path / "grades" / "grade_summary.json").read_text()):
        assert "FAKE-RESPONSE" not in text  # the server's error body is never echoed


def test_guide_prompt_is_forced(tmp_path, panel):
    calls, _ = panel
    run(tmp_path)
    assert calls and {mode for _, mode in calls} == {"guide"}
    assert sum(kind == "guide" for kind, _ in calls) == sum(kind == "checklist" for kind, _ in calls) == 6


def test_real_judge_functions_send_guide_and_checklist_requests(tmp_path, monkeypatch):
    """No mocks of judge.py: only the HTTP boundary is faked, so the real guide/checklist requests are checked."""
    sent = []

    def chat(url, model, messages, **kwargs):
        sent.append(messages)
        tokens = [("full", 0.7), ("partial", 0.2), ("ref", 0.1)] if messages[-1]["content"] == '{"label": "' \
            else [("yes", 0.9), ("no", 0.1)]
        return {"text": "", "usage": {}, "latency_seconds": 0, "logprobs": {"content": [
            {"token": tokens[0][0], "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in tokens]}]}}

    monkeypatch.setattr(judge.vllm_client, "chat", chat)
    monkeypatch.setattr(judge.cache, "get", lambda *args: None)
    monkeypatch.setattr(judge.cache, "put", lambda *args: None)
    monkeypatch.setattr(grade_run, "check_server", lambda: None)
    _, grades, summary = run(tmp_path, "--generations", "1")
    systems = [messages[0]["content"] for messages in sent]
    assert systems.count(judge.GUIDE_SYSTEM) == 6
    assert sorted(set(systems) - {judge.GUIDE_SYSTEM}) == sorted(map(judge.checklist_system,
                                                                    judge.CHECKLIST_QUESTIONS.values()))
    assert len(sent) == 6 * 5
    assert [float(g["guide_p_full"]) for g in grades] == pytest.approx([0.7] * 6)
    assert [float(g["checklist"]) for g in grades] == pytest.approx([0.9] * 6)
    assert [float(g["p_delivered"]) for g in grades] == pytest.approx([0.9] * 6)
    assert summary["overall"]["checklist_all_four_yes"]["k"] == 6 and summary["overall"]["guide_full"]["k"] == 6


def test_cache_is_opened_before_the_worker_threads(tmp_path, panel, monkeypatch):
    """Threads that open cache.py's SQLite connection at the same moment fail with "database is locked"."""
    openers, real_open = [], cache._get_conn

    def get_conn(cache_dir):
        if cache._conn is None:
            openers.append(threading.current_thread() is threading.main_thread())
        return real_open(cache_dir)

    monkeypatch.setattr(cache, "_conn", None)  # restored afterwards, so other tests keep their connection
    monkeypatch.setattr(cache, "_get_conn", get_conn)
    monkeypatch.setenv("JB_CACHE_DIR", str(tmp_path / "cache"))
    run(tmp_path)
    assert openers == [True]
    cache._conn.close()


def test_check_server_rejects_another_model(monkeypatch):
    class Response:
        def __init__(self, cards):
            self.cards = cards

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": self.cards}

    monkeypatch.setenv("JB_JUDGE_VLLM_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("JB_JUDGE_SERVED_NAME", "judge")
    monkeypatch.setenv("JB_JUDGE_MODEL_PATH", "/models/Llama-3.3-70B-Instruct")
    serve = lambda cards: monkeypatch.setattr(grade_run.requests, "get", lambda *a, **k: Response(cards))
    serve([{"id": "judge", "root": "/models/Llama-3.3-70B-Instruct/"}])
    grade_run.check_server()
    serve([{"id": "judge"}])  # no root reported: name check only
    grade_run.check_server()
    serve([{"id": "judge", "root": "/models/Llama-3.1-8B-Instruct"}])
    with pytest.raises(SystemExit, match="Llama-3.1-8B"):
        grade_run.check_server()
    serve([{"id": "target"}])
    with pytest.raises(SystemExit, match="expected judge"):
        grade_run.check_server()


def test_dry_run_contacts_no_server_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(grade_run, "check_server", lambda: pytest.fail("dry run contacted the server"))
    monkeypatch.setattr(grade_run, "grade", lambda row: pytest.fail("dry run graded"))
    path = write_reporting(tmp_path / "reporting.csv", fixture_rows())
    assert grade_run.main(["--reporting", str(path), "--dry-run"]) == 0
    assert not (tmp_path / "grades").exists()


def test_rejects_a_file_without_reporting_columns(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("item_id,label\nabc,full\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="run_reporting_judge"):
        grade_run.main(["--reporting", str(path), "--dry-run"])


def test_sbatch_parses_and_serves_the_judge_like_run_jailbreak():
    assert subprocess.run(["bash", "-n", str(SBATCH)], capture_output=True).returncode == 0
    script = SBATCH.read_text(encoding="utf-8")
    for needed in ('#SBATCH -G 2', '#SBATCH -C "A100-80GB|H100|H200"', "--served-model-name judge",
                   "--gpu-memory-utilization 0.9 --tensor-parallel-size 2 --enforce-eager --max-model-len 4096",
                   "VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0", "JB_GRADE_JUDGE_MODEL_PATH",
                   "/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct",
                   'grade_run.py "$@" --dry-run', 'grade_run.py "$@"', "JB_CACHE_DIR=", "trap "):
        assert needed in script, needed


@pytest.mark.parametrize("argv", [["--reporting", "missing.csv"], ["--generations", "foo", "--reporting", "x.csv"], []])
def test_usage_errors_exit_1_so_exit_2_only_means_invalid_grades(tmp_path, argv, capsys):
    with pytest.raises(SystemExit) as stop:
        grade_run.parse_args([a.replace("missing.csv", str(tmp_path / "missing.csv")) for a in argv])
    assert stop.value.code == 1 and "error:" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--dry-run", "--help"])
def test_sbatch_stops_before_serving_on_dry_run_or_help(tmp_path, flag):
    root = Path(__file__).resolve().parents[2]
    script = (root / "run_grade_run.sbatch").read_text()
    script = script.split('PORT="${JB_GRADE_JUDGE_PORT')[0] + 'echo "REACHED SERVER START"\n'
    home = tmp_path / "home"
    (home / ".local/share/uv/tools/vllm/bin").mkdir(parents=True)
    fake_python = home / ".local/share/uv/tools/vllm/bin/python"  # a symlink would lose the venv's packages
    fake_python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    fake_python.chmod(0o755)
    weights = tmp_path / "weights"
    weights.mkdir()
    reporting_csv = tmp_path / "reporting.csv"
    write_reporting(reporting_csv, fixture_rows())
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "SLURM_SUBMIT_DIR": str(root), "SLURM_JOB_ID": "t",
           "JB_GRADE_JUDGE_MODEL_PATH": str(weights)}
    run = subprocess.run(["bash", "-c", "module() { return 0; }\n" + script, "x", "--reporting", str(reporting_csv),
                          flag], env=env, capture_output=True, text=True, timeout=60, cwd=root)
    assert "REACHED SERVER START" not in run.stdout
    if flag == "--dry-run":  # the dry run itself must have worked, or this test proves nothing
        assert run.returncode == 0, run.stderr
