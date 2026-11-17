"""eval.py's __main__ against fake servers: with the flag off it makes no checklist requests and writes what it
always wrote; with the flag on it adds the column, the verdicts and the log line. No real inference."""
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
JB = ROOT / "sota" / "Jailbreak"
QUESTIONS = ("engaged", "on_target", "specific", "delivered")
P_YES = math.exp(-0.4) / (math.exp(-0.4) + math.exp(-1.3))  # from the fake judge's top logprobs below
DRIVER = r'''
import json, os, runpy, sys, threading
jb_dir, out, root = sys.argv[1:4]
sys.path.insert(0, jb_dir)
sys.path.append(root)
import vllm_client
lock, seen = threading.Lock(), []
TOP = [{"token": "full", "logprob": -0.2}, {"token": "refusal", "logprob": -2.0},
       {"token": "yes", "logprob": -0.4}, {"token": "no", "logprob": -1.3}]
def chat(base_url, model, messages, **kwargs):
    with lock:
        seen.append({"model": model, "messages": messages, "kwargs": sorted(kwargs)})
    if model == "target":
        return {"text": "Here is a long structured answer with several steps and examples. " * 4, "prompt_tokens": 100,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50}, "latency_seconds": 0.5, "logprobs": None}
    return {"text": "x", "usage": {"prompt_tokens": 300, "completion_tokens": 1}, "latency_seconds": 0.25,
            "logprobs": {"content": [{"token": "x", "logprob": -0.2, "top_logprobs": TOP}]}}
vllm_client.chat = chat
vllm_client.completion = lambda *a, **k: {"text": "no", "usage": {}, "latency_seconds": 0.1}
vllm_client.wait_ready = lambda *a, **k: True
vllm_client.assert_model = lambda *a, **k: None
sys.argv = ["eval.py", "--model", "prompt", "--variant_dir", jb_dir, "--run-id", "run", "--generation", "1"]
try:
    runpy.run_path(os.path.join(jb_dir, "eval.py"), run_name="__main__")
finally:
    with open(os.path.join(out, "requests.json"), "w") as f:
        json.dump(seen, f)
'''


def run_eval(tmp_path, flag):
    jb = tmp_path / "jb"
    if jb.exists():
        shutil.rmtree(jb)
    shutil.copytree(JB, jb, ignore=shutil.ignore_patterns("results", "__pycache__"))
    out = tmp_path / f"out_{flag}"
    out.mkdir()
    driver = tmp_path / "drive.py"
    driver.write_text(DRIVER)
    env = {k: v for k, v in os.environ.items() if not k.startswith("JB_")}
    env.update(JB_TARGET_VLLM_URL="http://target.invalid/v1", JB_JUDGE_VLLM_URL="http://judge.invalid/v1",
               JB_REPORTING_VLLM_URL="http://reporting.invalid/v1", JB_N_BEHAVIORS_PER_EVAL="6",
               JB_MAX_CONCURRENCY="4", JB_JUDGE_PROMPT="guide", JB_CACHE_DIR=str(out / "cache"),
               JB_GENE_RESULTS_DIR=str(out / "genes"))
    if flag is not None:
        env["JB_CHECKLIST_OBJECTIVE"] = flag
    run = subprocess.run([sys.executable, str(driver), str(jb), str(out), str(ROOT)], env=env,
                         capture_output=True, text=True, timeout=180)
    result = {"returncode": run.returncode, "stdout": run.stdout.replace(str(out), "OUT"), "stderr": run.stderr,
              "requests": json.loads((out / "requests.json").read_text())}
    if run.returncode == 0:
        candidates = (jb / "results" / "run" / "candidates.jsonl").read_text().splitlines()
        result.update(rows=[json.loads(line) for line in candidates],
                      csv=(out / "genes" / "seed_results.csv").read_text(),
                      sidecar=json.loads((out / "genes" / "seed_results.json").read_text()))
    return result


def is_checklist(request):
    return request["messages"][-1]["content"] == '{"answer": "'


def test_flag_off_main_path_is_unchanged_and_makes_no_checklist_calls(tmp_path):
    unset, zero = run_eval(tmp_path, None), run_eval(tmp_path, "0")
    assert unset["returncode"] == zero["returncode"] == 0, unset["stderr"][-2000:]
    norm = lambda requests: sorted(json.dumps(r, sort_keys=True) for r in requests)
    assert norm(unset["requests"]) == norm(zero["requests"])
    for key in ("csv", "rows", "sidecar", "stdout"):
        assert unset[key] == zero[key], key
    assert unset["rows"] and not any(is_checklist(r) for r in unset["requests"])
    assert unset["csv"].splitlines()[0] == "mean_judge_score,diversity,avg_prompt_tokens,easy_score,medium_score,hard_score"
    assert all("checklist" not in row for row in unset["rows"])
    assert "mean_checklist_score" not in unset["sidecar"]
    assert all("checklist" not in behavior for behavior in unset["sidecar"]["behaviors"])
    assert "checklist" not in unset["stdout"]


def test_flag_on_main_path_adds_the_objective(tmp_path):
    off, on = run_eval(tmp_path, None), run_eval(tmp_path, "1")
    assert on["returncode"] == 0, on["stderr"][-2000:]
    checklist = [r for r in on["requests"] if is_checklist(r)]
    assert len(checklist) == 4 * len(on["rows"]) == len(on["requests"]) - len(off["requests"])
    header, values = on["csv"].splitlines()
    assert header == ("mean_judge_score,diversity,avg_prompt_tokens,checklist_score,"
                      "easy_score,medium_score,hard_score")
    assert values.split(",")[:3] == off["csv"].splitlines()[1].split(",")[:3]
    assert float(values.split(",")[3]) == pytest.approx(P_YES)
    assert on["sidecar"]["mean_checklist_score"] == pytest.approx(P_YES)
    for row in on["rows"]:
        assert row["checklist"]["p_yes"] == pytest.approx(dict.fromkeys(QUESTIONS, P_YES))
    assert all(b["checklist"]["score"] == pytest.approx(P_YES) for b in on["sidecar"]["behaviors"])
    assert f"mean_checklist_score={P_YES:.4f} checklist_parse_failures=0" in on["stdout"]


def test_bad_flag_fails_before_any_model_call(tmp_path):
    bad = run_eval(tmp_path, "true")
    assert bad["returncode"] != 0 and "JB_CHECKLIST_OBJECTIVE must be '0' or '1'" in bad["stderr"]
    assert bad["requests"] == []
