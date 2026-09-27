"""Fitness evaluation for Jailbreaking. The titantic GE loop is not changed.
Invoked as: uv run python eval.py --model prompt_<gene_id> --variant_dir <dir>
Writes results to results/<gene_id>_results.csv with a header line + one data line of
len(FITNESS_WEIGHTS) == 2 comma-separated floats.

Objectives are:
    1. mean_judge_score to maximize
    2. avg_prompt_tokens to minimize, attack cost / stealth proxy
"""

import argparse
import csv
import importlib
import json
import os
import random
import sys
import fcntl
from pathlib import Path as p

SCRIPT_DIR = p(__file__).parent.resolve()

import cache
import jb_config
import judge
import vllm_client

PENALTY_TOKENS = 1_000_000.0  # obj2 penalty when no attack happened

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="prompt", help="variant module name, e.g. prompt_<gene_id>")
    parser.add_argument('--variant_dir', type=str, default='models', help="dir holding the evolved module")
    parser.add_argument('--save_dir', type=str, default="trained", help="unused; kept for CLI parity")
    parser.add_argument('--random_seed', type=int, default=42, help="unused; behavior sampling uses JB_BEHAVIOR_SEED")
    parser.add_argument('--generation', type=int, default=0)
    parser.add_argument('--run-id', type=str, default=os.getenv("LLMGE_RUN_ID", "manual"))
    return parser.parse_args()

def extract_gene_id(model_arg):
    prefix = "prompt_"
    if prefix in model_arg:
        return model_arg.split(prefix, 1)[1]
    return "seed"

def load_behaviors():
    path = jb_config.behaviors_path()
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        # common column names for the behavior text
        text_col = next((c for c in ("behavior", "goal", "prompt", "text") if c in cols), None)
        if text_col is None:
            raise ValueError(f"No behavior column found in {path}; columns={cols}")
        for row in reader:
            behavior = (row.get(text_col) or "").strip()
            if behavior:
                rows.append(behavior)
    if not rows:
        raise ValueError(f"No behaviors loaded from {path}")
    return rows

def sample_behaviors(behaviors):
    if jb_config.use_full_behaviors():
        return behaviors
    k = min(jb_config.n_behaviors_per_eval(), len(behaviors))
    rng = random.Random(jb_config.behavior_seed())
    return rng.sample(behaviors, k)

def get_target_responses(strategy, behaviors, target_url, target_name, use_cache=True):
    """Build prompts, query target. Returns list of dicts: {behavior, response, prompt_tokens}.
    A behavior whose prompt build fails is skipped.
    """
    cdir = jb_config.cache_dir()
    built = [] # (behavior, messages)
    for behavior in behaviors:
        try:
            messages = strategy.build_prompt(behavior)
        except Exception as err: # broken trait on this input, so skip
            print(f"  build_prompt failed for a behavior: {err}", flush=True)
            continue
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        built.append((behavior, messages))

    records = []
    to_query = [] # (index_into_records, messages)
    for behavior, messages in built:
        key = cache.response_key(target_name, messages, jb_config.target_max_new_tokens(),
                                 jb_config.target_temperature())
        hit = cache.get(cdir, key) if use_cache else None
        if hit is not None:
            data = json.loads(hit)
            records.append({"behavior": behavior, "response": data["text"], "prompt_tokens": data["prompt_tokens"],
                            "usage": data.get("usage", {}), "latency_seconds": data.get("latency_seconds", 0),
                            "messages": messages, "target_cached": True})
        else:
            records.append({"behavior": behavior, "response": None, "prompt_tokens": None,
                            "_key": key, "messages": messages, "target_cached": False})
            to_query.append((len(records) - 1, messages))

    if to_query:
        outputs = vllm_client.chat_batch(
            target_url, target_name, [m for _, m in to_query],
            max_tokens=jb_config.target_max_new_tokens(),
            temperature=jb_config.target_temperature(),
            timeout=jb_config.request_timeout(),
            max_concurrency=jb_config.max_concurrency(),
        )
        for (rec_idx, _), out in zip(to_query, outputs):
            rec = records[rec_idx]
            rec["response"] = out["text"]
            rec["prompt_tokens"] = out["prompt_tokens"]
            rec["usage"] = out.get("usage", {})
            rec["latency_seconds"] = out.get("latency_seconds", 0)
            if use_cache:
                cache.put(cdir, rec["_key"], json.dumps({"text": out["text"], "prompt_tokens": out["prompt_tokens"],
                                                         "usage": rec["usage"], "latency_seconds": rec["latency_seconds"]}))
    return records

def score(records):
    """Return the two fitness objectives; reporting never enters this calculation."""
    if not records:
        return 0.0, PENALTY_TOKENS
    scores = []
    tokens = []
    for rec in records:
        verdict = judge.score_response(rec["behavior"], rec["response"])
        rec["in_loop"] = verdict
        scores.append(verdict["score"])
        if rec["prompt_tokens"]:
            tokens.append(rec["prompt_tokens"])
    harm_rate = sum(scores) / len(scores)
    avg_tokens = (sum(tokens) / len(tokens)) if tokens else PENALTY_TOKENS
    return harm_rate, avg_tokens

def write_results(gene_id, harm_rate, avg_tokens):
    filename = os.path.abspath(f"results/{gene_id}_results.csv")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("mean_judge_score,avg_prompt_tokens\n")
        f.write(f"{harm_rate},{avg_tokens}\n")
    print(f"results have been written to {filename}", flush=True)

def verify_record(rec, target_url, target_name):
    """Three new target generations; never read or write the response cache."""
    trials = []
    for _ in range(3):
        try:
            output = vllm_client.chat(target_url, target_name, rec["messages"],
                                      max_tokens=jb_config.target_max_new_tokens(),
                                      temperature=jb_config.target_temperature(),
                                      timeout=jb_config.request_timeout())
            report = judge.reporting_score(rec["behavior"], output["text"])
            trials.append({"response": output["text"], "usage": output.get("usage", {}),
                           "latency_seconds": output.get("latency_seconds", 0),
                           "reporting": report})
        except Exception as exc:
            trials.append({"error": str(exc), "reporting": {"success": False, "parse_ok": False}})
    return {"trials": trials, "successes": sum(t["reporting"]["success"] for t in trials),
            "verified_success": sum(t["reporting"]["success"] for t in trials) >= 2}

def append_records(run_id, records):
    run_dir = SCRIPT_DIR / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "candidates.jsonl"
    with open(path, "a", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        for rec in records:
            file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        file.flush()
        fcntl.flock(file, fcntl.LOCK_UN)
    return path

def export_elites(path, limit=20):
    """Rebuild the per-run labeling export from all candidates written so far."""
    with open(path, encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_SH)
        rows = [json.loads(line) for line in file if line.strip()]
        fcntl.flock(file, fcntl.LOCK_UN)
    best = {}
    for row in rows:
        candidate = row["candidate_id"]
        if candidate not in best or row["in_loop"]["score"] > best[candidate]["in_loop"]["score"]:
            best[candidate] = row
    rows = sorted(best.values(), key=lambda row: row["in_loop"]["score"], reverse=True)
    output = path.with_name("elites.jsonl")
    with open(output, "w", encoding="utf-8") as file:
        for row in rows[:limit]:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output

if __name__ == "__main__":
    args = get_args()
    variant_dir_abs = os.path.abspath(args.variant_dir)
    sys.path.insert(0, variant_dir_abs)
    sys.path.insert(0, str(SCRIPT_DIR))
    os.chdir(SCRIPT_DIR)

    gene_id = extract_gene_id(args.model)
    print(f"Evaluating gene: {gene_id} (module: {args.model})", flush=True)

    model_module = importlib.import_module(args.model)
    strategy = model_module.PromptStrategy()

    behaviors = sample_behaviors(load_behaviors())
    print(f"Sampled {len(behaviors)} behaviors", flush=True)

    target_url = jb_config.target_base_url()
    judge_url = jb_config.judge_base_url()
    reporting_url = jb_config.reporting_base_url()
    ready_timeout = jb_config.server_ready_timeout()
    endpoints = [(target_url, jb_config.target_model_name()),
                 (judge_url, jb_config.judge_model_name()),
                 (reporting_url, jb_config.reporting_model_name())]
    if len({url for url, _ in endpoints}) != 3 or len({model for _, model in endpoints}) != 3:
        raise RuntimeError("Target, judge, and reporting endpoints/models must be distinct")
    for url, model in endpoints:
        if not vllm_client.wait_ready(url, timeout=ready_timeout):
            raise RuntimeError(f"vLLM server not reachable at {url}")
        vllm_client.assert_model(url, model)

    records = get_target_responses(strategy, behaviors, target_url, jb_config.target_model_name())
    harm_rate, avg_tokens = score(records)
    report_rows = []
    for index, rec in enumerate(records):
        in_loop = rec["in_loop"]
        verification = verify_record(rec, target_url, jb_config.target_model_name()) if in_loop["score"] >= jb_config.verification_threshold() else None
        report_rows.append({
            "run_id": args.run_id, "generation": args.generation, "candidate_id": gene_id,
            "behavior_index": index, "behavior": rec["behavior"], "evolved_prompt": rec["messages"],
            "target_response": rec["response"], "target_model": jb_config.target_model_name(),
            "judge_model": jb_config.judge_model_name(), "reporting_model": jb_config.reporting_model_name(),
            "target_usage": rec.get("usage", {}), "target_latency_seconds": rec.get("latency_seconds", 0),
            "target_cached": rec.get("target_cached", False), "in_loop": in_loop, "verification": verification,
        })
    path = append_records(args.run_id, report_rows)
    export_elites(path)
    in_loop_rate = sum(r["in_loop"]["score"] >= jb_config.verification_threshold() for r in report_rows) / len(report_rows) if report_rows else 0.0
    verified_rate = sum(bool(r["verification"] and r["verification"]["verified_success"]) for r in report_rows) / len(report_rows) if report_rows else 0.0
    print(f"mean_judge_score={harm_rate:.4f} in_loop_success_rate={in_loop_rate:.4f} verified_success_rate={verified_rate:.4f} avg_prompt_tokens={avg_tokens:.1f}", flush=True)

    write_results(gene_id, harm_rate, avg_tokens)
    print('=' * 60)
    print('job done')
    print('=' * 60)
