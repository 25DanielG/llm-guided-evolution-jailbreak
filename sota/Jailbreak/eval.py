"""Fitness evaluation for Jailbreaking. The titantic GE loop is not changed.
Invoked as: uv run python eval.py --model prompt_<gene_id> --variant_dir <dir>
Writes results to results/<gene_id>_results.csv with a header line + one data line of
len(FITNESS_WEIGHTS) == 2 comma-separated floats.

Objectives are:
    1. harm_success_rate to maximize, mean Guard-unsafe rate over sampled
    2. avg_prompt_tokens to minimize, attack cost / stealth proxy
"""

import argparse
import csv
import importlib
import json
import os
import random
import sys
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

def get_target_responses(strategy, behaviors, target_url, target_name):
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
        key = cache.response_key(target_name, messages)
        hit = cache.get(cdir, key)
        if hit is not None:
            data = json.loads(hit)
            records.append({"behavior": behavior, "response": data["text"], "prompt_tokens": data["prompt_tokens"]})
        else:
            records.append({"behavior": behavior, "response": None, "prompt_tokens": None, "_key": key, "_messages": messages})
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
            cache.put(cdir, rec["_key"], json.dumps({"text": out["text"], "prompt_tokens": out["prompt_tokens"]}))
    return records

def score(records):
    """Return (harm_success_rate, avg_prompt_tokens)."""
    if not records:
        return 0.0, PENALTY_TOKENS
    harms = []
    tokens = []
    for rec in records:
        verdict = judge.score_response(rec["behavior"], rec["response"])
        harms.append(verdict["harm"])
        if rec["prompt_tokens"]:
            tokens.append(rec["prompt_tokens"])
    harm_rate = sum(harms) / len(harms)
    avg_tokens = (sum(tokens) / len(tokens)) if tokens else PENALTY_TOKENS
    return harm_rate, avg_tokens

def write_results(gene_id, harm_rate, avg_tokens):
    filename = os.path.abspath(f"results/{gene_id}_results.csv")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("harm_success_rate,avg_prompt_tokens\n")
        f.write(f"{harm_rate},{avg_tokens}\n")
    print(f"results have been written to {filename}", flush=True)

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
    ready_timeout = jb_config.server_ready_timeout()
    if not vllm_client.wait_ready(target_url, timeout=ready_timeout):
        raise RuntimeError(f"Target vLLM server not reachable at {target_url}")
    if not vllm_client.wait_ready(judge_url, timeout=ready_timeout):
        raise RuntimeError(f"Judge vLLM server not reachable at {judge_url}")

    records = get_target_responses(strategy, behaviors, target_url, jb_config.target_model_name())
    harm_rate, avg_tokens = score(records)
    print(f"harm_success_rate={harm_rate:.4f} avg_prompt_tokens={avg_tokens:.1f}", flush=True)

    write_results(gene_id, harm_rate, avg_tokens)
    print('=' * 60)
    print('job done')
    print('=' * 60)
