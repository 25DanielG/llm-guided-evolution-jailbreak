"""Fitness evaluation for Jailbreaking. The titantic GE loop is not changed.
Invoked as: uv run python eval.py --model prompt_<gene_id> --variant_dir <dir>
Writes results to results/<gene_id>_results.csv with a header line + one data line of
len(FITNESS_WEIGHTS) == 2 comma-separated floats.

Objectives are:
    1. harm_success_rate to maximize, mean Guard-unsafe rate over sampled
    2. avg_prompt_tokens to minimize, attack cost / stealth proxy
"""

import argparse
import ast
import csv
import importlib
import json
import math
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
    return parser.parse_args()

def check_no_duplicate_traits(module_path):
    """A mutation that names its assignment to an existing trait_* name causes
    silent shadowing, dropping the trait that was meant to use that slot. Raise
    an error so the gene is marked invalid. Traits are plain string constants
    (tag text), not functions -- look for `trait_x = ...` assignments.
    """
    tree = ast.parse(open(module_path).read())
    seen = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("trait_"):
                    seen[target.id] = seen.get(target.id, 0) + 1
    dupes = [name for name, count in seen.items() if count > 1]
    if dupes:
        raise ValueError(f"duplicate trait assignments (collide, later wins): {dupes}")

def extract_gene_id(model_arg):
    prefix = "prompt_"
    if prefix in model_arg:
        return model_arg.split(prefix, 1)[1]
    return "seed"

def load_behaviors():
    path = jb_config.behaviors_path()
    rows = []  # (difficulty, behavior)
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        # common column names for the behavior text
        text_col = next((c for c in ("behavior", "goal", "prompt", "text") if c in cols), None)
        if text_col is None:
            raise ValueError(f"No behavior column found in {path}; columns={cols}")
        has_difficulty = "difficulty" in cols
        for i, row in enumerate(reader):
            behavior = (row.get(text_col) or "").strip()
            if not behavior:
                continue
            try:
                diff = float(row["difficulty"]) if has_difficulty else i
            except (TypeError, ValueError):
                diff = i
            rows.append((diff, behavior))
    if not rows:
        raise ValueError(f"No behaviors loaded from {path}")
    # order easy->hard
    rows.sort(key=lambda r: r[0])
    return [b for _, b in rows]

def sample_behaviors(behaviors):
    if jb_config.use_full_behaviors():
        return behaviors
    n = jb_config.n_behaviors_per_eval()
    if jb_config.use_curriculum():
        gen = int(os.getenv("JB_GEN", "1"))
        total = max(1, int(os.getenv("JB_NUM_GENS", "1")))
        frac = min(1.0, max(0.0, gen / total))
        window = max(n, math.ceil(frac * len(behaviors)))
        pool = behaviors[:window]
        seed = jb_config.behavior_seed() + gen
    else:
        pool = behaviors
        seed = jb_config.behavior_seed()
    k = min(n, len(pool))
    return random.Random(seed).sample(pool, k)

def _unpack_build_prompt(result):
    """PromptStrategy.build_prompt returns (messages, decoding_overrides) in the
    tag-based design; also accept a bare messages list/str for robustness."""
    if isinstance(result, tuple) and len(result) == 2:
        messages, decoding = result
    else:
        messages, decoding = result, None
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    return messages, (decoding or {})

def resolve_decoding(strategy):
    """trait_decoding does not depend on the behavior text, so resolve it once
    via a harmless probe call instead of per-behavior. Falls back to jb_config
    defaults on any failure."""
    try:
        _, decoding = _unpack_build_prompt(strategy.build_prompt("(decoding probe)"))
    except Exception:
        decoding = {}
    return {
        "temperature": decoding.get("temperature", jb_config.target_temperature()),
        "top_p": decoding.get("top_p"),  # None -> omitted from the request (server default)
    }

def get_target_responses(strategy, behaviors, target_url, target_name, decoding=None):
    """Build prompts, query target. Returns list of dicts: {behavior, response, prompt_tokens}.
    A behavior whose prompt build fails is skipped.
    """
    cdir = jb_config.cache_dir()
    decoding = decoding or {}
    max_tokens = jb_config.target_max_new_tokens()
    temperature = decoding.get("temperature", jb_config.target_temperature())
    top_p = decoding.get("top_p")
    built = [] # (behavior, messages_or_None)
    for behavior in behaviors:
        try:
            messages, _ = _unpack_build_prompt(strategy.build_prompt(behavior))
        except Exception as err: # broken trait on this input
            print(f"  build_prompt failed for a behavior: {err}", flush=True)
            messages = None  # keep the behavior; it scores as a failed attack (harm=0)
        built.append((behavior, messages))

    records = []
    to_query = [] # (index_into_records, messages)
    for behavior, messages in built:
        if messages is None:
            records.append({"behavior": behavior, "response": None, "prompt_tokens": None, "messages": None})
            continue
        key = cache.response_key(target_name, messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        hit = cache.get(cdir, key)
        if hit is not None:
            data = json.loads(hit)
            records.append({"behavior": behavior, "response": data["text"], "prompt_tokens": data["prompt_tokens"], "messages": messages})
        else:
            records.append({"behavior": behavior, "response": None, "prompt_tokens": None, "messages": messages, "_key": key, "_messages": messages})
            to_query.append((len(records) - 1, messages))

    if to_query:
        outputs = vllm_client.chat_batch(
            target_url, target_name, [m for _, m in to_query],
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
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
    """Return (harm_success_rate, avg_prompt_tokens, details).
    details is a per-behavior list: {behavior, harm, refused, stage, prompt_tokens,
    messages} -- the first-class per-behavior success vector (feeds directed
    mutation, the diversity objective, and the individual-card exporter).
    """
    if not records:
        return 0.0, PENALTY_TOKENS, []
    harms = []
    tokens = []
    details = []
    for rec in records:
        verdict = judge.score_response(rec["behavior"], rec["response"])
        harms.append(verdict["harm"])
        if rec["prompt_tokens"]:
            tokens.append(rec["prompt_tokens"])
        details.append({
            "behavior": rec["behavior"],
            "harm": verdict["harm"],
            "refused": verdict.get("refused"),
            "stage": verdict.get("stage"),
            "prompt_tokens": rec["prompt_tokens"],
            "messages": rec.get("messages"),
            "response": rec["response"],
        })
    harm_rate = sum(harms) / len(harms)
    avg_tokens = (sum(tokens) / len(tokens)) if tokens else PENALTY_TOKENS
    return harm_rate, avg_tokens, details

def write_results(gene_id, harm_rate, avg_tokens, diversity=0.0):
    """CSV columns match FITNESS_WEIGHTS order: harm, diversity, tokens.
    diversity is a placeholder here; run_improved.py fills it across the
    generation (the per-gene eval has no view of the population).
    """
    filename = os.path.abspath(f"results/{gene_id}_results.csv")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("harm_success_rate,diversity,avg_prompt_tokens\n")
        f.write(f"{harm_rate},{diversity},{avg_tokens}\n")
    print(f"results have been written to {filename}", flush=True)

def write_details(gene_id, harm_rate, avg_tokens, details, decoding=None):
    """Sidecar per-behavior JSON. Read by run_improved (diversity), directed
    mutation, the individual-card exporter, and the post-run report."""
    filename = os.path.abspath(f"results/{gene_id}_results.json")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    payload = {
        "gene_id": gene_id,
        "harm_success_rate": harm_rate,
        "avg_prompt_tokens": avg_tokens,
        "n_behaviors": len(details),
        "n_harmful": sum(d["harm"] for d in details),
        "decoding": decoding or {},
        "behaviors": details,
    }
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"details written to {filename}", flush=True)

if __name__ == "__main__":
    args = get_args()
    variant_dir_abs = os.path.abspath(args.variant_dir)
    sys.path.insert(0, variant_dir_abs)
    sys.path.insert(0, str(SCRIPT_DIR))
    os.chdir(SCRIPT_DIR)

    gene_id = extract_gene_id(args.model)
    print(f"Evaluating gene: {gene_id} (module: {args.model})", flush=True)

    module_file = os.path.join(variant_dir_abs, f"{args.model}.py")
    if os.path.exists(module_file):
        check_no_duplicate_traits(module_file)
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

    decoding = resolve_decoding(strategy)
    print(f"decoding overrides: {decoding}", flush=True)
    records = get_target_responses(strategy, behaviors, target_url, jb_config.target_model_name(), decoding=decoding)
    harm_rate, avg_tokens, details = score(records)
    print(f"harm_success_rate={harm_rate:.4f} avg_prompt_tokens={avg_tokens:.1f}", flush=True)

    write_results(gene_id, harm_rate, avg_tokens)
    write_details(gene_id, harm_rate, avg_tokens, details, decoding=decoding)
    print('=' * 60)
    print('job done')
    print('=' * 60)
