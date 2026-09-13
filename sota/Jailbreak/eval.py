"""Evaluate evolved jailbreak prompt strategies against persistent vLLM services."""

import argparse
import csv
import importlib
import json
import os
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

import cache
import jb_config
import judge
import vllm_client

PENALTY_TOKENS = 1_000_000.0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prompt", help="variant module, e.g. prompt_<gene_id>")
    parser.add_argument("--variant_dir", default="models", help="directory holding evolved modules")
    parser.add_argument("--save_dir", default="trained", help="unused; retained for CLI compatibility")
    parser.add_argument("--random_seed", type=int, default=42, help="unused; use JB_BEHAVIOR_SEED")
    return parser.parse_args()


def extract_gene_id(model_arg):
    return model_arg.split("prompt_", 1)[1] if "prompt_" in model_arg else "seed"


def load_behaviors(path=None):
    path = path or jb_config.behaviors_path()
    rows = []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        text_column = next(
            (column for column in ("behavior", "goal", "prompt", "text") if column in columns),
            None,
        )
        if text_column is None:
            raise ValueError(f"No behavior column found in {path}; columns={columns}")
        for index, row in enumerate(reader, 1):
            behavior = (row.get(text_column) or "").strip()
            if behavior:
                rows.append({
                    "id": (row.get("id") or f"behavior_{index:03d}").strip(),
                    "source": (row.get("source") or "unknown").strip(),
                    "behavior": behavior,
                })
    if not rows:
        raise ValueError(f"No behaviors loaded from {path}")
    return rows


def sample_behaviors(behaviors):
    if jb_config.use_full_behaviors():
        return behaviors
    count = min(jb_config.n_behaviors_per_eval(), len(behaviors))
    return random.Random(jb_config.behavior_seed()).sample(behaviors, count)


def get_target_responses(strategy, behaviors, target_url, target_name):
    """Build attack prompts and query the target in a bounded concurrent batch."""
    cache_dir = jb_config.cache_dir()
    max_tokens = jb_config.target_max_new_tokens()
    temperature = jb_config.target_temperature()
    records = []
    to_query = []

    for item in behaviors:
        record = dict(item)
        record.update({"messages": [], "response": "", "prompt_tokens": 0, "target_cache_hit": False})
        try:
            messages = strategy.build_prompt(item["behavior"])
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            if not isinstance(messages, list) or not messages:
                raise ValueError("build_prompt must return a nonempty message list or string")
            record["messages"] = messages
        except Exception as err:
            record["build_error"] = str(err)
            records.append(record)
            continue

        key = cache.response_key(target_name, messages, max_tokens, temperature)
        cached = cache.get(cache_dir, key)
        if cached is not None:
            data = json.loads(cached)
            record.update({
                "response": data["text"],
                "prompt_tokens": data["prompt_tokens"],
                "target_cache_hit": True,
            })
        else:
            record["_cache_key"] = key
            to_query.append((len(records), messages))
        records.append(record)

    if to_query:
        outputs = vllm_client.chat_batch(
            target_url,
            target_name,
            [messages for _, messages in to_query],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=jb_config.request_timeout(),
            max_concurrency=jb_config.max_concurrency(),
        )
        for (record_index, _), output in zip(to_query, outputs):
            record = records[record_index]
            record["response"] = output["text"]
            record["prompt_tokens"] = output["prompt_tokens"]
            cache.put(
                cache_dir,
                record.pop("_cache_key"),
                json.dumps({"text": output["text"], "prompt_tokens": output["prompt_tokens"]}),
            )
    return records


def score(records):
    """Attach judge results and return the two-objective fitness tuple."""
    if not records:
        return 0.0, PENALTY_TOKENS
    verdicts = judge.score_responses(
        [(record["behavior"], record["response"]) for record in records]
    )
    for record, verdict in zip(records, verdicts):
        record["judge"] = verdict
    harm_rate = sum(verdict["harm"] for verdict in verdicts) / len(verdicts)
    prompt_tokens = [record["prompt_tokens"] for record in records if record["prompt_tokens"] > 0]
    average_tokens = sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else PENALTY_TOKENS
    return harm_rate, average_tokens


def write_results(gene_id, harm_rate, average_tokens, records, results_dir="results"):
    output_dir = Path(results_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{gene_id}_results.csv"
    summary_path.write_text(
        "harm_success_rate,avg_prompt_tokens\n"
        f"{harm_rate},{average_tokens}\n"
    )
    audit_path = output_dir / f"{gene_id}_audit.jsonl"
    with audit_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"results have been written to {summary_path}", flush=True)
    print(f"audit has been written to {audit_path}", flush=True)
    return summary_path, audit_path


def main():
    args = get_args()
    variant_dir = Path(args.variant_dir).resolve()
    sys.path.insert(0, str(variant_dir))
    sys.path.insert(0, str(SCRIPT_DIR))
    os.chdir(SCRIPT_DIR)

    gene_id = extract_gene_id(args.model)
    print(f"Evaluating gene: {gene_id} (module: {args.model})", flush=True)
    strategy = importlib.import_module(args.model).PromptStrategy()
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
    harm_rate, average_tokens = score(records)
    print(f"harm_success_rate={harm_rate:.4f} avg_prompt_tokens={average_tokens:.1f}", flush=True)
    write_results(gene_id, harm_rate, average_tokens, records, jb_config.results_dir())
    print("=" * 60)
    print("job done")
    print("=" * 60)


if __name__ == "__main__":
    main()
