"""Print the raw judge output for responses that were scored `malformed_rubric`.

Usage: uv run python diagnose_malformed.py results/<run_id>/candidates.jsonl [--n 20]

Needs the judge vLLM server (JB_JUDGE_VLLM_URL / JB_JUDGE_SERVED_NAME). Re-judges
a sample of the unique failing responses and shows what the judge returned, whether
the tolerant parser now accepts it, and its completion token count.
"""

import argparse
import json
import random

import judge
import jb_config
import vllm_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seen = {}
    with open(args.candidates, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["in_loop"].get("error") == "malformed_rubric":
                seen.setdefault(row["target_response"], row["behavior"])
    items = list(seen.items())
    random.Random(args.seed).shuffle(items)
    print(f"{len(items)} unique malformed responses; sampling {min(args.n, len(items))}")
    fixed = 0
    for response, behavior in items[:args.n]:
        out = vllm_client.chat(jb_config.judge_base_url(), jb_config.judge_model_name(),
                               judge.judge_messages(behavior, response), max_tokens=128, temperature=0)
        ok = judge.parse_rubric(out["text"]) is not None
        fixed += ok
        print(f"--- tokens={out['usage'].get('completion_tokens')} parses_now={ok}\n{out['text']!r}")
    print(f"{fixed}/{min(args.n, len(items))} now parse")


if __name__ == "__main__":
    main()
