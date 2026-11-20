"""Populate behaviors/curated.csv from a public harmful-behavior benchmark.
    uv run python pull_behaviors.py --n 30 --out behaviors/curated.csv
    uv run python pull_behaviors.py --url <other_csv_url> --n 40
    uv run python pull_behaviors.py --n 90 --extend   # keep existing rows, add new ones up to 90 total
"""

import argparse
import csv
import io
import os
import random
import urllib.request

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=ADVBENCH_URL, help="CSV url with a 'goal' or 'behavior' column")
    parser.add_argument("--n", type=int, default=30, help="number of behaviors to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "behaviors", "curated.csv"))
    parser.add_argument("--source", default="advbench")
    parser.add_argument("--extend", action="store_true",
                        help="keep the rows already in --out (incl. tier columns) and append new "
                             "behaviors until --n total; existing behaviors stay put")
    args = parser.parse_args()

    print(f"Downloading {args.url}")
    with urllib.request.urlopen(args.url, timeout=60) as resp:
        content = resp.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(content))
    cols = reader.fieldnames or []
    text_col = next((c for c in ("behavior", "goal", "prompt", "text") if c in cols), None)
    if text_col is None:
        raise SystemExit(f"No behavior column in downloaded CSV; columns={cols}")

    behaviors = [(r.get(text_col) or "").strip() for r in reader]
    behaviors = [b for b in behaviors if b]
    existing, fields = [], ["id", "source", "behavior"]
    if args.extend and os.path.exists(args.out):
        with open(args.out, newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or fields
            existing = list(reader)
    have = {r["behavior"].strip() for r in existing}
    fresh = [b for b in behaviors if b not in have]
    rng = random.Random(args.seed)
    want = max(0, args.n - len(existing))
    if want < len(fresh):
        fresh = rng.sample(fresh, want)

    rows = list(existing)
    next_id = len(existing) + 1
    for b in fresh:
        rows.append({"id": f"{args.source}_{next_id:03d}", "source": args.source, "behavior": b})
        next_id += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} behaviors to {args.out} ({len(fresh)} new)")

if __name__ == "__main__":
    main()
