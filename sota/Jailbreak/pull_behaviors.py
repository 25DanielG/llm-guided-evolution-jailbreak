"""Populate behaviors/curated.csv from a public harmful-behavior benchmark.
    uv run python pull_behaviors.py --n 30 --out behaviors/curated.csv
    uv run python pull_behaviors.py --url <other_csv_url> --n 40
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
    rng = random.Random(args.seed)
    if args.n < len(behaviors):
        behaviors = rng.sample(behaviors, args.n)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "source", "behavior"])
        for i, b in enumerate(behaviors, 1):
            writer.writerow([f"{args.source}_{i:03d}", args.source, b])

    print(f"Wrote {len(behaviors)} behaviors to {args.out}")

if __name__ == "__main__":
    main()
