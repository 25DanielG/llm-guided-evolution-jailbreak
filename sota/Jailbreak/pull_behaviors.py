"""Populate behaviors/curated.csv from a public harmful-behavior benchmark.
    uv run python pull_behaviors.py --n 30 --out behaviors/curated.csv
    uv run python pull_behaviors.py --url <other_csv_url> --n 40
"""

import argparse
import csv
import hashlib
import io
import os
import random
import urllib.request

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)
ADVBENCH_SHA256 = "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=ADVBENCH_URL, help="CSV url with a 'goal' or 'behavior' column")
    parser.add_argument("--sha256", default=ADVBENCH_SHA256, help="required SHA-256 of the source CSV")
    parser.add_argument("--n", type=int, default=30, help="number of behaviors to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "behaviors", "advbench_30.csv"))
    parser.add_argument("--source", default="advbench")
    args = parser.parse_args()

    print(f"Downloading {args.url}")
    with urllib.request.urlopen(args.url, timeout=60) as resp:
        payload = resp.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest.lower() != args.sha256.lower():
        raise SystemExit(f"SHA-256 mismatch: expected {args.sha256}, received {digest}")
    content = payload.decode("utf-8")

    reader = csv.DictReader(io.StringIO(content))
    cols = reader.fieldnames or []
    text_col = next((c for c in ("behavior", "goal", "prompt", "text") if c in cols), None)
    if text_col is None:
        raise SystemExit(f"No behavior column in downloaded CSV; columns={cols}")

    behaviors = []
    for row_number, row in enumerate(reader, 1):
        behavior = (row.get(text_col) or "").strip()
        if behavior:
            behaviors.append({
                "id": (row.get("id") or f"{args.source}_{row_number:03d}").strip(),
                "source": (row.get("source") or args.source).strip(),
                "behavior": behavior,
            })
    rng = random.Random(args.seed)
    if args.n < len(behaviors):
        behaviors = rng.sample(behaviors, args.n)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "source", "behavior"])
        for item in behaviors:
            writer.writerow([item["id"], item["source"], item["behavior"]])

    print(f"Wrote {len(behaviors)} behaviors to {args.out}")

if __name__ == "__main__":
    main()
