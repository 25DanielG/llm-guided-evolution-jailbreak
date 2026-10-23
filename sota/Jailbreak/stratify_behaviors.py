"""Tier behaviors as easy/medium/hard from how often earlier runs broke them.

Usage: uv run python stratify_behaviors.py results/<run_id>/candidates.jsonl [more.jsonl ...]

Per behavior, solveis re_rate = fraction of distinct genes that broke it (HarmBench
verified by default; --metric inloop uses the rubric score >= threshold instead).
Behaviors are ranked by solve_rate and split into equal-sized terciles: the
lowest third is `hard`, the highest `easy`. solve_rate, n_genes and tier are
written back to the CSV.

Measured using genes from previous run.
"""

import argparse
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TIER_COLS = ["solve_rate", "n_genes", "tier"]

def load_outcomes(paths, metric, threshold):
    """{behavior: {gene_id: bool}}; a re-evaluated gene keeps its last row."""
    outcomes = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if metric == "verified":
                    ok = bool(row.get("verification") and row["verification"].get("verified_success"))
                else:
                    ok = row["in_loop"]["score"] >= threshold
                outcomes.setdefault(row["behavior"].strip(), {})[row["candidate_id"]] = ok
    return outcomes

def assign_tiers(rates):
    """rates: {behavior: solve_rate}. Equal-sized terciles, hardest first"""
    ranked = sorted(rates, key=lambda b: (rates[b], b))
    n = len(ranked)
    tiers = {}
    for i, b in enumerate(ranked):
        tiers[b] = ("hard", "medium", "easy")[min(2, i * 3 // n)]
    return tiers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", nargs="+", help="candidates.jsonl file(s) from earlier runs")
    parser.add_argument("--csv", default=os.path.join(HERE, "behaviors", "curated.csv"))
    parser.add_argument("--out", default=None, help="defaults to overwriting --csv")
    parser.add_argument("--metric", choices=("verified", "inloop"), default="verified")
    parser.add_argument("--threshold", type=float, default=0.5, help="in-loop score cutoff for --metric inloop")
    parser.add_argument("--min-genes", type=int, default=10,
                        help="behaviors seen by fewer genes stay unrated (rate too noisy)")
    args = parser.parse_args()

    outcomes = load_outcomes(args.candidates, args.metric, args.threshold)
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = [c for c in (reader.fieldnames or []) if c not in TIER_COLS]

    rates, counts = {}, {}
    for row in rows:
        b = row["behavior"].strip()
        seen = outcomes.get(b, {})
        if len(seen) >= args.min_genes:
            rates[b] = sum(seen.values()) / len(seen)
            counts[b] = len(seen)
    tiers = assign_tiers(rates) if rates else {}

    for row in rows:
        b = row["behavior"].strip()
        row["solve_rate"] = f"{rates[b]:.4f}" if b in rates else ""
        row["n_genes"] = counts.get(b, "")
        row["tier"] = tiers.get(b, "")

    out = args.out or args.csv
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols + TIER_COLS)
        writer.writeheader()
        writer.writerows(rows)

    for t in ("easy", "medium", "hard"):
        members = [rates[b] for b in tiers if tiers[b] == t]
        span = f"solve_rate {min(members):.2f}-{max(members):.2f}" if members else "empty"
        print(f"{t:6s} {len(members):3d} behaviors  {span}")
    print(f"unrated: {len(rows) - len(rates)}  (seen by < {args.min_genes} genes)")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
