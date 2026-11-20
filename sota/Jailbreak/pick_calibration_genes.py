"""Pick genes to calibrate behavior difficulty with, spread across the verified-ASR range.

uv run python pick_calibration_genes.py results/<run_id>/candidates.jsonl [more ...] -k 20 > calibration_genes.txt

Genes are ranked by verified ASR (fraction of their rows HarmBench-verified) and
k are taken at evenly spaced quantiles, so weak, middling and strong attackers
all vote on how hard each behavior is. Prints one gene id per line, `seed` first.
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", nargs="+")
    parser.add_argument("-k", type=int, default=20)
    parser.add_argument("--variant-dir", default=None,
                        help="keep only genes with prompt_<id>.py in this dir (e.g. models/llmge_models)")
    args = parser.parse_args()

    per_gene = {}  # gene -> {behavior: verified}
    for path in args.candidates:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    ok = bool(row.get("verification") and row["verification"].get("verified_success"))
                    per_gene.setdefault(row["candidate_id"], {})[row["behavior"]] = ok
    asr = {g: sum(v.values()) / len(v) for g, v in per_gene.items() if g != "seed"}
    if args.variant_dir:
        asr = {g: a for g, a in asr.items()
               if os.path.exists(os.path.join(args.variant_dir, f"prompt_{g}.py"))}
    ranked = sorted(asr, key=lambda g: (asr[g], g))
    if not ranked:
        raise SystemExit("no usable genes")
    k = min(args.k, len(ranked))
    picks = [ranked[round(i * (len(ranked) - 1) / max(1, k - 1))] for i in range(k)]
    print("seed")
    for g in dict.fromkeys(picks):
        print(g)
    import sys
    print(f"# picked {len(set(picks))} genes, ASR {asr[picks[0]]:.2f}..{asr[picks[-1]]:.2f}", file=sys.stderr)

if __name__ == "__main__":
    main()
