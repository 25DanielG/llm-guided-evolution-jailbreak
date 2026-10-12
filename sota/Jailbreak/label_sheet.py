"""Draw a stratified, blind human-labeling sample from reporting-judge CSVs.

Import labeler_*.csv into Google Sheets with "Convert text to numbers, dates, and formulas"
unchecked, or some item_ids become numbers and responses starting with -, +, = become formulas.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

STRATA = ("inloop_yes_hb_no", "inloop_no_hb_yes", "both_yes", "both_no")
STRATUM_OF = {(1, 0): STRATA[0], (0, 1): STRATA[1], (1, 1): STRATA[2], (0, 0): STRATA[3]}
RULE_COLUMNS = {"threshold": "inloop_yes", "gate": "inloop_gate"}
SHEET_COLUMNS = ["item_id", "behavior_id", "behavior", "response", "label", "confidence", "notes"]
KEY_COLUMNS = ["item_id", "run_id", "stratum", "weight", "N_h", "n_h", "behavior_id", "inloop_score",
               "inloop_yes", "inloop_gate", "harmbench_score", "harmbench_p_yes", "harmbench_card_score",
               "harmbench_card_p_yes", "generation", "candidate_id"]
VERDICT_COLUMNS = ("inloop_yes", "inloop_gate", "harmbench_score", "harmbench_card_score")


def read_csvs(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8", newline="") as file:
            rows.extend(csv.DictReader(file))
    return rows


def write_csv(path, columns, rows):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def binary(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number in (0, 1) else None


def behavior_key(row):
    # behavior_id is "" when the text matched no behaviors-CSV row; fall back to the text itself.
    return row["behavior_id"] or row["behavior"]


def eligible_strata(rows, rule, excluded):
    # frame = eligible population before --exclude-key (what weights represent); strata = still sampleable.
    frame, strata, skipped, seen, conflicts = defaultdict(list), defaultdict(list), Counter(), {}, set()
    for row in rows:
        verdicts = tuple(row[column] for column in VERDICT_COLUMNS)
        if row["item_id"] in seen:
            if seen[row["item_id"]] != verdicts:
                conflicts.add(row["item_id"])
            skipped["duplicate_item"] += 1
            continue
        seen[row["item_id"]] = verdicts
        inloop, harmbench = binary(row[RULE_COLUMNS[rule]]), binary(row["harmbench_score"])
        if inloop is None:
            skipped["inloop_missing"] += 1
        elif harmbench is None:
            skipped["harmbench_unusable"] += 1
        elif not row["response"].strip():
            skipped["empty_response"] += 1
        else:
            h = (row["run_id"], STRATUM_OF[(inloop, harmbench)])
            frame[h].append(row)
            if row["item_id"] in excluded:
                skipped["excluded"] += 1
            else:
                strata[h].append(row)
    if conflicts:
        raise ValueError(f"{len(conflicts)} item_ids repeat across --reporting rows with different judge verdicts "
                         "(e.g. a --dry-run CSV and a scored CSV of the same run); pass each run once")
    return frame, strata, skipped


def largest_remainder(total, weights):
    weight_sum = sum(weights)
    if total <= 0 or weight_sum <= 0:
        return [0] * len(weights)
    quotas = [Fraction(total) * weight / weight_sum for weight in weights]
    counts = [int(quota) for quota in quotas]
    order = sorted(range(len(weights)), key=lambda i: (counts[i] - quotas[i], i))
    for i in order[:total - sum(counts)]:
        counts[i] += 1
    return counts


def allocate(total, available, alloc):
    target = largest_remainder(total, alloc)
    take = [min(t, a) for t, a in zip(target, available)]
    while sum(take) < total:
        open_weights = [w if t < a else 0 for w, t, a in zip(alloc, take, available)]
        if not any(open_weights):
            break
        extra = largest_remainder(total - sum(take), open_weights)
        take = [min(t + e, a) for t, e, a in zip(take, extra, available)]
    return target, take


def select(population, k, cap, rng):
    pool = sorted(population, key=lambda row: row["item_id"])
    rng.shuffle(pool)
    taken, counts, limit = [], Counter(), cap
    while len(taken) < k:
        for index, row in enumerate(pool):
            if len(taken) < k and index not in taken and counts[behavior_key(row)] < limit:
                taken.append(index)
                counts[behavior_key(row)] += 1
        if len(taken) < k:
            limit += 1
    return [pool[i] for i in taken], limit - cap


def post_stratified_weights(population, sample):
    population_counts = Counter(behavior_key(row) for row in population)
    sample_counts = Counter(behavior_key(row) for row in sample)
    raw = {row["item_id"]: population_counts[behavior_key(row)] / sample_counts[behavior_key(row)] for row in sample}
    scale = len(population) / sum(raw.values()) if raw else 0
    return {item_id: weight * scale for item_id, weight in raw.items()}


def build(rows, n, alloc, rule, cap, seed, excluded=()):
    rng = random.Random(seed)
    frame, strata, skipped = eligible_strata(rows, rule, set(excluded))
    runs = sorted({run for run, _ in frame})
    # Even split across runs; a run without enough items passes its unfilled share to the others.
    capacity = [sum(len(strata[(run, name)]) for name, weight in zip(STRATA, alloc) if weight > 0) for run in runs]
    shares, run_takes = allocate(n, capacity, [1] * len(runs))
    sample, cells, run_summary = [], [], {}
    for run, share, run_n in zip(runs, shares, run_takes):
        populations = [strata[(run, name)] for name in STRATA]
        target, take = allocate(run_n, [len(p) for p in populations], alloc)
        for name, population, allocated, k in zip(STRATA, populations, target, take):
            chosen, relaxations = select(population, k, cap, rng)
            # Weights represent the pre-exclusion population, so keys from successive batches can be concatenated.
            size = len(frame[(run, name)])
            weights = post_stratified_weights(frame[(run, name)], chosen)
            for row in chosen:
                sample.append(dict(row, stratum=name, weight=weights[row["item_id"]], N_h=size, n_h=len(chosen)))
            cells.append({"run_id": run, "stratum": name, "N_h": size, "available": len(population),
                          "allocated": allocated, "n_h": len(chosen), "cap_relaxations": relaxations})
        run_summary[run] = {"requested": share, "sampled": sum(take),
                            "eligible": sum(len(p) for p in populations)}
    sheet = list(sample)
    rng.shuffle(sheet)
    summary = {"rule": rule, "inloop_column": RULE_COLUMNS[rule], "seed": seed,
               "alloc": dict(zip(STRATA, (float(w) for w in alloc))), "per_behavior_cap": cap,
               "n_requested": n, "n_sampled": len(sample), "input_rows": len(rows),
               "eligible": sum(len(p) for p in strata.values()), "skipped": dict(skipped),
               "runs": run_summary, "strata": cells}
    return sample, sheet, summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reporting", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--alloc", default="35,35,15,15")
    parser.add_argument("--rule", choices=sorted(RULE_COLUMNS), default="threshold")
    parser.add_argument("--per-behavior-cap", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude-key", nargs="+", default=[])
    args = parser.parse_args(argv)
    alloc = [Fraction(part.strip()) for part in args.alloc.split(",")]
    if len(alloc) != len(STRATA) or min(alloc) < 0 or sum(alloc) <= 0:
        parser.error(f"--alloc needs {len(STRATA)} non-negative proportions for {','.join(STRATA)}")

    excluded = {row["item_id"] for row in read_csvs(args.exclude_key)}
    sample, sheet, summary = build(read_csvs(args.reporting), args.n, alloc, args.rule,
                                   args.per_behavior_cap, args.seed, excluded)
    summary.update(reporting=args.reporting, exclude_key=args.exclude_key)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("labeler_1.csv", "labeler_2.csv"):
        write_csv(out_dir / name, SHEET_COLUMNS, sheet)
    write_csv(out_dir / "key.csv", KEY_COLUMNS, sample)
    (out_dir / "sampling_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"label_sheet eligible={summary['eligible']} sampled={summary['n_sampled']}/{args.n} "
          f"runs={len(summary['runs'])} skipped={summary['skipped']} out_dir={out_dir}", flush=True)
    print("Google Sheets import: uncheck 'Convert text to numbers, dates, and formulas' "
          "so item_ids and responses stay verbatim", flush=True)
    return summary


if __name__ == "__main__":
    main()
