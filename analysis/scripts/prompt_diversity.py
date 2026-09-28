#!/usr/bin/env python3
"""Measure retained/offspring prompt diversity without changing raw run outputs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys

# This analysis uses PyTorch; avoid importing the repo's unrelated TF/Keras stack.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.utils.prompt_diversity import DEFAULT_EMBEDDING_MODEL, SentenceBertEmbedder, load_run, measure_cohort
from sota.Jailbreak.prompt_records import SERIALIZATION_VERSION, atomic_json, checkpoint_fingerprint, digest


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="Run directory containing ckpt/ or checkpoints/")
    parser.add_argument("--records-dir", type=Path, help="Defaults to sota/Jailbreak/results/<run_id>")
    parser.add_argument("--candidates", type=Path, help="Alternate historical candidates.jsonl")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-tokenizer", help="Actual target checkpoint path or Hugging Face tokenizer ID")
    parser.add_argument("--tokenizer-revision", help="Immutable revision for a remote target tokenizer")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-revision", help="Override pinned default revision; custom models resolve once and retain a model lock")
    parser.add_argument("--device", default="cpu", help="Embedding device, e.g. cpu or cuda:0")
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--ngram-size", type=positive_int, default=3)
    parser.add_argument("--ngram-only", action="store_true")
    return parser.parse_args(argv)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_trends(output, summaries):
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output / "system-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    generations = range(min(row["generation"] for row in summaries), max(row["generation"] for row in summaries) + 1)
    for metric, title in (("ngram", "Token n-gram diversity"), ("semantic", "Semantic diversity")):
        fig, ax = plt.subplots(figsize=(9, 5))
        for cohort, color in (("retained", "tab:blue"), ("offspring", "tab:orange")):
            rows = {row["generation"]: row for row in summaries if row["cohort"] == cohort}
            if not rows:
                continue
            for stat, style, label in (("mean", "-", "mean pair distance"),
                                       ("nearest_neighbor", "--", "mean nearest-neighbor distance")):
                values = [rows.get(g, {}).get(f"{metric}_{stat}") for g in generations]
                ax.plot(list(generations), [np.nan if v is None else v for v in values],
                        linestyle=style, marker="o", color=color, label=f"{cohort}: {label}")
        ax.set(title=title, xlabel="Generation", ylabel="Jaccard distance" if metric == "ngram" else "Cosine distance")
        if metric == "ngram":
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(bottom=0)
        ax.set_xticks(list(generations))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        if not any(row[f"{metric}_mean"] is not None for row in summaries):
            ax.text(0.5, 0.5, "Unavailable — see diagnostics and metadata", transform=ax.transAxes, ha="center")
        fig.tight_layout()
        fig.savefig(output / f"{metric}_diversity.png", dpi=180)
        plt.close(fig)


def main(argv=None):
    args = get_args(argv)
    run_dir = args.run_directory.resolve()
    run_id = run_dir.parent.name if run_dir.name in ("ckpt", "checkpoints") else run_dir.name
    records_dir = args.records_dir or REPO_ROOT / "sota" / "Jailbreak" / "results" / run_id
    cohorts, notes = load_run(run_dir, records_dir, args.candidates)
    output = (args.output_dir or REPO_ROOT / "results" / "analysis" / run_id / "prompt_diversity").resolve()
    # Never overwrite or derive caches inside the experiment's raw records/checkpoints.
    if output == records_dir.resolve() or records_dir.resolve() in output.parents or output == run_dir or run_dir in output.parents:
        raise ValueError("Output directory must be outside raw run and prompt-record directories")
    output.mkdir(parents=True, exist_ok=True)
    contexts = [cohort.get("target_context", {}) for cohort in cohorts]
    recorded_paths = {context["target_checkpoint"] for context in contexts if context.get("target_checkpoint")}
    tokenizer_name = args.target_tokenizer or os.getenv("JB_TARGET_MODEL_PATH")
    if not tokenizer_name and len(recorded_paths) == 1:
        tokenizer_name = recorded_paths.pop()
    if not tokenizer_name:
        raise ValueError("Specify --target-tokenizer with the actual target checkpoint; the served alias 'target' is insufficient")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=args.tokenizer_revision)
    local_fingerprint = checkpoint_fingerprint(tokenizer_name)
    for context in contexts:
        if context.get("target_fingerprint") and context["target_fingerprint"] != local_fingerprint:
            raise ValueError("Target tokenizer checkpoint does not match the recorded target fingerprint")
    tokenizer_fingerprint = digest(tokenizer.backend_tokenizer.to_str()) if hasattr(tokenizer, "backend_tokenizer") else local_fingerprint
    semantic_error = None
    embedder = None
    if not args.ngram_only:
        try:
            embedder = SentenceBertEmbedder(output / "embedding_cache", args.embedding_model,
                                           args.embedding_revision, args.device, args.batch_size)
        except Exception as exc:
            semantic_error = f"{type(exc).__name__}: {exc}"
            print(f"WARNING: semantic diversity unavailable: {semantic_error}", file=sys.stderr)
    summaries, case_rows, diagnostics = [], [], []
    features = {}
    for cohort in cohorts:
        print(f"Measuring generation {cohort['generation']} {cohort['cohort']} ({len(cohort['strategy_ids'])} strategies)", flush=True)
        summary, cases, prompts = measure_cohort(
            cohort, lambda text: tokenizer.encode(text, add_special_tokens=False, truncation=False),
            embedder, args.ngram_size, features, "model_unavailable" if semantic_error else "disabled")
        summaries.append(summary)
        case_rows.extend(cases)
        diagnostics.extend(prompts)
    write_csv(output / "generation_diversity.csv", summaries)
    write_csv(output / "case_diversity.csv", case_rows)
    write_csv(output / "prompt_diagnostics.csv", diagnostics)
    packages = {}
    for name in ("numpy", "torch", "transformers", "sentence-transformers", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    atomic_json(output / "metadata.json", {
        "schema_version": 1, "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir), "records_directory": str(records_dir.resolve()),
        "candidates_path": str((args.candidates or records_dir / "candidates.jsonl").resolve()),
        "ngram_size": args.ngram_size, "serialization": SERIALIZATION_VERSION,
        "tokenizer": {"name": tokenizer_name, "revision": args.tokenizer_revision or tokenizer.init_kwargs.get("_commit_hash"),
                      "fingerprint": tokenizer_fingerprint, "checkpoint_fingerprint": local_fingerprint},
        "embedding": embedder.metadata if embedder else {"model": args.embedding_model, "status": "unavailable" if semantic_error else "disabled"},
        "semantic_error": semantic_error, "packages": packages, "notes": notes,
        "limitations": ["Chunk pooling approximates whole-prompt meaning.",
                        "The default embedding checkpoint is English-oriented.",
                        "Cohort differences do not imply post-evaluation survivor selection."],
        "aggregation": "all unordered same-case strategy pairs; equal weight per eligible case; metric-specific coverage",
        "unavailable_csv_value": "empty field",
    })
    plot_trends(output, summaries)
    print(f"Saved diversity reports to {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc))
