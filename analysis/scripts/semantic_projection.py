#!/usr/bin/env python3
"""Render case-matched semantic clusters and their evolution from cached embeddings."""

from __future__ import annotations

import argparse
from collections import Counter
import colorsys
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from analysis.utils.semantic_projection import (
    choose_representative_case, generation_alpha, load_cached_analysis, project_case,
)
from analysis.scripts.prompt_diversity import write_csv
from sota.Jailbreak.prompt_records import atomic_json


def cluster_color(label):
    if label < 0:
        return "#737b88"
    rgb = colorsys.hls_to_rgb((0.58 + label * 0.61803398875) % 1, 0.43, 0.68)
    return "#" + "".join(f"{round(value * 255):02x}" for value in rgb)


def bounds(projected):
    points = np.array([[p["x"], p["y"]] for p in projected.values()])
    low, high = points.min(axis=0), points.max(axis=0)
    padding = np.maximum(high - low, 1) * 0.10
    return low - padding, high + padding


def draw_map(ax, case, generation, history=True):
    projected, rows = case["points"], case["observations"]
    generations = sorted({r["generation"] for r in rows if r["generation"] <= generation})
    for g in generations if history else [generation]:
        counts = Counter(r["prompt_hash"] for r in rows if r["generation"] == g)
        alpha = generation_alpha(g, generations) if history else 1.0
        for h, multiplicity in sorted(counts.items()):
            point = projected[h]
            latest = g == generation
            ax.scatter(point["x"], point["y"], c=cluster_color(point["cluster"]),
                       s=(72 if latest else 20) * (1 + 0.35 * (multiplicity - 1)),
                       alpha=alpha, edgecolors="white" if latest else "none",
                       linewidths=0.8, zorder=3 if latest else 2)
    low, high = bounds(projected)
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]),
           xlabel="Projection axis 1", ylabel="Projection axis 2")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)


def plot_outputs(output, payload, selected):
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    case = payload["cases"][selected]
    generations = [s["generation"] for s in case["stats"]]
    first, last = generations[0], generations[-1]
    fig = plt.figure(figsize=(15, 9), facecolor="white")
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], hspace=0.40, wspace=0.4)
    ax = fig.add_subplot(grid[:, :2])
    draw_map(ax, case, last)
    ax.set_title(f"Case {case['case_number']:02d} · generations {first}–{last}", loc="left", pad=14, fontsize=14)
    handles = [Line2D([], [], marker="o", linestyle="none", color=cluster_color(0),
                      alpha=generation_alpha(g, generations), markersize=9,
                      label=f"Generation {g}" + (" (latest)" if g == last else ""))
               for g in sorted({first, generations[len(generations) // 2], last})]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10)
    trend = fig.add_subplot(grid[0, 2])
    summary = payload["summary"]
    x = [s["generation"] for s in summary]
    values = [s["semantic_nearest_neighbor"] for s in summary]
    trend.plot(x, values, "o-", color="#2563a6", lw=2)
    trend.set(title="Nearest neighbors tighten overall", xlabel="Generation", ylabel="Mean cosine distance", ylim=(0, max(values) * 1.32))
    trend.text(0.04, 0.92, f"{values[0]:.4f} → {values[-1]:.4f}\n{(values[-1] / values[0] - 1) * 100:.1f}% relative change", transform=trend.transAxes, va="top", fontsize=11)
    trend.set_xticks(x)
    trend.grid(alpha=0.15)
    lower = fig.add_subplot(grid[1, 2])
    means = [s["semantic_mean"] for s in summary]
    lower.plot(x, means, "o-", color="#b97022", lw=2)
    lower.set(title="Mean pair distance remains higher", xlabel="Generation", ylabel="Mean cosine distance", ylim=(0, max(means) * 1.32))
    lower.text(0.04, 0.92, f"{means[0]:.4f} → {means[-1]:.4f}\n{(means[-1] / means[0] - 1) * 100:+.1f}% relative change", transform=lower.transAxes, va="top", fontsize=11)
    lower.set_xticks(x)
    lower.grid(alpha=0.15)
    labels = sorted({p["cluster"] for p in case["points"].values()})
    cluster_handles = [Line2D([], [], marker="o", linestyle="none", color=cluster_color(label),
                              label=f"Group {label + 1}" if label >= 0 else "Unassigned") for label in labels]
    fig.legend(handles=cluster_handles, loc="lower left", bbox_to_anchor=(0.045, 0.075),
               ncol=min(6, len(cluster_handles)), frameon=False, fontsize=10)
    fig.suptitle(f"Semantic diversity over time · {payload['run_id']}", x=0.055, ha="left", y=0.98, fontsize=21, weight="bold")
    fig.text(0.055, 0.934, "Earlier generations fade; the latest completed generation is fully opaque. Identical prompts overlap.", fontsize=12, color="#4b5563")
    q = case["quality"]
    fig.text(0.055, 0.035, f"Fixed densMAP for one case across all generations · colors: original-space HDBSCAN groups · neighborhood trustworthiness: {q['trustworthiness']:.3f}\nRight: original embedding distances, equally weighted across all {len(payload['cases'])} cases. Lower nearest-neighbor distance = closer local variants.", fontsize=10, color="#4b5563")
    fig.subplots_adjust(top=0.87, bottom=0.18, left=0.065, right=0.975)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"semantic_projection.{suffix}", dpi=180)
    plt.close(fig)

    cols = min(5, len(generations))
    rows = int(np.ceil(len(generations) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.35, rows * 3.6), squeeze=False)
    for ax, stat in zip(axes.flat, case["stats"]):
        for p in case["points"].values():
            ax.scatter(p["x"], p["y"], s=6, color="#d4d9df", alpha=0.25)
        draw_map(ax, case, stat["generation"], history=False)
        ax.set_title(f"Generation {stat['generation']} · n={stat['valid_strategies']}\nNN cosine distance: {stat['semantic_nearest_neighbor']:.4f}", fontsize=11)
        ax.set_xlabel("Axis 1", fontsize=9)
        ax.set_ylabel("Axis 2", fontsize=9)
    for ax in list(axes.flat)[len(generations):]:
        ax.set_visible(False)
    fig.suptitle(f"Semantic population snapshots · case {case['case_number']:02d} · shared axes across generations", fontsize=17, weight="bold")
    fig.text(0.04, 0.035, "Colors identify groups inferred in the original embedding space; faint gray points show all observed locations.\nNN is measured within each generation in the original embedding space. Groups and 2D shapes are exploratory.", fontsize=10, color="#4b5563")
    fig.tight_layout(rect=(0.02, 0.10, 0.99, 0.92))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"semantic_generations.{suffix}", dpi=180)
    plt.close(fig)


def write_interactive(output, payload, selected):
    template = (ROOT / "analysis" / "utils" / "semantic_projection.html").read_text()
    # Prevent an embedded value from closing the JSON script element.
    data = json.dumps({**payload, "selected": selected}, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    (output / "semantic_projection.html").write_text(template.replace("__PROJECTION_DATA__", data), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_directory", type=Path, help="Existing successful prompt_diversity output")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cohort", choices=("retained", "offspring"), default="retained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--case-id", help="Case for static PNG/PDF; default: median endpoint nearest-neighbor change")
    args = parser.parse_args(argv)
    if args.n_neighbors < 2 or args.min_cluster_size < 3:
        parser.error("n-neighbors must be >=2 and min-cluster-size >=3")
    source = args.analysis_directory.resolve()
    metadata, vectors, grouped, unavailable, summaries, source_cases = load_cached_analysis(source, args.cohort)
    output = (args.output_dir or source / "semantic_projection").resolve()
    for raw in (metadata.get("run_directory"), metadata.get("records_directory")):
        if raw and (output == Path(raw).resolve() or Path(raw).resolve() in output.parents):
            raise ValueError("Projection output must be outside raw experiment directories")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(output / "numba-cache"))
    payload = {"run_id": metadata["run_id"], "cohort": args.cohort, "cases": {}, "summary": []}
    quality_rows, point_rows, stats_rows = [], [], []
    for number, case_id in enumerate(sorted(grouped), 1):
        rows = sorted(grouped[case_id], key=lambda r: (r["generation"], r["candidate_id"]))
        print(f"Projecting case {number}/{len(grouped)}", flush=True)
        points, stats, quality = project_case(rows, vectors, args.seed, args.n_neighbors, args.min_cluster_size)
        for stat in stats:
            expected = source_cases[(stat["generation"], case_id)]
            for key in ("semantic_mean", "semantic_nearest_neighbor"):
                if expected[key] and not np.isclose(stat[key], float(expected[key]), atol=5e-7):
                    raise ValueError(f"Cached embeddings disagree with original metric: {case_id} {key}")
            stats_rows.append({"case_id": case_id, **stat})
        for row in rows:
            point_rows.append({"case_id": case_id, "generation": row["generation"], "candidate_id": row["candidate_id"],
                               "prompt_hash": row["prompt_hash"], **points[row["prompt_hash"]]})
        payload["cases"][case_id] = {"case_number": number, "points": points,
                                     "observations": [{k: r[k] for k in ("generation", "candidate_id", "prompt_hash")} for r in rows],
                                     "stats": stats, "quality": quality}
        quality_rows.append({"case_id": case_id, **quality})
    for row in sorted(summaries, key=lambda r: int(r["generation"])):
        payload["summary"].append({"generation": int(row["generation"]),
                                    **{k: float(row[k]) if row[k] else None for k in ("semantic_mean", "semantic_nearest_neighbor", "semantic_pair_coverage")}})
    selected = args.case_id or choose_representative_case({k: v["stats"] for k, v in payload["cases"].items()})
    if selected not in payload["cases"]:
        raise ValueError(f"Case not found: {selected}")
    write_csv(output / "projection_points.csv", point_rows)
    write_csv(output / "projection_quality.csv", quality_rows)
    write_csv(output / "projection_case_metrics.csv", stats_rows)
    atomic_json(output / "projection_data.json", payload)
    atomic_json(output / "metadata.json", {
        "run_id": metadata["run_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_analysis": str(source), "embedding": metadata["embedding"], "cohort": args.cohort,
        "selected_case": selected, "case_selection": "explicit" if args.case_id else "median relative endpoint nearest-neighbor change",
        "seed": args.seed, "n_neighbors": args.n_neighbors, "min_dist": 0.1,
        "densmap": True, "dens_lambda": 2.0, "min_cluster_size": args.min_cluster_size, "min_samples": 3,
        "fit": "one joint map per case on unique vectors from all completed generations; duplicate strategies restored afterward",
        "clustering": "HDBSCAN on original cosine distances across all generations within each case",
        "unavailable_observations": len(unavailable), "packages": {p: version(p) for p in ("umap-learn", "scikit-learn", "numpy", "matplotlib")},
        "limitations": ["2D distances and densities are approximations; confirm tightening with original-space metrics.",
                        "Cases have separate coordinate systems and must not be compared by map location.",
                        "Group labels are exploratory and depend on clustering parameters.",
                        "Repeated vectors are fit once, then restored with population multiplicity.",
                        "Maps use future generations descriptively; they are not predictive projections.",
                        "Generation 7 in this historical run has incomplete coverage; see source analysis."],
    })
    plot_outputs(output, payload, selected)
    write_interactive(output, payload, selected)
    (output / "README.md").write_text(
        f"# Semantic projection: {metadata['run_id']}\n\n"
        "Open `semantic_projection.png` for the temporal overlay: older generations fade and the newest is opaque. "
        "`semantic_generations.png` shows each generation on the same axes. PDFs are also available. "
        "Open `semantic_projection.html` in a browser for all cases, a generation slider, hover IDs, and playback. "
        "The HTML is self-contained and makes no network requests.\n\n"
        f"Static figures use case {payload['cases'][selected]['case_number']:02d}, `{selected}`, selected by "
        "the median relative endpoint nearest-neighbor change unless explicitly overridden. Cases are fit separately "
        "to avoid confusing differences between evaluation topics with strategy diversity.\n\n"
        "Colors come from HDBSCAN groups in the original embedding space, not clusters inferred from the drawing. "
        "Coordinates are shared across generations within a case, with exact repeated vectors mapped to the same location. "
        "Point size increases for identical prompts from multiple strategies in the displayed generation. "
        "The side plots use the original cosine distances with equal case weighting. A lower nearest-neighbor mean "
        "supports closer local variants; it does not prove that every group contracts.\n\n"
        "The densMAP objective encourages local neighborhood and relative density preservation, but does not guarantee "
        "faithful 2D geometry. `projection_quality.csv` records trustworthiness and top-5 neighbor overlap per case. "
        "See https://umap-learn.readthedocs.io/en/latest/densmap_demo.html for the method. "
        "Group boundaries and counts are exploratory; no selection or fitness was changed.\n", encoding="utf-8")
    print(f"Saved projections to {output}; selected case {payload['cases'][selected]['case_number']:02d}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
