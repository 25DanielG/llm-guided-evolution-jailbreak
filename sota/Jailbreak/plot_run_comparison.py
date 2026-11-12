"""Create presentation-ready figures comparing two completed jailbreak runs.

Example::

    uv run python sota/Jailbreak/plot_run_comparison.py \
        --map-run map_elites_20261025_183444 \
        --baseline-run nsga2_baseline_20261029_222504

The script expects ``make_baseline_report.py`` outputs under
``comparison/map_elites`` and ``comparison/nsga2`` by default. It writes PNG,
SVG, and a compact CSV summary under ``comparison/figures``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / f"xdg-cache-{os.getuid()}"),
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402


FORMAT_STRATEGIES = (
    "freeform",
    "base64",
    "rot13",
    "homoglyph",
    "json_steps",
    "transcript",
)
TURN_DEPTHS = ("zero", "one_or_two", "three_plus")
DEPTH_LABELS = (
    "0 prior messages",
    "1–2 prior messages",
    "3+ prior messages",
)
METHOD_COLORS = {
    "MAP-Elites": "#B3A369",
    "NSGA-II": "#003057",
}
CHECKPOINT_RE = re.compile(r"checkpoint_gen_(\d+)\.pkl$")


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_genes(report_dir: Path, method: str) -> pd.DataFrame:
    path = report_dir / "genes.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run sota/Jailbreak/make_baseline_report.py first"
        )
    frame = pd.read_csv(path)
    required = {
        "generation",
        "judge_mean",
        "verified_asr",
        "avg_prompt_tokens",
        "diversity",
        "rendered_prompt_sig",
        "easy_verified",
        "medium_verified",
        "hard_verified",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

    numeric = required - {"rendered_prompt_sig"}
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["method"] = method
    return frame


def pareto_mask(frame: pd.DataFrame) -> np.ndarray:
    """Return the 3-objective nondominated mask.

    Judge score and diversity are maximized; token count is minimized.
    Rows with a missing objective are never marked nondominated.
    """

    columns = ["judge_mean", "diversity", "avg_prompt_tokens"]
    values = frame[columns].to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1)
    mask = np.zeros(len(frame), dtype=bool)
    valid_indices = np.flatnonzero(valid)

    for index in valid_indices:
        judge, diversity, tokens = values[index]
        others = values[valid_indices]
        no_worse = (
            (others[:, 0] >= judge)
            & (others[:, 1] >= diversity)
            & (others[:, 2] <= tokens)
        )
        strictly_better = (
            (others[:, 0] > judge)
            | (others[:, 1] > diversity)
            | (others[:, 2] < tokens)
        )
        mask[index] = not np.any(no_worse & strictly_better)
    return mask


def final_generation_summary(
    frame: pd.DataFrame, generation: int
) -> dict[str, float | int]:
    final = frame.loc[frame["generation"] == generation].copy()
    if final.empty:
        raise ValueError(f"no genes were assigned to generation {generation}")

    return {
        "generation": generation,
        "genes": len(final),
        "judge_mean": final["judge_mean"].mean(),
        "verified_mean": final["verified_asr"].mean(),
        "verified_best": final["verified_asr"].max(),
        "easy_verified": final["easy_verified"].mean(),
        "medium_verified": final["medium_verified"].mean(),
        "hard_verified": final["hard_verified"].mean(),
        "mean_tokens": final["avg_prompt_tokens"].mean(),
        "distinct_prompts": final["rendered_prompt_sig"].nunique(),
    }


def load_archive_summary(root: Path, run_id: str) -> dict:
    path = root / "sota" / "Jailbreak" / "results" / run_id / "map_elites.json"
    if not path.exists():
        raise FileNotFoundError(f"missing final MAP-Elites summary: {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def archive_matrix(payload: dict, quality_index: int = 0) -> np.ndarray:
    matrix = np.full((len(TURN_DEPTHS), len(FORMAT_STRATEGIES)), np.nan)
    format_index = {
        name: index for index, name in enumerate(FORMAT_STRATEGIES)
    }
    depth_index = {name: index for index, name in enumerate(TURN_DEPTHS)}
    for elite in payload.get("elites", []):
        format_name = elite.get("format_strategy")
        depth_name = elite.get("conversation_depth")
        quality = elite.get("quality", [])
        if (
            format_name in format_index
            and depth_name in depth_index
            and len(quality) > quality_index
        ):
            matrix[depth_index[depth_name], format_index[format_name]] = float(
                quality[quality_index]
            )
    return matrix


_CHECKPOINT_TYPES_REGISTERED = False


def _register_checkpoint_types(root: Path) -> None:
    global _CHECKPOINT_TYPES_REGISTERED
    if _CHECKPOINT_TYPES_REGISTERED:
        return
    sys.path.insert(0, str(root))
    # Importing run_improved registers the dynamic DEAP Individual classes used
    # by checkpoint pickles. It does not start an evolution run on import.
    __import__("run_improved")
    _CHECKPOINT_TYPES_REGISTERED = True


def checkpoint_coverage(root: Path, run_id: str) -> pd.DataFrame:
    checkpoint_dir = root / run_id / "ckpt"
    paths: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("checkpoint_gen_*.pkl"):
        match = CHECKPOINT_RE.search(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    if not paths:
        raise FileNotFoundError(f"no checkpoints found under {checkpoint_dir}")

    _register_checkpoint_types(root)
    rows = []
    for generation, path in sorted(paths):
        with path.open("rb") as file:
            payload = pickle.load(file)
        archive = (
            payload.get("map_elites_archive")
            if isinstance(payload, dict)
            else None
        )
        if archive is None:
            continue
        rows.append(
            {
                "generation": generation,
                "occupied_cells": len(archive),
                "capacity": archive.capacity,
            }
        )
    if not rows:
        raise ValueError(
            f"checkpoints in {checkpoint_dir} do not contain an archive"
        )
    return pd.DataFrame(rows)


def checkpoint_population_fitness(root: Path, run_id: str) -> pd.DataFrame:
    """Load the population-relative fitness saved after every generation.

    Diversity is computed by the controller after independent evaluations
    finish. Historical result CSVs therefore contain a zero placeholder, while
    checkpoints contain the exact diversity values used by selection.
    """
    checkpoint_dir = root / run_id / "ckpt"
    paths: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("checkpoint_gen_*.pkl"):
        match = CHECKPOINT_RE.search(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    if not paths:
        raise FileNotFoundError(f"no checkpoints found under {checkpoint_dir}")

    _register_checkpoint_types(root)
    rows = []
    for generation, path in sorted(paths):
        with path.open("rb") as file:
            payload = pickle.load(file)
        population = (
            payload.get("population", []) if isinstance(payload, dict) else []
        )
        for individual in population:
            fitness = tuple(getattr(individual.fitness, "values", ()))
            if len(fitness) < 3:
                continue
            rows.append(
                {
                    "generation": generation,
                    "gene_id": individual[0],
                    "judge_mean": float(fitness[0]),
                    "diversity": float(fitness[1]),
                    "avg_prompt_tokens": float(fitness[2]),
                }
            )
    if not rows:
        raise ValueError(
            f"checkpoints in {checkpoint_dir} do not contain population fitness"
        )
    return pd.DataFrame(rows).drop_duplicates(
        subset=["generation", "gene_id"], keep="last"
    )


def _scaled_marker_sizes(tokens: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(tokens), dtype=float)
    finite = np.isfinite(values)
    sizes = np.full(len(values), 38.0)
    if not finite.any():
        return sizes
    low, high = np.nanmin(values[finite]), np.nanmax(values[finite])
    if high == low:
        sizes[finite] = 58.0
    else:
        sizes[finite] = (
            28.0 + 72.0 * (values[finite] - low) / (high - low)
        )
    return sizes


def save_figure(
    figure: plt.Figure, output_dir: Path, stem: str, dpi: int
) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def plot_pareto(
    frames: dict[str, pd.DataFrame], output_dir: Path, dpi: int
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 6.0))
    figure.subplots_adjust(bottom=0.19)
    legend_handles = []

    for method, frame in frames.items():
        objectives = frame[
            ["judge_mean", "diversity", "avg_prompt_tokens"]
        ].to_numpy(dtype=float)
        candidates = frame.loc[
            frame["generation"].between(0, 98)
            & np.isfinite(objectives).all(axis=1)
        ].copy()
        sizes = _scaled_marker_sizes(candidates["avg_prompt_tokens"])
        color = METHOD_COLORS[method]
        axis.scatter(
            candidates["diversity"],
            candidates["judge_mean"],
            s=sizes,
            color=color,
            alpha=0.34,
            linewidths=0,
        )
        front_mask = pareto_mask(candidates)
        front = candidates.loc[front_mask]
        front_sizes = sizes[front_mask]
        axis.scatter(
            front["diversity"],
            front["judge_mean"],
            s=front_sizes + 24,
            facecolors="none",
            edgecolors=color,
            linewidths=1.8,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=8,
                label=method,
            )
        )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#333333",
            markersize=9,
            label="3-objective nondominated",
        )
    )
    axis.legend(handles=legend_handles, frameon=False, loc="best")
    axis.set_title("Quality–diversity candidate trade-off")
    axis.set_xlabel("Prompt diversity objective (higher is better)")
    axis.set_ylabel("Mean judge score (higher is better)")
    axis.grid(alpha=0.2)
    figure.text(
        0.5,
        0.025,
        "Checkpoint population snapshots. Marker size represents prompt "
        "length; outlined candidates are nondominated on judge ↑, "
        "diversity ↑, tokens ↓.",
        ha="center",
        fontsize=9,
    )
    save_figure(figure, output_dir, "pareto_front", dpi)


def plot_coverage(
    histories: dict[str, pd.DataFrame], output_dir: Path, dpi: int
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    capacity = 18
    for method, history in histories.items():
        capacity = max(capacity, int(history["capacity"].max()))
        axis.plot(
            history["generation"],
            history["occupied_cells"],
            marker="o",
            linewidth=2.2,
            markersize=4.5,
            color=METHOD_COLORS[method],
            label=method,
        )
        final = history.iloc[-1]
        axis.annotate(
            f"{int(final['occupied_cells'])}/{int(final['capacity'])}",
            (final["generation"], final["occupied_cells"]),
            xytext=(6, 6),
            textcoords="offset points",
            color=METHOD_COLORS[method],
            fontweight="bold",
        )
    axis.set_title("Archive coverage over generations")
    axis.set_xlabel("Generation")
    axis.set_ylabel("Occupied archive cells")
    axis.set_ylim(0, capacity)
    first_generation = min(
        int(history["generation"].min()) for history in histories.values()
    )
    last_generation = max(
        int(history["generation"].max()) for history in histories.values()
    )
    axis.set_xticks(range(first_generation, last_generation + 1))
    axis.set_yticks(range(0, capacity + 1, 3))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    save_figure(figure, output_dir, "coverage_by_generation", dpi)


def plot_heatmaps(
    archives: dict[str, dict], output_dir: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for axis, (method, payload) in zip(axes, archives.items()):
        matrix = archive_matrix(payload)
        image = axis.imshow(
            matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto"
        )
        occupied = int(
            payload.get("occupied_cells", np.isfinite(matrix).sum())
        )
        capacity = int(payload.get("capacity", matrix.size))
        axis.set_title(f"{method}: {occupied}/{capacity} cells")
        axis.set_xticks(range(len(FORMAT_STRATEGIES)))
        axis.set_xticklabels(
            [
                "Freeform",
                "Base64",
                "ROT13",
                "Homoglyph",
                "JSON\nsteps",
                "Transcript",
            ],
            rotation=25,
            ha="right",
        )
        axis.set_yticks(range(len(TURN_DEPTHS)))
        axis.set_yticklabels(DEPTH_LABELS)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                label = "—" if np.isnan(value) else f"{value:.2f}"
                color = (
                    "#222222"
                    if np.isnan(value) or value > 0.65
                    else "#FFFFFF"
                )
                axis.text(
                    column, row, label, ha="center", va="center", color=color
                )
        axis.set_xlabel("Prompt-format strategy")
    axes[0].set_ylabel("Prior conversation messages")
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, shrink=0.86)
        colorbar.set_label("Elite judge score")
    figure.suptitle("Final MAP-Elites archives")
    save_figure(figure, output_dir, "archive_heatmaps", dpi)


def plot_verified_performance(
    summaries: dict[str, dict[str, float | int]], output_dir: Path, dpi: int
) -> None:
    labels = ["Overall", "Easy", "Medium", "Hard", "Best candidate"]
    keys = [
        "verified_mean",
        "easy_verified",
        "medium_verified",
        "hard_verified",
        "verified_best",
    ]
    positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9.2, 5.5))
    figure.subplots_adjust(bottom=0.18)

    for offset, (method, summary) in zip(
        (-width / 2, width / 2), summaries.items()
    ):
        values = [float(summary[key]) for key in keys]
        bars = axis.bar(
            positions + offset,
            values,
            width,
            color=METHOD_COLORS[method],
            label=f"{method} (n={summary['genes']})",
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.0%}" for value in values],
            padding=3,
        )

    axis.set_title("Verified performance at the final generation")
    axis.set_ylabel("Verified attack success rate")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1.0)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    map_tokens = float(summaries["MAP-Elites"]["mean_tokens"])
    baseline_tokens = float(summaries["NSGA-II"]["mean_tokens"])
    figure.text(
        0.5,
        0.025,
        f"Mean prompt length: MAP-Elites {map_tokens:.0f} tokens; "
        f"NSGA-II {baseline_tokens:.0f} tokens.",
        ha="center",
        fontsize=9,
    )
    save_figure(figure, output_dir, "verified_performance", dpi)


def write_summary(
    summaries: dict[str, dict[str, float | int]],
    archives: dict[str, dict],
    output_dir: Path,
) -> None:
    fields = [
        "algorithm",
        "generation",
        "genes",
        "judge_mean",
        "verified_mean",
        "verified_best",
        "easy_verified",
        "medium_verified",
        "hard_verified",
        "mean_tokens",
        "distinct_prompts",
        "occupied_cells",
        "capacity",
        "coverage",
    ]
    with (output_dir / "comparison_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for method, summary in summaries.items():
            archive = archives[method]
            writer.writerow(
                {
                    "algorithm": method,
                    **summary,
                    "occupied_cells": archive["occupied_cells"],
                    "capacity": archive["capacity"],
                    "coverage": archive["coverage"],
                }
            )


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Plot a completed MAP-Elites run against an NSGA-II baseline."
        )
    )
    parser.add_argument(
        "--root", default=str(default_root), help="repository root"
    )
    parser.add_argument("--map-run", required=True, help="MAP-Elites run ID")
    parser.add_argument("--baseline-run", required=True, help="NSGA-II run ID")
    parser.add_argument("--map-report", default="comparison/map_elites")
    parser.add_argument("--baseline-report", default="comparison/nsga2")
    parser.add_argument("--output", default="comparison/figures")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = resolve_path(root, args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        "MAP-Elites": load_genes(
            resolve_path(root, args.map_report), "MAP-Elites"
        ),
        "NSGA-II": load_genes(
            resolve_path(root, args.baseline_report), "NSGA-II"
        ),
    }
    archives = {
        "MAP-Elites": load_archive_summary(root, args.map_run),
        "NSGA-II": load_archive_summary(root, args.baseline_run),
    }
    histories = {
        "MAP-Elites": checkpoint_coverage(root, args.map_run),
        "NSGA-II": checkpoint_coverage(root, args.baseline_run),
    }
    fitness_histories = {
        "MAP-Elites": checkpoint_population_fitness(root, args.map_run),
        "NSGA-II": checkpoint_population_fitness(root, args.baseline_run),
    }
    final_generations = {
        method: int(history["generation"].max())
        for method, history in histories.items()
    }
    summaries = {
        method: final_generation_summary(frame, final_generations[method])
        for method, frame in frames.items()
    }

    plot_coverage(histories, output_dir, args.dpi)
    plot_heatmaps(archives, output_dir, args.dpi)
    plot_pareto(fitness_histories, output_dir, args.dpi)
    plot_verified_performance(summaries, output_dir, args.dpi)
    write_summary(summaries, archives, output_dir)
    for method, history in fitness_histories.items():
        stem = method.lower().replace("-", "").replace(" ", "_")
        history.to_csv(
            output_dir / f"{stem}_fitness_history.csv", index=False
        )

    print(f"Wrote comparison figures to {output_dir}")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
