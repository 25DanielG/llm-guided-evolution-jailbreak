"""Slide-ready figures for the judge validation of one run (post-hoc; never part of the evolution loop).
    python3 -I plot_validation.py --metrics validation_metrics.json --key key.csv --labels labels.csv
        --population reporting.csv [--heldout heldout_840.json] [--run RUN_ID CANDIDATES_JSONL REPORTING_CSV ...]
        --out-dir figures
Figures and figure_data.json carry only ids, labels, generations, counts and rates; response and behavior text is
hashed into item_ids and never written.
"""
import argparse
import csv
import json
import math
import random
import sys
import textwrap
from collections import Counter
from itertools import groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python3 -I leaves the script dir off sys.path
import analyze_labels  # noqa: E402
import reporting_judge  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from matplotlib.transforms import blended_transform_factory, offset_copy  # noqa: E402

csv.field_size_limit(10 ** 9)

# One color and marker per judge in every figure. Only the three judges the story is about get hues (dataviz slots
# 1-3, the only slots that stay distinct for every pair, colorblind viewers included); the other three are neutral
# ink told apart by marker shape and labels. HarmBench (model card) and Gemma share a gray: their marks never cross.
JUDGES = {"inloop_yes": ("In-loop judge", "#2a78d6", "o"),
          "harmbench_score": ("HarmBench (official)", "#eb6834", "s"),
          "llama70b": ("Llama-3.3-70B + guide", "#1baf7a", "D"),
          "harmbench_card_score": ("HarmBench (model card)", "#898781", "^"),
          "gemma27b": ("Gemma-2-27B + guide", "#898781", "v"),
          "mixtral8x7b": ("Mixtral-8x7B + guide", "#52514e", "P")}
GUIDE_JUDGES = ("llama70b", "gemma27b", "mixtral8x7b")
DEFINITIONS = {"primary": "Success = full answer", "sensitivity": "Success = partial or full answer"}
HUMAN, TEAM = "#52514e", "#6250d6"
INK, SECONDARY, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RUN_INFO = {"jb_5956355": ("The in-loop judge's extra yeses grow over the run",
                           "jb_5956355: target Llama-3.1-8B, in-loop judge Llama-3.3-70B"),
            "jb_strat_base1": ("Old baseline with scoring bugs: in-loop {gap:+.0f} pts over HarmBench",
                               "jb_strat_base1 (older NSGA-II baseline): target Mistral-7B, "
                               "in-loop judge Llama-3.1-8B")}
FIGSIZE, DPI, THRESHOLD, WRAP = (12, 6.75), 200, 0.5, 110
NAMING = ("HarmBench (official) and (model card) are one classifier with two prompt wordings. The in-loop judge is "
          "Llama-3.3-70B with the loop's prompt; '+ guide' = an LLM labeling with the human labeling guide.")
UNSHOWN = "Not shown: Claude's partial labels (40 non-random items), in-loop gate (same as the in-loop judge)."


# ---- data (pure; no text leaves these functions) -------------------------------------------------------------

def load_items(key_path, labels_path, population):
    """Labeled key items with the population-rebased weights analyze_labels uses."""
    key = analyze_labels.load_key([key_path])
    weights, _ = analyze_labels.population_weights(key, population)
    labels, _ = analyze_labels.load_labels(labels_path, key)
    return [dict(item, weight=weights[item_id], label=labels[item_id])
            for item_id, item in key.items() if labels.get(item_id)]


def read_reporting(path):
    """reporting.csv -> item_id -> (inloop_yes, harmbench_score), each 0/1 or None."""
    rows = analyze_labels.read_csv(path, ("item_id", "inloop_yes", "harmbench_score"))
    return {row["item_id"]: (analyze_labels.parse_verdict(row["inloop_yes"]),
                             analyze_labels.parse_verdict(row["harmbench_score"])) for row in rows}


def read_candidates(path):
    """candidates.jsonl -> one dict per row with ids, generation and in-loop verdict; texts are hashed, then dropped."""
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            response, verdict = row["target_response"] or "", row["in_loop"]
            rows.append({"generation": row["generation"], "candidate_id": row["candidate_id"],
                         "item_id": reporting_judge.item_id(row["run_id"], row["behavior"], response),
                         "empty": not response.strip(), "parse_ok": bool(verdict.get("parse_ok")),
                         "score": verdict.get("score")})
    return rows


def generation_rates(rows, harmbench):
    """In-loop (score >= 0.5) and HarmBench official yes rates per generation on the same candidate-behavior rows.
    Each candidate counts in the generation it was first evaluated in; rows with an empty response, an unparsed
    in-loop reply or no usable HarmBench verdict are skipped, counted by reason and per generation ("excluded").
    skipped["excluded_scored_0"] counts the skipped first-generation rows the run itself scored 0 (a "no")."""
    first, cells, skipped = {}, {}, Counter()
    for row in rows:
        first[row["candidate_id"]] = min(row["generation"], first.get(row["candidate_id"], row["generation"]))
    for row in rows:
        if row["generation"] != first[row["candidate_id"]]:
            skipped["re_evaluated"] += 1
            continue
        verdict = harmbench.get(row["item_id"])
        reason = ("empty_response" if row["empty"] else "inloop_unparsed" if not row["parse_ok"] else
                  "harmbench_missing" if verdict is None else None)
        cell = cells.setdefault(row["generation"], [0, 0, 0, 0])
        if reason:
            skipped[reason] += 1
            skipped["excluded_scored_0"] += row["score"] is not None and float(row["score"]) == 0
            cell[3] += 1
            continue
        cell[0] += 1
        cell[1] += float(row["score"]) >= THRESHOLD
        cell[2] += verdict
    return [{"generation": generation, "n": n, "inloop_yes": inloop, "harmbench_yes": harmbench_yes,
             "inloop_rate": inloop / n, "harmbench_rate": harmbench_yes / n, "excluded": excluded}
            for generation, (n, inloop, harmbench_yes, excluded) in sorted(cells.items()) if n], dict(skipped)


def pooled(rates):
    """generation_rates rows summed over generations (every row counts once, not a mean of generation rates)."""
    n = sum(row["n"] for row in rates)
    inloop, harmbench = sum(row["inloop_yes"] for row in rates), sum(row["harmbench_yes"] for row in rates)
    return {"n": n, "inloop_yes": inloop, "harmbench_yes": harmbench, "inloop_rate": inloop / n,
            "harmbench_rate": harmbench / n, "gap": (inloop - harmbench) / n}


def weighted_rate(items, success):
    return analyze_labels.ratio(sum(item["weight"] for item in items if item["label"] in success),
                                sum(item["weight"] for item in items))


def bootstrap_rate(items, success, reps=2000, seed=0):
    """Weighted human success rate with a stratified-bootstrap 95% CI (resampling as analyze_labels.add_intervals)."""
    rng, strata = random.Random(seed), {}
    for item in items:
        strata.setdefault((item["run_id"], item["stratum"]), []).append(item)
    draws = [weighted_rate([pick for key in sorted(strata) for pick in rng.choices(strata[key], k=len(strata[key]))],
                           success) for _ in range(reps)]
    return {"estimate": weighted_rate(items, success), "ci95": analyze_labels.percentile_interval(draws)}


def roc_points(rows):
    """rows = (score, positive, weight) -> ([(fpr, tpr), ...] from (0, 0) to (1, 1), trapezoid AUC).
    Tied scores move together, so the AUC equals analyze_labels.weighted_auc (ties count 0.5)."""
    positive = sum(w for _, p, w in rows if p)
    negative = sum(w for _, p, w in rows if not p)
    points, tp, fp = [(0.0, 0.0)], 0.0, 0.0
    for _, group in groupby(sorted(rows, key=lambda row: -row[0]), key=lambda row: row[0]):
        for _, p, w in group:
            tp, fp = (tp + w, fp) if p else (tp, fp + w)
        points.append((fp / negative, tp / positive))
    return points, sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(points, points[1:]))


def row_shares(matrix):
    return [[cell / (sum(row) or 1) for cell in row] for row in matrix]


def overused_label(matrix, labels):
    """The label a judge gives most often to items the human labeled otherwise (largest off-diagonal column)."""
    col = max(range(len(labels)), key=lambda c: sum(matrix[r][c] for r in range(len(labels)) if r != c))
    others = [r for r in range(len(labels)) if r != col]
    return {"label": labels[col], "count": sum(matrix[r][col] for r in others),
            "of": sum(sum(matrix[r]) for r in others), "human_labels": [labels[r] for r in others]}


def same(value, expected, what):
    if not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{what}: computed {value} but validation_metrics.json has {expected}")


# ---- drawing ---------------------------------------------------------------------------------------------------

def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 20, "xtick.labelsize": 16,
                         "ytick.labelsize": 16, "legend.fontsize": 16, "text.color": INK, "hatch.linewidth": 2,
                         "axes.labelcolor": SECONDARY, "figure.facecolor": "white", "axes.facecolor": "white"})


def wrap(*paragraphs):
    return "\n".join(textwrap.fill(paragraph, WRAP, break_on_hyphens=False) for paragraph in paragraphs if paragraph)


def figure(title, subtitle=None, footnote=None):
    fig = plt.figure(figsize=FIGSIZE)
    fig.text(0.035, 0.955, title, fontsize=26, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.035, 0.875, subtitle, fontsize=17, color=SECONDARY, va="top", linespacing=1.35)
    if footnote:
        fig.text(0.035, 0.025, footnote, fontsize=16, color=SECONDARY, va="bottom", linespacing=1.35)
    return fig


def tidy(ax, grid=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=1)
        ax.set_axisbelow(True)
    ax.tick_params(colors=SECONDARY, length=0, pad=8)


def save(fig, out_dir, name):
    fig.savefig(out_dir / name, dpi=DPI, facecolor="white")
    plt.close(fig)


def tint(color, t):
    return tuple(1 - (1 - c) * t for c in to_rgb(color))


def ink_on(fill):
    """Black or white text, whichever contrasts more with the fill (WCAG relative luminance)."""
    luminance = sum(w * (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
                    for w, c in zip((0.2126, 0.7152, 0.0722), to_rgb(fill)))
    return "white" if luminance < 0.18 else INK


def pct(value, ci):
    return f"{value:.0%} [{ci[0] * 100:.0f}–{ci[1] * 100:.0f}]"


def two_lines(name):
    return name.replace(" + guide", "\n+ guide").replace(" (", "\n(")


def ranked(metrics):
    judges = metrics["pooled"]["primary"]["judges"]
    return sorted(JUDGES, key=lambda judge: -judges[judge]["metrics"]["balanced_accuracy"])


def plot_forest(metrics, out_dir, note):
    order = ranked(metrics)
    data = {"order": order, "best": {}}
    fig = figure(f"{JUDGES[order[0]][0]} is closest to the human on full answers",
                 "Balanced accuracy = average of catch rate and correct-no rate (0.5 = coin flip), with 95% CIs.\n"
                 "The ranking depends on what counts as success, and most CIs overlap.", wrap(note + ".", NAMING))
    axes = fig.subplots(1, 2, sharey=True)
    fig.subplots_adjust(left=0.235, right=0.975, top=0.67, bottom=0.3, wspace=0.08)
    for ax, (definition, heading) in zip(axes, DEFINITIONS.items()):
        judges, data[definition] = metrics["pooled"][definition]["judges"], {}
        best = max(order, key=lambda judge: judges[judge]["metrics"]["balanced_accuracy"])
        data["best"][definition] = best
        ax.axvline(0.5, color=MUTED, lw=1.5, zorder=1)
        ax.text(0.505, -0.75, "chance", fontsize=15, color=SECONDARY, va="center")
        for y, judge in enumerate(order):
            _, color, marker = JUDGES[judge]
            value = judges[judge]["metrics"]["balanced_accuracy"]
            low, high = judges[judge]["ci95"]["balanced_accuracy"]
            ax.plot([low, high], [y, y], color=color, lw=3, solid_capstyle="round", zorder=2)
            ax.plot(value, y, marker=marker, ms=13, color=color, mec="white", mew=2, zorder=3)
            ax.text(value, y - 0.24, f"{value:.2f}", ha="center", va="bottom", fontsize=15, zorder=4,
                    color=INK if judge == best else SECONDARY, fontweight="bold" if judge == best else "normal",
                    bbox={"boxstyle": "square,pad=0.05", "fc": "white", "ec": "none"})
            data[definition][judge] = {"balanced_accuracy": value, "ci95": [low, high]}
        ax.set_title(heading, loc="left", pad=10)
        ax.set_xlim(0.4, 1.0)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_ylim(len(order) - 0.5, -1.0)
        ax.set_xlabel("Balanced accuracy")
        tidy(ax, "x")
    axes[0].set_yticks(range(len(order)), [JUDGES[judge][0] for judge in order], fontsize=17, color=INK)
    save(fig, out_dir, "accuracy_forest.png")
    return data


def plot_error_tradeoff(metrics, out_dir, note):
    order, data, split = ranked(metrics), {}, {}
    for definition in DEFINITIONS:
        counts = metrics["pooled"][definition]["judges"][order[0]]["counts"]
        split[definition] = {"success": counts["tp"] + counts["fn"], "not_success": counts["fp"] + counts["tn"]}
    fig = figure("Two ways to be wrong",
                 "Each judge's false alarms (x) and misses (y) against the human label, with 95% CIs.\n"
                 "Left is the main definition; closer to the bottom-left corner is better.",
                 wrap(note + ".", f"Sample split: {split['primary']['success']} full vs "
                      f"{split['primary']['not_success']} not full (left), {split['sensitivity']['success']} partial "
                      f"or full vs {split['sensitivity']['not_success']} refusal (right); the small groups make the "
                      "CIs wide."))
    axes = [fig.add_axes([0.1, 0.25, 0.32, 0.47]), fig.add_axes([0.47, 0.25, 0.32, 0.47])]
    headings = {"primary": "Success = full answer", "sensitivity": "Success = partial or full"}
    for ax, definition in zip(axes, DEFINITIONS):
        judges, data[definition] = metrics["pooled"][definition]["judges"], {"sample": split[definition]}
        for judge in order:
            _, color, marker = JUDGES[judge]
            rates, ci = judges[judge]["metrics"], judges[judge]["ci95"]
            x, y = rates["fpr"], rates["fnr"]
            ax.errorbar(x, y, xerr=[[x - ci["fpr"][0]], [ci["fpr"][1] - x]], yerr=[[y - ci["fnr"][0]],
                        [ci["fnr"][1] - y]], fmt="none", ecolor=color, elinewidth=2.5, alpha=0.85, zorder=2,
                        clip_on=False)
            ax.plot(x, y, marker=marker, ms=14, color=color, mec="white", mew=2, zorder=4, clip_on=False)
            data[definition][judge] = {"fpr": x, "fnr": y, "fpr_ci95": ci["fpr"], "fnr_ci95": ci["fnr"]}
        ax.set_title(headings[definition], loc="left", pad=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("False positive rate (false alarms)")
        tidy(ax, "both")
    axes[0].set_ylabel("False negative rate (misses)")
    axes[1].tick_params(labelleft=False)
    axes[0].text(0.98, 0.17, "Says yes too often\n(false alarms)", ha="right", va="bottom", fontsize=15,
                 color=SECONDARY)
    axes[0].text(0.30, 0.97, "Too strict\n(misses real jailbreaks)", ha="left", va="top", fontsize=15,
                 color=SECONDARY)
    axes[0].annotate("Better", (0.012, 0.012), xytext=(0.09, 0.16), fontsize=15, color=SECONDARY, ha="center",
                     arrowprops={"arrowstyle": "->", "color": SECONDARY, "lw": 1.5})
    handles = [Line2D([], [], marker=JUDGES[judge][2], ms=13, color=JUDGES[judge][1], mec="white", mew=2, lw=0,
                      label=two_lines(JUDGES[judge][0])) for judge in order]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.805, 0.75), frameon=False, labelspacing=0.75,
               fontsize=16, handletextpad=0.5)
    save(fig, out_dir, "fpr_fnr.png")
    return data


def draw_matrix(ax, texts, shades, color, rows, columns, captions=None, outline=()):
    """texts[r][c] is the big label, shades[r][c] in [0, 1] its fill on one scale shared by every panel."""
    for r, row in enumerate(texts):
        for c, text in enumerate(row):
            fill = tint(color, 0.06 + 0.8 * shades[r][c])
            ax.add_patch(Rectangle((c + 0.02, r + 0.02), 0.96, 0.96, fc=fill, ec="none"))
            ax.text(c + 0.5, r + (0.36 if captions else 0.5), text, ha="center", va="center", fontsize=26,
                    fontweight="bold", color=ink_on(fill))
            if captions:
                ax.text(c + 0.5, r + 0.73, captions[r][c], ha="center", va="center", fontsize=15, color=ink_on(fill),
                        linespacing=1.2)
    for r, c in outline:
        ax.add_patch(Rectangle((c + 0.03, r + 0.03), 0.94, 0.94, fc="none", ec=INK, lw=2.5, zorder=3))
    ax.set_xlim(0, len(columns))
    ax.set_ylim(len(rows), 0)
    ax.set_aspect("equal")
    ax.xaxis.tick_top()
    ax.set_xticks([c + 0.5 for c in range(len(columns))], columns)
    ax.set_yticks([r + 0.5 for r in range(len(rows))], rows)
    ax.tick_params(length=0, colors=SECONDARY, labelsize=16)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_confusion_2x2(metrics, out_dir, eligible):
    judges, data = metrics["pooled"]["primary"]["judges"], {}
    fig = figure("What the judges get right and wrong", "Human label vs judge verdict (success = full answer)",
                 wrap(f"Big numbers: % of each human row, weighted to all {eligible:,} responses in the run, so "
                      "bottom-left = FPR and top-right = FNR. Brackets: 95% CI.",
                      f"n = how many of the {metrics['labels']['labeled']} hand-labeled responses fall in the cell; "
                      "the sample over-represents judge disagreements, so n and % differ."))
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.67, bottom=0.31, wspace=0.42)
    captions = [["caught", "missed"], ["false alarm", "correct no"]]
    for i, (ax, judge) in enumerate(zip(axes, ("inloop_yes", "harmbench_score", "llama70b"))):
        name, color, _ = JUDGES[judge]
        counts, weighted = judges[judge]["counts"], judges[judge]["weighted"]
        rates, ci = judges[judge]["metrics"], judges[judge]["ci95"]
        matrix = [[counts["tp"], counts["fn"]], [counts["fp"], counts["tn"]]]
        shares = row_shares([[weighted["tp"], weighted["fn"]], [weighted["fp"], weighted["tn"]]])
        same(shares[1][0], rates["fpr"], f"{judge} weighted FPR")
        same(shares[0][1], rates["fnr"], f"{judge} weighted FNR")
        draw_matrix(ax, [[f"{share:.0%}" for share in row] for row in shares], shares, color,
                    ["Human:\nfull", "Human:\nnot full"] if i == 0 else ["", ""], ["Judge\nyes", "Judge\nno"],
                    [[f"{caption}\nn={n}" for caption, n in zip(caption_row, count_row)]
                     for caption_row, count_row in zip(captions, matrix)])
        ax.set_title(name, fontsize=20, fontweight="bold", pad=14)
        ax.text(1, 2.12, f"FPR {pct(rates['fpr'], ci['fpr'])}\nFNR {pct(rates['fnr'], ci['fnr'])}", ha="center",
                va="top", fontsize=18, linespacing=1.3)
        data[judge] = {"counts_rows_human_full_notfull_cols_yes_no": matrix, "weighted_row_share": shares,
                       "fpr_weighted": rates["fpr"], "fpr_ci95": ci["fpr"], "fnr_weighted": rates["fnr"],
                       "fnr_ci95": ci["fnr"]}
    save(fig, out_dir, "confusion_2x2.png")
    return data


def plot_confusion_3x3(metrics, out_dir):
    data = {}
    fig = figure("Open LLMs labeling with the human guide", "Human label (rows) vs the LLM's label (columns), 3 classes",
                 wrap(f"Counts of the {metrics['labels']['labeled']} hand-labeled responses, unweighted: the sample "
                      "over-represents judge disagreements, so these are sample figures, not run-wide rates.",
                      "Outlined diagonal = same label as the human. Shading = share of the human row. "
                      "κ = Cohen's kappa."))
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(left=0.125, right=0.96, top=0.63, bottom=0.31, wspace=0.3)
    for i, (ax, judge) in enumerate(zip(axes, GUIDE_JUDGES)):
        stats = metrics["extra_judges"][judge]
        names, matrix = stats["confusion_3class"]["labels"], stats["confusion_3class"]["matrix"]
        agreement, kappa = stats["3class"]["agreement"], stats["3class"]["kappa"]
        over = overused_label(matrix, names)
        draw_matrix(ax, [[str(n) for n in row] for row in matrix], row_shares(matrix), JUDGES[judge][1],
                    names if i == 0 else [""] * len(names), names, outline=[(k, k) for k in range(len(names))])
        ax.set_title(JUDGES[judge][0], fontsize=20, fontweight="bold", pad=12)
        ax.set_xlabel("LLM's label", labelpad=8, fontsize=16)
        ax.xaxis.set_label_position("top")
        ax.text(1.5, 3.12, f"sample agreement {agreement:.0%} · κ {kappa:.2f}\nLabels {over['count']} of the "
                f"{over['of']} {'/'.join(over['human_labels'])}\nresponses '{over['label']}'", ha="center", va="top",
                fontsize=16, linespacing=1.3)
        data[judge] = {"labels": names, "matrix_rows_human_cols_judge": matrix, "row_share": row_shares(matrix),
                       "sample_agreement": agreement, "kappa": kappa, "most_overused_label": over}
    axes[0].set_ylabel("Human label", labelpad=10)
    save(fig, out_dir, "confusion_3x3.png")
    return data


def plot_rates(panels, out_dir, note):
    fig = figure("How often did the jailbreaks really work?",
                 "Share judged a jailbreak. Left: each distinct response counted once (identical responses once).\n"
                 "Right: each candidate × behavior pair (one candidate prompt on one behavior) counted once.", note)
    axes = fig.subplots(1, len(panels), sharey=True, squeeze=False,
                        gridspec_kw={"width_ratios": [len(panel["bars"]) for panel in panels]})[0]
    fig.subplots_adjust(left=0.08, right=0.98, top=0.69, bottom=0.35, wspace=0.12)
    for ax, panel in zip(axes, panels):
        bars = panel["bars"]
        for x, bar in enumerate(bars):
            if bar.get("hatch"):
                ax.bar(x, bar["rate"], width=0.6, fc="white", ec=bar["color"], hatch=bar["hatch"], lw=2, zorder=2)
            else:
                ax.bar(x, bar["rate"], width=0.6, color=bar["color"], zorder=2)
            top, lift = bar["rate"], 4
            if bar.get("ci95"):
                low, top = bar["ci95"]
                ax.errorbar(x, bar["rate"], yerr=[[bar["rate"] - low], [top - bar["rate"]]], color=INK, lw=2,
                            capsize=8, capthick=2, zorder=3)
                ax.annotate(f"CI {low:.0%}–{top:.0%}", (x, top), xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=15, color=SECONDARY)
                lift = 24
            ax.annotate(f"{bar['rate']:.0%}", (x, top), xytext=(0, lift), textcoords="offset points", ha="center",
                        va="bottom", fontsize=20, fontweight="bold")
        ax.set_xticks(range(len(bars)), [bar["label"] for bar in bars], fontsize=16)
        ax.set_xlim(-0.6, len(bars) - 0.4)
        ax.set_title(panel["heading"], loc="left", pad=10)
        if panel.get("note"):
            ax.text(0.02, 0.97, panel["note"], transform=ax.transAxes, ha="left", va="top", fontsize=15,
                    color=SECONDARY)
        tidy(ax, "y")
    axes[0].set_ylim(0, 1.0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    save(fig, out_dir, "jailbreak_rate.png")
    return panels


def plot_generation_gap(run_id, rates, skipped, out_dir):
    total = pooled(rates)
    gaps = [row["inloop_rate"] - row["harmbench_rate"] for row in rates]
    title, subtitle = RUN_INFO.get(run_id, (f"In-loop judge vs HarmBench by generation ({run_id})", None))
    title = title.format(gap=total["gap"] * 100)
    gens = [row["generation"] for row in rates]
    excluded = [f"{skipped[reason]:,} {text}" for reason, text in (
        ("empty_response", "empty responses"), ("inloop_unparsed", "unparsed in-loop replies"),
        ("harmbench_missing", "rows without a HarmBench verdict")) if skipped.get(reason)]
    shares = [row["excluded"] / (row["n"] + row["excluded"]) for row in rates]
    low, high = shares.index(min(shares)), shares.index(max(shares))
    n_excluded = sum(row["excluded"] for row in rates)
    zero = skipped.get("excluded_scored_0", 0)
    exclusion = (f"Excluded: {' and '.join(excluded)}, from {shares[low]:.0%} (gen {gens[low]}) to {shares[high]:.0%} "
                 f"(gen {gens[high]}) of a generation's rows. The run scored "
                 f"{'all' if zero == n_excluded else f'{zero:,} of the'} {n_excluded:,} excluded rows 0 (no), so the "
                 "loop's own yes rate was lower than the in-loop line." if excluded else "No rows excluded.")
    footnote = wrap(f"Over all {total['n']:,} rows: in-loop {total['inloop_rate']:.0%} vs HarmBench "
                    f"{total['harmbench_rate']:.0%} ({total['gap'] * 100:+.0f} pts, rows pooled); gap "
                    f"{gaps[0] * 100:+.0f} pts in gen {gens[0]}, {gaps[-1] * 100:+.0f} pts in gen {gens[-1]}. "
                    "Rows = candidate × behavior pairs, each candidate in its first generation; both lines use the "
                    "same rows.", exclusion)
    fig = figure(title, subtitle, footnote)
    bottom = 0.47 if excluded else 0.32
    ax = fig.add_axes([0.085, bottom, 0.76, 0.77 - bottom])
    blue, orange = JUDGES["inloop_yes"][1], JUDGES["harmbench_score"][1]
    inloop, harmbench = [row["inloop_rate"] for row in rates], [row["harmbench_rate"] for row in rates]
    ax.fill_between(gens, harmbench, inloop, color=blue, alpha=0.25, lw=0, zorder=1)
    ax.plot(gens, inloop, color=blue, lw=3, marker="o", ms=9, mec="white", mew=2, zorder=3,
            label="In-loop judge says yes")
    ax.plot(gens, harmbench, color=orange, lw=3, marker="s", ms=9, mec="white", mew=2, zorder=3,
            label="HarmBench (official) says yes")
    for values in (inloop, harmbench):
        ax.annotate(f"{values[-1]:.0%}", (gens[-1], values[-1]), xytext=(12, 0), textcoords="offset points",
                    va="center", fontsize=16)
    ax.plot([gens[-1]] * 2, [harmbench[-1], inloop[-1]], transform=offset_copy(ax.transData, fig, x=58, units="points"),
            color=SECONDARY, lw=2, marker="_", ms=12, mew=2, clip_on=False)
    ax.text(gens[-1], (inloop[-1] + harmbench[-1]) / 2, f"{gaps[-1] * 100:+.0f} pts", va="center", fontsize=16,
            fontweight="bold", transform=offset_copy(ax.transData, fig, x=66, units="points"))
    below = blended_transform_factory(ax.transData, ax.transAxes)
    size = 16 if len(gens) <= 12 else 14
    counts = [("rows", "n")] + ([("dropped", "excluded")] if excluded else [])
    for line, (label, field) in enumerate(counts):
        y = -(0.42 + 0.3 * line) / ((0.77 - bottom) * FIGSIZE[1])  # 0.42 in below the axis, 0.3 in per line
        ax.text(gens[0] - 0.6, y, label, transform=below, ha="right", va="top", fontsize=size, color=SECONDARY)
        for row in rates:
            ax.text(row["generation"], y, f"{row[field]:,}", transform=below, ha="center", va="top", fontsize=size,
                    color=SECONDARY)
    ax.set_xticks(gens)
    ax.set_xlim(gens[0] - 0.4, gens[-1] + 0.4)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Generation", labelpad=26 + 22 * (len(counts) - 1))
    ax.set_ylabel("Rows judged a jailbreak")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=16)
    tidy(ax, "y")
    save(fig, out_dir, f"generation_gap_{run_id}.png")
    return {"rows": rates, "skipped": skipped, "pooled": total, "gap_by_generation": gaps,
            "excluded_share_by_generation": shares}


def plot_roc(items, metrics, out_dir, note):
    judges = metrics["pooled"]["primary"]["judges"]
    fig = figure("How well do the raw scores rank real jailbreaks?",
                 "Weighted ROC against the human label (success = full answer)",
                 wrap(note + ".", "Brackets: 95% stratified-bootstrap CI. Open LLM judges output labels, not "
                      "scores, so they have no ROC curve."))
    ax = fig.add_axes([0.085, 0.24, 0.47, 0.57])
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1.5, label="Chance (AUC 0.50)")
    data, cutoffs = {}, []
    for score, judge in (("inloop_score", "inloop_yes"), ("harmbench_p_yes", "harmbench_score")):
        name, color, marker = JUDGES[judge]
        rows = [(item[score], item["label"] == "full", item["weight"]) for item in items if item[score] is not None]
        points, auc = roc_points(rows)
        same(auc, metrics["pooled"]["primary"]["auc"][score]["auc"], f"{score} AUC")
        low, high = metrics["pooled"]["primary"]["auc"][score]["ci95"]["auc"]
        ax.plot(*zip(*points), color=color, lw=3, solid_joinstyle="round", zorder=3,
                label=f"{name}\nAUC {auc:.2f} [95% CI {low:.2f}–{high:.2f}]")
        cut = judges[judge]["metrics"]
        ax.plot(cut["fpr"], cut["recall"], marker=marker, ms=14, color=color, mec="white", mew=2, zorder=4)
        cutoffs.append(Line2D([], [], marker=marker, ms=12, color=color, mec="white", mew=2, lw=0))
        _, above = roc_points([row for row in rows if row[0] >= THRESHOLD])
        data[score] = {"auc": auc, "auc_ci95": [low, high], "points_fpr_tpr": points,
                       "yes_no_cutoff_fpr_tpr": [cut["fpr"], cut["recall"]], "auc_above_cutoff": above}
    ax.annotate("Above the cutoff, a higher in-loop\nscore is no more likely to be\na real jailbreak",
                (0.25, 0.25), xytext=(0.97, 0.04), ha="right", va="bottom", fontsize=15, color=SECONDARY,
                linespacing=1.3, arrowprops={"arrowstyle": "->", "color": SECONDARY, "lw": 1.5,
                                             "shrinkA": 4, "shrinkB": 6})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("False positive rate (false alarms)")
    ax.set_ylabel("True positive rate (jailbreaks caught)")
    tidy(ax, "both")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [tuple(cutoffs)], labels + ["Yes/no cutoff used\nin the other charts"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)}, loc="upper left",
              bbox_to_anchor=(1.05, 1.02), frameon=False, labelspacing=0.6, fontsize=16)
    inloop = judges["inloop_yes"]["metrics"]["balanced_accuracy"]
    fig.text(0.6, 0.2, "\n".join(textwrap.wrap(
        "AUC = chance that a real jailbreak gets a higher score than a non-jailbreak (1.0 = perfect, 0.5 = random). "
        f"The in-loop AUC ({data['inloop_score']['auc']:.2f}) is below its balanced accuracy at the cutoff "
        f"({inloop:.2f}) because only its yes/no split carries signal.", 44)), fontsize=15, color=SECONDARY,
        va="bottom", linespacing=1.35)
    data["inloop_balanced_accuracy_at_cutoff"] = inloop
    save(fig, out_dir, "roc.png")
    return data


def table_rows(metrics):
    rows = []
    for definition in DEFINITIONS:
        judges = metrics["pooled"][definition]["judges"]
        for judge in ranked(metrics):
            result = judges[judge]
            rows.append({"definition": definition, "judge": judge, "display_name": JUDGES[judge][0], "n": result["n"],
                         **{m: result["metrics"][m] for m in ("precision", "recall", "fpr", "fnr", "specificity",
                                                              "f1", "balanced_accuracy")},
                         **{f"{m}_ci_{end}": result["ci95"][m][i] for m in ("fpr", "fnr", "balanced_accuracy")
                            for i, end in enumerate(("low", "high"))},
                         "w_yes": result["fusion"]["w_yes"], "w_no": result["fusion"]["w_no"]})
    return rows


def write_tables(rows, out_dir, note):
    with open(out_dir / "metrics_table.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Judge metrics against human labels", "", f"{note}. 95% stratified-bootstrap CIs in brackets. "
             "w_yes / w_no are the Phase 2 fusion weights (log-likelihood ratios).", ""]
    for definition, heading in DEFINITIONS.items():
        lines += [f"## {heading}", "", "| Judge | n | Precision | Recall | FPR [95% CI] | FNR [95% CI] | Specificity "
                  "| F1 | Balanced acc. [95% CI] | w_yes | w_no |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {row['display_name']} | {row['n']} | {row['precision']:.2f} | {row['recall']:.2f} | "
                  + " | ".join(f"{row[m]:.2f} [{row[m + '_ci_low']:.2f}, {row[m + '_ci_high']:.2f}]"
                               for m in ("fpr", "fnr")) + f" | {row['specificity']:.2f} | {row['f1']:.2f} | "
                  f"{row['balanced_accuracy']:.2f} [{row['balanced_accuracy_ci_low']:.2f}, "
                  f"{row['balanced_accuracy_ci_high']:.2f}] | {row['w_yes']:.2f} | {row['w_no']:.2f} |"
                  for row in rows if row["definition"] == definition]
        lines.append("")
    lines += ["Footnotes:", "", "- claude_partial (Claude's labels) is excluded from this table and every figure: it "
              "labeled only 40 items (n=40), chosen non-randomly, so its numbers are not comparable (see "
              "validation_metrics.json).", "- inloop_gate gives the same verdicts as the in-loop judge on this run "
              "and is omitted.", f"- {NAMING}", ""]
    (out_dir / "metrics_table.md").write_text("\n".join(lines), encoding="utf-8")


def plot_table(rows, out_dir, note):
    fig = figure("Judge scorecard", "Success = full answer, sorted by balanced accuracy. Rates in %; F1 and balanced "
                 "accuracy on 0–1.",
                 wrap(f"{note}.", "Brackets: 95% CI. Recall (not shown) = 100% − FNR. For success = partial or "
                      "full answer, see the right panels of the accuracy and error-rate charts.", UNSHOWN))
    ax = fig.add_axes([0.035, 0.25, 0.93, 0.56])
    ax.axis("off")
    columns = [("Precision", 0.4, lambda row: f"{row['precision']:.0%}"),
               ("FPR\n[95% CI]", 0.555, lambda row: pct(row["fpr"], (row["fpr_ci_low"], row["fpr_ci_high"]))),
               ("FNR\n[95% CI]", 0.715, lambda row: pct(row["fnr"], (row["fnr_ci_low"], row["fnr_ci_high"]))),
               ("F1", 0.785, lambda row: f"{row['f1']:.2f}")]
    shown = [row for row in rows if row["definition"] == "primary"]
    step = 1 / (len(shown) + 1)
    y = 1 - step / 2
    ax.text(0.03, y, "Judge", fontsize=17, fontweight="bold", va="center")
    for header, x, _ in columns:
        ax.text(x, y, header, fontsize=17, fontweight="bold", va="center", ha="right")
    ax.text(1.0, y, "Balanced acc.\n[95% CI]", fontsize=17, fontweight="bold", va="center", ha="right")
    ax.axhline(y - step / 2, color=AXIS, lw=1.5)
    for row in shown:
        y -= step
        _, color, marker = JUDGES[row["judge"]]
        ax.plot(0.008, y, marker=marker, ms=13, color=color, mec="white", mew=2, transform=ax.transAxes)
        ax.text(0.03, y, row["display_name"], fontsize=17, va="center")
        for _, x, text in columns:
            ax.text(x, y, text(row), fontsize=17, va="center", ha="right")
        ax.text(1.0, y, f"{row['balanced_accuracy']:.2f} [{row['balanced_accuracy_ci_low']:.2f}–"
                f"{row['balanced_accuracy_ci_high']:.2f}]", fontsize=17, va="center", ha="right", fontweight="bold")
        ax.axhline(y - step / 2, color=GRID, lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, out_dir, "metrics_table.png")
    return shown


def write_readme(out_dir, runs):
    lines = ["# Judge-validation figures", "",
             "Built by `sota/Jailbreak/plot_validation.py`; every plotted number is in `figure_data.json`. Rates "
             "against the human label are weighted to all responses in the run unless a figure says otherwise.", "",
             "- `accuracy_forest.png`: how often each judge agrees with the human label (balanced accuracy = average "
             "of catch rate and correct-no rate; 0.5 = coin flip), with 95% CIs; left counts only full answers as a "
             "jailbreak, right also counts partial ones. The ranking changes between the two, and most CIs overlap.",
             "- `fpr_fnr.png`: the two kinds of mistakes per judge, false alarms (x) vs missed jailbreaks (y), with "
             "95% CIs, for both success definitions (left = full answer, the main one); closer to the bottom-left "
             "corner is better.",
             "- `confusion_2x2.png`: right and wrong calls of the in-loop judge, HarmBench (official) and "
             "Llama-3.3-70B + guide. Big numbers are run-weighted % of each human row (bottom-left = FPR, top-right "
             "= FNR); n is the raw count among the 100 labeled responses.",
             "- `confusion_3x3.png`: how the three open LLMs given the human labeling guide label refusal / partial / "
             "full compared with the human (raw sample counts, unweighted), with each one's most over-used label.",
             "- `jailbreak_rate.png`: the jailbreak success rate each source reports. Left: the human-label estimate "
             "(full answer, with CI; hatched = partial or full) vs HarmBench vs the in-loop judge over the run's "
             "distinct responses. Right: in-loop vs team verification vs HarmBench over the final population's 840 "
             "candidate x behavior pairs (no human labels there). The two panels count different things; the "
             "footnote gives the per-attempt rates.",
             "- `roc.png`: how well the raw in-loop score and HarmBench's P(yes) rank real jailbreaks above "
             "non-jailbreaks (AUC with 95% CI; 1.0 perfect, 0.5 random). Above its cutoff the in-loop score does not "
             "rank better than chance."]
    for run_id in runs:
        lines.append(f"- `generation_gap_{run_id}.png`: per generation, the share of rows the in-loop judge calls a "
                     "jailbreak vs the share HarmBench does on the same rows; the shaded band is how much more often "
                     "the in-loop judge says yes."
                     + (" Old baseline with scoring bugs (different target and in-loop judge); backup slide only."
                        if run_id == "jb_strat_base1" else ""))
    lines += ["- `metrics_table.png` / `metrics_table.md` / `metrics_table.csv`: the full per-judge numbers "
              "(PNG = success = full answer only; MD/CSV = both definitions plus specificity, recall and fusion "
              "weights).",
              "- `figure_data.json`: every number shown in every figure, keyed by figure file name.", ""]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", required=True, help="analyze_labels validation_metrics.json")
    parser.add_argument("--key", required=True)
    parser.add_argument("--labels", required=True, help="one labeler's export (item_id,label)")
    parser.add_argument("--population", nargs="+", required=True, help="reporting CSV(s) the weights are rebased on")
    parser.add_argument("--heldout", help="final-population counts JSON (in_loop_yes, team_verified, ...)")
    parser.add_argument("--run", nargs=3, action="append", default=[],
                        metavar=("RUN_ID", "CANDIDATES_JSONL", "REPORTING_CSV"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    items = load_items(args.key, args.labels, args.population)
    same(weighted_rate(items, {"full"}), metrics["pooled"]["primary"]["human_success_rate"], "human success rate")
    verdicts = {}
    for path in args.population:
        verdicts.update(read_reporting(path))
    run = {"n": len(verdicts), "inloop_yes": sum(v[0] == 1 for v in verdicts.values()),
           "harmbench_yes": sum(v[1] == 1 for v in verdicts.values())}
    eligible, labeled, labelers = metrics["population"]["eligible"], metrics["labels"]["labeled"], len(
        metrics["inputs"]["labels"])
    note = (f"{labeled} hand-labeled responses ({labelers} labeler{'s' * (labelers > 1)}), weighted to all "
            f"{eligible:,} responses in the run")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    style()

    data = {"accuracy_forest.png": plot_forest(metrics, out_dir, note),
            "fpr_fnr.png": plot_error_tradeoff(metrics, out_dir, note),
            "confusion_2x2.png": plot_confusion_2x2(metrics, out_dir, eligible),
            "confusion_3x3.png": plot_confusion_3x3(metrics, out_dir)}
    gaps = {}
    for run_id, candidates, reporting in args.run:
        harmbench = {item_id: verdict[1] for item_id, verdict in read_reporting(reporting).items()}
        gaps[run_id] = generation_rates(read_candidates(candidates), harmbench)
        data[f"generation_gap_{run_id}.png"] = plot_generation_gap(run_id, *gaps[run_id], out_dir)
    human = {name: bootstrap_rate(items, success, args.bootstrap, args.seed)
             for name, success in analyze_labels.DEFINITIONS.items()}
    left = [{"label": "Human:\nfull answer", "rate": human["primary"]["estimate"], "ci95": human["primary"]["ci95"],
             "color": HUMAN},
            {"label": "Human: partial\nor full", "rate": human["sensitivity"]["estimate"],
             "ci95": human["sensitivity"]["ci95"], "color": HUMAN, "hatch": "//"},
            {"label": "HarmBench\n(official)", "rate": run["harmbench_yes"] / run["n"], "count": run["harmbench_yes"],
             "of": run["n"], "color": JUDGES["harmbench_score"][1]},
            {"label": "In-loop\njudge", "rate": run["inloop_yes"] / run["n"], "count": run["inloop_yes"],
             "of": run["n"], "color": JUDGES["inloop_yes"][1]}]
    panels = [{"heading": f"{', '.join(metrics['inputs']['runs'])}: {run['n']:,} distinct responses", "bars": left}]
    notes = [f"Human bars: {labeled} hand-labeled responses weighted to the run, 95% stratified-bootstrap CI "
             f"({args.bootstrap:,} reps); solid = main definition (full answer), hatched = partial or full."]
    attempts = [pooled(gaps[run_id][0]) for run_id in metrics["inputs"]["runs"] if run_id in gaps]
    notes.append("Judge bars are exact counts, so no CI." + (
        f" Counted per attempt instead ({attempts[0]['n']:,} candidate × behavior rows): in-loop "
        f"{attempts[0]['inloop_rate']:.0%}, HarmBench {attempts[0]['harmbench_rate']:.0%}."
        if len(attempts) == 1 else ""))
    if args.heldout:
        heldout = json.loads(Path(args.heldout).read_text(encoding="utf-8"))
        right = [{"label": label, "rate": heldout[field][0] / heldout[field][1], "count": heldout[field][0],
                  "of": heldout[field][1], "color": color} for label, field, color in (
                     ("Team\nverification", "team_verified", TEAM),
                     ("HarmBench\n(official)", "harmbench_official_yes", JUDGES["harmbench_score"][1]),
                     ("In-loop\njudge", "in_loop_yes", JUDGES["inloop_yes"][1]))]
        panels.append({"heading": f"Final population: {heldout['in_loop_yes'][1]:,} pairs", "bars": right,
                       "note": "No human labels here"})
        notes.append("Team verification = success in at least 2 of 3 re-runs (before the verification-temperature "
                     "fix).")
    data["jailbreak_rate.png"] = {"panels": plot_rates(panels, out_dir, wrap(*notes)),
                                  "per_attempt": attempts[0] if len(attempts) == 1 else None}
    data["roc.png"] = plot_roc(items, metrics, out_dir, note)
    rows = table_rows(metrics)
    write_tables(rows, out_dir, note)
    data["metrics_table.png"] = plot_table(rows, out_dir, note)
    data["metrics_table.csv"] = rows
    (out_dir / "figure_data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    write_readme(out_dir, [run_id for run_id, _, _ in args.run])
    print(f"human_full={human['primary']['estimate']:.4f} ci={human['primary']['ci95']} "
          f"inloop_yes={run['inloop_yes']}/{run['n']} harmbench_yes={run['harmbench_yes']}/{run['n']} "
          f"runs={len(args.run)} wrote={out_dir}", flush=True)
    return data


if __name__ == "__main__":
    main()
