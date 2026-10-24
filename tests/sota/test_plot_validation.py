import importlib
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sota" / "Jailbreak"))
plot_validation = importlib.import_module("plot_validation")
analyze_labels = importlib.import_module("analyze_labels")
reporting_judge = importlib.import_module("reporting_judge")


def row(generation, candidate, item, score=0.0, empty=False, parse_ok=True):
    return {"generation": generation, "candidate_id": candidate, "item_id": item, "empty": empty,
            "parse_ok": parse_ok, "score": score}


def test_generation_rates_use_same_rows_and_skip_by_reason():
    rows = [row(0, "a", "i1", 0.9), row(0, "a", "i2", 0.1), row(0, "b", "i3", 0.5),
            row(1, "c", "i4", 0.7), row(1, "c", "i5", empty=True), row(1, "c", "i6", parse_ok=False, score=None),
            row(1, "c", "i7", 0.8), row(2, "a", "i1", 0.9)]  # candidate a re-evaluated later: counted in gen 0 only
    harmbench = {"i1": 1, "i2": 0, "i3": 0, "i4": 1, "i5": 1, "i6": 1, "i7": None}
    rates, skipped = plot_validation.generation_rates(rows, harmbench)
    assert rates == [
        {"generation": 0, "n": 3, "inloop_yes": 2, "harmbench_yes": 1, "inloop_rate": 2 / 3, "harmbench_rate": 1 / 3,
         "excluded": 0},
        {"generation": 1, "n": 1, "inloop_yes": 1, "harmbench_yes": 1, "inloop_rate": 1.0, "harmbench_rate": 1.0,
         "excluded": 3}]
    assert skipped == {"empty_response": 1, "inloop_unparsed": 1, "harmbench_missing": 1, "re_evaluated": 1,
                       "excluded_scored_0": 1}  # only i5 (scored 0.0): i6 has no score, i7 scored 0.8


def test_pooled_counts_every_row_once_not_a_mean_of_generations():
    rates = [{"n": 10, "inloop_yes": 5, "harmbench_yes": 1}, {"n": 90, "inloop_yes": 9, "harmbench_yes": 9}]
    total = plot_validation.pooled(rates)
    assert (total["n"], total["inloop_yes"], total["harmbench_yes"]) == (100, 14, 10)
    assert total["gap"] == pytest.approx(0.04)  # the mean of the per-generation gaps would be 0.2


def test_row_shares_and_overused_label():
    matrix = [[28, 14, 1], [0, 30, 1], [0, 19, 7]]  # rows = human refusal/partial/full, columns = judge
    assert plot_validation.row_shares(matrix)[2] == pytest.approx([0, 19 / 26, 7 / 26])
    assert plot_validation.row_shares([[0, 0]]) == [[0, 0]]
    assert plot_validation.overused_label(matrix, ["refusal", "partial", "full"]) == {
        "label": "partial", "count": 33, "of": 69, "human_labels": ["refusal", "full"]}


def test_roc_auc_matches_weighted_mann_whitney_with_ties():
    rows = [(0.9, True, 2.0), (0.9, False, 1.0), (0.7, True, 1.0), (0.4, False, 3.0), (0.4, True, 0.5),
            (0.1, False, 1.5), (0.0, False, 1.0), (0.0, True, 1.0)]
    points, auc = plot_validation.roc_points(rows)
    assert points[0] == (0.0, 0.0) and points[-1] == pytest.approx((1.0, 1.0))
    assert auc == pytest.approx(analyze_labels.weighted_auc(rows))
    perfect, auc_perfect = plot_validation.roc_points([(1, True, 1), (0, False, 1)])
    assert auc_perfect == 1.0 and perfect == [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]


def test_bootstrap_rate_is_weighted_and_resamples_within_strata():
    items = [{"run_id": "r", "stratum": "a", "weight": 3.0, "label": "full"},
             {"run_id": "r", "stratum": "a", "weight": 3.0, "label": "full"},
             {"run_id": "r", "stratum": "b", "weight": 1.0, "label": "refusal"},
             {"run_id": "r", "stratum": "b", "weight": 1.0, "label": "partial"}]
    full = plot_validation.bootstrap_rate(items, {"full"}, reps=200, seed=0)
    assert full["estimate"] == pytest.approx(0.75)
    assert full["ci95"] == pytest.approx([0.75, 0.75])  # each stratum's full share is fixed, so every replicate agrees
    lenient = plot_validation.bootstrap_rate(items, {"partial", "full"}, reps=200, seed=0)
    assert lenient["estimate"] == pytest.approx(7 / 8)
    assert lenient["ci95"][0] == pytest.approx(0.75) and lenient["ci95"][1] == pytest.approx(1.0)
    assert plot_validation.bootstrap_rate(items, {"partial", "full"}, reps=200, seed=0) == lenient


def test_read_candidates_hashes_text_into_reporting_item_ids(tmp_path):
    lines = [{"run_id": "run-x", "generation": 3, "candidate_id": "c1", "behavior": "B", "target_response": "R",
              "in_loop": {"score": 0.75, "parse_ok": True}},
             {"run_id": "run-x", "generation": 4, "candidate_id": "c2", "behavior": "B", "target_response": "",
              "in_loop": {"parse_ok": False}}]
    path = tmp_path / "candidates.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    rows = plot_validation.read_candidates(path)
    assert rows == [{"generation": 3, "candidate_id": "c1", "item_id": reporting_judge.item_id("run-x", "B", "R"),
                     "empty": False, "parse_ok": True, "score": 0.75},
                    {"generation": 4, "candidate_id": "c2", "item_id": reporting_judge.item_id("run-x", "B", ""),
                     "empty": True, "parse_ok": False, "score": None}]


def scored_items(n=40, seed=1):
    """Synthetic labeled items in two strata with a key score per judge (verdict = score >= 0.5, as in the key)."""
    rng, items = random.Random(seed), []
    for i in range(n):
        label = rng.choice(["refusal", "partial", "full"])
        inloop, harmbench = rng.random(), min(1.0, rng.random() * 0.6 + 0.5 * (label == "full"))
        items.append({"item_id": f"i{i}", "run_id": "r", "stratum": "ab"[i % 2], "weight": 1.0 + i % 3,
                      "label": label, "inloop_score": inloop, "inloop_yes": int(inloop >= 0.5), "inloop_gate": None,
                      "harmbench_p_yes": harmbench, "harmbench_score": int(harmbench >= 0.5),
                      "harmbench_card_score": None})
    return items


def test_add_score_reads_a_csv_column_or_a_key_score(tmp_path):
    items = [{"item_id": "a", "inloop_score": 0.25}, {"item_id": "b", "inloop_score": 0.75}]
    path = tmp_path / "probs.csv"
    path.write_text("item_id,label,p_full\na,full,0.9\nb,refusal,\nc,full,0.1\n", encoding="utf-8")
    plot_validation.add_score(items, "guide", f"{path}:p_full")
    plot_validation.add_score(items, "logged", "inloop_score")
    assert [(item["guide"], item["logged"]) for item in items] == [(0.9, 0.25), (None, 0.75)]  # blank -> None
    with pytest.raises(ValueError, match="1 labeled item_ids have no row"):
        plot_validation.add_score(items + [{"item_id": "z"}], "guide", f"{path}:p_full")
    with pytest.raises(ValueError, match="missing columns"):
        plot_validation.add_score(items, "guide", f"{path}:p_partial")


def test_add_score_counts_unparsed_replies_scored_0(tmp_path):
    items = [{"item_id": i} for i in "abcd"]
    path = tmp_path / "rubric.csv"
    path.write_text("item_id,score,parse_ok\na,0.0,0\nb,0.0,1\nc,0.5,0\nd,0.0,\n", encoding="utf-8")
    assert plot_validation.add_score(items, "rubric", f"{path}:score") == 1  # c is unparsed but not scored 0
    assert [item["rubric"] for item in items] == [0.0, 0.0, 0.5, 0.0]
    keyed = [dict(item, inloop_score=0.0) for item in items]
    assert plot_validation.add_score(keyed, "logged", "inloop_score") == 0  # key scores carry no parse_ok


@pytest.mark.parametrize("lows, helped, title", [
    ({"A": -0.1, "B": -0.2}, [], "No judge model clearly gains from the labeling guide"),
    ({"A": -0.1, "B": 0.05}, ["B"], "Only B clearly gains from the labeling guide"),
    ({"A": 0.1, "B": 0.05}, ["A", "B"], "The labeling guide helps A and B")])
def test_guide_verdict_counts_a_gain_only_when_its_ci_excludes_0(lows, helped, title):
    changes = {model: {"change": 0.1, "ci95": [low, 0.3]} for model, low in lows.items()}
    got_helped, models, got_title = plot_validation.guide_verdict(changes)
    assert (got_helped, got_title) == (helped, title)
    assert models == helped + [model for model in lows if model not in helped]  # gaining models listed first


def test_score_metrics_weighted_auc_and_cutoff_split():
    items = [{"label": "full", "weight": 2.0, "s": 0.9}, {"label": "full", "weight": 1.0, "s": 0.3},
             {"label": "partial", "weight": 1.0, "s": 0.6}, {"label": "refusal", "weight": 3.0, "s": 0.1},
             {"label": "full", "weight": 5.0, "s": None}]
    result = plot_validation.score_metrics(items, "s", {"full"})
    assert result["n"] == 4
    assert result["auc"] == pytest.approx((2 * 4 + 1 * 3) / (3 * 4))  # 0.9 beats both; 0.3 beats only 0.1
    assert (result["recall"], result["fpr"]) == pytest.approx((2 / 3, 1 / 4))  # yes = score >= 0.5
    assert result["balanced_accuracy"] == pytest.approx((2 / 3 + 3 / 4) / 2)


def test_bootstrap_scores_reproduce_analyze_labels_intervals():
    items = [dict(item, extra=item["harmbench_p_yes"] if i else None) for i, item in enumerate(scored_items())]
    expected = analyze_labels.evaluate(items, ["r"])
    analyze_labels.add_intervals(expected, items, ["r"], 200, 3)
    stats, draws = plot_validation.bootstrap_scores(items, [*analyze_labels.SCORES, "extra"], {"full"}, 200, 3)
    primary = expected["pooled"]["primary"]
    for score, judge in (("inloop_score", "inloop_yes"), ("harmbench_p_yes", "harmbench_score")):
        assert stats[score]["auc"] == pytest.approx(primary["auc"][score]["auc"])
        assert stats[score]["auc_ci95"] == pytest.approx(primary["auc"][score]["ci95"]["auc"])
        judged = primary["judges"][judge]  # the key's verdict is its score's split at 0.5
        assert stats[score]["balanced_accuracy"] == pytest.approx(judged["metrics"]["balanced_accuracy"])
        assert stats[score]["balanced_accuracy_ci95"] == pytest.approx(judged["ci95"]["balanced_accuracy"])
    assert stats["extra"]["n"] == len(items) - 1 and len(draws["extra"]) == 200
    assert plot_validation.bootstrap_scores(items, ["extra"], {"full"}, 200, 3)[0] == {"extra": stats["extra"]}


def test_paired_change_uses_shared_replicates():
    items = [dict(item, same=item["inloop_score"], better=float(item["label"] == "full")) for item in scored_items()]
    stats, draws = plot_validation.bootstrap_scores(items, ["inloop_score", "same", "better"], {"full"}, 200, 0)
    assert plot_validation.paired_change(stats, draws, "inloop_score", "same") == {"change": 0.0, "ci95": [0.0, 0.0]}
    better = plot_validation.paired_change(stats, draws, "inloop_score", "better")
    assert stats["better"]["auc"] == 1.0
    assert better["change"] == pytest.approx(1.0 - stats["inloop_score"]["auc"]) and better["ci95"][0] > 0


@pytest.mark.parametrize("extra", [["--extra-score", "llama8b=probs.csv:p_full"],
                                   ["--extra-score", "llama70b=a.csv:p_full", "--extra-score", "llama70b=b.csv:p_full"],
                                   ["--model-prompt", "M", "rubric", "inloop_score"],
                                   ["--model-prompt", "M", "rubric", "a.csv:s", "--model-prompt", "M", "prompt",
                                    "b.csv:s"]])
def test_main_rejects_bad_score_options(extra):
    with pytest.raises(SystemExit):
        plot_validation.main(["--metrics", "m.json", "--key", "k.csv", "--labels", "l.csv", "--population", "p.csv",
                              "--out-dir", "out", *extra])
