import importlib
import json
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
