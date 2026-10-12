import csv
import importlib
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sota" / "Jailbreak"))
analyze_labels = importlib.import_module("analyze_labels")

KEY_COLUMNS = ["item_id", "run_id", "stratum", "weight", "N_h", "n_h", "behavior_id", "inloop_score", "inloop_yes",
               "inloop_gate", "harmbench_score", "harmbench_p_yes", "harmbench_card_score", "harmbench_card_p_yes",
               "generation", "candidate_id"]
SHEET_COLUMNS = ["item_id", "behavior_id", "behavior", "response", "label", "confidence", "notes"]


def item(label, weight=1.0, run="run-a", stratum="both_yes", **values):
    row = {"item_id": f"id{id(values)}", "run_id": run, "stratum": stratum, "weight": weight, "label": label}
    row.update({name: None for name in analyze_labels.JUDGES + analyze_labels.SCORES})
    row.update(values)
    return row


def write_csv(path, columns, rows):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def test_cohen_kappa_textbook_example():
    # Classic 2x2 example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> p_o = 0.7, p_e = 0.5, kappa = 0.4.
    pairs = [("yes", "yes")] * 20 + [("yes", "no")] * 5 + [("no", "yes")] * 10 + [("no", "no")] * 15
    assert analyze_labels.agreement(pairs) == pytest.approx(0.7)
    assert analyze_labels.cohen_kappa(pairs, ("yes", "no")) == pytest.approx(0.4)
    assert analyze_labels.cohen_kappa(pairs, ("yes", "no"), [3.0] * len(pairs)) == pytest.approx(0.4)
    assert analyze_labels.cohen_kappa([("a", "a")] * 3, ("a", "b")) is None


def test_weighted_confusion_metrics_hand_computed():
    items = [item("full", 2, inloop_yes=1), item("full", 1, inloop_yes=1), item("full", 1, inloop_yes=0),
             item("refusal", 1, inloop_yes=1), item("refusal", 3, inloop_yes=0), item("partial", 2, inloop_yes=1),
             item("full", 5, inloop_yes=None)]
    primary = analyze_labels.judge_metrics(items, "inloop_yes", {"full"})
    assert primary["weighted"] == {"tp": 3, "fp": 3, "fn": 1, "tn": 3}
    assert primary["counts"] == {"tp": 2, "fp": 2, "fn": 1, "tn": 1}
    assert (primary["n"], primary["excluded"]) == (6, 1)
    expected = {"precision": 0.5, "recall": 0.75, "fpr": 0.5, "fnr": 0.25, "specificity": 0.5, "accuracy": 0.6,
                "balanced_accuracy": 0.625, "f1": 0.6}
    assert primary["metrics"] == pytest.approx(expected)
    sensitivity = analyze_labels.judge_metrics(items, "inloop_yes", {"partial", "full"})
    assert sensitivity["weighted"] == {"tp": 5, "fp": 1, "fn": 1, "tn": 3}
    assert sensitivity["metrics"]["precision"] == pytest.approx(5 / 6)
    assert sensitivity["metrics"]["recall"] == pytest.approx(5 / 6)
    assert sensitivity["metrics"]["fpr"] == pytest.approx(0.25)
    assert sensitivity["metrics"]["accuracy"] == pytest.approx(0.8)


def test_blank_and_minus_one_excluded():
    assert analyze_labels.parse_verdict("") is None
    assert analyze_labels.parse_verdict("-1") is None
    assert analyze_labels.parse_verdict("1") == 1 and analyze_labels.parse_verdict("0.0") == 0
    assert analyze_labels.parse_float("") is None and analyze_labels.parse_float("0.25") == 0.25
    items = [item("full", harmbench_score=1, harmbench_p_yes=0.9), item("refusal", harmbench_score=None),
             item("refusal", harmbench_score=0, harmbench_p_yes=None)]
    judged = analyze_labels.judge_metrics(items, "harmbench_score", {"full"})
    assert (judged["n"], judged["excluded"]) == (2, 1)
    auc = analyze_labels.score_auc(items, "harmbench_p_yes", {"full"})
    assert (auc["n"], auc["excluded"], auc["auc"]) == (1, 2, None)


def test_weighted_auc_with_ties():
    # Pairs (pos, neg): 0.9>0.5 w=1*1, 0.9>0.1 w=1*3, 0.5=0.5 tie 0.5*2*1, 0.5>0.1 w=2*3 -> 11 / (3*4).
    rows = [(0.9, True, 1.0), (0.5, True, 2.0), (0.5, False, 1.0), (0.1, False, 3.0)]
    assert analyze_labels.weighted_auc(rows) == pytest.approx(11 / 12)
    assert analyze_labels.weighted_auc([(0.2, True, 1.0), (0.1, False, 1.0)]) == 1.0
    assert analyze_labels.weighted_auc([(0.4, True, 1.0), (0.4, False, 5.0)]) == 0.5
    assert analyze_labels.weighted_auc([(0.4, True, 1.0)]) is None


def test_llr_fusion_weights_with_smoothing():
    # 3 TP, 1 FN, 0 FP, 4 TN at weight 10 -> rescaled to weight 1, +0.5 per cell: sens 0.7, spec 0.9.
    items = ([item("full", 10, inloop_yes=1)] * 3 + [item("full", 10, inloop_yes=0)]
             + [item("refusal", 10, inloop_yes=0)] * 4)
    block = analyze_labels.analyze(items, {"full"})
    fusion = block["judges"]["inloop_yes"]["fusion"]
    assert fusion["sensitivity"] == pytest.approx(0.7) and fusion["specificity"] == pytest.approx(0.9)
    assert fusion["w_yes"] == pytest.approx(math.log(7))
    assert fusion["w_no"] == pytest.approx(math.log(1 / 3))
    assert block["human_success_rate"] == pytest.approx(0.5) and block["prior_logit"] == pytest.approx(0)
    assert block["judges"]["harmbench_score"]["fusion"]["w_yes"] is None
    skewed = analyze_labels.analyze(items + [item("full", 20, inloop_yes=1)], {"full"})
    assert skewed["prior_logit"] == pytest.approx(math.log(60 / 40))


def test_lower_label_rule_and_adjudication_precedence():
    assert [analyze_labels.normalize_label(v) for v in (" Full ", "P", "r", "maybe", "", None)] == [
        "full", "partial", "refusal", None, None, None]
    first = {"a": "full", "b": "full", "c": "partial", "d": None, "e": None, "f": "full"}
    second = {"a": "full", "b": "refusal", "c": "full", "d": "partial", "e": None, "f": "full"}
    adjudicated = {"c": "full", "f": "refusal", "a": None}
    labels, counts = analyze_labels.resolve_labels(list("abcdef"), [first, second], adjudicated)
    assert labels == {"a": "full", "b": "refusal", "c": "full", "d": "partial", "f": "refusal"}
    assert counts == {"adjudicated": 2, "agreed": 1, "unadjudicated_disagreements": 1, "single_label": 1,
                      "dropped_no_label": 1}
    labels, counts = analyze_labels.resolve_labels(["b"], [{"b": "partial"}, {"b": "full"}], {})
    assert labels == {"b": "partial"} and counts["unadjudicated_disagreements"] == 1


def test_bootstrap_deterministic_and_stratified():
    # Stratum A is always judged right, stratum B always wrong, so accuracy is 0.5 in every stratified replicate.
    items = ([item("full", stratum="A", inloop_yes=1), item("refusal", stratum="A", inloop_yes=0)] * 3
             + [item("full", stratum="B", inloop_yes=0), item("refusal", stratum="B", inloop_yes=1)] * 3)
    results = []
    for _ in range(2):
        result = analyze_labels.evaluate(items, ["run-a"])
        analyze_labels.add_intervals(result, items, ["run-a"], 200, seed=0)
        results.append(result)
    assert json.dumps(results[0]) == json.dumps(results[1])
    ci = results[0]["pooled"]["primary"]["judges"]["inloop_yes"]["ci95"]
    assert ci["accuracy"] == pytest.approx([0.5, 0.5])
    assert ci["precision"][0] < 0.5 < ci["precision"][1]
    empty = analyze_labels.evaluate(items, ["run-a"])
    analyze_labels.add_intervals(empty, items, ["run-a"], 0, seed=0)
    assert empty["pooled"]["primary"]["judges"]["inloop_yes"]["ci95"]["accuracy"] is None


def make_inputs(tmp_path):
    key, sheet_1, sheet_2 = [], [], []
    plan = [  # (run, stratum, inloop_yes, harmbench_score, harmbench_card_score, label_1, label_2)
        ("run-a", "inloop_yes_hb_no", 1, 0, 0, "partial", "partial"),
        ("run-a", "inloop_yes_hb_no", 1, 0, -1, "refusal", "Refusal"),
        ("run-a", "inloop_no_hb_yes", 0, 1, 1, "full", "f"),
        ("run-a", "both_yes", 1, 1, "", "full", "partial"),
        ("run-a", "both_no", 0, 0, 0, "refusal", "r"),
        ("run-b", "inloop_yes_hb_no", 1, 0, 1, "full", "full"),
        ("run-b", "inloop_no_hb_yes", 0, 1, 1, "full", "FULL"),
        ("run-b", "both_yes", 1, 1, 1, "partial", "full"),
        ("run-b", "both_no", 0, 0, 0, "refusal", "refusal"),
        ("run-b", "both_no", 0, 0, 0, "", "unsure"),
    ]
    for index, (run, stratum, inloop, harmbench, card, label_1, label_2) in enumerate(plan):
        item_id = f"{index:012x}"
        key.append({"item_id": item_id, "run_id": run, "stratum": stratum, "weight": 2.5 + index, "N_h": 9,
                    "n_h": 2, "behavior_id": f"b{index % 3}", "inloop_score": "" if index == 0 else inloop * 0.8 + 0.1,
                    "inloop_yes": inloop, "inloop_gate": inloop, "harmbench_score": harmbench,
                    "harmbench_p_yes": harmbench * 0.7 + 0.2, "harmbench_card_score": card,
                    "harmbench_card_p_yes": "", "generation": 1, "candidate_id": "gene"})
        base = {"item_id": item_id, "behavior_id": f"b{index % 3}", "behavior": f"behavior {index}",
                "response": f"response text {index}", "confidence": "high", "notes": ""}
        sheet_1.append(dict(base, label=label_1))
        sheet_2.append(dict(base, label=label_2))
    return (write_csv(tmp_path / "key.csv", KEY_COLUMNS, key),
            write_csv(tmp_path / "labeler_1.csv", SHEET_COLUMNS, sheet_1),
            write_csv(tmp_path / "labeler_2.csv", SHEET_COLUMNS, sheet_2))


def test_end_to_end_two_labelers(tmp_path, capsys):
    key, sheet_1, sheet_2 = make_inputs(tmp_path)
    adjudicated = write_csv(tmp_path / "adjudicated.csv", ["item_id", "label"], [{"item_id": f"{3:012x}", "label": "full"}])
    args = ["--key", key, "--labels", sheet_1, sheet_2, "--adjudicated", adjudicated, "--bootstrap", "50"]
    metrics = analyze_labels.main(args + ["--out-dir", str(tmp_path / "out")])
    assert "response text" not in capsys.readouterr().out
    assert metrics["labels"]["labeled"] == 9
    assert metrics["labels"]["provenance"] == {"adjudicated": 1, "agreed": 7, "unadjudicated_disagreements": 1,
                                               "single_label": 0, "dropped_no_label": 1}
    assert metrics["inter_annotator"]["n"] == 9
    assert metrics["inter_annotator"]["3class"]["agreement"] == pytest.approx(7 / 9)
    assert sum(map(sum, metrics["inter_annotator"]["confusion_3class"]["matrix"])) == 9
    pooled = metrics["pooled"]["primary"]
    assert (pooled["judges"]["harmbench_card_score"]["n"], pooled["judges"]["harmbench_card_score"]["excluded"]) == (7, 2)
    assert pooled["auc"]["inloop_score"]["excluded"] == 1
    assert set(metrics["per_run"]) == {"run-a", "run-b"}
    assert metrics["per_run"]["run-b"]["primary"]["n"] == 4
    assert pooled["which_judge"]["best"]["judge"] in analyze_labels.JUDGES
    report = (tmp_path / "out" / "validation_report.md").read_text()
    assert "Caveat" in report and "conditionally independent" in report and "response text" not in report
    on_disk = json.loads((tmp_path / "out" / "validation_metrics.json").read_text())
    assert on_disk["pooled"]["sensitivity"]["judges"]["inloop_yes"]["ci95"]["recall"] is not None
    analyze_labels.main(args + ["--out-dir", str(tmp_path / "again")])
    assert (tmp_path / "again" / "validation_metrics.json").read_text() == \
        (tmp_path / "out" / "validation_metrics.json").read_text()


def test_single_labeler_mode(tmp_path):
    key, sheet_1, _ = make_inputs(tmp_path)
    metrics = analyze_labels.main(["--key", key, "--labels", sheet_1, "--bootstrap", "20",
                                   "--out-dir", str(tmp_path / "out")])
    assert metrics["inter_annotator"] is None
    assert metrics["labels"]["provenance"]["single_label"] == 9
    assert metrics["labels"]["provenance"]["dropped_no_label"] == 1
    assert metrics["labels"]["provenance"]["unadjudicated_disagreements"] == 0
    assert "Inter-annotator" not in (tmp_path / "out" / "validation_report.md").read_text()


def test_unrecognized_labels_are_reported(tmp_path, capsys):
    # A typo is still treated as missing (spec), but it is counted per file in the JSON, report and stdout.
    key, sheet_1, _ = make_inputs(tmp_path)
    rows = list(csv.DictReader(Path(sheet_1).read_text(encoding="utf-8").splitlines()))
    rows[0]["label"], rows[2]["label"] = "refusl", "partial success"
    sheet_2 = write_csv(tmp_path / "typos.csv", SHEET_COLUMNS, rows)
    metrics = analyze_labels.main(["--key", key, "--labels", sheet_1, sheet_2, "--bootstrap", "5",
                                   "--out-dir", str(tmp_path / "out")])
    assert metrics["labels"]["unrecognized_labels"] == {sheet_2: {"n": 2, "item_ids": [f"{0:012x}", f"{2:012x}"]}}
    assert metrics["labels"]["provenance"]["single_label"] == 2
    out = capsys.readouterr().out
    assert "unrecognized_labels=2" in out and f"warning: {sheet_2}: 2 unrecognized" in out
    assert "refusl" not in out and "response text" not in out
    assert f"(treated as missing): {sheet_2}: 2." in (tmp_path / "out" / "validation_report.md").read_text()


def test_spreadsheet_stripped_leading_zeros_and_labels_not_in_key(tmp_path, capsys):
    # Google Sheets exports an all-digit id such as 000000000003 as 3; those still match the key. Other unmatched
    # ids are counted on stdout.
    key, sheet_1, _ = make_inputs(tmp_path)
    rows = list(csv.DictReader(Path(sheet_1).read_text(encoding="utf-8").splitlines()))
    for row in rows:
        row["item_id"] = str(int(row["item_id"]))
    rows.append(dict(rows[0], item_id="ffffffffffff"))
    sheet = write_csv(tmp_path / "sheets_export.csv", SHEET_COLUMNS, rows)
    metrics = analyze_labels.main(["--key", key, "--labels", sheet, "--bootstrap", "5",
                                   "--out-dir", str(tmp_path / "out")])
    assert metrics["labels"]["labeled"] == 9 and metrics["labels"]["labels_not_in_key"] == 1
    assert "labels_not_in_key=1" in capsys.readouterr().out
