import csv
import hashlib
import importlib
import json
import random
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sota" / "Jailbreak"))
label_sheet = importlib.import_module("label_sheet")

HEADER = ["item_id", "behavior_id", "behavior", "response", "inloop_score", "harmbench_score", "run_id",
          "generation", "candidate_id", "behavior_index", "n_rows", "inloop_yes", "inloop_gate",
          "inloop_parse_ok", "inloop_filter", "rubric_refused", "rubric_on_topic", "rubric_specificity",
          "rubric_convincingness", "rating_method", "judge_model", "harmbench_p_yes", "harmbench_raw",
          "harmbench_truncated", "harmbench_card_score", "harmbench_card_p_yes", "team_verify_successes",
          "team_verified_success", "team_audit_success", "source"]
ALLOC = [Fraction(35), Fraction(35), Fraction(15), Fraction(15)]
VERDICTS = {"inloop_yes_hb_no": (1, 0), "inloop_no_hb_yes": (0, 1), "both_yes": (1, 1), "both_no": (0, 0)}


def item(run, behavior, index, inloop, harmbench, gate=None, response=None):
    response = f"response text {index}" if response is None else response
    row = dict.fromkeys(HEADER, "")
    row.update(item_id=hashlib.sha256(f"{run}\x00behavior {behavior}\x00{response}".encode()).hexdigest()[:12],
               behavior_id=f"b_{behavior}", behavior=f"behavior {behavior}", response=response,
               inloop_score="0.75" if inloop == 1 else "0.25", harmbench_score=str(harmbench), run_id=run,
               generation="1", candidate_id=f"cand_{index}", behavior_index=str(index), n_rows="1",
               inloop_yes=str(inloop), inloop_gate=str(inloop if gate is None else gate), inloop_parse_ok="1",
               harmbench_p_yes="0.9" if harmbench == 1 else "0.1", harmbench_card_score=str(harmbench),
               source="candidates")
    return row


def population(runs=("run_alpha", "run_beta"), per_stratum=30, behaviors=8):
    rows, index = [], 0
    for run in runs:
        for inloop, harmbench in VERDICTS.values():
            for _ in range(per_stratum):
                rows.append(item(run, chr(65 + index % behaviors), index, inloop, harmbench))
                index += 1
    return rows


def write_reporting(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def read(path):
    with open(path, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_allocation_and_rounding():
    assert label_sheet.largest_remainder(100, ALLOC) == [35, 35, 15, 15]
    assert label_sheet.largest_remainder(50, ALLOC) == [18, 18, 7, 7]
    assert label_sheet.largest_remainder(7, ALLOC) == [3, 2, 1, 1]
    assert label_sheet.largest_remainder(10, [1, 1, 1]) == [4, 3, 3]
    _, _, summary = label_sheet.build(population(runs=("r3", "r1", "r2")), 10, ALLOC, "threshold", 3, 0)
    assert {run: s["requested"] for run, s in summary["runs"].items()} == {"r1": 4, "r2": 3, "r3": 3}
    assert [c["allocated"] for c in summary["strata"] if c["run_id"] == "r1"] == [1, 1, 1, 1]
    assert summary["n_sampled"] == 10


def test_shortfall_redistribution():
    target, take = label_sheet.allocate(20, [2, 100, 100, 1], ALLOC)
    assert target == [7, 7, 3, 3] and take == [2, 12, 5, 1]
    assert label_sheet.allocate(20, [2, 3, 1, 1], ALLOC)[1] == [2, 3, 1, 1]
    rows = [r for r in population(runs=("run_alpha",)) if r["inloop_yes"] == "0" or r["harmbench_score"] == "1"]
    rows += population(runs=("run_alpha",), per_stratum=2)[:2]
    sample, _, summary = label_sheet.build(rows, 40, ALLOC, "threshold", 3, 0)
    cells = {c["stratum"]: c for c in summary["strata"]}
    assert cells["inloop_yes_hb_no"]["N_h"] == cells["inloop_yes_hb_no"]["n_h"] == 2
    assert [cells[s]["n_h"] for s in label_sheet.STRATA] == [2, 20, 9, 9]
    assert len(sample) == 40


def test_cap_and_relaxation():
    rows = [item("r", "A", i, 1, 0) for i in range(10)] + [item("r", "B", 10, 1, 0), item("r", "C", 11, 1, 0)]
    chosen, relaxations = label_sheet.select(rows, 4, 2, random.Random(0))
    assert relaxations == 0 and max(Counter(r["behavior"] for r in chosen).values()) == 2
    chosen, relaxations = label_sheet.select(rows, 6, 2, random.Random(0))
    assert relaxations == 2
    assert Counter(r["behavior"] for r in chosen) == {"behavior A": 4, "behavior B": 1, "behavior C": 1}
    _, _, summary = label_sheet.build(rows, 6, [1, 0, 0, 0], "threshold", 2, 0)
    assert summary["strata"][0]["cap_relaxations"] == 2


def test_weights_sum_to_population_per_stratum():
    pop = [item("r", "A", i, 1, 0) for i in range(6)] + [item("r", b, i, 1, 0) for b, i in
                                                          (("B", 6), ("B", 7), ("C", 8), ("C", 9))]
    weights = label_sheet.post_stratified_weights(pop, [pop[0], pop[1], pop[6]])
    assert weights == {pop[0]["item_id"]: 3.75, pop[1]["item_id"]: 3.75, pop[6]["item_id"]: 2.5}
    rows = population(behaviors=5) + [item("run_alpha", "Z", 999, 0, 0)]
    sample, _, summary = label_sheet.build(rows, 60, ALLOC, "threshold", 3, 0)
    for cell in summary["strata"]:
        members = [r for r in sample if (r["run_id"], r["stratum"]) == (cell["run_id"], cell["stratum"])]
        assert len(members) == cell["n_h"] and all(r["N_h"] == cell["N_h"] and r["n_h"] == cell["n_h"] for r in members)
        assert sum(r["weight"] for r in members) == pytest.approx(cell["N_h"])


def test_blind_sheets_and_outputs(tmp_path, capsys):
    rows = population()
    rows[0]["response"] = "response text, with comma\nand a second line"
    reporting = write_reporting(tmp_path / "reporting.csv", rows)
    summary = label_sheet.main(["--reporting", reporting, "--out-dir", str(tmp_path / "out"), "--n", "40"])
    printed = capsys.readouterr().out
    assert "response text" not in printed and "Convert text to numbers, dates, and formulas" in printed
    out = tmp_path / "out"
    sheet_text = (out / "labeler_1.csv").read_text(encoding="utf-8")
    assert sheet_text == (out / "labeler_2.csv").read_text(encoding="utf-8")
    sheet, key = read(out / "labeler_1.csv"), read(out / "key.csv")
    assert list(sheet[0]) == label_sheet.SHEET_COLUMNS and list(key[0]) == label_sheet.KEY_COLUMNS
    assert all(r["label"] == r["confidence"] == r["notes"] == "" for r in sheet)
    for leaked in ("run_alpha", "run_beta", "cand_", *label_sheet.STRATA):
        assert leaked not in sheet_text
    assert len(sheet) == len(key) == summary["n_sampled"] == 40
    by_id = {r["item_id"]: r for r in rows}
    assert all(r["response"] == by_id[r["item_id"]]["response"] for r in sheet)
    groups = {r["item_id"]: (r["run_id"], r["stratum"]) for r in key}
    sequence = [groups[r["item_id"]] for r in sheet]
    blocks = 1 + sum(a != b for a, b in zip(sequence, sequence[1:]))
    assert sequence != sorted(sequence) and blocks > len(set(sequence)) * 2
    written = json.loads((out / "sampling_summary.json").read_text(encoding="utf-8"))
    assert written["rule"] == "threshold" and written["seed"] == 0 and len(written["strata"]) == 8


def test_determinism_with_seed():
    rows = population()
    first = label_sheet.build(rows, 40, ALLOC, "threshold", 3, 7)
    again = label_sheet.build(list(reversed(rows)), 40, ALLOC, "threshold", 3, 7)
    other = label_sheet.build(rows, 40, ALLOC, "threshold", 3, 8)
    ids = lambda result: [r["item_id"] for r in result[1]]
    assert ids(first) == ids(again) and first[0] == again[0] and first[2] == again[2]
    assert ids(first) != ids(other)


def test_exclude_key(tmp_path):
    reporting = write_reporting(tmp_path / "reporting.csv", population(per_stratum=12))
    label_sheet.main(["--reporting", reporting, "--out-dir", str(tmp_path / "b1"), "--n", "40"])
    summary = label_sheet.main(["--reporting", reporting, "--out-dir", str(tmp_path / "b2"), "--n", "40",
                                "--exclude-key", str(tmp_path / "b1" / "key.csv")])
    first = {r["item_id"] for r in read(tmp_path / "b1" / "key.csv")}
    second = {r["item_id"] for r in read(tmp_path / "b2" / "key.csv")}
    assert len(first) == len(second) == 40 and not first & second
    assert summary["skipped"]["excluded"] == 40 and summary["eligible"] == 96 - 40


def test_exclude_key_batches_pool_to_population():
    rows = [item("r", chr(65 + i % 4), i, 1, 0) for i in range(10)]
    rows += [item("r", chr(65 + i % 4), 100 + i, 0, 0) for i in range(100)]
    first, _, _ = label_sheet.build(rows, 16, [1, 0, 0, 1], "threshold", 3, 0)
    second, _, summary = label_sheet.build(rows, 16, [1, 0, 0, 1], "threshold", 3, 0, {r["item_id"] for r in first})
    assert {r["item_id"] for r in first}.isdisjoint(r["item_id"] for r in second) and len(second) == 16
    cell = summary["strata"][0]
    assert (cell["N_h"], cell["available"], cell["n_h"]) == (10, 2, 2)
    for name, size in (("inloop_yes_hb_no", 10), ("both_no", 100)):
        assert all(r["N_h"] == size for r in first + second if r["stratum"] == name)
        assert sum(r["weight"] for r in first + second if r["stratum"] == name) == pytest.approx(2 * size)


def test_unfilled_run_share_goes_to_other_runs():
    rows = population(runs=("big",), per_stratum=20) + [item("tiny", "A", 900 + i, 0, 0) for i in range(5)]
    sample, _, summary = label_sheet.build(rows, 40, ALLOC, "threshold", 3, 0)
    assert len(sample) == summary["n_sampled"] == 40
    assert {run: (s["requested"], s["sampled"]) for run, s in summary["runs"].items()} == {
        "big": (20, 35), "tiny": (20, 5)}


def test_duplicate_items_do_not_depend_on_order():
    rows = population(runs=("r",), per_stratum=6)
    copies = [dict(r) for r in rows]
    results = [label_sheet.build(order, 12, ALLOC, "threshold", 3, 0) for order in (rows + copies, copies + rows)]
    assert [r["item_id"] for r in results[0][1]] == [r["item_id"] for r in results[1][1]]
    assert all(s["skipped"] == {"duplicate_item": 24} and s["eligible"] == 24 for _, _, s in results)
    dry_run = [dict(r, harmbench_score="", harmbench_card_score="") for r in rows]
    for order in (rows + dry_run, dry_run + rows):
        with pytest.raises(ValueError, match="24 item_ids"):
            label_sheet.build(order, 12, ALLOC, "threshold", 3, 0)


def test_rule_gate_and_eligibility():
    rows = [item("r", "A", 0, 1, 0, gate=0), item("r", "B", 1, 0, 1, gate=1), item("r", "C", 2, 1, 1, gate=""),
            item("r", "D", 3, 1, -1), item("r", "E", 4, 0, 0, response=" "), item("r", "F", 5, 0, 0)]
    rows[5]["inloop_yes"] = ""
    sample, _, summary = label_sheet.build(rows, 10, ALLOC, "threshold", 3, 0)
    assert {r["behavior"]: r["stratum"] for r in sample} == {
        "behavior A": "inloop_yes_hb_no", "behavior B": "inloop_no_hb_yes", "behavior C": "both_yes"}
    assert summary["skipped"] == {"inloop_missing": 1, "harmbench_unusable": 1, "empty_response": 1}
    sample, _, summary = label_sheet.build(rows, 10, ALLOC, "gate", 3, 0)
    assert {r["behavior"]: r["stratum"] for r in sample} == {
        "behavior A": "both_no", "behavior B": "both_yes", "behavior F": "both_no"}
    assert summary["inloop_column"] == "inloop_gate"
    assert summary["skipped"] == {"inloop_missing": 1, "harmbench_unusable": 1, "empty_response": 1}
