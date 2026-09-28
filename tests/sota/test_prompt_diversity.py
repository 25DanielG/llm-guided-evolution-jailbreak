import importlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sota" / "Jailbreak"))

from sota.Jailbreak import prompt_records as capture
from analysis.utils import prompt_diversity as diversity

evaluation = importlib.import_module("eval")


def observation(candidate, case="one", generation=0, text="a b c", evaluation_id=None, status="ok", **extra):
    attempt = evaluation_id or f"attempt-{candidate}-{generation}"
    return {"record_id": f"{attempt}:{case}", "evaluation_id": attempt, "candidate_id": candidate,
            "case_id": case, "generation": generation, "prompt_text": text, "status": status,
            "messages": [{"role": "user", "content": text}], "behavior": case,
            "dataset_fingerprint": "dataset", "target_context": {"target_model": "test"}, **extra}


def cohort(rows, ids=None, cases=None):
    return {"generation": 1, "cohort": "retained", "source": "manifest",
            "strategy_ids": ids or sorted({row["candidate_id"] for row in rows}),
            "cases": [{"case_id": case} for case in (cases or sorted({row["case_id"] for row in rows}))],
            "prompts": rows}


def test_ngram_sets_and_short_prompts():
    a = diversity.ngrams([1, 2, 1, 2, 1], 2)
    assert a == {(1, 2), (2, 1)}
    assert diversity.jaccard_distance(a, a) == 0
    assert diversity.jaccard_distance(a, {(3, 4)}) == 1
    assert diversity.jaccard_distance(a, {(1, 2), (3, 4)}) == pytest.approx(2 / 3)
    assert diversity.ngrams([1, 2], 3) == set()
    with pytest.raises(ValueError):
        diversity.ngrams([1], 0)
    summary, _, diagnostics = diversity.measure_cohort(cohort([observation("a", text="a b"), observation("b")]), str.split)
    assert summary["ngram_mean"] is None
    assert summary["ngram_pair_coverage"] == 0
    assert diagnostics[0]["ngram_status"] == "short_prompt"


def test_cosine_extremes_and_invalid_vectors():
    assert diversity.cosine_distance([1, 0], [1, 0]) == 0
    assert diversity.cosine_distance([1, 0], [0, 1]) == 1
    assert diversity.cosine_distance([1, 0], [-1, 0]) == 2
    for vector in ([0, 0], [np.nan, 1], [np.inf, 1]):
        with pytest.raises(ValueError):
            diversity.normalize(vector)


def test_equal_case_weighting_and_nearest_neighbors():
    # Case one has three disjoint pairs (mean 1); case two one identical pair (mean 0).
    rows = [observation("a", text="a b c"), observation("b", text="d e f"), observation("c", text="g h i"),
            observation("a", "two", text="same text here"), observation("b", "two", text="same text here"),
            observation("c", "two", status="build_failed")]
    summary, cases, _ = diversity.measure_cohort(cohort(rows), str.split)
    assert summary["ngram_mean"] == 0.5  # Pair weighting would incorrectly give .75.
    assert summary["ngram_nearest_neighbor"] == 0.5
    assert summary["ngram_compared_pairs"] == 4
    assert summary["possible_pairs"] == 6
    assert summary["ngram_pair_coverage"] == pytest.approx(2 / 3)
    assert cases[0]["ngram_nearest_neighbor"] == 1  # No self-neighbor zeros.


def test_cases_never_cross_and_unavailable_not_zero():
    summary, _, _ = diversity.measure_cohort(cohort([observation("a", "one"), observation("b", "two")]), str.split)
    assert summary["ngram_mean"] is None
    assert summary["ngram_valid_cases"] == 0
    summary, _, _ = diversity.measure_cohort(cohort([observation("a")]), str.split)
    assert summary["ngram_mean"] is None and summary["ngram_pair_coverage"] is None


def test_missing_prompt_reduces_coverage_without_excluding_valid_pairs():
    rows = [observation("a"), observation("b"), {"candidate_id": "c", "case_id": "one", "status": "missing_prompt"}]
    summary, cases, _ = diversity.measure_cohort(cohort(rows), str.split)
    assert summary["ngram_mean"] == 0
    assert summary["ngram_compared_pairs"] == 1
    assert summary["ngram_pair_coverage"] == pytest.approx(1 / 3)
    assert cases[0]["status"] == "ok"


def test_identical_strategies_count_and_embedding_cache():
    class Embedder:
        calls = 0
        def embed(self, text):
            self.calls += 1
            return [1, 0], 1
    embedder = Embedder()
    summary, _, _ = diversity.measure_cohort(cohort([observation("a"), observation("b")]), str.split, embedder)
    assert summary["ngram_mean"] == summary["semantic_mean"] == 0
    assert summary["semantic_compared_pairs"] == 1 and embedder.calls == 1


def test_chunk_token_coverage_and_pooling():
    class Tokenizer:
        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False, "truncation": False}
            return list(range(9))
        def num_special_tokens_to_add(self, pair):
            return 2
    chunks = diversity.token_chunks(Tokenizer(), "long prompt", 6)
    assert chunks == [[0, 1, 2, 3], [4, 5, 6, 7], [8]]
    expected = np.array([2, 1]) / np.sqrt(5)
    assert np.allclose(diversity.pooled_embedding([[1, 0], [0, 1]], [2, 1]), expected)


def test_persistent_cache_identity(tmp_path):
    # The cache is keyed by content and every model/preprocessing setting.
    embedder = diversity.SentenceBertEmbedder.__new__(diversity.SentenceBertEmbedder)
    embedder.cache_dir = tmp_path
    embedder.cache_config = {"revision": "one", "pooling": "first"}
    first = embedder.cache_path("prompt")
    assert first != embedder.cache_path("different")
    embedder.cache_config["revision"] = "two"
    assert first != embedder.cache_path("prompt")
    embedder.cache_config = {"revision": "one", "pooling": "second"}
    assert first != embedder.cache_path("prompt")


def test_persistent_cache_hit_skips_model_inference(tmp_path):
    embedder = diversity.SentenceBertEmbedder.__new__(diversity.SentenceBertEmbedder)
    embedder.cache_dir = tmp_path
    embedder.cache_config = {"revision": "pinned", "pooling": diversity.POOLING_VERSION}
    capture.atomic_json(embedder.cache_path("text"), {"vector": [1, 0], "chunk_count": 2})
    # No tokenizer/model attributes exist: a miss would fail the test.
    vector, chunks = embedder.embed("text")
    assert np.array_equal(vector, [1, 0]) and chunks == 2


def test_stable_case_ids_and_fences(tmp_path):
    path = tmp_path / "cases.csv"
    path.write_text("id,behavior\nfirst,First case\nsecond,Second case\n")
    cases, fingerprint = capture.load_cases(path)
    assert [case["case_id"] for case in cases] == ["first", "second"]
    assert fingerprint == capture.digest(path.read_bytes())
    messages = [{"role": "system", "content": "instructions"}, {"role": "user", "content": "```\nkeep this\n```"}]
    assert capture.serialize_messages(messages) == "[system]\ninstructions\n[user]\n```\nkeep this\n```"


def test_duplicate_case_rows_preserve_evaluation_inputs(tmp_path):
    path = tmp_path / "cases.csv"
    path.write_text("behavior\nsame case\nsame case\n")
    cases, _ = capture.load_cases(path)
    assert [case["behavior"] for case in cases] == ["same case", "same case"]
    assert len({case["case_id"] for case in cases}) == 2
    assert cases == capture.load_cases(path)[0]


def test_capture_before_target_inference_and_build_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(evaluation.cache, "get", lambda *args: None)
    monkeypatch.setattr(evaluation.cache, "put", lambda *args: None)
    class Strategy:
        def build_prompt(self, behavior):
            if behavior == "bad":
                raise ValueError("broken")
            return behavior
    context = {"run_id": "run", "generation": 3, "candidate_id": "a", "dataset_fingerprint": "dataset",
               "target_context": {"target_model": "test"}, "sampled_case_ids": ["first", "second"]}
    def query(*args, **kwargs):
        rows = capture.read_jsonl(tmp_path / "results/run/prompt_records.jsonl")
        assert [row["status"] for row in rows] == ["build_failed", "ok"]
        assert [row["case_id"] for row in rows] == ["first", "second"]
        return [{"text": "answer", "prompt_tokens": 4}]
    monkeypatch.setattr(evaluation.vllm_client, "chat_batch", query)
    rows = evaluation.get_target_responses(Strategy(), [{"case_id": "first", "behavior": "bad"},
                                                       {"case_id": "second", "behavior": "good"}], "url", "test", capture_context=context)
    assert rows[0]["case_id"] == "second" and rows[0]["behavior_index"] == 1


def test_capture_error_does_not_change_fitness_inputs(monkeypatch):
    monkeypatch.setattr(evaluation.prompt_records, "append_jsonl", lambda *args: (_ for _ in ()).throw(OSError("disk error")))
    monkeypatch.setattr(evaluation.cache, "get", lambda *args: json.dumps({"text": "answer", "prompt_tokens": 5}))
    strategy = type("Strategy", (), {"build_prompt": lambda self, behavior: behavior})()
    rows = evaluation.get_target_responses(strategy, ["case"], "url", "test", capture_context={"run_id": "run"})
    assert rows[0]["response"] == "answer" and rows[0]["prompt_tokens"] == 5


def test_manifest_carried_survivors_failed_reevaluation_and_resume(tmp_path):
    cases = [{"case_id": "one", "behavior": "one"}, {"case_id": "two", "behavior": "two"}]
    context = {"target_model": "test"}
    old = [observation("a"), observation("a", "two"), observation("b")]
    fresh = [observation("a", generation=2, status="build_failed"),
             observation("a", "two", generation=2)]
    capture.append_jsonl(tmp_path / "prompt_records.jsonl", old + fresh)
    path = capture.save_population_manifest(tmp_path, "run", 2, {"retained": ["a", "b", "b"], "offspring": ["a"]}, cases, "dataset", context)
    manifest = json.loads(path.read_text())
    refs = manifest["cohorts"]["retained"]["references"]
    assert refs[0]["status"] == "build_failed"  # Not filled from old successful prompt.
    assert refs[2]["record_id"] == old[2]["record_id"]  # Carried survivor.
    assert refs[3]["status"] == "missing_prompt"
    before = path.read_bytes()
    capture.save_population_manifest(tmp_path, "run", 2, {"retained": []}, [], "changed", {})
    assert path.read_bytes() == before


def test_context_mismatches_excluded():
    rows = [observation("a"), observation("b", target_context={"target_model": "other"})]
    summary, cases, _ = diversity.measure_cohort(cohort(rows), str.split)
    assert summary["ngram_mean"] is None and cases[0]["status"] == "context_mismatch"
    rows = [observation("a"), observation("b", dataset_fingerprint="different")]
    assert diversity.measure_cohort(cohort(rows), str.split)[0]["ngram_mean"] is None


def test_historical_conflicts_and_missing_cases():
    rows = [observation("a"), observation("a", text="different text here"), observation("b")]
    # Old logs have no evaluation attempt identifiers.
    for row in rows:
        row.pop("evaluation_id")
    result = diversity.historical_cohort(rows, 1, ["a", "b", "c"], [{"case_id": "one"}])
    assert [row["status"] for row in result] == ["conflicting_records", "ok", "missing_prompt"]


def test_historical_checkpoint_reader_and_native_manifest(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "ckpt").mkdir(parents=True)
    with (run_dir / "ckpt/checkpoint_gen_1.pkl").open("wb") as file:
        pickle.dump({"population": [["a"], ["b"]]}, file)
    records_dir = tmp_path / "records"
    legacy = [{"candidate_id": candidate, "generation": 0, "behavior": "one", "target_model": "test",
               "evolved_prompt": [{"role": "user", "content": "a b c"}]} for candidate in ("a", "b")]
    capture.append_jsonl(records_dir / "candidates.jsonl", legacy)
    cohorts, notes = diversity.load_run(run_dir, records_dir)
    assert len(cohorts) == 1 and cohorts[0]["source"] == "checkpoint_reconstruction"
    assert notes and cohorts[0]["prompts"][0]["generation"] == 0
    rows = [observation("a"), observation("b")]
    capture.append_jsonl(records_dir / "prompt_records.jsonl", rows)
    capture.save_population_manifest(records_dir, "run", 1, {"retained": ["a", "b"], "offspring": ["b"]},
                                     [{"case_id": "one"}], "dataset", {"target_model": "test"})
    cohorts, notes = diversity.load_run(run_dir, records_dir)
    assert len(cohorts) == 2 and not notes
    assert all(c["source"] == "manifest" for c in cohorts)


def test_missing_native_record_remains_unavailable(tmp_path):
    rows = [observation("a"), observation("b")]
    capture.append_jsonl(tmp_path / "records/prompt_records.jsonl", rows)
    capture.save_population_manifest(tmp_path / "records", "run", 1, {"retained": ["a", "b"]},
                                     [{"case_id": "one"}], "dataset", {"target_model": "test"})
    (tmp_path / "records/prompt_records.jsonl").unlink()
    cohorts, _ = diversity.load_run(tmp_path / "run", tmp_path / "records")
    summary, _, diagnostics = diversity.measure_cohort(cohorts[0], str.split)
    assert summary["ngram_mean"] is None
    assert all(row["ngram_status"] == "missing_record" for row in diagnostics)


def test_cli_preserves_raw_files_when_embedding_unavailable(tmp_path, monkeypatch):
    from analysis.scripts import prompt_diversity as cli
    import transformers
    raw = tmp_path / "raw"
    rows = [observation("a"), observation("b")]
    capture.append_jsonl(raw / "prompt_records.jsonl", rows)
    capture.save_population_manifest(raw, "run", 1, {"retained": ["a", "b"]},
                                     [{"case_id": "one"}], "dataset", {"target_model": "test"})
    original = {path: path.read_bytes() for path in raw.rglob("*") if path.is_file()}
    class Tokenizer:
        init_kwargs = {}
        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False, "truncation": False}
            return text.split()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(cli, "SentenceBertEmbedder", lambda *args: (_ for _ in ()).throw(RuntimeError("model unavailable")))
    output = tmp_path / "analysis"
    assert cli.main([str(tmp_path / "run"), "--records-dir", str(raw), "--output-dir", str(output),
                     "--target-tokenizer", "fake-target"]) == 0
    assert original == {path: path.read_bytes() for path in raw.rglob("*") if path.is_file()}
    assert (output / "ngram_diversity.png").exists() and (output / "semantic_diversity.png").exists()
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["semantic_error"] == "RuntimeError: model unavailable"
    import csv
    summary = list(csv.DictReader((output / "generation_diversity.csv").open()))[0]
    assert float(summary["ngram_mean"]) == 0 and summary["semantic_mean"] == ""
