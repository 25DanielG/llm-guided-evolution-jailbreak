import os

import run_improved


def test_gene_result_path_defaults_under_sota_root(monkeypatch):
    monkeypatch.delenv("JB_GENE_RESULTS_DIR", raising=False)
    monkeypatch.setattr(run_improved, "SOTA_ROOT", "/tmp/jailbreak")

    assert run_improved._gene_result_path("gene-a", "json") == os.path.join(
        "/tmp/jailbreak", "results", "gene-a_results.json"
    )


def test_gene_result_path_respects_relative_override(monkeypatch):
    monkeypatch.setenv("JB_GENE_RESULTS_DIR", "custom-results")
    monkeypatch.setattr(run_improved, "SOTA_ROOT", "/tmp/jailbreak")

    assert run_improved._gene_result_path("gene-a", "csv") == os.path.join(
        "/tmp/jailbreak", "custom-results", "gene-a_results.csv"
    )


def test_gene_result_path_respects_absolute_override(monkeypatch):
    monkeypatch.setenv("JB_GENE_RESULTS_DIR", "/tmp/shared-results")

    assert run_improved._gene_result_path("gene-a", "csv") == os.path.join(
        "/tmp/shared-results", "gene-a_results.csv"
    )
