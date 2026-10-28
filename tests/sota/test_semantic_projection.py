import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.utils.semantic_projection import (
    choose_representative_case, cosine_matrix, generation_alpha, load_cached_analysis,
    neighborhood_quality, project_case,
)


def test_joint_coordinates_restore_duplicates_and_preserve_population_weights():
    vectors = {"a": np.array([1., 0.]), "b": np.array([0., 1.]), "same": np.array([1., 0.])}
    rows = [{"generation": 1, "candidate_id": "one", "prompt_hash": "a"},
            {"generation": 1, "candidate_id": "two", "prompt_hash": "b"},
            {"generation": 2, "candidate_id": "three", "prompt_hash": "a"},
            {"generation": 2, "candidate_id": "four", "prompt_hash": "same"},
            {"generation": 2, "candidate_id": "five", "prompt_hash": "b"}]
    coordinates, stats, quality = project_case(rows, vectors)
    assert coordinates["a"] == coordinates["same"]
    assert quality["unique_vectors"] == 2
    assert stats[0]["semantic_nearest_neighbor"] == 1
    assert stats[1]["semantic_mean"] == pytest.approx(2 / 3)
    assert stats[1]["semantic_nearest_neighbor"] == pytest.approx(1 / 3)
    assert stats[1]["valid_strategies"] == 3
    assert stats[1]["unique_prompts"] == 3


def test_original_cosine_distances_and_neighborhood_check():
    xy = np.array([[i, i * i] for i in range(8)], dtype=float)
    distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
    quality = neighborhood_quality(distances, xy, k=2)
    assert quality["trustworthiness"] == 1
    assert quality["neighbor_overlap"] == 1
    matrix = cosine_matrix([[1, 0], [0, 1], [1, 0]])
    np.testing.assert_allclose(matrix, [[0, 1, 0], [1, 0, 1], [0, 1, 0]])


def test_latest_generation_is_opaque_older_generations_are_faded():
    generations = [1, 3, 10]
    values = [generation_alpha(g, generations) for g in generations]
    assert values[-1] == 1
    assert 0 < values[0] < values[1] < values[-1]
    assert generation_alpha(10, [10]) == 1


def test_case_selection_uses_median_metric_change_not_projection_appearance():
    def stats(first, last):
        return [{"semantic_nearest_neighbor": first}, {"semantic_nearest_neighbor": last}]
    assert choose_representative_case({"dramatic": stats(1, .1), "median": stats(1, .7), "expands": stats(1, 1.5)}) == "median"


def test_cache_loader_filters_case_and_rejects_missing_provenance(tmp_path):
    config = {"model": "fixed", "revision": "one"}
    (tmp_path / "metadata.json").write_text(json.dumps({"embedding": {**config, "device": "cpu"}}))
    cache = tmp_path / "embedding_cache"
    cache.mkdir()
    (cache / "one.json").write_text(json.dumps({"config": config, "prompt_hash": "a", "vector": [1, 0]}))
    (cache / "two.json").write_text(json.dumps({"config": config, "prompt_hash": "b", "vector": [0, 1]}))
    (tmp_path / "prompt_diagnostics.csv").write_text(
        "generation,cohort,case_id,candidate_id,semantic_status,prompt_hash\n"
        "1,retained,one,s1,ok,a\n1,retained,two,s2,ok,b\n"
        "1,retained,one,s3,missing_prompt,\n1,offspring,one,s4,ok,a\n")
    (tmp_path / "generation_diversity.csv").write_text("generation,cohort\n1,retained\n")
    (tmp_path / "case_diversity.csv").write_text("generation,cohort,case_id\n1,retained,one\n1,retained,two\n")
    _, _, grouped, unavailable, _, _ = load_cached_analysis(tmp_path)
    assert len(grouped["one"]) == len(grouped["two"]) == 1
    assert len(unavailable) == 1
    (cache / "one.json").write_text(json.dumps({"config": {**config, "revision": "wrong"}, "prompt_hash": "a", "vector": [1, 0]}))
    with pytest.raises(ValueError, match="Missing matching cached embedding"):
        load_cached_analysis(tmp_path)
