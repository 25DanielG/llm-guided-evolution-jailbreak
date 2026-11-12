import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "sota"
    / "Jailbreak"
    / "plot_run_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("plot_run_comparison", SCRIPT)
plotting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotting)


def test_pareto_mask_uses_all_three_objectives():
    frame = pd.DataFrame(
        [
            {"judge_mean": 0.8, "diversity": 0.6, "avg_prompt_tokens": 200},
            {"judge_mean": 0.7, "diversity": 0.5, "avg_prompt_tokens": 220},
            {"judge_mean": 0.7, "diversity": 0.7, "avg_prompt_tokens": 180},
            {"judge_mean": np.nan, "diversity": 0.9, "avg_prompt_tokens": 100},
        ]
    )

    assert plotting.pareto_mask(frame).tolist() == [True, False, True, False]


def test_final_generation_summary_ignores_other_generations():
    frame = pd.DataFrame(
        [
            {
                "generation": 15,
                "judge_mean": 0.1,
                "verified_asr": 0.1,
                "easy_verified": 0.1,
                "medium_verified": 0.1,
                "hard_verified": 0.1,
                "avg_prompt_tokens": 100,
                "rendered_prompt_sig": "old",
            },
            {
                "generation": 16,
                "judge_mean": 0.6,
                "verified_asr": 0.4,
                "easy_verified": 0.5,
                "medium_verified": 0.4,
                "hard_verified": 0.3,
                "avg_prompt_tokens": 200,
                "rendered_prompt_sig": "a",
            },
            {
                "generation": 16,
                "judge_mean": 0.8,
                "verified_asr": 0.8,
                "easy_verified": 0.9,
                "medium_verified": 0.8,
                "hard_verified": 0.7,
                "avg_prompt_tokens": 300,
                "rendered_prompt_sig": "b",
            },
        ]
    )

    summary = plotting.final_generation_summary(frame, 16)

    assert summary["genes"] == 2
    assert summary["judge_mean"] == 0.7
    assert summary["verified_mean"] == pytest.approx(0.6)
    assert summary["verified_best"] == 0.8
    assert summary["mean_tokens"] == 250
    assert summary["distinct_prompts"] == 2


def test_archive_matrix_places_elites_in_expected_cells():
    payload = {
        "elites": [
            {
                "format_strategy": "base64",
                "conversation_depth": "zero",
                "quality": [0.75, 0.2],
            },
            {
                "format_strategy": "transcript",
                "conversation_depth": "three_plus",
                "quality": [0.9, 0.4],
            },
        ]
    }

    matrix = plotting.archive_matrix(payload)

    assert matrix.shape == (3, 6)
    assert matrix[0, 1] == 0.75
    assert matrix[2, 5] == 0.9
    assert np.isnan(matrix[1, 0])


class _FakeArchive:
    capacity = 18

    def __init__(self, occupied):
        self.occupied = occupied

    def __len__(self):
        return self.occupied


class _FakeFitness:
    def __init__(self, values):
        self.values = values


class _FakeIndividual(list):
    def __init__(self, gene_id, fitness):
        super().__init__([gene_id])
        self.fitness = _FakeFitness(fitness)


def test_checkpoint_coverage_orders_generations(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "run" / "ckpt"
    checkpoint_dir.mkdir(parents=True)
    for generation, occupied in ((10, 8), (2, 3)):
        path = checkpoint_dir / f"checkpoint_gen_{generation}.pkl"
        with path.open("wb") as file:
            pickle.dump({"map_elites_archive": _FakeArchive(occupied)}, file)
    monkeypatch.setattr(
        plotting, "_register_checkpoint_types", lambda root: None
    )

    history = plotting.checkpoint_coverage(tmp_path, "run")

    assert history["generation"].tolist() == [2, 10]
    assert history["occupied_cells"].tolist() == [3, 8]


def test_checkpoint_population_fitness_uses_saved_diversity(
    tmp_path, monkeypatch
):
    checkpoint_dir = tmp_path / "run" / "ckpt"
    checkpoint_dir.mkdir(parents=True)
    with (checkpoint_dir / "checkpoint_gen_3.pkl").open("wb") as file:
        pickle.dump(
            {
                "population": [
                    _FakeIndividual("a", (0.72, 0.41, 220.0)),
                    _FakeIndividual("b", (0.65, 0.63, 180.0)),
                ]
            },
            file,
        )
    monkeypatch.setattr(
        plotting, "_register_checkpoint_types", lambda root: None
    )

    history = plotting.checkpoint_population_fitness(tmp_path, "run")

    assert history["generation"].tolist() == [3, 3]
    assert history["gene_id"].tolist() == ["a", "b"]
    assert history["diversity"].tolist() == [0.41, 0.63]


def test_plot_functions_write_png_and_svg(tmp_path):
    base_rows = [
        {
            "generation": 16,
            "judge_mean": 0.6,
            "verified_asr": 0.5,
            "easy_verified": 0.6,
            "medium_verified": 0.5,
            "hard_verified": 0.4,
            "avg_prompt_tokens": 200,
            "diversity": 0.3,
            "rendered_prompt_sig": "a",
        },
        {
            "generation": 16,
            "judge_mean": 0.7,
            "verified_asr": 0.7,
            "easy_verified": 0.8,
            "medium_verified": 0.7,
            "hard_verified": 0.6,
            "avg_prompt_tokens": 250,
            "diversity": 0.5,
            "rendered_prompt_sig": "b",
        },
    ]
    frames = {
        "MAP-Elites": pd.DataFrame(base_rows),
        "NSGA-II": pd.DataFrame(base_rows),
    }
    histories = {
        "MAP-Elites": pd.DataFrame(
            {
                "generation": [1, 16],
                "occupied_cells": [2, 12],
                "capacity": [18, 18],
            }
        ),
        "NSGA-II": pd.DataFrame(
            {
                "generation": [1, 16],
                "occupied_cells": [2, 10],
                "capacity": [18, 18],
            }
        ),
    }
    archive = {
        "occupied_cells": 1,
        "capacity": 18,
        "coverage": 1 / 18,
        "elites": [
            {
                "format_strategy": "base64",
                "conversation_depth": "zero",
                "quality": [0.75],
            }
        ],
    }
    archives = {"MAP-Elites": archive, "NSGA-II": archive}
    summaries = {
        method: plotting.final_generation_summary(frame, 16)
        for method, frame in frames.items()
    }

    plotting.plot_coverage(histories, tmp_path, 72)
    plotting.plot_heatmaps(archives, tmp_path, 72)
    plotting.plot_pareto(frames, tmp_path, 72)
    plotting.plot_verified_performance(summaries, tmp_path, 72)

    for stem in (
        "coverage_by_generation",
        "archive_heatmaps",
        "pareto_front",
        "verified_performance",
    ):
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.svg").exists()
