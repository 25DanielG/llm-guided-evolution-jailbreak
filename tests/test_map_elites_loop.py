import json
import pickle

import run_improved
from src.map_elites import MapElitesArchive, PromptDescriptor


def _individual(gene_id, fitness):
    individual = run_improved.creator.Individual([gene_id])
    individual.fitness.values = fitness
    return individual


def _write_prompt(variant_dir, gene_id, format_type="base64", turns=0):
    turn_text = '<turn role="user">x</turn>' * turns
    (variant_dir / f"prompt_{gene_id}.py").write_text(
        f'trait_format = \'<format type="{format_type}">{{ASK}}</format>\'\n'
        f'trait_turns = \'<turns>{turn_text}</turns>\'\n',
        encoding="utf-8",
    )


def _write_result(results_dir, gene_id, easy, medium, hard):
    (results_dir / f"{gene_id}_results.json").write_text(
        json.dumps({
            "tier_scores": {"easy": easy, "medium": medium, "hard": hard},
            "behaviors": [],
        }),
        encoding="utf-8",
    )


def test_evaluated_prompts_populate_and_replace_archive(tmp_path, monkeypatch):
    variants = tmp_path / "variants"
    results = tmp_path / "results"
    variants.mkdir()
    results.mkdir()
    monkeypatch.setattr(run_improved, "VARIANT_DIR", str(variants))
    monkeypatch.setattr(run_improved, "MODEL", "prompt")
    monkeypatch.setattr(run_improved, "box_print", lambda *args, **kwargs: None)
    monkeypatch.setenv("JB_GENE_RESULTS_DIR", str(results))

    _write_prompt(variants, "a")
    _write_prompt(variants, "b")
    _write_result(results, "a", easy=0.9, medium=0.7, hard=0.4)
    _write_result(results, "b", easy=0.7, medium=0.6, hard=0.5)
    first = _individual("a", (0.60, 0.30, 100.0))
    second = _individual("b", (0.70, 0.20, 120.0))
    archive = MapElitesArchive()

    assert run_improved.update_map_elites_archive([first], archive)["inserted"] == 1
    assert run_improved.update_map_elites_archive([second], archive)["replaced"] == 1

    elite = archive.get(PromptDescriptor("base64", "zero"))
    assert elite.gene_id == "b"
    assert elite.quality == (0.70, 0.5, 0.6, 0.7, 0.20, -120.0)


def test_parent_selection_uses_isolated_archive_copies(monkeypatch):
    archive = MapElitesArchive()
    first = _individual("a", (0.6, 0.3, 100.0))
    second = _individual("b", (0.7, 0.2, 120.0))
    archive.add("a", PromptDescriptor("base64", "zero"), (0.6,), individual=first)
    archive.add("b", PromptDescriptor("transcript", "three_plus"), (0.7,), individual=second)
    monkeypatch.setattr(run_improved, "MAP_ELITES_ARCHIVE", archive)
    monkeypatch.setattr(run_improved.random, "choice", lambda values: values[-1])

    selected = run_improved.map_elites_select([first, second], 3)

    assert [individual[0] for individual in selected] == ["b", "b", "b"]
    selected[0][0] = "changed"
    assert archive.get(PromptDescriptor("transcript", "three_plus")).gene_id == "b"
    assert archive.get(PromptDescriptor("transcript", "three_plus")).individual[0] == "b"


def test_archive_is_saved_in_checkpoint(tmp_path, monkeypatch):
    archive = MapElitesArchive()
    individual = _individual("a", (0.6, 0.3, 100.0))
    archive.add("a", PromptDescriptor("base64", "zero"), (0.6,), individual=individual)
    monkeypatch.setattr(run_improved, "population", [individual], raising=False)
    monkeypatch.setattr(run_improved, "hof", [], raising=False)
    monkeypatch.setattr(run_improved, "MAP_ELITES_ARCHIVE", archive)

    run_improved.save_checkpoint(2, folder_name=str(tmp_path), global_path=None)

    with (tmp_path / "checkpoint_gen_2.pkl").open("rb") as file:
        restored = pickle.load(file)
    restored_elite = restored["map_elites_archive"].get(PromptDescriptor("base64", "zero"))
    assert restored_elite.gene_id == "a"


def test_summary_reports_coverage(tmp_path, monkeypatch):
    archive = MapElitesArchive()
    individual = _individual("a", (0.6, 0.3, 100.0))
    descriptor = PromptDescriptor("base64", "zero")
    archive.add("a", descriptor, (0.6,), fitness=individual.fitness.values, individual=individual)
    monkeypatch.setenv("JB_GENE_RESULTS_DIR", str(tmp_path))

    output = run_improved.write_map_elites_summary(archive, generation=3, run_id="run")

    payload = json.loads((tmp_path / "run" / "map_elites.json").read_text())
    assert output == str(tmp_path / "run" / "map_elites.json")
    assert payload["generation"] == 3
    assert payload["occupied_cells"] == 1
    assert payload["capacity"] == 18
    assert payload["elites"][0]["gene_id"] == "a"
