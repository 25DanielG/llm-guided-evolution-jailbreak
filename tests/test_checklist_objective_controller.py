"""Controller side of the opt-in checklist objective (JB_CHECKLIST_OBJECTIVE=1); no inference."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_improved as evolution

pytestmark = pytest.mark.mock_inference
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEADER = "mean_judge_score,diversity,avg_prompt_tokens,easy_score,medium_score,hard_score"
CHECKLIST_HEADER = "mean_judge_score,diversity,avg_prompt_tokens,checklist_score,easy_score,medium_score,hard_score"


def _constants(value):
    env = {k: v for k, v in os.environ.items() if k != "JB_CHECKLIST_OBJECTIVE"}
    if value is not None:
        env["JB_CHECKLIST_OBJECTIVE"] = value
    return subprocess.run([sys.executable, "-c", "from src.cfg import constants as c; "
                           "print(c.FITNESS_WEIGHTS, len(c.INVALID_FITNESS_MAX), len(c.PLACEHOLDER_FITNESS))"],
                          cwd=ROOT, env=env, capture_output=True, text=True)


@pytest.mark.parametrize("value, weights", [(None, (1.0, 1.0, -1.0)), ("0", (1.0, 1.0, -1.0)),
                                            ("1", (1.0, 1.0, -1.0, 1.0))])
def test_fitness_weights_follow_the_flag(value, weights):
    result = _constants(value)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{weights} {len(weights)} {len(weights)}"


def test_bad_flag_stops_the_controller():
    result = _constants("true")
    assert result.returncode != 0 and "JB_CHECKLIST_OBJECTIVE must be '0' or '1'" in result.stderr


@pytest.fixture
def gene_results(tmp_path, monkeypatch):
    monkeypatch.setenv("JB_GENE_RESULTS_DIR", str(tmp_path))
    def run(flag, header, row):
        monkeypatch.setattr(evolution, "CHECKLIST_OBJECTIVE", flag)
        monkeypatch.setattr(evolution, "FITNESS_WEIGHTS", (1.0, 1.0, -1.0) + ((1.0,) if flag == "1" else ()))
        monkeypatch.setattr(evolution, "GLOBAL_DATA", {"gene": {"results_job": 1, "local_output": "job done"}})
        (tmp_path / "gene_results.csv").write_text(f"{header}\n{row}\n")
        evolution.check4results("gene")
        return evolution.GLOBAL_DATA["gene"]["fitness"]
    return run


def test_checklist_is_the_fourth_fitness_value(gene_results):
    assert gene_results("1", CHECKLIST_HEADER, "0.5,0.0,12.0,0.3,0.8,0.5,0.2") == (0.5, 0.0, 12.0, 0.3)


def test_flag_on_without_checklist_column_fails_loudly(gene_results):
    # an eval job that never got the flag would otherwise hand easy_score to NSGA-II as the 4th objective
    with pytest.raises(RuntimeError, match="no checklist_score column"):
        gene_results("1", DEFAULT_HEADER, "0.5,0.0,12.0,0.8,0.5,0.2")


@pytest.mark.parametrize("header, row", [(DEFAULT_HEADER, "0.5,0.0,12.0,0.8,0.5,0.2"),
                                         (CHECKLIST_HEADER, "0.5,0.0,12.0,0.3,0.8,0.5,0.2")])
def test_flag_off_keeps_three_objectives(gene_results, header, row):
    assert gene_results("0", header, row) == (0.5, 0.0, 12.0)


def test_resume_stops_when_the_checkpoint_has_another_objective_count(monkeypatch):
    # DEAP unpickles each individual with the weights it was saved with, so a 3-objective checkpoint
    # loaded into a 4-objective run looks like this
    class SavedFitness(evolution.base.Fitness):
        weights = (1.0, 1.0, -1.0)
    saved = [SimpleNamespace(fitness=SavedFitness())]
    monkeypatch.setattr(evolution, "FITNESS_WEIGHTS", (1.0, 1.0, -1.0, 1.0))
    with pytest.raises(ValueError, match="checkpoint has 3 objectives but this run has 4"):
        evolution.check_objective_count(saved)
    evolution.check_objective_count([])  # nothing restored, nothing to compare
    monkeypatch.setattr(evolution, "FITNESS_WEIGHTS", (1.0, 1.0, -1.0))
    evolution.check_objective_count(saved)  # same count resumes


def _sbatch_prefix(tmp_path, **env_overrides):
    """Run run_jailbreak.sbatch's setup up to the first service start (as test_shared_model_serving does)."""
    script = (ROOT / "run_jailbreak.sbatch").read_text().split("# Retry a service's own startup")[0]
    script = 'module() { return 0; }; python3() { "$REAL_PYTHON" "$@"; }\n' + script
    env = {k: v for k, v in os.environ.items() if not k.startswith("JB_")}
    env.update(REAL_PYTHON=sys.executable, SLURM_SUBMIT_DIR=str(ROOT), SLURM_JOB_ID="cpu-test", TMPDIR=str(tmp_path),
               CUDA_VISIBLE_DEVICES="GPU-a,GPU-b,GPU-c,GPU-d,GPU-e", **env_overrides)
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=30)


def test_sbatch_flag_off_adds_no_output(tmp_path):
    unset, zero = _sbatch_prefix(tmp_path), _sbatch_prefix(tmp_path, JB_CHECKLIST_OBJECTIVE="0")
    assert unset.returncode == zero.returncode == 0, unset.stderr
    assert (unset.stdout, unset.stderr) == (zero.stdout, zero.stderr)
    assert "hecklist" not in unset.stdout + unset.stderr


@pytest.mark.parametrize("value", ["true", "", "2"])
def test_sbatch_rejects_bad_values(tmp_path, value):
    run = _sbatch_prefix(tmp_path, JB_CHECKLIST_OBJECTIVE=value)
    assert run.returncode != 0 and "JB_CHECKLIST_OBJECTIVE must be 0 or 1" in run.stderr


def test_sbatch_flag_on_uses_the_70b_judge_and_warns_about_the_selector(tmp_path):
    run = _sbatch_prefix(tmp_path, JB_CHECKLIST_OBJECTIVE="1")
    assert run.returncode == 0, run.stderr
    assert "Checklist objective: on" in run.stdout
    assert "Llama-3.3-70B-Instruct" in run.stdout
    assert "only steers parent selection" in run.stderr
    nsga2 = _sbatch_prefix(tmp_path, JB_CHECKLIST_OBJECTIVE="1", JB_SELECTION="NSGA2")
    assert nsga2.returncode == 0 and "only steers parent selection" not in nsga2.stderr


def test_sbatch_flag_on_refuses_a_non_70b_judge_profile(tmp_path):
    profile = dict(JB_VLLM_BIN="/fake/bin/vllm", JB_CUDA_MODULE="fake-cuda",
                   JB_JUDGE_PROMPT="rubric")
    run = _sbatch_prefix(tmp_path, JB_CHECKLIST_OBJECTIVE="1", JB_RUN_JUDGE_PROFILE="gpt-oss-20b", **profile)
    assert run.returncode != 0 and "checklist objective is validated only with llama-3.3-70b" in run.stderr
    ok = _sbatch_prefix(tmp_path, JB_CHECKLIST_OBJECTIVE="1", JB_RUN_JUDGE_PROFILE="llama-3.3-70b", **profile)
    assert "checklist objective is validated only" not in ok.stderr


def test_checkpoint_identity_only_changes_when_flag_is_on(monkeypatch):
    monkeypatch.setattr(evolution, "CHECKLIST_OBJECTIVE", "0")
    off = evolution.checkpoint_identity()
    assert "JB_CHECKLIST_OBJECTIVE" not in off["environment"]  # checkpoints from before the flag still resume
    monkeypatch.setattr(evolution, "CHECKLIST_OBJECTIVE", "1")
    on = evolution.checkpoint_identity()
    assert on["environment"] == {**off["environment"], "JB_CHECKLIST_OBJECTIVE": "1"}
    assert on != off  # so a run never resumes across a change in the number of objectives
