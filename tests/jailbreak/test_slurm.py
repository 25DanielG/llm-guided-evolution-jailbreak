from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_generated_slurm_resources_and_entrypoints():
    run_script = (ROOT / "run.sh").read_text()
    eval_template = (ROOT / "slurm-config/slurm_config.yaml").read_text()
    vllm_script = (ROOT / "jailbreak_vllm.sbatch").read_text()
    server_script = (ROOT / "server.sh").read_text()

    assert "run_improved.py jailbreak_test" in run_script
    assert "JB_JUDGE_MODE" in run_script
    assert "#SBATCH --gres=gpu" not in eval_template.split("python_bash_script:", 1)[1].split("llm_bash_script:", 1)[0]
    assert vllm_script.count("#SBATCH --gres=gpu:2") == 1
    assert 'H200|H100|A100' in vllm_script
    assert 'H200|H100' in server_script
    assert "SUBMIT_ISLAND_CONTROLLER:-0" in server_script
