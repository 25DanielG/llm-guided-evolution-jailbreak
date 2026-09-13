"""Preflight checks for the local test suite and PACE deployment."""

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import jb_config
import vllm_client
from src.cfg import constants


def require(condition, message, errors):
    if condition:
        print(f"PASS: {message}")
    else:
        print(f"FAIL: {message}")
        errors.append(message)


def local_checks():
    errors = []
    require(Path(constants.SOTA_ROOT).resolve() == HERE, "jailbreak constants are active", errors)
    require(Path(constants.SEED_NETWORK).is_file(), "seed prompt module exists", errors)
    require(bool(list((ROOT / "templates/Jailbreak/traits").glob("*.txt"))), "trait templates exist", errors)
    require(Path(jb_config.behaviors_path()).is_file(), "behavior CSV exists", errors)
    require(jb_config.n_behaviors_per_eval() == 4, "pilot behavior count defaults to four", errors)
    require(jb_config.max_concurrency() == 2, "pilot concurrency defaults to two", errors)
    require(jb_config.judge_mode() == "llm", "LLM judge mode is active", errors)
    require(jb_config.judge_model_name() == "judge", "judge served name is consistent", errors)
    for relative in ("run.sh", "server.sh", "jailbreak_vllm.sbatch", "slurm-config/slurm_config.yaml"):
        require((ROOT / relative).is_file(), f"generated artifact exists: {relative}", errors)
    result_dir = HERE / "results"
    result_dir.mkdir(exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=result_dir):
            pass
        require(True, "results directory is writable", errors)
    except OSError:
        require(False, "results directory is writable", errors)
    spec = importlib.util.spec_from_file_location("preflight_prompt", constants.SEED_NETWORK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(hasattr(module, "PromptStrategy"), "seed exports PromptStrategy", errors)
    return errors


def pace_checks():
    errors = local_checks()
    try:
        jb_config.api_key()
        require(True, "JB_VLLM_API_KEY is set", errors)
    except RuntimeError:
        require(False, "JB_VLLM_API_KEY is set", errors)
    vllm_bin = os.getenv("JB_VLLM_BIN", "vllm")
    require(bool(shutil.which(vllm_bin) or Path(vllm_bin).is_file()), "vLLM executable is available", errors)
    for label, model_path in (
        ("mutator", constants.MODEL_PATH),
        ("target", constants.TARGET_MODEL_PATH),
        ("judge", constants.JUDGE_MODEL_PATH),
    ):
        model_dir = Path(model_path)
        require(model_dir.is_dir() and (model_dir / "config.json").is_file(), f"{label} model is readable", errors)
    for label, path in (
        ("target", Path(jb_config.TARGET_HOST_FILE)),
        ("judge", Path(jb_config.JUDGE_HOST_FILE)),
    ):
        require(path.is_file() and bool(path.read_text().strip()), f"{label} host file exists", errors)
    if not errors:
        require(vllm_client.wait_ready(jb_config.target_base_url(), timeout=15, check_interval=2), "target endpoint is ready", errors)
        require(vllm_client.wait_ready(jb_config.judge_base_url(), timeout=15, check_interval=2), "judge endpoint is ready", errors)
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("local", "pace"), default="local")
    args = parser.parse_args()
    errors = pace_checks() if args.mode == "pace" else local_checks()
    if errors:
        raise SystemExit(f"Preflight failed with {len(errors)} error(s)")
    print(f"{args.mode.upper()} preflight passed")


if __name__ == "__main__":
    main()
