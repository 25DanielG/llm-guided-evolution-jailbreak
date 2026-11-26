"""Shared, manually maintained serving profiles. No torch/vLLM import on CPU.

Run from the repository root: python -m src.model_serving --help.
Launches only staged local checkpoints; never downloads weights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex

REGISTRY = Path(__file__).resolve().parents[1] / "config" / "served_models.json"
PACE_REGISTRY = REGISTRY.parents[1] / "scripts" / "serving" / "pace" / "config" / "models.json"


def registry():
    return json.loads(REGISTRY.read_text())


def profile(name):
    models = registry()["models"]
    if name in models:
        return models[name]
    for item in models.values():
        if item["model_id"] == name:
            return item
    raise ValueError(f"Unknown model {name!r}; choose from {', '.join(models)}")


def endpoint_profile(name):
    """Resolve a model already running behind an OpenAI-compatible endpoint."""
    try:
        return profile(name)
    except ValueError:
        pace = json.loads(PACE_REGISTRY.read_text())
        for alias, item in pace.items():
            if name in (alias, item["repo"]):
                # PACE serves by alias. Its template and parser choices
                # belong to the guarded launcher, separate from GPT-OSS.
                return {"model_id": alias}
        raise


def model_path(name):
    item = profile(name)
    if item.get("checkpoint_env"):
        configured = os.getenv(item["checkpoint_env"])
        if configured:
            return str(Path(configured).expanduser())
    root = os.getenv("JB_SHARED_MODEL_ROOT", registry()["shared_model_root"])
    return str(Path(root) / item["model_id"])


def command(name, path=None, served_name=None, port=8001, host="0.0.0.0", tp=None,
            max_len=None):
    """Build generic vLLM arguments for inspection or an already checked launch."""
    item = profile(name)
    path = path or model_path(name)
    return [os.getenv("JB_VLLM_BIN", "vllm"), "serve", path,
            "--served-model-name", served_name or item.get("served_model_name", item["model_id"]),
            "--host", host, "--port", str(port),
            "--tensor-parallel-size", str(tp or item["tensor_parallel_size"]),
            "--max-model-len", str(max_len or item["max_model_len"]),
            "--max-num-seqs", str(item["max_num_seqs"]),
            "--gpu-memory-utilization", "0.9", *item["vllm_args"]]


def guarded_launcher_message(name):
    return (f"{name} requires the qualified guarded PACE Inkling launcher. "
            "Generic serving does not apply its worker and kernel policy.")


def generic_launch_error(name, item):
    if item.get("smoke_mode", "generic") != "generic":
        return guarded_launcher_message(name)
    return None


def check_legacy_launch_source(source):
    """Check a raw vLLM model identifier or local directory before launch."""
    for name, item in registry()["models"].items():
        if source in (name, item["model_id"], item.get("served_model_name")):
            error = generic_launch_error(name, item)
            if error:
                raise ValueError(error)
            break

    # A raw path can point to Inkling under a different directory or symlink.
    # Stock and Unbound share this guarded architecture and checkpoint config.
    try:
        config = json.loads((Path(source) / "config.json").read_text())
    except (OSError, ValueError):
        return  # The existing launcher handles missing or malformed checkpoints.
    if isinstance(config, dict) and config.get("model_type") == "inkling2_loop":
        raise ValueError(guarded_launcher_message(source))


def validate_checkpoint(name, path):
    item = profile(name)
    path = Path(path)
    native_mistral = "mistral" in item["vllm_args"]
    config_name = "params.json" if native_mistral else "config.json"
    if not path.is_dir() or not (path / config_name).is_file():
        raise ValueError(f"Checkpoint not staged at {path}; ask the shared storage maintainer")
    if native_mistral and not (path / "tekken.json").is_file():
        raise ValueError(f"Native Mistral tokenizer tekken.json missing at {path}")
    weight_pattern = "consolidated*.safetensors" if native_mistral else "*.safetensors"
    if not any(path.glob(weight_pattern)):
        raise ValueError(f"No safetensors weights at {path}")
    if item.get("checkpoint_manifest"):
        manifest_path = REGISTRY.parent / item["checkpoint_manifest"]
        if not manifest_path.is_file():
            raise ValueError(f"Missing checkpoint integrity manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("revision") != item.get("revision") or manifest.get("repo_id") != item["model_id"]:
            raise ValueError("Checkpoint integrity manifest identity does not match registry")
        for filename, expected in manifest["files"].items():
            file_path = path / filename
            if not file_path.is_file():
                raise ValueError(f"Checkpoint file missing: {file_path}")
            if file_path.stat().st_size != expected["bytes"]:
                raise ValueError(f"Checkpoint file missing or size mismatch: {file_path}")
            with file_path.open("rb") as checkpoint_file:
                if hasattr(hashlib, "file_digest"):
                    digest = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
                else:  # Python < 3.11 fallback; bounded to 1 MiB per read.
                    hasher = hashlib.sha256()
                    for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                        hasher.update(chunk)
                    digest = hasher.hexdigest()
            if digest != expected["sha256"]:
                raise ValueError(f"Checkpoint SHA-256 mismatch: {file_path}")
    if "revision" in item and not item.get("checkpoint_manifest"):
        manifest = path / "deployment.json"
        if not manifest.is_file():
            raise ValueError(f"Missing {manifest}; record model_id and revision after staging")
        staged = json.loads(manifest.read_text())
        if staged.get("model_id") != item["model_id"] or staged.get("revision") != item["revision"]:
            raise ValueError(f"{manifest} does not match the registry model/revision")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "path", "tp", "check", "launch-check", "command", "serve"],
                        help="command prints arguments for inspection; launch-check validates generic launch eligibility")
    parser.add_argument("--model")
    parser.add_argument("--model-path")
    parser.add_argument("--launch-source", action="append", default=[],
                        help="raw model ID or local path for launch-check; repeat for multiple sources")
    parser.add_argument("--served-name")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--tp", type=int)
    parser.add_argument("--max-len", type=int)
    args = parser.parse_args()
    if args.action == "list":
        print(json.dumps(registry(), indent=2))
        return
    try:
        if args.action == "launch-check":
            if not args.model and not args.launch_source:
                raise ValueError("launch-check requires --model or --launch-source")
            if args.model:
                error = generic_launch_error(args.model, profile(args.model))
                if error:
                    raise ValueError(error)
            for source in args.launch_source:
                check_legacy_launch_source(source)
            return
        if not args.model:
            raise ValueError("--model is required")
        item = profile(args.model)
        launch_error = generic_launch_error(args.model, item)
        if args.action == "serve" and launch_error:
            raise ValueError(launch_error)
        path = args.model_path or model_path(args.model)
        if args.action == "path":
            print(path)
        elif args.action == "tp":
            print(item["tensor_parallel_size"])
        elif args.action == "check":
            validate_checkpoint(args.model, path)
            print(f"Staged checkpoint metadata matches: {path}")
        else:
            if (args.max_len is not None and args.max_len < 1) or (args.tp is not None and args.tp < 1):
                raise ValueError("--max-len and --tp must be positive")
            cmd = command(args.model, path, args.served_name, args.port, args.host, args.tp, args.max_len)
            if args.action == "command":
                if launch_error:
                    print(f"# Inspection only. {launch_error}")
                    # Keep every line a comment, even if an argument contains a newline.
                    print("# " + shlex.join(cmd).replace("\n", "\\n"))
                else:
                    print(shlex.join(cmd))
                return
            check_legacy_launch_source(path)
            if not os.getenv("SLURM_JOB_ID"):
                raise ValueError("serve requires a Slurm GPU allocation")
            devices = [d for d in os.getenv("CUDA_VISIBLE_DEVICES", "").split(",") if d and d != "-1"]
            if len(devices) < (args.tp or item["tensor_parallel_size"]):
                raise ValueError("CUDA_VISIBLE_DEVICES has fewer GPUs than the profile requires")
            validate_checkpoint(args.model, path)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            print(shlex.join(cmd), flush=True)
            os.execvp(cmd[0], cmd)
    except ValueError as err:
        parser.error(str(err))


if __name__ == "__main__":
    main()
