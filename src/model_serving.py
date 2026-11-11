"""Shared, manually maintained serving profiles. No torch/vLLM import on CPU.

Run from the repository root: python -m src.model_serving --help.
Launches only staged local checkpoints; never downloads weights.
"""
import argparse
import json
import os
from pathlib import Path
import shlex

REGISTRY = Path(__file__).resolve().parents[1] / "config" / "served_models.json"


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


def model_path(name):
    root = os.getenv("JB_SHARED_MODEL_ROOT", registry()["shared_model_root"])
    return str(Path(root) / profile(name)["model_id"])


def command(name, path=None, served_name=None, port=8001, host="0.0.0.0", tp=None,
            max_len=None):
    item = profile(name)
    path = path or model_path(name)
    return [os.getenv("JB_VLLM_BIN", "vllm"), "serve", path,
            "--served-model-name", served_name or item["model_id"],
            "--host", host, "--port", str(port),
            "--tensor-parallel-size", str(tp or item["tensor_parallel_size"]),
            "--max-model-len", str(max_len or item["max_model_len"]),
            "--max-num-seqs", str(item["max_num_seqs"]),
            "--gpu-memory-utilization", "0.9", *item["vllm_args"]]


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
    if "revision" in item:
        manifest = path / "deployment.json"
        if not manifest.is_file():
            raise ValueError(f"Missing {manifest}; record model_id and revision after staging")
        staged = json.loads(manifest.read_text())
        if staged.get("model_id") != item["model_id"] or staged.get("revision") != item["revision"]:
            raise ValueError(f"{manifest} does not match the registry model/revision")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "path", "tp", "check", "command", "serve"])
    parser.add_argument("--model")
    parser.add_argument("--model-path")
    parser.add_argument("--served-name")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--tp", type=int)
    parser.add_argument("--max-len", type=int)
    args = parser.parse_args()
    if args.action == "list":
        print(json.dumps(registry(), indent=2))
        return
    if not args.model:
        parser.error("--model is required")
    try:
        item = profile(args.model)
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
                print(shlex.join(cmd))
                return
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
