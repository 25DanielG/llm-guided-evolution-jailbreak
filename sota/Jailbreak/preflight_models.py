"""Fail before GPU startup if a configured model has missing weight shards."""

import json
import sys
from pathlib import Path


def check_model(path_string):
    path = Path(path_string)
    if not (path / "config.json").is_file():
        raise ValueError(f"Missing model config: {path / 'config.json'}")

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if index_path.is_file():
            with index_path.open() as handle:
                filenames = set(json.load(handle)["weight_map"].values())
            missing = [name for name in sorted(filenames) if not (path / name).is_file()
                       or (path / name).stat().st_size == 0]
            if missing:
                raise ValueError(f"Incomplete model at {path}: missing weight shards {missing}")
            print(f"Model preflight passed: {path} ({len(filenames)} weight shards)", flush=True)
            return

    if any((path / name).is_file() and (path / name).stat().st_size > 0
           for name in ("model.safetensors", "pytorch_model.bin")):
        print(f"Model preflight passed: {path} (single weight file)", flush=True)
        return
    raise ValueError(f"No model weights found at {path}")


if __name__ == "__main__":
    for model_path in dict.fromkeys(sys.argv[1:]):
        check_model(model_path)
