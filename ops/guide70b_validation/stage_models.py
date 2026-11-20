"""Stage only the indexed files required by the fixed VIP serving validation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import os

GIB = 1024 ** 3
MARGIN_BYTES = 32 * GIB
CHECKPOINTS = {
    "judge": ("meta-llama/Llama-3.3-70B-Instruct", Path(
        "/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct"
    )),
    "target": ("openai/gpt-oss-20b", Path(
        "/storage/ice1/8/3/pablo3/pr3-models/openai/gpt-oss-20b"
    )),
    "reporter": ("cais/HarmBench-Llama-2-13b-cls", Path(
        "/storage/ice1/8/3/pablo3/pr3-models/cais/HarmBench-Llama-2-13b-cls"
    )),
    "mutator": ("meta-llama/Llama-3.1-8B-Instruct", Path(
        "/storage/ice1/8/3/pablo3/pr3-models/meta-llama/Llama-3.1-8B-Instruct"
    )),
}
INDEX = "model.safetensors.index.json"
REQUIRED = (INDEX, "config.json", "tokenizer_config.json")
OPTIONAL = (
    "tokenizer.json", "tokenizer.model", "special_tokens_map.json",
    "generation_config.json", "chat_template.jinja", "added_tokens.json",
    "deployment.json",
)


def emit(event, **fields):
    print(json.dumps({
        "utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": event,
        **fields,
    }, allow_nan=False), flush=True)


def mount_of(path):
    output = subprocess.check_output(
        ["findmnt", "--json", "--target", str(path), "--output", "TARGET,SOURCE,FSTYPE"],
        text=True, timeout=15,
    )
    return json.loads(output)["filesystems"][-1]


def files_for(source):
    raw = (source / INDEX).read_bytes()
    weights = sorted(set(json.loads(raw)["weight_map"].values()))
    if not weights or any(Path(name).name != name or not name.endswith(".safetensors")
                          for name in weights):
        raise ValueError(f"Invalid indexed weight shard list in {source / INDEX}")
    files = (*weights, *REQUIRED, *(name for name in OPTIONAL if (source / name).is_file()))
    if not {"tokenizer.json", "tokenizer.model"}.intersection(files):
        raise ValueError(f"No tokenizer assets in {source}")
    sizes = {}
    for name in files:
        path = source / name
        if not path.is_file() or (size := path.stat().st_size) <= 0:
            raise ValueError(f"Missing, empty, or non-regular checkpoint file {path}")
        sizes[name] = size
    return weights, sizes, hashlib.sha256(raw).hexdigest()


def inventory():
    result = {}
    for role, (model_id, source) in CHECKPOINTS.items():
        source = source.resolve(strict=True)
        weights, sizes, index_sha256 = files_for(source)
        result[role] = {
            "model_id": model_id, "source": str(source), "source_mount": mount_of(source),
            "index_sha256": index_sha256, "weight_shards": weights, "sizes": sizes,
            "bytes": sum(sizes.values()),
        }
    return result


def check_local_disk(path, required):
    path = path.resolve(strict=True)
    mount = mount_of(path)
    free = shutil.disk_usage(path).free
    if mount["fstype"] not in {"xfs", "ext4", "btrfs"} or not mount["source"].startswith("/dev/"):
        raise ValueError(f"Job scratch is not a verified disk-backed local filesystem: {path} {mount}")
    if free < required:
        raise ValueError(f"Job scratch has {free} bytes free at {path}, requires {required}")
    return mount, free


def copy_checkpoint(item, destination):
    source = Path(item["source"])
    if destination.parent.stat().st_dev == source.stat().st_dev:
        raise ValueError(f"Staging destination still uses source filesystem: {destination}")
    destination.mkdir(parents=True)
    started = time.monotonic()
    for name, expected in item["sizes"].items():
        original = (source / name).resolve(strict=True)
        if destination.stat().st_dev == original.stat().st_dev:
            raise ValueError(f"Checkpoint file is already on the destination filesystem: {original}")
        before = original.stat()
        if before.st_size != expected:
            raise ValueError(f"Checkpoint changed before copy: {original}")
        copied = destination / name
        shutil.copyfile(original, copied)
        with copied.open("rb") as file:
            os.fsync(file.fileno())
        after = original.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ino) != (
            before.st_size, before.st_mtime_ns, before.st_ino
        ):
            raise ValueError(f"Checkpoint changed during copy: {original}")
        if copied.is_symlink() or copied.resolve(strict=True).parent != destination.resolve():
            raise ValueError(f"Staged checkpoint escapes its local directory: {copied}")
        if copied.stat().st_size != expected:
            raise ValueError(f"Staged file size differs: {copied}")
        if name in item["weight_shards"]:
            emit("staged_shard", model=item["model_id"], shard=name, bytes=expected,
                 elapsed_seconds=round(time.monotonic() - started, 3))
    weights, sizes, index_hash = files_for(destination)
    if (weights, sizes, index_hash) != (item["weight_shards"], item["sizes"], item["index_sha256"]):
        raise ValueError(f"Staged checkpoint differs from source inventory: {destination}")
    return round(time.monotonic() - started, 3)


def stage(destination, summary):
    started = time.monotonic()
    models = inventory()
    total = sum(item["bytes"] for item in models.values())
    mount, free = check_local_disk(destination, total + MARGIN_BYTES)
    emit("staging_started", path=str(destination), mount=mount, free_bytes=free,
         checkpoint_bytes=total, margin_bytes=MARGIN_BYTES)
    for role, item in models.items():
        local = destination / role / Path(item["model_id"]).name
        local.parent.mkdir()
        item["staged_path"] = str(local)
        item["staging_seconds"] = copy_checkpoint(item, local)
        if mount_of(local) != mount:
            raise ValueError(f"Local checkpoint mount changed: {local}")
        emit("staged_checkpoint", role=role, model_id=item["model_id"],
             seconds=item["staging_seconds"], shards=len(item["weight_shards"]))
    seconds = round(time.monotonic() - started, 3)
    summary.write_text(json.dumps({
        "staging_seconds": seconds, "source_payload_bytes": total,
        "local_mount": mount, "local_bytes_free_before": free, "models": models,
    }, indent=2) + "\n")
    emit("staging_complete", seconds=seconds, bytes=total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    stage(args.destination, args.summary)
