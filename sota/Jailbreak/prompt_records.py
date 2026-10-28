"""Auditable prompt observations and population manifests; no fitness dependencies."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path

SERIALIZATION_VERSION = "role-content-v1"


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def serialize_messages(messages) -> str:
    """Keep target-visible content, including intentional fences, unchanged."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected a nonempty list of text chat messages")
    parts = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            raise ValueError("Only text messages with string role and content are supported")
        parts.append(f"[{message['role']}]\n{message['content']}")
    return "\n".join(parts)


def load_cases(path) -> tuple[list[dict], str]:
    path = Path(path)
    fingerprint = digest(path.read_bytes())
    cases = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        column = next((c for c in ("behavior", "goal", "prompt", "text") if c in (reader.fieldnames or [])), None)
        if column is None:
            raise ValueError(f"No behavior column found in {path}")
        for row_index, row in enumerate(reader):
            text = (row.get(column) or "").strip()
            if not text:
                continue
            case_id = row.get("id") or row.get("case_id") or "text:" + digest(text)
            if case_id in seen:
                # Preserve previously accepted duplicate evaluation rows without
                # conflating their identity in reports or changing fitness inputs.
                base = case_id
                case_id = f"{base}:row:{row_index}"
                while case_id in seen:
                    case_id += ":duplicate"
            seen.add(case_id)
            cases.append({"case_id": case_id, "behavior": text})
    if not cases:
        raise ValueError(f"No behaviors loaded from {path}")
    return cases, fingerprint


def checkpoint_fingerprint(path) -> str | None:
    """Hash configuration/tokenizer assets, rather than multi-GB model weights."""
    if not path or not Path(path).is_dir():
        return None
    try:
        files = sorted(p for p in Path(path).iterdir() if p.is_file() and
                       (p.name.startswith(("tokenizer", "vocab", "merges", "special_tokens", "added_tokens")) or p.name == "config.json"))
        if not files:
            return None
        h = hashlib.sha256()
        for file in files:
            h.update(file.name.encode())
            h.update(file.read_bytes())
        return h.hexdigest()
    except OSError:
        # A reporting fingerprint must not make an otherwise usable target fail.
        return None


def target_context(model_name, model_path=None) -> dict:
    return {"target_model": model_name, "target_checkpoint": str(model_path) if model_path else None,
            "target_fingerprint": checkpoint_fingerprint(model_path)}


def append_jsonl(path, rows):
    """Write one entire evaluation attempt under a lock, before target inference."""
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.write(payload)
        file.flush()
        fcntl.flock(file, fcntl.LOCK_UN)


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_SH)
        return [json.loads(line) for line in file if line.strip()]


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = file.name
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def latest_attempts(rows, generation):
    """Never fill failed/missing cases from an earlier evaluation attempt."""
    latest = {}
    for row in rows:
        if row["generation"] > generation:
            continue
        candidate = row["candidate_id"]
        old = latest.get(candidate)
        if old is None or row["generation"] > old["generation"] or (
                row["generation"] == old["generation"] and row["evaluation_id"] != old["evaluation_id"]):
            latest[candidate] = {"generation": row["generation"], "evaluation_id": row["evaluation_id"], "cases": {}}
        attempt = latest[candidate]
        if attempt["evaluation_id"] == row["evaluation_id"]:
            attempt["cases"][row["case_id"]] = row
    return latest


def population_manifest(run_id, generation, cohorts, cases, dataset_fingerprint, context, observations):
    latest = latest_attempts(observations, generation)
    result = {"schema_version": 1, "run_id": run_id, "generation": generation,
              "dataset_fingerprint": dataset_fingerprint, "target_context": context,
              "expected_cases": cases, "cohorts": {}}
    for cohort, candidates in cohorts.items():
        ids = sorted(set(candidates))
        references = []
        for candidate in ids:
            attempt = latest.get(candidate, {})
            for case in cases:
                row = attempt.get("cases", {}).get(case["case_id"])
                compatible = row is not None and row["dataset_fingerprint"] == dataset_fingerprint and row["target_context"] == context
                references.append({"candidate_id": candidate, "case_id": case["case_id"],
                                   "record_id": row["record_id"] if compatible else None,
                                   "status": row["status"] if compatible else ("context_mismatch" if row else "missing_prompt")})
        result["cohorts"][cohort] = {"strategy_ids": ids, "references": references}
    return result


def save_population_manifest(run_dir, run_id, generation, cohorts, cases, dataset_fingerprint, context):
    run_dir = Path(run_dir)
    path = run_dir / "manifests" / f"generation_{generation:04d}.json"
    # Completed snapshots are immutable, including on resume.
    if path.exists():
        return path
    manifest = population_manifest(run_id, generation, cohorts, cases, dataset_fingerprint,
                                   context, read_jsonl(run_dir / "prompt_records.jsonl"))
    atomic_json(path, manifest)
    return path
