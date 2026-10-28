"""Case-matched prompt diversity, population reconstruction, and SBERT pooling."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np

from sota.Jailbreak.prompt_records import (
    SERIALIZATION_VERSION, atomic_json, digest, latest_attempts, read_jsonl, serialize_messages,
)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_EMBEDDING_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
POOLING_VERSION = "nonoverlapping-token-weighted-normalized-chunks-v1"


def ngrams(tokens, n=3):
    if n < 1:
        raise ValueError("n-gram size must be positive")
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard_distance(a, b):
    union = a | b
    if not union:
        raise ValueError("Empty n-gram sets are unavailable")
    return 1 - len(a & b) / len(union)


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("Embedding must be a finite vector")
    norm = np.linalg.norm(vector)
    if not math.isfinite(float(norm)) or norm == 0:
        raise ValueError("Embedding has zero or invalid norm")
    return vector / norm


def cosine_distance(a, b):
    return 1 - float(np.clip(np.dot(normalize(a), normalize(b)), -1, 1))


def pooled_embedding(vectors, weights):
    if not weights or len(vectors) != len(weights) or any(w <= 0 for w in weights):
        raise ValueError("Each chunk requires a positive token count")
    return normalize(np.average(np.stack([normalize(v) for v in vectors]), axis=0, weights=weights))


def token_chunks(tokenizer, text, max_length):
    """Return content token IDs without decoding/re-tokenizing chunk boundaries."""
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    capacity = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity <= 0:
        raise ValueError("Model limit leaves no space for content tokens")
    if not ids:
        raise ValueError("Empty prompt")
    return [ids[i:i + capacity] for i in range(0, len(ids), capacity)]


class SentenceBertEmbedder:
    """Pinned, local model; exact token chunks bypass SBERT's default truncation."""

    def __init__(self, cache_dir, model_name=DEFAULT_EMBEDDING_MODEL, revision=None,
                 device="cpu", batch_size=8):
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
        import torch

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.cache_dir / "model_lock.json"
        lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
        # Resolve once, then retain the exact immutable revision across reruns.
        if revision is None and lock.get("model") == model_name:
            revision = lock["revision"]
        if revision is None and model_name == DEFAULT_EMBEDDING_MODEL:
            revision = DEFAULT_EMBEDDING_REVISION
        if Path(model_name).is_dir():
            model_path = Path(model_name).resolve()
            # Local checkpoints are versioned by their complete asset contents.
            import hashlib
            h = hashlib.sha256()
            for file in sorted(model_path.rglob("*")):
                if file.is_file():
                    h.update(str(file.relative_to(model_path)).encode())
                    with file.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            h.update(block)
            resolved_revision = "local-sha256:" + h.hexdigest()
            if revision is not None and revision != resolved_revision:
                raise ValueError("Local embedding checkpoint does not match requested revision")
        else:
            model_path = Path(snapshot_download(model_name, revision=revision, cache_dir=self.cache_dir / "hub",
                                               allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))
            resolved_revision = model_path.name
        self.model = SentenceTransformer(str(model_path), device=device)
        self.model.float().eval()
        self.tokenizer = self.model.tokenizer
        self.max_length = int(self.model.max_seq_length)
        self.batch_size = batch_size
        self.torch = torch
        self.metadata = {"model": model_name, "revision": resolved_revision, "max_length": self.max_length,
                         "pooling": POOLING_VERSION, "serialization": SERIALIZATION_VERSION,
                         "device": device, "dtype": "float32"}
        # Device does not affect cache identity; dtype and preprocessing do.
        self.cache_config = {k: v for k, v in self.metadata.items() if k != "device"}
        atomic_json(lock_path, {"model": model_name, "revision": resolved_revision})

    def cache_path(self, text):
        key = digest(json.dumps(self.cache_config, sort_keys=True) + "\n" + text)
        return self.cache_dir / f"{key}.json"

    def embed(self, text):
        path = self.cache_path(text)
        if path.exists():
            try:
                cached = json.loads(path.read_text())
                return normalize(cached["vector"]), cached["chunk_count"]
            except (ValueError, KeyError, json.JSONDecodeError):
                pass  # A damaged derived cache entry can be safely recomputed.
        chunks = token_chunks(self.tokenizer, text, self.max_length)
        vectors = []
        for offset in range(0, len(chunks), self.batch_size):
            batch = chunks[offset:offset + self.batch_size]
            features = [self.tokenizer.prepare_for_model(ids, add_special_tokens=True, truncation=False,
                                                       return_attention_mask=True) for ids in batch]
            inputs = self.tokenizer.pad(features, padding=True, return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                output = self.model(inputs)["sentence_embedding"].detach().cpu().float().numpy()
            vectors.extend(output)
        vector = pooled_embedding(vectors, [len(chunk) for chunk in chunks])
        atomic_json(path, {"vector": vector.tolist(), "chunk_count": len(chunks), "config": self.cache_config,
                           "prompt_hash": digest(text)})
        return vector, len(chunks)


def _historical_observations(rows):
    observations = []
    for index, row in enumerate(rows):
        try:
            text = serialize_messages(row["evolved_prompt"])
            status = "ok"
        except (KeyError, ValueError):
            text, status = None, "invalid_prompt"
        observations.append({
            **row, "record_id": f"legacy:{index}", "case_id": "text:" + digest(row["behavior"]),
            "messages": row.get("evolved_prompt"), "prompt_text": text, "status": status,
            "target_context": row.get("target_context", {"target_model": row.get("target_model")}),
        })
    return observations


def historical_cohort(rows, generation, strategy_ids, cases):
    """Use latest eligible evaluations, flag ambiguity, and forbid topic mixing."""
    by_candidate = defaultdict(list)
    for row in rows:
        if row["generation"] <= generation:
            by_candidate[row["candidate_id"]].append(row)
    references = []
    for candidate in sorted(set(strategy_ids)):
        available = by_candidate[candidate]
        latest_gen = max((row["generation"] for row in available), default=-1)
        attempt = [row for row in available if row["generation"] == latest_gen]
        # Logs with evaluation IDs identify the last attempt even when it omitted a case.
        if attempt and attempt[-1].get("evaluation_id"):
            attempt = [row for row in attempt if row.get("evaluation_id") == attempt[-1]["evaluation_id"]]
        for case in cases:
            matches = [row for row in attempt if row["case_id"] == case["case_id"]]
            signatures = {json.dumps([r["prompt_text"], r["target_context"], r.get("dataset_fingerprint")], sort_keys=True) for r in matches}
            if len(signatures) > 1:
                references.append({"candidate_id": candidate, "case_id": case["case_id"], "status": "conflicting_records"})
            elif matches:
                references.append(matches[-1])
            else:
                references.append({"candidate_id": candidate, "case_id": case["case_id"], "status": "missing_prompt"})
    return references


def load_run(run_dir, records_dir, candidates_path=None):
    """Read only raw artifacts. Existing workspace checkpoints are trusted pickles."""
    run_dir, records_dir = Path(run_dir), Path(records_dir)
    observations = read_jsonl(records_dir / "prompt_records.jsonl")
    records = {row["record_id"]: row for row in observations}
    cohorts = []
    manifest_generations = set()
    notes = []
    for path in sorted((records_dir / "manifests").glob("generation_*.json")):
        manifest = json.loads(path.read_text())
        generation = manifest["generation"]
        manifest_generations.add(generation)
        for name, cohort in manifest["cohorts"].items():
            resolved = []
            for ref in cohort["references"]:
                row = records.get(ref.get("record_id"))
                compatible = row is not None and row["candidate_id"] == ref["candidate_id"] and row["case_id"] == ref["case_id"] and row["target_context"] == manifest["target_context"] and row["dataset_fingerprint"] == manifest["dataset_fingerprint"]
                missing_status = ref["status"] if ref["status"] != "ok" else "missing_record"
                resolved.append(row if compatible else {**ref, "status": missing_status if row is None else "context_mismatch"})
            cohorts.append({"generation": generation, "cohort": name, "strategy_ids": cohort["strategy_ids"],
                            "cases": manifest["expected_cases"], "prompts": resolved, "source": "manifest",
                            "target_context": manifest["target_context"]})
    candidate_rows = read_jsonl(candidates_path or records_dir / "candidates.jsonl")
    legacy = _historical_observations(candidate_rows)
    # Infer historical case scope from case texts recorded anywhere in this run;
    # absent cases are unknown, and this limitation is explicit in metadata.
    case_lookup = {row["case_id"]: {"case_id": row["case_id"], "behavior": row["behavior"]} for row in legacy}
    cases = [case_lookup[key] for key in sorted(case_lookup)]
    checkpoint_dirs = [run_dir / "ckpt", run_dir / "checkpoints"]
    if run_dir.name in ("ckpt", "checkpoints"):
        checkpoint_dirs = [run_dir]
    for directory in checkpoint_dirs:
        for path in sorted(directory.glob("checkpoint_gen_*.pkl")):
            generation = int(path.stem.rsplit("_", 1)[1])
            if generation in manifest_generations:
                continue
            with path.open("rb") as file:
                checkpoint = pickle.load(file)
            ids = sorted({ind[0] for ind in checkpoint["population"]})
            if observations:
                # Reconstruct interrupted/missing manifests from captured attempts.
                attempts = latest_attempts(observations, generation)
                captured_cases = {}
                prompts = []
                for attempt in attempts.values():
                    for case_id, row in attempt["cases"].items():
                        captured_cases[case_id] = {"case_id": case_id, "behavior": row["behavior"]}
                capture_cases = [captured_cases[key] for key in sorted(captured_cases)]
                for candidate in ids:
                    for case in capture_cases:
                        row = attempts.get(candidate, {}).get("cases", {}).get(case["case_id"])
                        prompts.append(row or {"candidate_id": candidate, "case_id": case["case_id"], "status": "missing_prompt"})
                cohort_cases = capture_cases
            else:
                prompts = historical_cohort(legacy, generation, ids, cases)
                cohort_cases = cases
            cohorts.append({"generation": generation, "cohort": "retained", "strategy_ids": ids,
                            "cases": cohort_cases, "prompts": prompts, "source": "checkpoint_reconstruction"})
            notes.append(f"Generation {generation}: retained membership from checkpoint; offspring unavailable; case scope inferred from logs.")
    if not cohorts:
        raise ValueError("No population manifests or checkpoints found; candidate logs alone do not establish retained membership")
    return sorted(cohorts, key=lambda c: (c["generation"], c["cohort"])), notes


def distance_stats(features, distance):
    if len(features) < 2:
        return None, None, 0
    nearest = [math.inf] * len(features)
    total, count = 0.0, 0
    for i, j in combinations(range(len(features)), 2):
        value = distance(features[i], features[j])
        total += value
        count += 1
        nearest[i] = min(nearest[i], value)
        nearest[j] = min(nearest[j], value)
    return total / count, sum(nearest) / len(nearest), count


def measure_cohort(cohort, tokenize, embedder=None, n=3, feature_cache=None, semantic_unavailable="disabled"):
    feature_cache = {} if feature_cache is None else feature_cache
    by_case = defaultdict(list)
    diagnostics = []
    for row in cohort["prompts"]:
        diagnostic = {"generation": cohort["generation"], "cohort": cohort["cohort"],
                      "candidate_id": row["candidate_id"], "case_id": row["case_id"],
                      "record_id": row.get("record_id"), "source_generation": row.get("generation"),
                      "prompt_hash": None, "token_count": None, "chunk_count": None,
                      "ngram_status": row["status"], "semantic_status": row["status"], "error": row.get("error")}
        features = {"ngram": None, "semantic": None,
                    "context": {"target": row.get("target_context"), "dataset": row.get("dataset_fingerprint")}}
        if row["status"] == "ok":
            text = row["prompt_text"]
            diagnostic["prompt_hash"] = digest(text)
            cached = feature_cache.get(text)
            if cached is None:
                cached = {"ngram": None, "semantic": None, "token_count": None, "chunk_count": None,
                          "ngram_status": "ok", "semantic_status": semantic_unavailable, "error": None}
                try:
                    tokens = tokenize(text)
                    cached["token_count"] = len(tokens)
                    if len(tokens) < n:
                        cached["ngram_status"] = "short_prompt"
                    else:
                        cached["ngram"] = ngrams(tokens, n)
                except Exception as exc:
                    cached["ngram_status"], cached["error"] = "tokenizer_error", str(exc)
                if embedder is not None:
                    try:
                        if not any(m.get("content", "").strip() for m in row.get("messages", [])):
                            raise ValueError("Empty prompt content")
                        vector, chunks = embedder.embed(text)
                        cached["semantic"] = normalize(vector)
                        cached["chunk_count"] = chunks
                        cached["semantic_status"] = "ok"
                    except Exception as exc:
                        cached["semantic_status"], cached["error"] = "embedding_error", str(exc)
                feature_cache[text] = cached
            features.update({metric: cached[metric] for metric in ("ngram", "semantic")})
            diagnostic.update({key: cached[key] for key in ("token_count", "chunk_count", "ngram_status", "semantic_status", "error")})
        by_case[row["case_id"]].append(features)
        diagnostics.append(diagnostic)

    case_rows = []
    population = len(set(cohort["strategy_ids"]))
    possible = population * (population - 1) // 2
    for case in cohort["cases"]:
        rows = by_case[case["case_id"]]
        result = {"generation": cohort["generation"], "cohort": cohort["cohort"], "case_id": case["case_id"],
                  "population_size": population, "possible_pairs": possible}
        mismatch = False
        for metric, distance in (("ngram", jaccard_distance), ("semantic", cosine_distance)):
            valid_rows = [row for row in rows if row[metric] is not None]
            contexts = {json.dumps(row["context"], sort_keys=True) for row in valid_rows}
            metric_mismatch = len(contexts) > 1
            mismatch = mismatch or metric_mismatch
            valid = [] if metric_mismatch else [row[metric] for row in valid_rows]
            avg, nn, count = distance_stats(valid, distance)
            result.update({f"{metric}_mean": avg, f"{metric}_nearest_neighbor": nn,
                           f"{metric}_valid_prompts": len(valid), f"{metric}_compared_pairs": count,
                           f"{metric}_pair_coverage": count / possible if possible else None})
        result["status"] = "context_mismatch" if mismatch else "ok"
        case_rows.append(result)
    summary = {"generation": cohort["generation"], "cohort": cohort["cohort"], "population_size": population,
               "expected_cases": len(case_rows), "possible_pairs": possible * len(case_rows), "source": cohort["source"]}
    for metric in ("ngram", "semantic"):
        valid_cases = [row for row in case_rows if row[f"{metric}_mean"] is not None]
        for stat in ("mean", "nearest_neighbor"):
            summary[f"{metric}_{stat}"] = float(np.mean([row[f"{metric}_{stat}"] for row in valid_cases])) if valid_cases else None
        count = sum(row[f"{metric}_compared_pairs"] for row in case_rows)
        summary.update({f"{metric}_valid_cases": len(valid_cases),
                        f"{metric}_valid_prompts": sum(row[f"{metric}_valid_prompts"] for row in case_rows),
                        f"{metric}_compared_pairs": count,
                        f"{metric}_pair_coverage": count / summary["possible_pairs"] if summary["possible_pairs"] else None})
    return summary, case_rows, diagnostics
