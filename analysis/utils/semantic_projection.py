"""Fixed, case-matched semantic maps from audited, cached prompt embeddings."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_cached_analysis(directory, cohort="retained"):
    """Require matching embedding provenance; never infer or regenerate embeddings."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("semantic_error") or metadata["embedding"].get("status"):
        raise ValueError("The input analysis has no successful semantic embeddings")
    config = {k: v for k, v in metadata["embedding"].items() if k != "device"}
    vectors = {}
    for path in sorted((directory / "embedding_cache").glob("*.json")):
        entry = json.loads(path.read_text())
        if entry.get("config") != config or "prompt_hash" not in entry:
            continue
        vector = np.asarray(entry["vector"], dtype=np.float64)
        norm = np.linalg.norm(vector)
        if vector.ndim != 1 or not np.isfinite(vector).all() or norm == 0:
            raise ValueError(f"Invalid cached vector: {path}")
        vector /= norm
        key = entry["prompt_hash"]
        if key in vectors and not np.allclose(vectors[key], vector, atol=1e-7):
            raise ValueError(f"Conflicting cached vectors for {key}")
        vectors[key] = vector
    grouped = defaultdict(list)
    unavailable = []
    keys = set()
    for row in read_csv(directory / "prompt_diagnostics.csv"):
        if row["cohort"] != cohort:
            continue
        row["generation"] = int(row["generation"])
        key = (row["generation"], row["case_id"], row["candidate_id"])
        if key in keys:
            raise ValueError(f"Duplicate strategy/case observation: {key}")
        keys.add(key)
        if row["semantic_status"] != "ok":
            unavailable.append(row)
            continue
        if row["prompt_hash"] not in vectors:
            raise ValueError(f"Missing matching cached embedding for {row['prompt_hash']}")
        grouped[row["case_id"]].append(row)
    if not grouped:
        raise ValueError(f"No valid observations for cohort {cohort}")
    summaries = [r for r in read_csv(directory / "generation_diversity.csv") if r["cohort"] == cohort]
    cases = {(int(r["generation"]), r["case_id"]): r
             for r in read_csv(directory / "case_diversity.csv") if r["cohort"] == cohort}
    return metadata, vectors, grouped, unavailable, summaries, cases


def cosine_matrix(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    distances = np.clip(1 - vectors @ vectors.T, 0, 2)
    np.fill_diagonal(distances, 0)
    return distances


def neighborhood_quality(distances, coordinates, k=5):
    """Check local preservation without assigning semantic units to projected axes."""
    from sklearn.manifold import trustworthiness

    count = len(coordinates)
    k = min(k, max(0, (count - 1) // 2))
    if k < 1:
        return {"quality_neighbors": 0, "trustworthiness": None, "neighbor_overlap": None}
    projected = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
    original = distances.copy()
    np.fill_diagonal(original, np.inf)
    np.fill_diagonal(projected, np.inf)
    near_original = np.argsort(original, axis=1, kind="stable")[:, :k]
    near_projected = np.argsort(projected, axis=1, kind="stable")[:, :k]
    overlap = np.mean([len(set(a) & set(b)) / k for a, b in zip(near_original, near_projected)])
    return {"quality_neighbors": k,
            "trustworthiness": float(trustworthiness(distances, coordinates, n_neighbors=k, metric="precomputed")),
            "neighbor_overlap": float(overlap)}


def project_case(rows, vectors, seed=42, n_neighbors=15, min_cluster_size=5):
    """One joint densMAP per case, fit on unique vectors across all generations."""
    hashes = sorted({r["prompt_hash"] for r in rows})
    matrix = np.stack([vectors[h] for h in hashes])
    # Exact repeated embeddings must share coordinates, even for different text hashes.
    unique, inverse = np.unique(matrix, axis=0, return_inverse=True)
    distances = cosine_matrix(unique)
    count = len(unique)
    if count >= 4:
        from umap import UMAP
        mapper = UMAP(n_components=2, n_neighbors=min(n_neighbors, count - 1),
                      metric="cosine", densmap=True, dens_lambda=2.0,
                      min_dist=0.1, random_state=seed, n_jobs=1)
        coordinates = mapper.fit_transform(unique.astype(np.float32)).astype(np.float64)
        method = "densMAP"
    else:
        # The nonlinear fit is undefined for very small cases. Preserve these exactly
        # when possible with centered SVD, and clearly report the fallback.
        centered = unique - unique.mean(axis=0)
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        coordinates = np.zeros((count, 2))
        width = min(2, len(s))
        coordinates[:, :width] = u[:, :width] * s[:width]
        method = "small-sample SVD"
    labels = np.full(count, -1, dtype=int)
    if count >= min_cluster_size:
        from sklearn.cluster import HDBSCAN
        labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=3,
                         metric="precomputed").fit_predict(distances)
        # Stable color order, largest group first. Grouping is in original space.
        counts = Counter(labels[labels >= 0])
        order = sorted(counts, key=lambda label: (-counts[label], int(label)))
        remap = {label: index for index, label in enumerate(order)}
        labels = np.array([remap.get(label, -1) for label in labels])
    projected = {h: {"x": float(coordinates[inverse[i], 0]), "y": float(coordinates[inverse[i], 1]),
                     "cluster": int(labels[inverse[i]])} for i, h in enumerate(hashes)}
    stats = []
    for generation in sorted({r["generation"] for r in rows}):
        active = [r for r in rows if r["generation"] == generation]
        original = cosine_matrix([vectors[r["prompt_hash"]] for r in active])
        xy = np.array([[projected[r["prompt_hash"]]["x"], projected[r["prompt_hash"]]["y"]] for r in active])
        original_mean = original_nn = projected_nn = None
        if len(active) >= 2:
            original_mean = float(original[np.triu_indices(len(active), 1)].mean())
            np.fill_diagonal(original, np.inf)
            original_nn = float(original.min(axis=1).mean())
            planar = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
            np.fill_diagonal(planar, np.inf)
            projected_nn = float(planar.min(axis=1).mean())
        stats.append({"generation": generation, "valid_strategies": len(active),
                      "unique_prompts": len({r["prompt_hash"] for r in active}),
                      "semantic_mean": original_mean, "semantic_nearest_neighbor": original_nn,
                      "projected_nearest_neighbor": projected_nn})
    quality = {"method": method, "unique_prompts": len(hashes), "unique_vectors": count,
               "clusters": len(set(labels) - {-1}), "unclustered_unique_vectors": int(sum(labels < 0)),
               **neighborhood_quality(distances, coordinates)}
    return projected, stats, quality


def choose_representative_case(case_stats):
    """Median endpoint NN change, selected without inspecting projected appearance."""
    changes = []
    for case, stats in case_stats.items():
        first, last = stats[0], stats[-1]
        if first["semantic_nearest_neighbor"] and last["semantic_nearest_neighbor"] is not None:
            changes.append((last["semantic_nearest_neighbor"] / first["semantic_nearest_neighbor"] - 1, case))
    if not changes:
        return sorted(case_stats)[0]
    median = float(np.median([change for change, _ in changes]))
    return min(changes, key=lambda pair: (abs(pair[0] - median), pair[1]))[1]


def generation_alpha(generation, generations):
    """Latest completed generation fully opaque; older generations progressively faint."""
    index = generations.index(generation)
    if index == len(generations) - 1:
        return 1.0
    return 0.10 + 0.35 * index / max(1, len(generations) - 2)
