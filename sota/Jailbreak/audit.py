"""Compare top candidates in a completed generation with HarmBench."""
import argparse
import fcntl
import json
from pathlib import Path

import jb_config
import judge
import vllm_client


def select_top_candidates(rows, generation, limit):
    best = {}
    for row in rows:
        if row["generation"] != generation:
            continue
        candidate = row["candidate_id"]
        if candidate not in best or row["in_loop"]["score"] > best[candidate]["in_loop"]["score"]:
            best[candidate] = row
    return sorted(best.values(), key=lambda row: row["in_loop"]["score"], reverse=True)[:limit]


def audit(run_id, generation):
    path = Path(__file__).resolve().parent / "results" / run_id / "candidates.jsonl"
    with open(path, encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_SH)
        rows = [json.loads(line) for line in file if line.strip()]
        fcntl.flock(file, fcntl.LOCK_UN)
    top = select_top_candidates(rows, generation, jb_config.audit_sample_size())
    vllm_client.assert_model(jb_config.reporting_base_url(), jb_config.reporting_model_name())
    results = []
    for row in top:
        report = judge.reporting_score(row["behavior"], row["target_response"])
        results.append({"run_id": run_id, "generation": generation,
                        "candidate_id": row["candidate_id"], "behavior_index": row["behavior_index"],
                        "in_loop_score": row["in_loop"]["score"], "reporting": report,
                        "score_gap": row["in_loop"]["score"] - float(report["success"])})
    output = path.with_name("audits.jsonl")
    with open(output, "a", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        fcntl.flock(file, fcntl.LOCK_UN)
    if results:
        print(f"audit_generation={generation} sample={len(results)} mean_score_gap={sum(r['score_gap'] for r in results)/len(results):.4f}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generation", required=True, type=int)
    args = parser.parse_args()
    audit(args.run_id, args.generation)
