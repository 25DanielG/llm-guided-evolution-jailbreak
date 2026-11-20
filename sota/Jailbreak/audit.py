"""Compare eligible top responses in a completed generation with HarmBench."""
import argparse
import fcntl
import json
from pathlib import Path

import jb_config
import judge
import vllm_client
from reporting_inputs import select_eligible, atomic_summary


def select_top_candidates(rows, generation, limit):
    return select_eligible(rows, limit, generation)[0]


def audit(run_id, generation):
    path = Path(__file__).resolve().parent / 'results' / run_id / 'candidates.jsonl'
    with path.open(encoding='utf-8') as file:
        fcntl.flock(file, fcntl.LOCK_SH)
        rows = [json.loads(line) for line in file if line.strip()]
    top, summary = select_eligible(rows, jb_config.audit_sample_size(), generation)
    summary.update(run_id=run_id, generation=generation)
    summary_path = path.with_name(f'audit_generation_{generation:02d}.json')
    atomic_summary(summary_path, summary)
    results = []
    try:
        if top:
            vllm_client.assert_model(jb_config.reporting_base_url(), jb_config.reporting_model_name())
        with path.with_name('audits.jsonl').open('a', encoding='utf-8') as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            for row in top:
                report = judge.reporting_score(row['behavior'], row['target_response'])
                if report.get('parse_ok') is not True or type(report.get('success')) is not bool:
                    raise RuntimeError('Independent reporter returned an invalid result: ' + repr(report))
                result = dict(run_id=run_id, generation=generation, candidate_id=row['candidate_id'],
                              behavior_index=row['behavior_index'], in_loop_score=row['in_loop']['score'],
                              reporting=report, score_gap=row['in_loop']['score'] - float(report['success']))
                file.write(json.dumps(result, ensure_ascii=False) + '\n'); file.flush()
                results.append(result)
                summary['completed'] = len(results)
                atomic_summary(summary_path, summary)
    except Exception as exc:
        summary.update(status='reporter_error', error=type(exc).__name__ + ': ' + str(exc))
        atomic_summary(summary_path, summary)
        raise
    gap = sum(row['score_gap'] for row in results)/len(results) if results else None
    print(f"audit_generation={generation} sample={len(results)} requested={summary['requested']} "
          f"excluded_records={len(summary['exclusions'])} status={summary['status']} mean_score_gap={gap}", flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--generation', required=True, type=int)
    args = parser.parse_args()
    audit(args.run_id, args.generation)
