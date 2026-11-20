"""Eligibility for independent reporting only; never changes in-loop fitness."""
import json
import os
from pathlib import Path
import tempfile


def text_error(value, name):
    if value is None:
        return 'missing_' + name
    if not isinstance(value, str):
        return 'non_string_' + name
    if not value.strip():
        return 'empty_' + name
    return None


def exclusion_reasons(row):
    reasons = [reason for reason in (text_error(row.get('behavior'), 'behavior'),
                                    text_error(row.get('target_response'), 'response')) if reason]
    if row.get('target_error_type'):
        reasons.append('target_error:' + str(row['target_error_type']))
    if row.get('target_finish_reason') != 'stop':
        reasons.append('finish_reason:' + str(row.get('target_finish_reason')))
    return reasons


def select_eligible(rows, limit, generation=None):
    if limit < 0:
        raise ValueError('Reporting sample limit must be nonnegative')
    best, exclusions = {}, []
    considered = 0
    for index, row in enumerate(rows, 1):
        if generation is not None and row['generation'] != generation:
            continue
        considered += 1
        reasons = exclusion_reasons(row)
        if reasons:
            exclusions.append(dict(record_index=index, candidate_id=row['candidate_id'],
                                   generation=row['generation'], behavior_index=row.get('behavior_index'),
                                   reasons=reasons))
            continue
        candidate = row['candidate_id']
        if candidate not in best or row['in_loop']['score'] > best[candidate]['in_loop']['score']:
            best[candidate] = row
    selected = sorted(best.values(), key=lambda row: row['in_loop']['score'], reverse=True)[:limit]
    summary = dict(schema_version=1, requested=limit, considered_records=considered,
                   eligible_candidates=len(best), selected=len(selected), completed=0,
                   exclusions=exclusions,
                   status='complete' if len(selected) == limit else 'insufficient_eligible_candidates')
    return selected, summary


def atomic_summary(path, summary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(summary, stream, indent=2); stream.write('\n')
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def validate_audits(folder, run_id, generations):
    """Require actual eligible, parseable audit results; disclose data shortfalls."""
    folder = Path(folder)
    candidates = [json.loads(line) for line in (folder/'candidates.jsonl').read_text().splitlines() if line.strip()]
    audits = [json.loads(line) for line in (folder/'audits.jsonl').read_text().splitlines() if line.strip()]
    reports = []
    for generation in generations:
        summary = json.loads((folder/f'audit_generation_{generation:02d}.json').read_text())
        selected, expected = select_eligible(candidates, summary['requested'], generation)
        for key in ['schema_version','requested','considered_records','eligible_candidates','selected','exclusions','status']:
            if summary[key] != expected[key]:
                raise ValueError('Reporting audit selection differs: ' + key)
        if summary['run_id'] != run_id or summary['generation'] != generation:
            raise ValueError('Reporting audit identity differs')
        records = [row for row in audits if row['generation'] == generation]
        if len(records) != len(selected) or summary['completed'] != len(selected):
            raise ValueError('Reporting audit is incomplete')
        for row, source in zip(records, selected):
            if (row['run_id'] != run_id or row['candidate_id'] != source['candidate_id']
                    or row['behavior_index'] != source['behavior_index']
                    or row['reporting'].get('parse_ok') is not True
                    or type(row['reporting'].get('success')) is not bool
                    or row['in_loop_score'] != source['in_loop']['score']
                    or row['score_gap'] != row['in_loop_score'] - float(row['reporting']['success'])):
                raise ValueError('Reporting audit result differs or failed')
        reports.append({key: summary[key] for key in ['generation','status','requested','eligible_candidates','selected','completed']}
                       | {'excluded_records':len(summary['exclusions'])})
    return reports
