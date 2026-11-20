"""Fail closed on serving evidence, frozen configuration, and live continuation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpointing import digest, inspect
from src.run_status import atomic_json, utc


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def verify_run(folder):
    folder = Path(folder)
    identity = read(folder / 'ops/experiment-identity.json')
    require(digest(folder / 'ops/common.env') == identity['common_env_sha256'], 'common.env changed')
    for name, expected in identity['source_files'].items():
        require(digest(folder / name) == expected, 'Prepared source changed: ' + name)
    sample = read(folder / 'ops/data/shared50/sample.json')
    selected = folder / 'ops/data/shared50/selected.csv'
    require(sample.get('status') == 'approved', 'Common behavior choice has not been approved')
    require(sample['count'] == 50 and sample['seed'] == 0 and sample['pool_count'] == 200,
            'Unexpected common sample definition')
    require(digest(selected) == sample['selected_sha256'] ==
            '1787d4db623a90d1374fdfb8415ff577d55b909bca89f165924aec0008205df5',
            'Frozen common sample checksum mismatch')
    rows = list(csv.DictReader(selected.open(newline='')))
    require(len(rows) == len({r['behavior'] for r in rows}) == 50 and
            [r['id'] for r in rows] == sample['ids'], 'Behavior count, order, or IDs differ')
    return identity


def preflight(chain):
    identity = verify_run(ROOT)
    expected_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    require(os.environ.get('LLMGE_EXPERIMENT_CONFIG_SHA256') == expected_identity,
            'Run environment and source identity differ')
    settings = dict(JB_N_BEHAVIORS_PER_EVAL='50', JB_FULL_BEHAVIORS='1', JB_STRATIFIED='0',
                    JB_CURRICULUM='0', JB_BEHAVIOR_SEED='0', LLMGE_EVOLUTION_SEED='0',
                    JB_TARGET_MAX_NEW_TOKENS='512', JB_SELECTION='map_elites',
                    JB_JUDGE_MODE='llm', JB_JUDGE_PROMPT='rubric', JB_RUN_TARGET_PROFILE='gpt-oss-20b',
                    LLMGE_START_POPULATION_SIZE='24', LLMGE_POPULATION_SIZE='24', LLMGE_NUM_GENERATIONS='17')
    for key, value in settings.items():
        require(os.environ.get(key) == value, f'{key} must be {value}')
    from src.model_serving import model_path, validate_checkpoint
    validate_checkpoint('gpt-oss-20b', model_path('gpt-oss-20b'))
    require(model_path('gpt-oss-20b') == chain['target_checkpoint'], '20B checkpoint path changed')
    require(os.environ['JB_EXPERIMENT_PYTHON'] == chain['experiment_python'] and
            os.environ['JB_SERVING_ENV'] == chain['serving_env'], 'Verified runtime paths changed')
    for key in ('JB_RUN_JUDGE_PROFILE', 'JB_RUN_MUTATOR_PROFILE'):
        require(not os.environ.get(key), 'Auxiliary profile override is forbidden: ' + key)
    pins = read(ROOT / 'ops/auxiliary-models.json')
    for role, keys in (('llama', ('JB_RUN_JUDGE_MODEL', 'JB_RUN_MUTATOR_MODEL')),
                       ('reporter', ('JB_RUN_REPORTING_MODEL',))):
        evidence = read(Path(chain['directory']) / (role + '-checkpoint.json'))
        require(all(evidence[k] == pins[role][k] for k in ('model_id', 'revision')), 'Auxiliary pin differs')
        require(all(os.environ.get(k) == evidence['path'] for k in keys), 'Auxiliary model path differs')
        for item in evidence['files']:
            stat = (Path(evidence['path']) / item['name']).stat()
            require(stat.st_size == item['size'] and stat.st_mtime_ns == item['mtime_ns'],
                    'Verified auxiliary file changed: ' + item['name'])
    require(verify_run(chain['pipeline_dir']) == verify_run(chain['full_dir']),
            'Pipeline and full run must use identical source and common configuration')


def serving(chain, verify_accounting=True):
    if not Path(chain['smoke_result']).is_file():
        # The queued job's exported result root is not exposed by scontrol.
        # Its standard launcher prints the exact path on successful completion.
        stdout = Path(chain['smoke_stdout'])
        if stdout.is_file() and not chain.get('smoke_result_explicit'):
            matches = re.findall(r'^passed: (.+/result\.json)\s*$', stdout.read_text(), re.M)
            require(len(matches) == 1, 'Cannot locate the serving result from the smoke stdout')
            chain['smoke_result'] = str((Path(chain['smoke_workdir']) / matches[0]).resolve())
    result = read(chain['smoke_result'])
    require(str(result.get('slurm_job_id')) == '7001848', 'Wrong serving job evidence')
    require(result.get('status') == 'passed' and result.get('inference_verified') is True,
            'Serving acceptance failed or incomplete')
    require(not any(value for key, value in result.items() if 'error' in key.lower()),
            'Serving result contains an error despite its passed status')
    require(result.get('model') == 'gpt-oss-20b' and result.get('model_id') == 'openai/gpt-oss-20b' and
            result.get('revision') == '6cee5e81ee83917806bbde320786a8fb61efebee' and
            result.get('checkpoint') == chain['target_checkpoint'], 'Wrong serving model/revision/path')
    parsed = result.get('parsed_response', {})
    require(parsed.get('finish_reason') == 'stop' and re.search(r'\b391\b', parsed.get('text', '')) and
            parsed.get('reasoning'), 'Serving final answer/reasoning separation not demonstrated')
    require(re.search(r'\bREADY\b', result.get('generate_response', {}).get('generated_text', ''), re.I),
            'Serving /generate contract failed')
    require(result.get('services_stopped') is True and result.get('resource_samples') and
            result.get('torch_cuda') == '12.9' and result.get('packages', {}).get('vllm') == '0.17.0',
            'Serving cleanup/resources/runtime evidence missing')
    if verify_accounting and chain.get('smoke_accounting_required'):
        evidence = read(Path(chain['directory']) / 'smoke-accounting.json')
        job = evidence.get('job', {})
        require(job.get('Source') == 'sacct' and job.get('JobId') == '7001848' and
                job.get('JobState') == 'COMPLETED' and job.get('ExitCode') == '0:0' and
                job.get('UserId') == os.environ.get('USER', 'pablo3'),
                'Saved serving accounting is missing or unsuccessful')
        require(evidence.get('result_sha256') == digest(chain['smoke_result']),
                'Serving artifact differs from the accounting acceptance record')
    atomic_json(Path(chain['directory']) / 'smoke-accepted.json',
                dict(job_id='7001848', result_path=chain['smoke_result'],
                     result_sha256=digest(chain['smoke_result']), checked_at=utc()))


def completed(chain, full=False):
    serving(chain)
    folder = Path(chain['full_dir' if full else 'pipeline_dir'])
    run_id = chain['full_run_id' if full else 'pipeline_run_id']
    run = folder / run_id
    job_file = Path(chain['directory']) / ('full.jobid' if full else 'pipeline.jobid')
    job_id = job_file.read_text().strip()
    state = read(run / 'status.json')
    last = 16 if full else 2
    require(state.get('state') == 'COMPLETED' and state.get('returncode') == 0 and
            state.get('latest_completed_generation') == last and str(state.get('job_id')) == job_id and
            state.get('run_id') == run_id, 'Run status, job ID, or final generation differs')
    identity = verify_run(folder)
    config_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    markers = []
    for generation in ((last,) if full else (0, 1, 2)):
        _, _, marker = inspect(run / 'ckpt', run / 'global', f'checkpoint_gen_{generation}.pkl')
        ident = marker['identity']
        require(ident['run_id'] == run_id and ident['behavior_sha256'] == identity['sample_sha256'] and
                ident['environment']['LLMGE_EXPERIMENT_CONFIG_SHA256'] == config_hash and
                ident['start_population_size'] == (24 if full else 8) and
                ident['population_size'] == (24 if full else 8), 'Checkpoint identity differs')
        markers.append(marker)
    if not full:
        require(all(m['identity'] == markers[0]['identity'] for m in markers),
                'Continuation changed checkpoint identity')
        probe = read(run / 'pipeline-probe.json')
        require(probe.get('status') == 'passed' and str(probe.get('slurm_job_id')) == job_id and
                probe.get('target_budget') == 512 and probe.get('target_final_present') is True and
                probe.get('target_cache_hit') is True and probe.get('judge_cache_hit') is True and
                probe.get('target_budget_check', {}).get('finish_reason') == 'stop' and
                probe.get('judge', {}).get('parse_ok') is True and
                probe.get('reporter', {}).get('parse_ok') is True and probe.get('mutator_response'),
                'Live pipeline evidence incomplete')
        logs = list((run / 'attempts').glob('*/console.log'))
        require(any('LIVE CONTINUATION PASSED' in log.read_text() for log in logs),
                'Required continuation did not finish')
    evidence = dict(job_id=job_id, checked_at=utc(), final_generation=last,
                    status_sha256=digest(run / 'status.json'),
                    smoke_sha256=digest(chain['smoke_result']),
                    checkpoint_markers=markers)
    atomic_json(Path(chain['directory']) / ('full-accepted.json' if full else 'pipeline-accepted.json'), evidence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain', required=True)
    parser.add_argument('--phase', choices=('preflight', 'serving', 'pipeline', 'full'), required=True)
    args = parser.parse_args()
    chain = read(Path(args.chain) / 'chain.json')
    if args.phase == 'preflight':
        preflight(chain)
    elif args.phase == 'serving':
        serving(chain)
    else:
        preflight(chain)
        if args.phase == 'full':
            completed(chain)
        completed(chain, args.phase == 'full')
    print(args.phase.upper() + ' CHECKS PASSED', flush=True)
