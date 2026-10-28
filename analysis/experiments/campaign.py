"""Durable Slurm campaign controller with bounded GPU worker concurrency."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from analysis.experiments.coverage import digest, load_experiment, write_json
from analysis.experiments.runtime import latest_complete

REPO = Path(__file__).resolve().parents[2]
GPUS_PER_WORKER = 6
SHARED_LLAMA_70B = '/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct'
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING', 'STAGE_OUT'}
CONTINUE = {'TIMEOUT', 'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL'}
SUBSETS = [f'{kind}_{i:02d}' for kind, count in [('random', 4), ('narrow', 3), ('broad', 3)] for i in range(1, count+1)]


def now():
    return datetime.now(timezone.utc).isoformat()


def minutes(value):
    """Slurm [days-]hours:minutes:seconds; bare integers are minutes."""
    days, clock = (value.split('-', 1) if '-' in value else ('0', value))
    parts = list(map(int, clock.split(':')))
    if len(parts) == 1:
        return int(days)*1440 + parts[0]
    if len(parts) == 2:
        return int(days)*1440 + parts[0] + parts[1]/60
    if len(parts) == 3:
        return int(days)*1440 + parts[0]*60 + parts[1] + parts[2]/60
    raise ValueError('Invalid Slurm time')


def stages(config):
    rows = [{'id': 'prepare', 'type': 'prepare', 'gpu': False},
            {'id': 'initial', 'type': 'freeze-initial', 'gpu': True}]
    for repeat, seed in enumerate(config['seeds'], 1):
        for subset in SUBSETS:
            rows.append({'id': f'{subset}_repeat_{repeat:02d}', 'type': 'run', 'gpu': True,
                         'subset': subset, 'seed': seed})
    rows.append({'id': 'analysis', 'type': 'analyze', 'gpu': False})
    return rows


def initialize(root, args):
    root = Path(root).resolve()
    path = root / 'campaign.json'
    if path.exists():
        return json.loads(path.read_text())
    if len(args.seeds) != len(set(args.seeds)) or not args.seeds or any(s < 0 or s >= 2**32 for s in args.seeds):
        raise ValueError('Use distinct nonnegative 32-bit evolution seeds')
    wall_minutes = minutes(args.worker_time)
    if wall_minutes <= 0 or wall_minutes * GPUS_PER_WORKER > args.gpu_hour_limit * 60:
        raise ValueError('Six GPUs times worker walltime exceeds the configured GPU-hour allowance')
    if args.generations < 1 or args.max_chunks < 1 or args.max_failures < 1:
        raise ValueError('Generations and retry limits must be positive')
    if not 1 <= args.max_gpu_workers <= 50:
        raise ValueError('Use between 1 and 50 GPU workers, including pending jobs')
    cfg = {'version': 1, 'root': str(root), 'created': now(), 'generations': args.generations,
           'seeds': args.seeds, 'worker_time': args.worker_time, 'gpu_hour_limit': args.gpu_hour_limit,
           'gpu_partition': args.gpu_partition, 'cpu_partition': args.cpu_partition,
           'constraint': args.constraint, 'exclude': args.exclude, 'account': args.account, 'qos': args.qos,
           'max_chunks': args.max_chunks, 'max_failures': args.max_failures,
           'max_gpu_workers': args.max_gpu_workers,
           'source_csv': str(Path(args.source_csv).resolve()) if args.source_csv else None,
           'source_revision': args.source_revision, 'embedding_model': args.embedding_model,
           'embedding_revision': args.embedding_revision,
           'models': {k: str(Path(v).expanduser().resolve()) for k, v in {
               'TARGET': args.target_model, 'JUDGE': args.judge_model,
               'REPORTING': args.reporting_model, 'MUTATOR': args.mutator_model}.items()}}
    if cfg['source_csv'] and not cfg['source_revision']:
        raise ValueError('--source-csv requires --source-revision')
    if cfg['source_csv'] and not Path(cfg['source_csv']).is_file():
        raise ValueError('Local source CSV does not exist')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    write_json(path, cfg)
    return cfg


def resource_args(config, stage):
    gpu = stage['gpu']
    args = ['--nodes=1', '--ntasks=1', '--time=' + (config['worker_time'] if gpu else '04:00:00'),
            '--cpus-per-task=' + ('24' if gpu else '4'), '--mem=' + ('192G' if gpu else '24G'),
            '--partition=' + config['gpu_partition' if gpu else 'cpu_partition']]
    if gpu:
        args += [f'--gpus={GPUS_PER_WORKER}', '--constraint=' + config['constraint']]
        if config.get('exclude'):
            args.append('--exclude=' + config['exclude'])
    for key in ('account', 'qos'):
        if config.get(key):
            args.append('--' + key + '=' + config[key])
    return args


def stage_command(config, stage):
    root = Path(config['root'])
    manifest = root / 'experiment.json'
    base = [sys.executable, str(REPO / 'analysis/scripts/coverage_experiment.py')]
    if stage['type'] == 'prepare':
        cmd = base + ['prepare', '--output', str(root), '--embedding-model', config['embedding_model']]
        for key in ('source_csv', 'source_revision', 'embedding_revision'):
            if config.get(key):
                cmd += ['--' + key.replace('_', '-'), config[key]]
        return cmd
    if stage['type'] == 'analyze':
        runs = [str(root / 'runs' / s['id']) for s in stages(config) if s['type'] == 'run']
        return base + ['analyze', '--manifest', str(manifest), '--runs', *runs,
                      '--output', str(root / f'analysis_g{config["generations"]:02d}.working'),
                      '--final-generation', str(config['generations'])]
    return ['bash', str(REPO / 'run_coverage_experiment.sbatch')]


def run_root(config, stage):
    root = Path(config['root'])
    return root / ('initial_creation' if stage['type'] == 'freeze-initial' else 'runs/' + stage['id'])


def progress(config, stage):
    """Durable progress only, so repeated dead jobs cannot burn resources forever."""
    root = Path(config['root'])
    if stage['type'] == 'prepare':
        return int((root / 'experiment.json').exists())
    if stage['type'] == 'freeze-initial':
        return int((root / 'initial/population.json').exists())
    if stage['type'] == 'analyze':
        return int((root / f'analysis_g{config["generations"]:02d}/report.txt').exists())
    rr = run_root(config, stage)
    completed = latest_complete(rr)
    # Count fully scored held-out bundles, not pre-inference observations.
    scored = 0
    for path in (rr / 'observations/heldout').glob('*/*.json'):
        try:
            record = json.loads(path.read_text())
            if record.get('evaluation_complete'):
                scored += 1
        except (OSError, ValueError):
            pass
    return (0 if completed is None else completed+1)*100000 + scored


def finished(config, stage):
    root = Path(config['root'])
    if stage['type'] == 'prepare':
        if not (root / 'experiment.json').exists():
            return False
        load_experiment(root / 'experiment.json')
        return True
    if stage['type'] == 'freeze-initial':
        path = root / 'initial/population.json'
        if not path.exists():
            return False
        pop = json.loads(path.read_text())
        if len(pop['members']) != 24 or not all(digest((path.parent / m['file']).read_bytes()) == m['sha256'] for m in pop['members']):
            raise ValueError('Frozen initial population checksum mismatch')
        return True
    if stage['type'] == 'analyze':
        return bool(progress(config, stage))
    rr = run_root(config, stage)
    completed = latest_complete(rr)
    if completed is None or completed < config['generations']:
        return False
    expected = set(load_experiment(root / 'experiment.json')['heldout_ids'])
    for generation in range(config['generations']+1):
        snap = json.loads((rr / 'snapshots' / f'{generation:04d}.json').read_text())
        for member in snap['members']:
            if not member['strategy_hash']:
                continue
            path = rr / 'observations/heldout' / f'{generation:04d}' / f'{member["id"]}.json'
            if not path.exists():
                return False
            obs = json.loads(path.read_text())
            if not obs.get('evaluation_complete') or obs['strategy_hash'] != member['strategy_hash'] or {r['behavior_id'] for r in obs['records']} != expected:
                return False
    return True


class Slurm:
    def command(self, args):
        return subprocess.run(args, check=True, text=True, capture_output=True, timeout=45).stdout.strip()

    def submit(self, args):
        result = self.command(['sbatch', '--parsable', *args])
        job = result.split(';')[0]
        if not job.isdigit():
            raise ValueError('Unexpected sbatch response: ' + result)
        return job

    def state(self, job):
        try:
            live = self.command(['squeue', '-h', '-j', job, '-o', '%T'])
        except subprocess.CalledProcessError as exc:
            if 'invalid job id' not in (exc.stderr or '').lower():
                raise
            live = ''
        if live:
            return live.splitlines()[0].strip()
        rows = self.command(['sacct', '-X', '-nP', '-j', job, '--format=JobIDRaw,State'])
        for row in rows.splitlines():
            fields = row.split('|')
            if fields[0] == job:
                return fields[1].split()[0].rstrip('+')
        return 'UNKNOWN'

    def find(self, name, created):
        # Recover a submission whose receipt was lost without submitting a duplicate.
        live = self.command(['squeue', '-h', '-u', os.environ['USER'], '--name', name, '-o', '%i'])
        ids = set(live.splitlines()) if live else set()
        rows = self.command(['sacct', '-X', '-nP', '-u', os.environ['USER'], '-S', datetime.fromisoformat(created).astimezone().strftime('%Y-%m-%dT%H:%M:%S'),
                             '--name', name, '--format=JobIDRaw,JobName%100'])
        ids.update(row.split('|')[0] for row in rows.splitlines() if row.split('|')[1] == name)
        if len(ids) > 1:
            raise RuntimeError('Multiple jobs for submission token ' + name)
        return next(iter(ids), None)


def save_state(root, state):
    state['updated'] = now()
    write_json(root / 'campaign_state.json', state)


def tick(config, state, slurm):
    """One reconciled scheduling step. Caller holds the campaign lock."""
    root = Path(config['root'])
    work = stages(config)
    if state.get('stop'):
        return 'stopped'
    active = state.get('active')
    if active:
        stage = next(s for s in work if s['id'] == active['stage'])
        if not active.get('job'):
            active['job'] = slurm.find(active['name'], config['created'])
            if not active['job']:
                # Submission ambiguity must not silently result in a duplicate GPU job.
                if time.time() - active['submitted_at'] > 600:
                    state['error'] = 'Submission receipt missing; inspect Slurm before clearing active intent'
                    return 'blocked'
                return 'waiting'
        status = slurm.state(active['job'])
        active['slurm_state'] = status
        if status in ACTIVE:
            return 'waiting'
        if status == 'UNKNOWN':
            active.setdefault('unknown_since', time.time())
            if time.time() - active['unknown_since'] > 600:
                state['error'] = 'Slurm cannot account for job ' + active['job']
                return 'blocked'
            return 'waiting'
        entry = state['stages'].setdefault(stage['id'], {})
        current = progress(config, stage)
        if status == 'CANCELLED':
            # Cancellation is intentional even if final artifacts were just written.
            state['stop'] = True
            entry['status'] = 'cancelled'
        elif finished(config, stage):
            entry['status'] = 'complete'
        else:
            advanced = current > active['progress']
            entry['no_progress'] = 0 if advanced else entry.get('no_progress', 0)+1
            entry['failures'] = entry.get('failures', 0) + int(status not in CONTINUE)
            entry['status'] = 'retry'
            entry['last_state'] = status
            if (entry['attempts'] - entry.get('attempt_offset', 0) >= config['max_chunks'] or entry['no_progress'] >= config['max_failures']
                    or entry['failures'] >= config['max_failures']):
                entry['status'] = 'failed'
        state.setdefault('history', []).append({**active, 'ended': now(), 'result': entry['status']})
        state['active'] = None
        save_state(root, state)
        if state.get('stop'):
            return 'stopped'
    for stage in work:
        entry = state['stages'].setdefault(stage['id'], {})
        if entry.get('status') in {'complete', 'failed'}:
            continue
        if finished(config, stage):
            entry['status'] = 'complete'
            continue
        prerequisites = ['prepare'] if stage['type'] == 'freeze-initial' else ['prepare', 'initial'] if stage['type'] == 'run' else []
        if stage['type'] == 'analyze':
            prerequisites = [s['id'] for s in work if s['type'] == 'run']
        if any(state['stages'].get(p, {}).get('status') == 'failed' for p in prerequisites):
            entry['status'] = 'failed'
            entry['reason'] = 'prerequisite failed'
            continue
        attempt = entry.get('attempts', 0)+1
        token = digest(str(root))[:8]
        name = f'cov-{token}-{work.index(stage):02d}-{attempt:02d}'
        active = {'stage': stage['id'], 'name': name, 'job': None, 'submitted_at': time.time(), 'progress': progress(config, stage)}
        state['active'] = active
        entry.update(attempts=attempt, status='submitted')
        save_state(root, state)  # Intent precedes sbatch.
        args = [*resource_args(config, stage), '--job-name=' + name, '--chdir=' + str(REPO),
                '--output=' + str(root / 'logs' / (name + '-%j.out')),
                str(REPO / 'run_coverage_stage.sbatch'), str(root), stage['id']]
        try:
            active['job'] = slurm.submit(args)
        except subprocess.CalledProcessError as exc:
            # An explicit rejection is safe to retry later; cap repeated rejections.
            entry['submission_failures'] = entry.get('submission_failures', 0)+1
            entry['attempts'] -= 1
            entry['status'] = 'retry'
            if entry['submission_failures'] >= config['max_failures']:
                entry['status'] = 'failed'
            state['active'] = None
            state['last_submission_error'] = exc.stderr
        save_state(root, state)
        return 'waiting'
    save_state(root, state)
    return 'failed' if any(s.get('status') == 'failed' for s in state['stages'].values()) else 'complete'


def parallel_tick(config, state, slurm):
    """Reconcile every worker before filling slots; caller holds the campaign lock.

    Adopt a serial controller's active job without resubmitting it. Both queued
    and running GPU jobs count toward the concurrency bound. Slurm enforces
    the account's aggregate remaining-GPU-minute allowance.
    """
    root = Path(config['root'])
    work = stages(config)
    by_id = {s['id']: s for s in work}
    if 'active_jobs' not in state:
        state['active_jobs'] = [state['active']] if state.get('active') else []
    jobs = state['active_jobs']
    if len({j['stage'] for j in jobs}) != len(jobs):
        raise RuntimeError('Multiple active workers for one stage; inspect Slurm')

    def save():
        # Keep the legacy field as an alias for status readers; the full list
        # is authoritative on restart and after a controller handoff.
        state['active'] = jobs[0] if jobs else None
        save_state(root, state)

    if state.get('stop'):
        return 'stopped'
    for active in list(jobs):
        stage = by_id[active['stage']]
        if not active.get('job'):
            active['job'] = slurm.find(active['name'], config['created'])
            if not active['job']:
                if time.time() - active['submitted_at'] > 600:
                    state['error'] = 'Submission receipt missing; inspect Slurm before clearing active intent'
                    return 'blocked'
                return 'waiting'
        status = slurm.state(active['job'])
        active['slurm_state'] = status
        if status in ACTIVE:
            continue
        if status == 'UNKNOWN':
            active.setdefault('unknown_since', time.time())
            if time.time() - active['unknown_since'] > 600:
                state['error'] = 'Slurm cannot account for job ' + active['job']
                return 'blocked'
            continue
        active.pop('unknown_since', None)
        entry = state['stages'].setdefault(stage['id'], {})
        current = progress(config, stage)
        if status == 'CANCELLED':
            state['stop'] = True
            entry['status'] = 'cancelled'
        elif finished(config, stage):
            entry['status'] = 'complete'
        else:
            advanced = current > active['progress']
            entry['no_progress'] = 0 if advanced else entry.get('no_progress', 0) + 1
            entry['failures'] = entry.get('failures', 0) + int(status not in CONTINUE)
            entry.update(status='retry', last_state=status)
            if (entry['attempts'] - entry.get('attempt_offset', 0) >= config['max_chunks']
                    or entry['no_progress'] >= config['max_failures']
                    or entry['failures'] >= config['max_failures']):
                entry['status'] = 'failed'
        state.setdefault('history', []).append({**active, 'ended': now(), 'result': entry['status']})
        jobs.remove(active)
        save()
    if state.get('stop'):
        return 'stopped'

    for index, stage in enumerate(work):
        entry = state['stages'].setdefault(stage['id'], {})
        if entry.get('status') in {'complete', 'failed'} or any(j['stage'] == stage['id'] for j in jobs):
            continue
        if finished(config, stage):
            entry['status'] = 'complete'
            continue
        required = (['prepare'] if stage['type'] == 'freeze-initial' else
                    ['prepare', 'initial'] if stage['type'] == 'run' else
                    [s['id'] for s in work if s['type'] == 'run'] if stage['type'] == 'analyze' else [])
        if any(state['stages'].get(p, {}).get('status') == 'failed' for p in required):
            entry.update(status='failed', reason='prerequisite failed')
            continue
        if any(state['stages'].get(p, {}).get('status') != 'complete' for p in required):
            continue
        if stage['gpu'] and sum(by_id[j['stage']]['gpu'] for j in jobs) >= config.get('max_gpu_workers', 1):
            continue
        attempt = entry.get('attempts', 0) + 1
        name = f'cov-{digest(str(root))[:8]}-{index:02d}-{attempt:02d}'
        active = {'stage': stage['id'], 'name': name, 'job': None,
                  'submitted_at': time.time(), 'progress': progress(config, stage)}
        jobs.append(active)
        entry.update(attempts=attempt, status='submitted')
        save()  # Persist every intent before submitting, including partial batches.
        args = [*resource_args(config, stage), '--job-name=' + name, '--chdir=' + str(REPO),
                '--output=' + str(root / 'logs' / (name + '-%j.out')),
                str(REPO / 'run_coverage_stage.sbatch'), str(root), stage['id']]
        try:
            active['job'] = slurm.submit(args)
        except subprocess.CalledProcessError as exc:
            entry['submission_failures'] = entry.get('submission_failures', 0) + 1
            entry['attempts'] -= 1
            entry['status'] = 'failed' if entry['submission_failures'] >= config['max_failures'] else 'retry'
            jobs.remove(active)
            state['last_submission_error'] = exc.stderr
        save()
    save()
    if jobs or any(state['stages'].get(s['id'], {}).get('status') not in {'complete', 'failed'} for s in work):
        return 'waiting'
    return 'failed' if any(s.get('status') == 'failed' for s in state['stages'].values()) else 'complete'


def controller(config, poll=60, lease=17*3600, slurm=None):
    root = Path(config['root'])
    slurm = slurm or Slurm()
    state_path = root / 'campaign_state.json'
    with (root / '.controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text()) if state_path.exists() else {'stages': {}, 'active': None}
        state['controller_job'] = os.getenv('SLURM_JOB_ID')
        state['controller_status'] = 'running'
        start = time.monotonic()
        while True:
            if (root / 'STOP').exists():
                state['stop'] = True
            try:
                step = parallel_tick if config.get('max_gpu_workers', 1) > 1 or 'active_jobs' in state else tick
                result = step(config, state, slurm)
                state.pop('scheduler_error', None)
            except (subprocess.SubprocessError, OSError) as exc:
                state['scheduler_error'] = str(exc)
                result = 'waiting'
            except (ValueError, RuntimeError) as exc:
                state['error'] = str(exc)
                result = 'blocked'
            state['controller_status'] = result
            save_state(root, state)
            if result != 'waiting':
                return 0 if result in {'complete', 'stopped'} else 1
            if time.monotonic() - start >= lease:
                current = os.getenv('SLURM_JOB_ID')
                if not current:
                    raise RuntimeError('Automatic controller continuation requires a Slurm allocation')
                args = ['--job-name=coverage-controller', '--chdir=' + str(REPO),
                        '--partition=' + config['cpu_partition'], '--dependency=afterany:' + current,
                        '--output=' + str(root / 'logs/controller-%j.out')]
                for key in ('account', 'qos'):
                    if config.get(key):
                        args.append('--' + key + '=' + config[key])
                successor = slurm.submit([*args, str(REPO / 'run_coverage_campaign.sbatch'), str(root)])
                state['controller_status'] = 'handed_off'
                state['successor_controller_job'] = successor
                save_state(root, state)
                return 0
            time.sleep(poll)


def work(config, stage_id):
    root = Path(config['root'])
    stage = next(s for s in stages(config) if s['id'] == stage_id)
    if finished(config, stage):
        return
    env = os.environ.copy()
    env.update({f'COVERAGE_{key}_MODEL': value for key, value in config['models'].items()})
    env.update(COVERAGE_MANIFEST=str(root / 'experiment.json'), COVERAGE_INITIAL=str(root / 'initial'),
               COVERAGE_RUN_ROOT=str(run_root(config, stage)), COVERAGE_ACTION=stage['type'],
               COVERAGE_FINAL_GENERATION=str(config['generations']), COVERAGE_SEED=str(stage.get('seed', 42)))
    if stage.get('subset'):
        env['COVERAGE_SUBSET'] = stage['subset']
    if stage['type'] == 'analyze':
        partial = root / f'analysis_g{config["generations"]:02d}.working'
        if partial.exists():
            # Retain partial reports for debugging, rather than overwriting evidence.
            partial.rename(root / (partial.name + '.failed-' + str(time.time_ns())))
    subprocess.run(stage_command(config, stage), cwd=REPO, env=env, check=True)
    if stage['type'] == 'analyze':
        partial.rename(root / f'analysis_g{config["generations"]:02d}')
    if not finished(config, stage):
        raise RuntimeError('Stage exited without all required artifacts: ' + stage_id)


def preflight(config):
    import importlib.util
    for package in ('numpy', 'sentence_transformers', 'sklearn', 'umap', 'statsmodels'):
        if importlib.util.find_spec(package) is None:
            raise ValueError('Missing analysis dependencies; run uv sync --frozen --extra diversity first')
    for name, path in config['models'].items():
        if not Path(path).is_dir():
            raise ValueError(f'{name} model directory does not exist: {path}')


def stage_inputs(config):
    """Download only on the submitting host; workers can operate offline."""
    import urllib.request
    from huggingface_hub import snapshot_download
    from analysis.utils.prompt_diversity import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_REVISION
    preflight(config)
    root = Path(config['root'])
    if (root / 'experiment.json').exists():
        load_experiment(root / 'experiment.json')
        return
    if (root / 'campaign_state.json').exists():
        raise ValueError('Do not change staged inputs after the controller has started')
    if not config.get('source_csv'):
        revision = config.get('source_revision')
        if not revision:
            request = urllib.request.Request('https://api.github.com/repos/llm-attacks/llm-attacks/commits/main',
                                             headers={'User-Agent': 'coverage-campaign'})
            with urllib.request.urlopen(request, timeout=60) as response:
                revision = json.load(response)['sha']
        if not re.fullmatch('[0-9a-fA-F]{40}', revision):
            raise ValueError('Source revision must be an immutable commit SHA')
        url = f'https://raw.githubusercontent.com/llm-attacks/llm-attacks/{revision}/data/advbench/harmful_behaviors.csv'
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        destination = root / 'upstream.csv'
        destination.write_bytes(raw)
        config.update(source_csv=str(destination), source_revision=revision)
    if not Path(config['embedding_model']).is_dir():
        revision = config.get('embedding_revision')
        if revision is None and config['embedding_model'] == DEFAULT_EMBEDDING_MODEL:
            revision = DEFAULT_EMBEDDING_REVISION
        local = snapshot_download(config['embedding_model'], revision=revision,
                                  cache_dir=root / 'embedding_cache/hub',
                                  allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])
        config.update(embedding_model=str(Path(local).resolve()), embedding_revision=None)
    write_json(root / 'campaign.json', config)


def submit_controller(config):
    preflight(config)
    root = Path(config['root'])
    args = ['--job-name=coverage-controller', '--chdir=' + str(REPO),
            '--partition=' + config['cpu_partition'], '--output=' + str(root / 'logs/controller-%j.out')]
    for key in ('account', 'qos'):
        if config.get(key):
            args.append('--' + key + '=' + config[key])
    job = Slurm().submit([*args, str(REPO / 'run_coverage_campaign.sbatch'), str(root)])
    print('Controller job ' + job + '; status: python -m analysis.experiments.campaign status ' + str(root))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['init', 'plan', 'stage-inputs', 'submit', 'control', 'work', 'status', 'resume'])
    p.add_argument('root')
    p.add_argument('--stage')
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--generations', type=int, default=5)
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--worker-time', default='02:40:00')
    p.add_argument('--gpu-hour-limit', type=float, default=16)
    p.add_argument('--gpu-partition', default='ice-gpu')
    p.add_argument('--cpu-partition', default='ice-cpu')
    p.add_argument('--constraint', default='H100|H200')
    p.add_argument('--exclude', help='Slurm node list to exclude from GPU workers')
    p.add_argument('--account')
    p.add_argument('--qos')
    p.add_argument('--max-chunks', type=int, default=12)
    p.add_argument('--max-failures', type=int, default=3)
    p.add_argument('--max-gpu-workers', type=int, default=1,
                   help='Maximum queued plus running GPU workers (default: serial)')
    p.add_argument('--source-csv')
    p.add_argument('--source-revision')
    p.add_argument('--embedding-model', default='sentence-transformers/all-mpnet-base-v2')
    p.add_argument('--embedding-revision')
    local_models = Path.home() / 'scratch/llm_storage'
    for key, path in [('target', local_models / 'Llama-3.1-8B-Instruct'),
                      ('judge', SHARED_LLAMA_70B),
                      ('reporting', local_models / 'HarmBench-Llama-2-13b-cls'),
                      ('mutator', SHARED_LLAMA_70B)]:
        p.add_argument('--' + key + '-model', default=str(path))
    return p


def main(argv=None):
    arguments = list(argv if argv is not None else sys.argv[1:])
    a = parser().parse_args(arguments)
    root = Path(a.root).resolve()
    if a.command == 'init':
        existing = (root / 'campaign.json').exists()
        config = initialize(root, a)
        if existing:
            for key in ('generations', 'seeds', 'worker_time', 'gpu_hour_limit', 'gpu_partition',
                        'cpu_partition', 'constraint', 'exclude', 'account', 'qos', 'max_chunks', 'max_failures', 'max_gpu_workers'):
                if any(v.split('=')[0] == '--' + key.replace('_', '-') for v in arguments) and getattr(a, key) != config.get(key):
                    raise ValueError('Campaign configuration is fixed; choose a new root to change ' + key)

    else:
        config = json.loads((root / 'campaign.json').read_text())
    if a.command in {'init', 'plan'}:
        print(json.dumps({'root': str(root), 'gpu_hours_per_worker': minutes(config['worker_time'])*GPUS_PER_WORKER/60,
                          'max_campaign_gpu_workers': config.get('max_gpu_workers', 1), 'stages': stages(config)}, indent=2))
    elif a.command == 'status':
        path = root / 'campaign_state.json'
        print(path.read_text() if path.exists() else 'Campaign has not started')
    elif a.command == 'stage-inputs':
        stage_inputs(config)
    elif a.command == 'submit':
        submit_controller(config)
    elif a.command == 'resume':
        with (root / '.controller.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state_path = root / 'campaign_state.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {'stages': {}, 'active': None}
            state['stop'] = False
            state.pop('error', None)
            (root / 'STOP').unlink(missing_ok=True)
            for item in state['stages'].values():
                if item.get('status') == 'cancelled' or (a.retry_failed and item.get('status') == 'failed'):
                    # Keep monotonically increasing submission names for reconciliation.
                    previous = item.get('attempts', 0)
                    item.clear()
                    item.update(status='retry', attempts=previous, attempt_offset=previous)
            save_state(root, state)
        submit_controller(config)
    elif a.command == 'control':
        preflight(config)
        lease = 17*3600
        if os.getenv('SLURM_JOB_ID'):
            remaining = Slurm().command(['squeue', '-h', '-j', os.environ['SLURM_JOB_ID'], '-o', '%L'])
            lease = max(0, minutes(remaining)*60 - 600)
        raise SystemExit(controller(config, lease=lease))
    elif a.command == 'work':
        work(config, a.stage)

if __name__ == '__main__':
    main()
