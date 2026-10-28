"""All scheduling tests use fake Slurm; no submissions or model calls."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.experiments import campaign as c


class Slurm:
    def __init__(self):
        self.submissions = []
        self.states = {}
        self.names = {}
    def submit(self, args):
        job = str(100 + len(self.submissions))
        self.submissions.append(args)
        self.states[job] = 'PENDING'
        self.names[next(a.split('=', 1)[1] for a in args if a.startswith('--job-name='))] = job
        return job
    def state(self, job):
        return self.states[job]
    def find(self, name, created):
        return self.names.get(name)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    args = c.parser().parse_args(['init', str(tmp_path / 'campaign')])
    config = c.initialize(args.root, args)
    complete, amounts = set(), {}
    monkeypatch.setattr(c, 'finished', lambda cfg, s: s['id'] in complete)
    monkeypatch.setattr(c, 'progress', lambda cfg, s: amounts.get(s['id'], 0))
    return config, {'stages': {}, 'active': None}, Slurm(), complete, amounts


def test_serial_entire_campaign(setup):
    cfg, state, slurm, done, _ = setup
    assert len(c.stages(cfg)) == 13
    for stage in c.stages(cfg):
        assert c.tick(cfg, state, slurm) == 'waiting'
        assert state['active']['stage'] == stage['id']
        count = len(slurm.submissions)
        assert c.tick(cfg, state, slurm) == 'waiting'
        assert len(slurm.submissions) == count
        done.add(stage['id'])
        slurm.states[state['active']['job']] = 'COMPLETED'
    assert c.tick(cfg, state, slurm) == 'complete'
    assert len(slurm.submissions) == 13
    assert sum('--gpus=6' in a for a in slurm.submissions) == 11


def test_resource_math_and_repeats(tmp_path):
    args = c.parser().parse_args(['init', str(tmp_path), '--worker-time', '04:00:00'])
    with pytest.raises(ValueError, match='allowance'):
        c.initialize(tmp_path, args)
    args.worker_time = '02:40:00'
    args.seeds = [42, 43]
    cfg = c.initialize(tmp_path, args)
    assert len(c.stages(cfg)) == 23
    assert c.minutes('1-02:30:00') == 1590
    assert c.minutes(cfg['worker_time'])*c.GPUS_PER_WORKER/60 == 16
    assert not any(arg.startswith('--gpus=') for arg in c.resource_args(cfg, c.stages(cfg)[0]))
    assert '--gpus=6' in c.resource_args(cfg, c.stages(cfg)[1])
    assert '--constraint=H100|H200' in c.resource_args(cfg, c.stages(cfg)[1])


def test_timeout_progress_resumes_same_run(setup):
    cfg, state, slurm, done, progress = setup
    done.update(['prepare', 'initial'])
    c.tick(cfg, state, slurm)
    active = state['active']
    name = active['stage']
    slurm.states[active['job']] = 'TIMEOUT'
    progress[name] = 200000
    c.tick(cfg, state, slurm)
    assert state['active']['stage'] == name
    assert state['active']['job'] != active['job']
    assert state['stages'][name]['no_progress'] == 0
    assert state['stages'][name]['attempts'] == 2
    assert state['stages'][name]['failures'] == 0


def test_excluded_node_applies_to_gpu_submissions(setup):
    cfg, state, slurm, done, _ = setup
    cfg['exclude'] = 'atl1-1-03-013-3-0'
    c.tick(cfg, state, slurm)
    assert not any(arg.startswith('--exclude=') for arg in slurm.submissions[-1])
    done.add('prepare')
    slurm.states[state['active']['job']] = 'COMPLETED'
    c.tick(cfg, state, slurm)
    assert '--exclude=atl1-1-03-013-3-0' in slurm.submissions[-1]
    assert '--gpus=6' in slurm.submissions[-1]


def test_retry_failed_preserves_active_worker(setup, monkeypatch):
    cfg, state, slurm, done, _ = setup
    done.update(['prepare', 'initial'])
    c.tick(cfg, state, slurm)
    active = dict(state['active'])
    state['stages']['random_02_repeat_01'] = {'status': 'failed', 'attempts': 3, 'failures': 3}
    c.save_state(Path(cfg['root']), state)
    submitted = []
    monkeypatch.setattr(c, 'submit_controller', lambda config: submitted.append(config['root']))
    c.main(['resume', cfg['root'], '--retry-failed'])
    saved = json.loads((Path(cfg['root']) / 'campaign_state.json').read_text())
    assert saved['active'] == active
    assert saved['stages']['random_02_repeat_01'] == {'status': 'retry', 'attempts': 3, 'attempt_offset': 3}
    assert submitted == [cfg['root']]
    assert len(slurm.submissions) == 1


def test_three_stalled_attempts_then_next_subset(setup):
    cfg, state, slurm, done, _ = setup
    done.update(['prepare', 'initial'])
    c.tick(cfg, state, slurm)
    first = state['active']['stage']
    for _ in range(3):
        slurm.states[state['active']['job']] = 'TIMEOUT'
        c.tick(cfg, state, slurm)
    assert state['stages'][first]['status'] == 'failed'
    assert state['active']['stage'] == 'random_02_repeat_01'


def test_initial_failure_blocks_dependent_work(setup):
    cfg, state, slurm, done, _ = setup
    done.add('prepare')
    c.tick(cfg, state, slurm)
    for _ in range(3):
        slurm.states[state['active']['job']] = 'FAILED'
        result = c.tick(cfg, state, slurm)
    assert result == 'failed'
    assert len(slurm.submissions) == 3
    assert state['stages']['analysis']['reason'] == 'prerequisite failed'


def test_cancel_does_not_resubmit(setup):
    cfg, state, slurm, _, _ = setup
    c.tick(cfg, state, slurm)
    slurm.states[state['active']['job']] = 'CANCELLED'
    assert c.tick(cfg, state, slurm) == 'stopped'
    assert len(slurm.submissions) == 1


def test_lost_submission_receipt_reconciles(setup):
    cfg, state, slurm, _, _ = setup
    original = slurm.submit
    def uncertain(args):
        original(args)
        raise subprocess.TimeoutExpired('sbatch', 45)
    slurm.submit = uncertain
    with pytest.raises(subprocess.TimeoutExpired):
        c.tick(cfg, state, slurm)
    assert state['active']['job'] is None
    assert c.tick(cfg, state, slurm) == 'waiting'
    assert state['active']['job'] == '100'
    assert len(slurm.submissions) == 1


def test_accounting_lag_does_not_duplicate(setup):
    cfg, state, slurm, _, _ = setup
    c.tick(cfg, state, slurm)
    slurm.states[state['active']['job']] = 'UNKNOWN'
    for _ in range(3):
        assert c.tick(cfg, state, slurm) == 'waiting'
    assert len(slurm.submissions) == 1


def test_controller_handoff_keeps_active_worker(setup, monkeypatch):
    cfg, state, slurm, _, _ = setup
    monkeypatch.setenv('SLURM_JOB_ID', '55')
    assert c.controller(cfg, lease=0, slurm=slurm) == 0
    saved = json.loads((Path(cfg['root']) / 'campaign_state.json').read_text())
    assert saved['controller_status'] == 'handed_off'
    assert saved['active']['job'] == '100'
    assert '--dependency=afterany:55' in slurm.submissions[1]
    assert not any(arg.startswith('--gpus=') for arg in slurm.submissions[1])


def test_explicit_submission_rejections_are_bounded(setup):
    cfg, state, slurm, _, _ = setup
    def reject(args):
        raise subprocess.CalledProcessError(1, 'sbatch', stderr='limit reached')
    slurm.submit = reject
    for _ in range(3):
        assert c.tick(cfg, state, slurm) == 'waiting'
    assert c.tick(cfg, state, slurm) == 'failed'
    assert state['stages']['prepare']['attempts'] == 0


def test_cancel_with_completed_artifacts_still_stops(setup):
    cfg, state, slurm, done, _ = setup
    c.tick(cfg, state, slurm)
    done.add(state['active']['stage'])
    slurm.states[state['active']['job']] = 'CANCELLED'
    assert c.tick(cfg, state, slurm) == 'stopped'
    assert len(slurm.submissions) == 1


def test_changed_existing_configuration_is_rejected(tmp_path):
    c.main(['init', str(tmp_path)])
    with pytest.raises(ValueError, match='fixed'):
        c.main(['init', str(tmp_path), '--generations', '10'])
    c.main(['init', str(tmp_path)])


def test_corrupted_initial_population_blocks_before_submission(tmp_path):
    cfg = {'root': str(tmp_path)}
    initial = tmp_path / 'initial'
    initial.mkdir()
    (initial / 'population.json').write_text(json.dumps({'members': []}))
    with pytest.raises(ValueError, match='checksum'):
        c.finished(cfg, {'type': 'freeze-initial'})


def test_parallel_adopts_serial_worker_and_bounds_pending_jobs(setup):
    cfg, state, slurm, done, _ = setup
    done.update(['prepare', 'initial'])
    c.tick(cfg, state, slurm)
    original = dict(state['active'])
    cfg['max_gpu_workers'] = 3
    assert c.parallel_tick(cfg, state, slurm) == 'waiting'
    assert len(slurm.submissions) == 3
    assert state['active_jobs'][0]['job'] == original['job']
    assert state['active_jobs'][0]['stage'] == original['stage']
    assert len({a['stage'] for a in state['active_jobs']}) == 3
    c.parallel_tick(cfg, state, slurm)
    assert len(slurm.submissions) == 3  # Pending jobs consume slots too.


def test_parallel_dependencies_and_final_analysis(setup):
    cfg, state, slurm, done, _ = setup
    cfg['max_gpu_workers'] = 4
    for _ in range(12):
        result = c.parallel_tick(cfg, state, slurm)
        if result == 'complete':
            break
        active = list(state['active_jobs'])
        if 'prepare' not in done:
            assert [a['stage'] for a in active] == ['prepare']
        elif 'initial' not in done:
            assert [a['stage'] for a in active] == ['initial']
        if any(a['stage'] == 'analysis' for a in active):
            assert all(s['id'] in done for s in c.stages(cfg) if s['type'] == 'run')
        assert len(active) <= 4
        for a in active:
            done.add(a['stage'])
            slurm.states[a['job']] = 'COMPLETED'
    assert result == 'complete'
    assert len(slurm.submissions) == 13


def test_parallel_timeout_resumes_only_interrupted_run(setup):
    cfg, state, slurm, done, amounts = setup
    cfg['max_gpu_workers'] = 3
    done.update(['prepare', 'initial'])
    c.parallel_tick(cfg, state, slurm)
    before = {a['stage']: a['job'] for a in state['active_jobs']}
    first = state['active_jobs'][0]
    slurm.states[first['job']] = 'TIMEOUT'
    amounts[first['stage']] = 400000
    c.parallel_tick(cfg, state, slurm)
    after = {a['stage']: a['job'] for a in state['active_jobs']}
    assert before.keys() == after.keys()
    assert after[first['stage']] != before[first['stage']]
    assert all(after[name] == job for name, job in before.items() if name != first['stage'])
    assert state['stages'][first['stage']]['no_progress'] == 0
    assert state['stages'][first['stage']]['failures'] == 0


def test_parallel_lost_submission_receipt_survives_reload(setup):
    cfg, state, slurm, done, _ = setup
    cfg['max_gpu_workers'] = 3
    done.update(['prepare', 'initial'])
    original = slurm.submit
    def uncertain(args):
        original(args)
        raise subprocess.TimeoutExpired('sbatch', 45)
    slurm.submit = uncertain
    with pytest.raises(subprocess.TimeoutExpired):
        c.parallel_tick(cfg, state, slurm)
    saved = json.loads((Path(cfg['root']) / 'campaign_state.json').read_text())
    assert saved['active_jobs'][0]['job'] is None
    slurm.submit = original
    c.parallel_tick(cfg, saved, slurm)
    assert len(slurm.submissions) == 3
    assert saved['active_jobs'][0]['job'] == '100'


def test_parallel_handoff_preserves_every_worker(setup, monkeypatch):
    cfg, _, slurm, done, _ = setup
    cfg['max_gpu_workers'] = 3
    done.update(['prepare', 'initial'])
    monkeypatch.setenv('SLURM_JOB_ID', '55')
    assert c.controller(cfg, lease=0, slurm=slurm) == 0
    saved = json.loads((Path(cfg['root']) / 'campaign_state.json').read_text())
    assert len(saved['active_jobs']) == 3
    assert saved['controller_status'] == 'handed_off'
    assert '--dependency=afterany:55' in slurm.submissions[-1]
    assert not any(a.startswith('--gpus=') for a in slurm.submissions[-1])


def test_parallel_cancellation_stops_new_submissions(setup):
    cfg, state, slurm, done, _ = setup
    cfg['max_gpu_workers'] = 3
    done.update(['prepare', 'initial'])
    c.parallel_tick(cfg, state, slurm)
    slurm.states[state['active_jobs'][0]['job']] = 'CANCELLED'
    assert c.parallel_tick(cfg, state, slurm) == 'stopped'
    assert len(slurm.submissions) == 3
    assert len(state['active_jobs']) == 2


def test_parallel_initial_failure_blocks_runs(setup):
    cfg, state, slurm, done, _ = setup
    cfg['max_gpu_workers'] = 3
    done.add('prepare')
    c.parallel_tick(cfg, state, slurm)
    for _ in range(3):
        slurm.states[state['active_jobs'][0]['job']] = 'FAILED'
        result = c.parallel_tick(cfg, state, slurm)
    assert result == 'failed'
    assert len(slurm.submissions) == 3
    assert state['stages']['analysis']['reason'] == 'prerequisite failed'
