"""Persistent, versioned run records and a bounded supervisor; never submits jobs."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import traceback


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def update(path, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {'schema_version': 1}
        data.update(fields)
        data['updated_at'] = utc()
        atomic_json(path, data)
    return data


def progress(generation, checkpoint):
    if os.getenv('JB_STATUS_FILE'):
        update(os.environ['JB_STATUS_FILE'], latest_completed_generation=generation,
               checkpoint=str(Path(checkpoint).resolve()), last_progress_at=utc(),
               phase='checkpoint_committed')


def health(data, now=None, slow_after=3600, stall_after=10800):
    """Staleness is evidence of a possible stall, not proof that a process hung."""
    if data.get('state') not in ('RUNNING', 'STOPPING'):
        return data.get('state', 'UNKNOWN').lower()
    now = now or datetime.now(timezone.utc)
    def age(key):
        value = data.get(key) or data.get('started_at')
        return (now - datetime.fromisoformat(value)).total_seconds() if value else 0
    if age('last_progress_at') < slow_after:
        return 'progressing'
    if age('last_activity_at') < slow_after:
        return 'slow_but_active'
    if age('last_progress_at') >= stall_after and age('last_activity_at') >= stall_after:
        return 'suspected_stall_inspect_logs'
    return 'slow_no_recent_activity'


def scheduler_state(job_id):
    """One bounded accounting lookup. Unreachable is not a FAILED job."""
    try:
        result = subprocess.run(['sacct', '-n', '-P', '-j', str(job_id),
                                 '--format=JobIDRaw,State,ExitCode,Reason,MaxRSS'],
                                capture_output=True, text=True, timeout=15)
        observation = {'checked_at': utc(), 'returncode': result.returncode,
                       'accounting': result.stdout[-16000:], 'error': result.stderr[-2000:]}
        for line in result.stdout.splitlines():
            fields = line.split('|')
            if len(fields) >= 4 and fields[0] == str(job_id):
                observation.update(job_state=fields[1], exit_code=fields[2], exit_reason=fields[3])
        return observation
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'checked_at': utc(), 'error': str(exc), 'accounting': ''}


def summary(data):
    keys = ('run_id', 'job_id', 'state', 'job_state', 'health', 'latest_completed_generation',
            'checkpoint', 'last_progress_at', 'last_activity_at', 'heartbeat_at',
            'exit_reason', 'returncode', 'exception', 'console_log')
    return '\n'.join(f'{key}: {data.get(key)}' for key in keys)


def supervise(status, run_id, command, interval=300, max_hours=12, stop_file=None):
    if interval < 60 or max_hours <= 0:
        raise ValueError('interval must be >=60 seconds and max-hours positive')
    status = Path(status).resolve()
    status.parent.mkdir(parents=True, exist_ok=True)
    stop_file = Path(stop_file or status.parent / 'STOP_AFTER_GENERATION')
    job = os.getenv('SLURM_JOB_ID', 'local')
    attempt = status.parent / 'attempts' / f'{job}-{time.time_ns()}'
    attempt.mkdir(parents=True)
    # Held for the entire attempt: a second supervisor cannot own the same run.
    with (status.parent / '.supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status.exists():
            atomic_json(attempt / 'previous-status.json', json.loads(status.read_text()))
        stop_file.unlink(missing_ok=True)
        start = utc()
        atomic_json(status, {'schema_version': 1, 'run_id': run_id, 'job_id': job,
                    'state': 'RUNNING', 'job_state': 'RUNNING' if job != 'local' else 'LOCAL',
                    'started_at': start, 'last_progress_at': start, 'last_activity_at': start,
                    'heartbeat_at': start, 'returncode': None, 'phase': 'starting',
                    'latest_completed_generation': None, 'checkpoint': None,
                    'host': socket.gethostname(), 'command': command,
                    'console_log': str(attempt / 'console.log'), 'exit_reason': None,
                    'exception': None})
        env = dict(os.environ, JB_STATUS_FILE=str(status), JB_STOP_FILE=str(stop_file))
        received = []
        proc = None
        def handle(sig, _frame):
            received.append(sig)
            if sig == signal.SIGUSR1:
                # Never acquire the status flock inside a signal handler.
                stop_file.touch()
            elif proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        signals = (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)
        old = {sig: signal.signal(sig, handle) for sig in signals}
        try:
            with (attempt / 'console.log').open('w') as log:
                proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                deadline = time.monotonic() + max_hours * 3600
                last_size = 0
                termination_at = None
                while proc.poll() is None:
                    size = (attempt / 'console.log').stat().st_size
                    fields = {'heartbeat_at': utc(), 'pid': proc.pid}
                    if signal.SIGUSR1 in received:
                        fields.update(state='STOPPING', phase='walltime_warning')
                    if size != last_size:
                        fields['last_activity_at'] = utc()
                        last_size = size
                    update(status, **fields)
                    if time.monotonic() >= deadline or any(s != signal.SIGUSR1 for s in received):
                        if termination_at is None:
                            termination_at = time.monotonic()
                            os.killpg(proc.pid, signal.SIGTERM)
                        elif time.monotonic() - termination_at >= 30:
                            os.killpg(proc.pid, signal.SIGKILL)
                    try:
                        remaining = max(0.1, deadline - time.monotonic())
                        proc.wait(timeout=min(interval, remaining, 5 if received or termination_at else interval))
                    except subprocess.TimeoutExpired:
                        pass
                code = proc.returncode
            reason = ('checkpoint_pause' if code == 75 else
                      'completed' if code == 0 else
                      'signal_or_deadline' if received or termination_at else 'command_failed')
            with (attempt / 'console.log').open('rb') as stream:
                stream.seek(max(0, stream.seek(0, 2) - 16000))
                tail = stream.read().decode(errors='replace')
            update(status, state='PAUSED' if code == 75 else 'COMPLETED' if code == 0 else 'FAILED',
                   returncode=code, exit_reason=reason, finished_at=utc(),
                   job_state='UNCONFIRMED_AFTER_EXIT',
                   received_signals=[signal.Signals(s).name for s in received],
                   exception=tail if code not in (0, 75) else None)
            atomic_json(attempt / 'status.json', json.loads(status.read_text()))
            return code if code >= 0 else 128 - code
        except BaseException:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            update(status, state='FAILED', exception=traceback.format_exc(), exit_reason='supervisor_exception')
            raise
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    run = sub.add_parser('supervise')
    run.add_argument('--status', required=True)
    run.add_argument('--run-id', required=True)
    run.add_argument('--interval', type=int, default=300)
    run.add_argument('--max-hours', type=float, default=12)
    run.add_argument('command', nargs=argparse.REMAINDER)
    show = sub.add_parser('summary')
    show.add_argument('--status', required=True)
    show.add_argument('--refresh', action='store_true')
    observe = sub.add_parser('observe')
    observe.add_argument('--status', required=True)
    observe.add_argument('--interval', type=int, default=300)
    observe.add_argument('--checks', type=int, default=144)
    args = parser.parse_args()
    if args.action == 'supervise':
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        if not command:
            parser.error('a command is required after --')
        raise SystemExit(supervise(args.status, args.run_id, command, args.interval, args.max_hours))
    if args.action == 'observe':
        if args.interval < 60 or not 1 <= args.checks <= 288:
            parser.error('observe requires interval>=60 and checks in 1..288')
        for index in range(args.checks):
            data = json.loads(Path(args.status).read_text())
            observation = scheduler_state(data['job_id'])
            atomic_json(Path(args.status).with_name('scheduler-observation.json'), observation)
            print(json.dumps(observation), flush=True)
            if observation.get('job_state', '').split('+')[0] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'):
                break
            if index + 1 < args.checks:
                time.sleep(args.interval)
        return
    data = json.loads(Path(args.status).read_text())
    data['health'] = health(data)
    if args.refresh:
        data['scheduler_observation'] = scheduler_state(data['job_id'])
        observation = data['scheduler_observation']
        if observation.get('job_state'):
            data['job_state'] = observation['job_state']
            if data['state'] in ('RUNNING', 'STOPPING') and observation['job_state'] not in ('RUNNING', 'PENDING', 'COMPLETING'):
                data.update(state='ENDED_BY_SCHEDULER', exit_reason=observation.get('exit_reason'),
                            scheduler_exit_code=observation.get('exit_code'))
                data['health'] = 'scheduler_reports_termination'
        print(json.dumps(data['scheduler_observation'], indent=2))
        atomic_json(Path(args.status).with_name('scheduler-observation.json'), data['scheduler_observation'])
    print(summary(data))


if __name__ == '__main__':
    main()
