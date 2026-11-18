"""Run one allocation and chain checkpoint continuations without changing frozen code."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.experiments.difficulty_tournament import load_manifest, slot
from src.run_status import atomic_json


def submit(stage, campaign, index, dependency=None):
    command = ['sbatch', '--parsable', f'--array={index}']
    if dependency:
        command.append(f'--dependency=afterany:{dependency}')
    command += [str(ROOT / 'run_difficulty_chain.sbatch'), stage, str(campaign)]
    result = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(';')[0]
    if not result.isdigit():
        raise RuntimeError(f'Unexpected sbatch response: {result!r}')
    return result


def run(stage, campaign, index):
    manifest = load_manifest(campaign)
    seed, arm = slot(manifest, stage, index)
    directory = campaign / ('initial' if stage == 'initial' else 'runs') / f'{seed}-{arm}'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.chain.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = directory / 'chain.json'
        state = json.loads(journal.read_text()) if journal.exists() else {'attempts': [], 'children': {}}
        if not (directory / 'complete.json').exists():
            work = directory / 'work'
            resume = 'required' if work.exists() else 'new'
            if resume == 'required' and not list((work / 'evolution/ckpt').glob('checkpoint_gen_*.json')):
                raise RuntimeError('Existing work has no committed checkpoint; investigate startup failure before retrying')
            # The original batch script forwards the stop request to the evolution driver.
            child = subprocess.Popen(['bash', str(ROOT / 'run_difficulty_tournament.sbatch'),
                                      stage, str(campaign), resume], cwd=ROOT)
            def forward(signum, _frame):
                if child.poll() is None:
                    child.send_signal(signum)
            previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
            try:
                code = child.wait()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
            event = {'job': os.environ['SLURM_JOB_ID'], 'exit_code': code, 'resume': resume}
            state['attempts'].append(event)
            atomic_json(journal, state)
            if code == 75:
                # afterany ensures this allocation releases resources before its continuation.
                event['continuation'] = submit(stage, campaign, index, os.environ['SLURM_JOB_ID'])
                atomic_json(journal, state)
                print(f'Checkpoint saved; continuation submitted: {event["continuation"]}', flush=True)
                return 75
            if code:
                print(f'Execution failed ({code}); no automatic retry for non-checkpoint failures', flush=True)
                return code
        complete = json.loads((directory / 'complete.json').read_text())
        if complete['identity'] != manifest['identity']:
            raise RuntimeError('Completion belongs to another campaign')
        if stage == 'initial':
            for child_index in (2 * index, 2 * index + 1):
                key = str(child_index)
                if key not in state['children']:
                    state['children'][key] = submit('run', campaign, child_index, os.environ['SLURM_JOB_ID'])
                    atomic_json(journal, state)
                    print(f'Submitted experiment index {key}: {state["children"][key]}', flush=True)
        print(f'Complete: {seed} {arm}', flush=True)
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['initial', 'run'])
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.stage, args.campaign.resolve(), int(os.environ['SLURM_ARRAY_TASK_ID'])))
