"""Recover an isolated runtime whose sources match an existing campaign exactly."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
ROOTS = {'analysis', 'src', 'sota', 'templates', 'slurm-config', 'scripts', 'tests'}
ROOT_FILES = {'run_improved.py', 'server.py', 'pyproject.toml', 'uv.lock',
              'run_coverage_campaign.sbatch', 'run_coverage_stage.sbatch', 'run_coverage_experiment.sbatch'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def head_version(repo, name):
    result = subprocess.run(['git', 'show', 'HEAD:' + name], cwd=repo, capture_output=True)
    return result.stdout if result.returncode == 0 else None


def recover_source(repo, name, expected):
    """Only accept bytes that match the recorded hash; never relax run binding."""
    current = (repo / name).read_bytes()
    if sha(current) == expected:
        return current
    original = head_version(repo, name)
    if original is not None and sha(original) == expected:
        return original
    if name == 'sota/Jailbreak/eval.py' and original is not None:
        # The working tree includes experiment instrumentation absent from HEAD.
        # Recover the old verification function while retaining that instrumentation.
        start, end = b'def verify_record(', b'def append_records('
        if all(marker in data for data in (current, original) for marker in (start, end)):
            candidate = (current[:current.index(start)]
                         + original[original.index(start):original.index(end)]
                         + current[current.index(end):])
            if sha(candidate) == expected:
                return candidate
    raise ValueError(f'Cannot recover the recorded version of {name} ({expected}); no files changed')


def source_files(repo, required):
    output = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                            cwd=repo, check=True, capture_output=True).stdout
    names = {os.fsdecode(name) for name in output.split(b'\0') if name}
    names.update(required)
    names.update(ROOT_FILES)
    names.add('src/cfg/constants.py')  # Selected configuration is deliberately gitignored.
    return sorted(name for name in names
                  if (name in ROOT_FILES or Path(name).parts[0] in ROOTS)
                  and (repo / name).is_file())


def prepare_runtime(root, repo=REPO):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    runs = sorted((root / 'runs').glob('*/run.json'))
    if not runs:
        raise ValueError('No saved run configurations found')
    recorded = [json.loads(path.read_text()) for path in runs]
    expected = recorded[0]['source_hashes']
    if any(Path(name).is_absolute() or '..' in Path(name).parts for name in expected):
        raise ValueError('Recorded source paths must stay inside the runtime')
    if any(config['source_hashes'] != expected for config in recorded):
        raise ValueError('Runs have different source versions; recover them as separate cohorts')
    recovered = {}
    for name, digest in expected.items():
        # Once recovered, retain exact versions even if HEAD or the working tree
        # changes again. Every archived candidate must still match the saved hash.
        archived = [path for path in sorted((root / 'runtimes').glob('*/' + name))
                    if path.is_file() and sha(path.read_bytes()) == digest]
        recovered[name] = archived[0].read_bytes() if archived else recover_source(repo, name, digest)
    files = {name: recovered[name] if name in recovered else (repo / name).read_bytes()
             for name in source_files(repo, expected)}
    # The selected constants must use the same recovered configuration content.
    if 'src/cfg/constants_jailbreak.py' in recovered:
        files['src/cfg/constants.py'] = recovered['src/cfg/constants_jailbreak.py']
    signature = sha(json.dumps({name: sha(data) for name, data in sorted(files.items())},
                               sort_keys=True).encode())
    destination = root / 'runtimes' / signature[:16]
    if not (repo / '.venv/bin/python').exists():
        raise ValueError('Repository Python environment is missing')
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = destination / name
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f'Frozen runtime was modified: {path}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    venv = destination / '.venv'
    if not venv.exists():
        venv.symlink_to(repo / '.venv', target_is_directory=True)
    metadata = {'runtime': str(destination), 'campaign': str(root), 'signature': signature,
                'files': {name: sha(data) for name, data in sorted(files.items())},
                'recorded_source_hashes': expected,
                'restored_files': [name for name in expected if (repo / name).read_bytes() != recovered[name]]}
    (destination / 'runtime_manifest.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    return destination


CHECK_SCRIPT = r'''
import json, os, sys
from pathlib import Path
from analysis.experiments import runtime, runner
root = Path(sys.argv[1]).resolve()
snapshot = Path.cwd().resolve()
assert Path(runtime.__file__).resolve().is_relative_to(snapshot), runtime.__file__
manifest = root / 'experiment.json'
for path in sorted((root / 'runs').glob('*/run.json')):
    saved = json.loads(path.read_text())
    for key in list(os.environ):
        if key.startswith(('JB_', 'LLMGE_', 'LLM_')):
            os.environ.pop(key)
    os.environ.update(saved['environment'])
    os.environ.update(runner.environment(path.parent, manifest.parent / saved['subset']['csv']))
    actual = runtime.configuration(manifest, root / 'initial', saved['subset']['id'], saved['seed'])
    # Operator code is independently validated by the source hash for run_improved.py.
    actual['operator_settings'] = saved['operator_settings']
    different = [key for key in set(saved) | set(actual) if saved.get(key) != actual.get(key)]
    assert not different, (path.parent.name, different)
    runtime.bind_run(path.parent, actual)
    print(path.parent.name + ': configuration matches; validated generation ' + str(runtime.latest_complete(path.parent)), flush=True)
'''


def validate_runtime(root, snapshot):
    subprocess.run([str(snapshot / '.venv/bin/python'), '-c', CHECK_SCRIPT, str(Path(root).resolve())],
                   cwd=snapshot, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--resume', action='store_true', help='Submit a controller after compatibility validation')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    snapshot = prepare_runtime(root)
    print('Recovered runtime:', snapshot, flush=True)
    validate_runtime(root, snapshot)
    print('All saved configurations and latest checkpoint checksums match.', flush=True)
    if args.resume:
        from analysis.experiments.campaign import Slurm, ACTIVE
        state_path = root / 'campaign_state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        receipt_path = root / 'recovery_submission.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        for controller in {state.get('controller_job'), receipt.get('controller_job')} - {None}:
            if Slurm().state(controller) in ACTIVE:
                print(f'Controller {controller} is already active; no duplicate submitted')
                return
        backup = root / 'recovery_backups' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup.mkdir(parents=True)
        for name in ('campaign.json', 'campaign_state.json'):
            if (root / name).exists():
                shutil.copy2(root / name, backup / name)
        result = subprocess.run([str(snapshot / '.venv/bin/python'), '-m', 'analysis.experiments.campaign',
                                 'resume', str(root), '--retry-failed'], cwd=snapshot, check=True,
                                text=True, capture_output=True)
        print(result.stdout, end='', flush=True)
        match = re.search(r'Controller job (\d+);', result.stdout)
        if not match:
            raise RuntimeError('Controller submission output was unexpected; inspect Slurm before retrying')
        receipt_path.write_text(json.dumps({'controller_job': match.group(1), 'runtime': str(snapshot),
                                          'backup': str(backup)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
