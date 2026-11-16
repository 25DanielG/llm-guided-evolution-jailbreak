"""Paired difficulty-weighted tournament campaign; see DIFFICULTY_TOURNAMENT.md.

No model calls or Slurm submissions occur on import, prepare, or summarize.
GPU stages require an explicit allocation and immutable prepared inputs.
"""
import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import time

from src.checkpointing import digest
from src.run_status import atomic_json

BASE_COMMIT = '7349752f1a246730056277ba066bf5cdd2f2950c'
ROOT = Path(__file__).resolve().parents[2]
ARMS = ('tournament', 'difficulty_tournament')
TIERS = ('easy', 'medium', 'hard')
PACKAGES = ('torch', 'transformers', 'numpy', 'deap', 'requests', 'accelerate')


def read_json(path):
    return json.loads(Path(path).read_text())


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def versions():
    result = {'python': sys.version}
    for package in PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def serving_fingerprint(vllm_bin):
    """Reject a CPU torch wheel without importing models or reading weight files."""
    executable = Path(vllm_bin).expanduser().resolve()
    if not os.access(executable, os.X_OK):
        raise ValueError(f'Missing vLLM executable: {executable}')
    probe = r"""
import ast, importlib.metadata as metadata, importlib.util, json
from pathlib import Path
versions = {p: metadata.version(p) for p in ('vllm', 'torch', 'transformers')}
torch_root = Path(importlib.util.find_spec('torch').origin).parent
cuda = None
for node in ast.parse((torch_root / 'version.py').read_text()).body:
    target = node.target if isinstance(node, ast.AnnAssign) else None
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    if isinstance(target, ast.Name) and target.id == 'cuda':
        cuda = ast.literal_eval(node.value)
versions['torch_cuda'] = cuda
versions['cuda_libraries_present'] = all((torch_root / 'lib' / name).is_file()
    for name in ('libtorch_cuda.so', 'libc10_cuda.so'))
print(json.dumps(versions))
"""
    result = json.loads(subprocess.check_output([str(executable.parent / 'python'), '-c', probe], text=True))
    if not result['torch_cuda'] or not result['cuda_libraries_present'] or '+cpu' in result['torch']:
        raise ValueError(f'CPU-only or incomplete PyTorch in {executable.parent.parent}; '
                         'use --vllm-bin pointing to a CUDA-enabled serving environment')
    return str(executable), result


def freeze_panel(source, n=None, stratified=False):
    with Path(source).open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'behavior', 'tier'} if stratified else {'behavior'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Calibrated behavior CSV must contain behavior and tier columns; '
                             'supply --behaviors PATH. Do not fabricate difficulty labels.')
        rows = [dict(row, behavior=row['behavior'].strip(), tier=(row.get('tier') or '').strip().lower() if stratified else '') for row in reader]
    if any(not row['behavior'] or (stratified and row['tier'] not in TIERS) for row in rows):
        raise ValueError('Every behavior must have a nonempty text and easy/medium/hard tier')
    if len({row['behavior'] for row in rows}) != len(rows):
        raise ValueError('Duplicate behavior text in calibrated CSV')
    if n is None:
        n = min(50, len(rows))
    if n < 1:
        raise ValueError('Behavior panel cannot be empty')
    if not stratified:
        if len(rows) < n:
            raise ValueError(f'Need at least {n} behaviors')
        return random.Random(0).sample(rows, n)
    grouped = {tier: [row for row in rows if row['tier'] == tier] for tier in TIERS}
    # Same quota policy as eval.py: distribute the remainder hard -> medium.
    base, remainder = divmod(n, 3)
    quotas = {tier: min(len(grouped[tier]), base + (i < remainder))
              for i, tier in enumerate(reversed(TIERS))}
    while sum(quotas.values()) < n:
        changed = False
        for tier in reversed(TIERS):
            if sum(quotas.values()) < n and quotas[tier] < len(grouped[tier]):
                quotas[tier] += 1
                changed = True
        if not changed:
            raise ValueError(f'Need at least {n} calibrated behaviors')
    if not all(quotas.values()):
        raise ValueError('Every tier must be represented')
    rng = random.Random(0)
    return [row for tier in TIERS for row in rng.sample(grouped[tier], quotas[tier])]


def model_inventory(path):
    path = Path(path).expanduser().resolve()
    if not (path / 'config.json').is_file():
        raise ValueError(f'Missing model checkpoint: {path}; stage it before prepare')
    files = sorted(p for p in path.iterdir() if p.is_file() and
                   (p.suffix in ('.json', '.safetensors', '.bin', '.model', '.txt', '.jinja')))
    if not any(p.suffix in ('.safetensors', '.bin') for p in files):
        raise ValueError(f'No model weights in {path}')
    # Never read multi-GB weight shards during preparation. Existing model files
    # stay in place; the resolved path, origin revision, config and stat metadata
    # identify the deployment without a costly full-checkpoint scan.
    small_configs = {'config.json', 'generation_config.json', 'tokenizer_config.json',
                     'special_tokens_map.json', 'model.safetensors.index.json',
                     'pytorch_model.bin.index.json', 'difficulty_model_origin.json'}
    inventory = {}
    for item in files:
        stat = item.stat()
        info = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if item.name in small_configs:
            info['sha256'] = digest(item)
        inventory[item.name] = info
    origin = path / 'difficulty_model_origin.json'
    return {'path': str(path), 'fingerprint': 'config-hashes-and-file-metadata',
            'origin': read_json(origin) if origin.exists() else None, 'files': inventory}


def snapshot_source(destination):
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    new = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    paths = sorted(set(p for p in tracked + new if p and Path(p).suffix in
                       ('.py', '.yaml', '.yml', '.json', '.toml', '.txt', '.md', '.sh', '.sbatch', '.csv', '.lock')))
    hashes = {}
    with tarfile.open(destination, 'w:gz') as archive:
        for name in paths:
            path = ROOT / name
            if not path.is_file():
                continue
            if path.is_symlink() and not path.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError(f'External source symlink: {path}')
            hashes[name] = digest(path)
            archive.add(path, arcname=name, recursive=False)
    return hashes


def stage_target(args):
    """Resolve a named HF revision once and persist the exact downloaded commit."""
    from huggingface_hub import HfApi, snapshot_download
    destination = args.destination.expanduser().absolute()
    if destination.exists():
        raise ValueError('Destination exists; use the existing checkpoint or a new empty path')
    repository = 'mistralai/Mistral-7B-Instruct-v0.2'
    revision = HfApi().model_info(repository, revision=args.revision).sha
    snapshot_download(repository, revision=revision, local_dir=destination,
                      allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja'])
    atomic_json(destination / 'difficulty_model_origin.json', {'repository': repository, 'revision': revision})
    print(f'Staged {repository}@{revision} at {destination}')


def prepare(args):
    campaign = args.campaign.resolve()
    if campaign.exists():
        raise ValueError('Campaign directory exists; use a new name, never overwrite prepared inputs')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        raise ValueError('Seeds must be distinct unsigned 32-bit integers')
    if args.final_generation < 1 or args.population < 6:
        raise ValueError('Need at least one generation and six initial individuals for retained elites')
    panel = freeze_panel(args.behaviors, args.n_behaviors, args.panel_mode == 'stratified')
    # Validate inputs before creating artifacts; never scan or copy model weights.
    models = {role: model_inventory(getattr(args, role + '_model'))
              for role in ('target', 'judge', 'reporting', 'mutator')}
    python_bin = str(Path(sys.executable).absolute())
    vllm_bin, serving_versions = serving_fingerprint(args.vllm_bin)
    if any(v is None for v in versions().values()):
        raise ValueError('Experiment environment is missing required packages')
    subprocess.run(['git', 'merge-base', '--is-ancestor', BASE_COMMIT, 'HEAD'], cwd=ROOT, check=True)
    campaign.mkdir(parents=True)
    hashes = snapshot_source(campaign / 'source.tar.gz')
    with (campaign / 'behaviors.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['behavior', 'tier'])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in panel)
    atomic_json(campaign / 'panel.json', [row['behavior'] for row in panel])
    manifest = dict(schema_version=1, base_commit=BASE_COMMIT,
                    code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    source_hashes=hashes, source_sha256=digest(campaign / 'source.tar.gz'),
                    behaviors_sha256=digest(campaign / 'behaviors.csv'), panel_sha256=digest(campaign / 'panel.json'),
                    original_behaviors_sha256=digest(args.behaviors), models=models, versions=versions(),
                    serving_versions=serving_versions, python=python_bin, vllm_bin=vllm_bin,
                    seeds=args.seeds, final_generation=args.final_generation, population=args.population,
                    tournament_size=3, concurrency_candidates=[16, 32, 64],
                    target_max_new_tokens=512, target_temperature=0.0, judge_prompt='guide',
                    mutation_probability=0.8, crossover_probability=0.15, elites=6,
                    verification_trials=3, behavior_seed=0, panel_mode=args.panel_mode,
                    primary_endpoint='overall_verified_asr', panel_size=len(panel))
    manifest['identity'] = json_hash(manifest)
    atomic_json(campaign / 'manifest.json', manifest)
    print(f'Prepared immutable campaign: {campaign}')


def load_manifest(campaign):
    manifest = read_json(campaign / 'manifest.json')
    payload = {k: v for k, v in manifest.items() if k != 'identity'}
    if json_hash(payload) != manifest['identity']:
        raise ValueError('Campaign manifest changed')
    for name, key in [('source.tar.gz', 'source_sha256'), ('behaviors.csv', 'behaviors_sha256'), ('panel.json', 'panel_sha256')]:
        if digest(campaign / name) != manifest[key]:
            raise ValueError(f'Frozen input changed: {name}')
    return manifest


def validate_runtime(manifest):
    for relative, expected in manifest['source_hashes'].items():
        if relative.startswith('analysis/experiments/difficulty') and digest(ROOT / relative) != expected:
            raise ValueError(f'Campaign driver changed since preparation: {relative}')
    if versions() != manifest['versions']:
        raise ValueError('Experiment Python/dependency identity changed')
    _, serving = serving_fingerprint(manifest['vllm_bin'])
    if serving != manifest['serving_versions']:
        raise ValueError('Serving environment changed')
    for model in manifest['models'].values():
        for name, info in model['files'].items():
            stat = (Path(model['path']) / name).stat()
            if (stat.st_size, stat.st_mtime_ns) != (info['size'], info['mtime_ns']):
                raise ValueError(f'Model file changed: {model["path"]}/{name}; prepare a new campaign')


def extract_source(campaign, work):
    work.mkdir(parents=True, exist_ok=False)
    with tarfile.open(campaign / 'source.tar.gz') as archive:
        archive.extractall(work, filter='data')


def slot(manifest, stage, index):
    limit = len(manifest['seeds']) * (2 if stage == 'run' else 1)
    if not 0 <= index < limit:
        raise ValueError(f'Index must be in [0, {limit})')
    return (manifest['seeds'][index // 2], ARMS[index % 2]) if stage == 'run' else (manifest['seeds'][index], 'initial')


def environment(campaign, run, work, attempt, manifest, seed, arm, port):
    # Do not inherit experiment/model overrides from the submitting shell.
    env = {k: v for k, v in os.environ.items() if not k.startswith(('JB_', 'LLMGE_', 'DT_'))}
    env.update(LOCAL='true', LOCAL_LLM='true', DELAYED_CHECK='false', USE_TF='0', USE_TORCH='1',
               DT_SEED=str(seed), DT_STRICT_EVAL='1', DT_IDENTITY=manifest['identity'], DT_CAMPAIGN_ID=manifest['identity'],
               DT_PANEL_JSON=str(campaign / 'panel.json'), DT_SELECTION_DIR=str(run / 'selection'),
               DT_WORK_ROOT=str(work), DT_REQUEST_LOG=str(run / 'requests.jsonl'), DT_PYTHON=manifest['python'], JB_BEHAVIORS_PATH=str(campaign / 'behaviors.csv'),
               JB_FULL_BEHAVIORS='1', JB_STRATIFIED='0', JB_CURRICULUM='0', JB_BEHAVIOR_SEED='0',
               JB_JUDGE_MODE='llm', JB_JUDGE_PROMPT='guide', JB_JUDGE_TEMPERATURE='0',
               JB_VERIFY_THRESHOLD='0.5', JB_AUDIT_SAMPLE_SIZE='3',
               JB_SELECTION='tournament' if arm == 'initial' else arm, JB_TOURNAMENT_SIZE='3',
               JB_TARGET_MAX_NEW_TOKENS='512', JB_TARGET_TEMPERATURE='0.0',
               JB_MUTATOR_MAX_NEW_TOKENS='850', JB_REQUEST_TIMEOUT='180',
               JB_MAX_CONCURRENCY='16', LLMGE_START_POPULATION_SIZE=str(manifest['population']),
               LLMGE_POPULATION_SIZE=str(manifest['population']),
               LLMGE_NUM_GENERATIONS='1' if arm == 'initial' else str(manifest['final_generation'] + 1),
               LLMGE_RUN_ID='evolution', SERVER_HOSTNAME='127.0.0.1', LLMGE_PORT=str(port + 3),
               JB_STOP_FILE=os.environ.get('DT_STOP_FILE', str(attempt / 'stop-requested')),
               JB_STATUS_FILE=str(run / 'status.json'), JB_CACHE_DIR=str(work / 'cache'),
               JB_GENE_RESULTS_DIR=str(work / 'sota/Jailbreak/results'),
               JB_BALANCE_TURN_DEPTHS='1', JB_STRUCTURAL_TURN_DEPTH_MUTATION='1',
               VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_ALLREDUCE_USE_FLASHINFER='0',
               UV_PROJECT_ENVIRONMENT=str(Path(manifest['python']).parent.parent),
               UV_NO_SYNC='1', UV_OFFLINE='1', PYTHONHASHSEED='0',
               PYTHONPATH=str(work), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for i, role in enumerate(('target', 'judge', 'reporting')):
        env[f'JB_{role.upper()}_MODEL_PATH'] = manifest['models'][role]['path']
        env[f'JB_{role.upper()}_SERVED_NAME'] = role
        env[f'JB_{role.upper()}_VLLM_URL'] = f'http://127.0.0.1:{port+i}/v1'
    env['JB_MUTATOR_MODEL_PATH'] = manifest['models']['mutator']['path']
    # Local CPU clients must use the existing environment, never synchronize uv.
    uv = shutil.which('uv') or str(Path.home() / '.local/bin/uv')
    env['PATH'] = str(Path(uv).parent) + os.pathsep + str(Path(manifest['vllm_bin']).parent) + os.pathsep + env.get('PATH', '')
    return env


def install_local_templates(work):
    # Replace cluster templates only inside the isolated runtime tree.
    # They otherwise load modules and force CUDA_VISIBLE_DEVICES=0 in CPU clients.
    config = work / 'runtime-slurm'
    config.mkdir(exist_ok=True)
    (config / 'slurm_config.yaml').write_text(json.dumps({
        'gpu_selection': '', 'python_bash_script': '#!/bin/bash\nset -euo pipefail\n{}\n',
        'llm_bash_script': '#!/bin/bash\nset -euo pipefail\n{}\n'}))
    return str(config)


def run_child(command, work, env, logfile):
    with Path(logfile).open('a') as stream:
        child_env = dict(env, CUDA_VISIBLE_DEVICES='')
        proc = subprocess.Popen(command, cwd=work, env=child_env, stdout=stream, stderr=subprocess.STDOUT)
        return proc.wait()


def import_bundle(initial, work):
    marker = read_json(initial / 'complete.json')
    if marker.get('stage') != 'initial':
        raise ValueError('Initial bundle is incomplete')
    for rel, expected in marker['bundle_hashes'].items():
        source = initial / 'work' / rel
        if digest(source) != expected:
            raise ValueError(f'Initial bundle changed: {rel}')
        destination = work / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return marker


def stage(args):
    from analysis.experiments.difficulty_services import Services, port_block
    campaign = args.campaign.resolve()
    manifest = load_manifest(campaign)
    validate_runtime(manifest)
    seed, arm = slot(manifest, args.command, args.index)
    run = campaign / ('initial' if arm == 'initial' else 'runs') / f'{seed}-{arm}'
    attempt = Path(os.environ.get('DT_ATTEMPT_DIR', run / 'attempts' / str(time.time_ns()))).resolve()
    attempt.mkdir(parents=True, exist_ok=True)
    with exclusive(run / '.lock'):
        work = run / 'work'
        resume = getattr(args, 'resume', 'new')
        if (run / 'complete.json').exists():
            raise ValueError('Stage is already complete')
        if resume == 'new':
            if work.exists():
                raise ValueError('Run exists; use required resume or a new campaign')
            extract_source(campaign, work)
        elif not work.is_dir():
            raise ValueError('No existing work directory to resume')
        for relative, expected in manifest['source_hashes'].items():
            if digest(work / relative) != expected:
                raise ValueError(f'Runtime source changed: {relative}')
        if arm != 'initial' and resume == 'new':
            initial = campaign / 'initial' / f'{seed}-initial'
            imported = import_bundle(initial, work)
            if imported['identity'] != manifest['identity'] or imported['seed'] != seed:
                raise ValueError('Initial population belongs to another campaign')
        with port_block() as port:
            env = environment(campaign, run, work, attempt, manifest, seed, arm, port)
            env['LLMGE_SLURM_CONFIG_DIR'] = install_local_templates(work)
            (work / 'hostname.log').write_text('127.0.0.1\n')
            env['JB_RESUME_MODE'] = resume
            if arm != 'initial' and resume == 'new':
                env['DT_INITIAL_ROOT'] = str(initial / 'work')
            # Arm-specific checkpoint identity includes the common calibrated throughput setting.
            if arm != 'initial':
                throughput = read_json(campaign / 'initial' / f'{seed}-initial' / 'throughput.json')
                env['JB_MAX_CONCURRENCY'] = str(throughput['chosen'])
                env['DT_IDENTITY'] = json_hash([manifest['identity'], throughput])
            record = dict(identity=manifest['identity'], seed=seed, arm=arm,
                          job_id=os.getenv('SLURM_JOB_ID'), started=time.time(),
                          versions=versions(), cuda_modules=os.getenv('LOADEDMODULES'), environment={k: v for k, v in env.items()
                              if k.startswith(('DT_', 'JB_', 'LLMGE_', 'UV_'))},
                          gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv'], text=True))
            atomic_json(attempt / 'runtime.json', record)
            old_term = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            started = time.monotonic()
            try:
                with Services(work, attempt, env, manifest['models'], manifest['vllm_bin'], port):
                    if arm == 'initial' and not (run / 'throughput.json').exists():
                        code = run_child([manifest['python'], '-m', 'analysis.experiments.difficulty_worker',
                                          'benchmark', '--output', str(run / 'throughput.json')],
                                         work, dict(env, DT_PHASE='benchmark'), attempt / 'benchmark.log')
                        if code:
                            raise RuntimeError('Throughput check failed; see benchmark log')
                    if arm == 'initial':
                        env['JB_MAX_CONCURRENCY'] = str(read_json(run / 'throughput.json')['chosen'])
                    record['effective_concurrency'] = int(env['JB_MAX_CONCURRENCY'])
                    atomic_json(attempt / 'runtime.json', record)
                    command = [manifest['python'], 'run_improved.py', '--checkpoints', 'evolution/ckpt',
                               '--global_path', 'evolution/global', '--llm_model', 'llama3',
                               '--prompt_group', 'Jailbreak/traits', '--run-id', 'evolution', '--resume-mode', resume]
                    code = run_child(command, work, env, attempt / 'evolution.log')
                    if code:
                        raise SystemExit(code)
                    if arm != 'initial':
                        code = run_child([manifest['python'], '-m', 'analysis.experiments.difficulty_worker',
                                          'audit', '--output', str(run / 'audit.json')], work, dict(env, DT_PHASE='audit'), attempt / 'audit.log')
                        if code:
                            raise RuntimeError('Final audit failed; resume the completed evolution to retry audit')
            finally:
                signal.signal(signal.SIGTERM, old_term)
                atomic_json(attempt / 'usage.json', {'elapsed_seconds': time.monotonic() - started, 'allocated_gpus': 5})
        expected_generation = 0 if arm == 'initial' else manifest['final_generation']
        checkpoint = work / 'evolution/ckpt' / f'checkpoint_gen_{expected_generation}.json'
        if not checkpoint.exists():
            raise ValueError(f'Missing final committed checkpoint: {checkpoint}')
        complete = dict(stage='initial' if arm == 'initial' else 'run', identity=manifest['identity'], seed=seed, arm=arm)
        if arm == 'initial':
            paths = []
            for directory in ('sota/Jailbreak/models/llmge_models', 'sota/Jailbreak/results', 'cache'):
                paths.extend(p for p in (work / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
            complete['bundle_hashes'] = {str(p.relative_to(work)): digest(p) for p in paths}
        atomic_json(run / 'complete.json', complete)
        print(f'Completed {run}')


def summarize(args):
    campaign = args.campaign.resolve()
    manifest = load_manifest(campaign)
    rows = []
    for index in range(2 * len(manifest['seeds'])):
        seed, arm = slot(manifest, 'run', index)
        run = campaign / 'runs' / f'{seed}-{arm}'
        complete = read_json(run / 'complete.json')
        if complete['identity'] != manifest['identity']:
            raise ValueError('Run identity mismatch')
        audit = read_json(run / 'audit.json')
        usage = [read_json(p) for p in (campaign / 'attempts' / f'run-{index}').glob('*/usage.json')]
        usage += [read_json(p) for p in (run / 'attempts').glob('*/usage.json')]
        requests = [json.loads(line) for line in (run / 'requests.jsonl').read_text().splitlines() if line]
        rows.append(dict(seed=seed, arm=arm, **audit['summary'],
                         inference_requests=len(requests), failed_requests=sum(bool(r.get('error')) for r in requests),
                         gpu_hours=sum(u['elapsed_seconds'] * u['allocated_gpus'] / 3600 for u in usage)))
    metrics = ['overall_verified_asr']
    if manifest['panel_mode'] == 'stratified':
        metrics.append('hard_verified_asr')
    differences = []
    for seed in manifest['seeds']:
        plain, weighted = [next(row for row in rows if row['seed'] == seed and row['arm'] == arm) for arm in ARMS]
        differences.append({'seed': seed, **{metric: weighted[metric] - plain[metric] for metric in metrics}})
    aggregate = {metric: {'mean_difference': statistics.mean(d[metric] for d in differences),
                          'min_difference': min(d[metric] for d in differences),
                          'max_difference': max(d[metric] for d in differences)}
                 for metric in metrics}
    output = campaign / 'report'
    output.mkdir(exist_ok=True)
    atomic_json(output / 'summary.json', dict(runs=rows, paired_differences=differences, aggregate=aggregate))
    with (output / 'runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    from analysis.experiments.difficulty_worker import plot_report
    plot_report(campaign, manifest, output)
    print(json.dumps(aggregate, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    staging = commands.add_parser('stage-target', help='Download the missing Mistral checkpoint at a pinned revision')
    staging.add_argument('--destination', type=Path, required=True)
    staging.add_argument('--revision', default='main')
    prep = commands.add_parser('prepare')
    prep.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    prep.add_argument('--behaviors', type=Path, default=ROOT / 'sota/Jailbreak/behaviors/curated.csv')
    prep.add_argument('--panel-mode', choices=('unstratified', 'stratified'), default='unstratified')
    prep.add_argument('--population', type=int, default=24)
    prep.add_argument('--final-generation', type=int, default=16)
    prep.add_argument('--n-behaviors', type=int, default=None, help='Default: all available behaviors, capped at 50')
    local = Path.home() / 'scratch/llm_storage'
    prep.add_argument('--target-model', default=str(local / 'Mistral-7B-Instruct-v0.2'))
    prep.add_argument('--judge-model', default='/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct')
    prep.add_argument('--reporting-model', default=str(local / 'HarmBench-Llama-2-13b-cls'))
    prep.add_argument('--mutator-model', default=str(local / 'Llama-3.1-8B-Instruct'))
    prep.add_argument('--vllm-bin', default=str(Path.home() / '.local/bin/vllm'))
    initial = commands.add_parser('initial')
    initial.add_argument('--index', type=int, required=True)
    initial.add_argument('--resume', choices=('new', 'required'), default='new')
    run = commands.add_parser('run')
    run.add_argument('--index', type=int, required=True)
    run.add_argument('--resume', choices=('new', 'required'), required=True)
    summary = commands.add_parser('summarize')
    for command in (prep, initial, run, summary):
        command.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args(argv)
    {'stage-target': stage_target, 'prepare': prepare, 'initial': stage, 'run': stage, 'summarize': summarize}[args.command](args)


if __name__ == '__main__':
    main()
