"""Serial execution against explicitly provisioned model services."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from analysis.experiments.coverage import digest, load_experiment, write_json
from analysis.experiments.runtime import latest_complete, environment_fingerprint

ROOT = Path(__file__).resolve().parents[2]


def environment(root, behaviors):
    root = Path(root).resolve()
    env = os.environ.copy()
    env.update({'JB_EXPERIMENT_ROOT': str(root), 'JB_BEHAVIORS_PATH': str(Path(behaviors).resolve()),
                'JB_FULL_BEHAVIORS': '1', 'JB_STRATIFIED': '0', 'JB_CURRICULUM': '0',
                'JB_N_BEHAVIORS_PER_EVAL': '30', 'JB_GENE_RESULTS_DIR': str(root / 'gene_results'),
                'JB_CANDIDATE_DIR': str(root / 'candidate_records'), 'JB_CACHE_DIR': str(root / 'cache'),
                'LOCAL': 'true', 'LOCAL_LLM': 'true'})
    return env


def execute(command, env, root, stage):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, env=env)
    timing = {'stage': stage, 'seconds': time.monotonic() - start, 'returncode': result.returncode}
    with (root / 'timings.jsonl').open('a') as f:
        f.write(json.dumps(timing) + '\n')
    result.check_returncode()


def require_services(env):
    for key in ('JB_TARGET_MODEL_PATH', 'JB_JUDGE_MODEL_PATH', 'JB_REPORTING_MODEL_PATH',
                'JB_MUTATOR_MODEL_PATH', 'JB_TARGET_VLLM_URL', 'JB_JUDGE_VLLM_URL', 'JB_REPORTING_VLLM_URL'):
        if not env.get(key):
            raise ValueError(f'{key} must be provided; use the experiment Slurm launcher')


def run(manifest, initial, subset, root, seed=42, final_generation=5, llm_model='llama3'):
    manifest, initial, root = map(lambda p: Path(p).resolve(), (manifest, initial, root))
    spec = load_experiment(manifest)
    selected = next(s for s in spec['subsets'] if s['id'] == subset)
    env = environment(root, manifest.parent / selected['csv'])
    require_services(env)
    command = [sys.executable, str(ROOT / 'run_improved.py'), '--experiment-manifest', str(manifest),
               '--initial-population', str(initial), '--subset', subset, '--evolution-seed', str(seed),
               '--final-generation', str(final_generation), '--checkpoints', str(root / 'ckpt'),
               '--global_path', str(root / 'global'), '--run-id', root.name, '--llm_model', llm_model,
               '--prompt_group', 'Jailbreak/traits']
    # An advisory lock prevents two jobs from mutating a single run concurrently.
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(command, env, root, 'evolution')
        evaluate_heldout(manifest, root, final_generation)


def freeze(manifest, destination, root, seed=42, llm_model='llama3'):
    manifest, destination, root = map(lambda p: Path(p).resolve(), (manifest, destination, root))
    load_experiment(manifest)
    if (destination / 'population.json').exists():
        raise ValueError('Frozen population already exists')
    env = environment(root, manifest.parent / 'heldout.csv')
    # No evaluation occurs here; only the existing strategy creation path is used.
    execute([sys.executable, str(ROOT / 'run_improved.py'), '--freeze-initial', str(destination),
             '--evolution-seed', str(seed), '--llm_model', llm_model, '--prompt_group', 'Jailbreak/traits'],
            env, root, 'freeze_initial')


def evaluate_heldout(manifest, root, final_generation=5):
    manifest, root = Path(manifest).resolve(), Path(root).resolve()
    spec = load_experiment(manifest)
    config = json.loads((root / 'run.json').read_text())
    if digest(manifest.read_bytes()) != config['experiment_hash']:
        raise ValueError('Held-out dataset does not match this run')
    env = environment(root, manifest.parent / 'heldout.csv')
    env['JB_CACHE_DIR'] = str(root / 'heldout_cache')
    env['JB_CANDIDATE_DIR'] = str(root / 'heldout_candidates')
    env['JB_GENE_RESULTS_DIR'] = str(root / 'heldout_results')
    require_services(env)
    if environment_fingerprint(env) != config['environment']:
        raise ValueError('Held-out runtime settings differ from evolution')
    from sota.Jailbreak.prompt_records import checkpoint_fingerprint
    for key, model in config['models'].items():
        if env.get(key) != model['path'] or checkpoint_fingerprint(env.get(key)) != model['fingerprint']:
            raise ValueError(f'Held-out model differs from evolution: {key}')
    complete = latest_complete(root)
    if complete is None:
        raise ValueError('No completed generations to evaluate')
    scratch = root / 'heldout_models'
    scratch.mkdir(exist_ok=True)
    for gen in range(min(complete, final_generation) + 1):
        snap = json.loads((root / 'snapshots' / f'{gen:04d}.json').read_text())
        for member in snap['members']:
            source_hash = member['strategy_hash']
            if not source_hash:
                continue
            gene = member['id']
            observation = root / 'observations/heldout' / f'{gen:04d}' / f'{gene}.json'
            if observation.exists():
                data = json.loads(observation.read_text())
                if data.get('evaluation_complete') and data['strategy_hash'] == source_hash and {r['behavior_id'] for r in data['records']} == set(spec['heldout_ids']):
                    continue
            source = root / 'strategies' / f'{source_hash}.py'
            if digest(source.read_bytes()) != source_hash:
                raise ValueError('Archived strategy changed')
            shutil.copy2(source, scratch / f'prompt_{gene}.py')
            execute([sys.executable, str(ROOT / 'sota/Jailbreak/eval.py'), '--model', f'prompt_{gene}',
                     '--variant_dir', str(scratch), '--generation', str(gen), '--run-id', root.name,
                     '--split', 'heldout', '--random_seed', str(config['seed'])], env, root, 'heldout')


def estimate(manifest, pilot=None, repeats=1, final_generation=5):
    spec = load_experiment(manifest)
    if repeats < 1 or final_generation < 0:
        raise ValueError('Invalid repeat count or generation limit')
    result = {'subsets': len(spec['subsets']), 'repeats': repeats,
              'runs': len(spec['subsets']) * repeats, 'snapshots_per_run': final_generation + 1,
              'behaviors_per_split': 30, 'initial_strategy_count': 24,
              'nominal_retained_prompt_observations': len(spec['subsets']) * repeats * (final_generation + 1) * 24 * 60,
              'note': 'Population size varies; estimates exclude service startup unless supplied by launcher. No jobs submitted.'}
    if pilot:
        rows = [json.loads(line) for line in (Path(pilot) / 'timings.jsonl').read_text().splitlines()]
        elapsed = sum(r['seconds'] for r in rows if r['returncode'] == 0 and r['stage'] in {'evolution', 'heldout'})
        generations = latest_complete(pilot)
        if generations is None or generations < 1:
            raise ValueError('Pilot needs at least one completed evolution generation')
        per_run = elapsed * (final_generation + 1) / (generations + 1)
        result.update(estimated_campaign_hours=per_run * result['runs'] / 3600,
                      estimated_gpu_hours=4 * per_run * result['runs'] / 3600,
                      pilot_generations=generations)
    return result
