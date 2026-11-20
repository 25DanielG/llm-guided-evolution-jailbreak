"""Pinned AdvBench sampling and independent generation-two acceptance gates."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import random

TIERS = ('easy', 'medium', 'hard')
POOL_RELATIVE = 'sota/Jailbreak/behaviors/curated_tiered_jb_strat_base1.csv'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sample_records(pool, seed):
    """Baris's ordering and Random(seed) draws; never reconstruct or retier data."""
    require(type(seed) is int and seed in (0, 1, 2), 'Expected seed 0, 1 or 2')
    with Path(pool).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 90, 'AdvBench requires the pinned 90-row pool')
    require(len({r['id'] for r in rows}) == len({r['behavior'].strip() for r in rows}) == 90,
            'Pool IDs and behavior texts must be unique')
    require(all(r['source'] == 'advbench' for r in rows), 'Expected AdvBench source')
    require(collections.Counter(r.get('tier') for r in rows) == dict(easy=30, medium=30, hard=30),
            'Expected original 30/30/30 difficulty tiers')
    ordered = []
    for index, row in enumerate(rows):
        row = dict(row)
        row['behavior'] = row['behavior'].strip()
        try:
            difficulty = float(row['difficulty']) if 'difficulty' in row else index
        except (TypeError, ValueError):
            difficulty = index
        ordered.append((difficulty, row))
    ordered.sort(key=lambda pair: pair[0])  # stable, exactly as load_behavior_rows
    rng = random.Random(seed)
    picked = []
    for tier, quota in zip(TIERS, (16, 17, 17)):
        picked.extend(rng.sample([r for _, r in ordered if r['tier'] == tier], quota))
    return picked


def panel_manifest(pool, seed):
    picked = sample_records(pool, seed)
    texts = [r['behavior'] for r in picked]
    return dict(pool_sha256=digest(pool), pool_count=90, seed=seed, sample_count=50,
                tier_counts=dict(collections.Counter(r['tier'] for r in picked)),
                selected_ids=[r['id'] for r in picked],
                ordered_behavior_sha256=hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest())


def verify_panel(folder, config, require_approval=False):
    actual = panel_manifest(Path(folder) / POOL_RELATIVE, config['protocol']['seed'])
    require(actual == config['sampling'], 'Pool, seed, tiers or selected behavior order differs')
    require(actual['pool_sha256'] == config['dataset_sha256'], 'Dataset identity differs')
    if require_approval:
        approval = read(Path(folder) / 'full-submission-approval.json')
        require(approval.get('approved') is True
                and approval.get('config_sha256') == config['config_sha256']
                and approval.get('run_id') == config['run_id'],
                'Explicit user approval for this exact full configuration is required')
    return actual


def verify_acceptance(folder, config):
    """Recheck reporter, actual audit rows and seed/dataset-bound checkpoints."""
    from src import checkpointing
    from sota.Jailbreak.reporting_inputs import validate_audits
    folder = Path(folder)
    verify_panel(folder, config)
    run = folder / config['run_id']
    probe = read(run / 'pipeline-probe.json')
    require(probe.get('status') == 'passed' and probe.get('target_budget') == 512
            and probe.get('target_budget_check', {}).get('finish_reason') == 'stop',
            'Complete-final probe at the actual 512-token budget is required')
    require(probe.get('reporter', {}).get('parse_ok') is True
            and type(probe['reporter'].get('success')) is bool
            and probe.get('judge', {}).get('parse_ok') is True,
            'Independent reporter and judge probes must parse')
    require(probe.get('target_cache_hit') is True and probe.get('judge_cache_hit') is True,
            'Target and judge cache checks must pass')
    for generation in (0, 1, 2):
        _, _, marker = checkpointing.inspect(run / 'ckpt', run / 'global',
                                             f'checkpoint_gen_{generation}.pkl')
        identity = marker['identity']
        environment = identity['environment']
        require(identity.get('selection') == 'nsga2'
                and identity.get('behavior_sha256') == config['dataset_sha256']
                and environment.get('LLMGE_EXPERIMENT_CONFIG_SHA256') == config['config_sha256']
                and environment.get('JB_BEHAVIOR_SEED') == str(config['protocol']['seed'])
                and environment.get('JB_STRATIFIED') == '1', 'Checkpoint protocol identity differs')
    continuation = read(run / 'continuation-passed.json')
    require(continuation.get('restored_generation') == 1 and continuation.get('completed_generation') == 2
            and continuation.get('run_id') == config['run_id']
            and continuation.get('config_sha256') == config['config_sha256'],
            'Fresh-process generation-one to generation-two continuation is required')
    candidates = folder / 'sota/Jailbreak/results' / config['run_id']
    audits = validate_audits(candidates, config['run_id'], [2])
    require(len(audits) == 1 and audits[0]['status'] == 'complete'
            and audits[0]['requested'] == audits[0]['selected'] == audits[0]['completed'] == 3,
            'Generation-two independent audit must complete all three requested checks')
    expected = [r['behavior'] for r in sample_records(folder / POOL_RELATIVE, config['protocol']['seed'])]
    groups = collections.defaultdict(list)
    for line in (candidates / 'candidates.jsonl').read_text().splitlines():
        row = json.loads(line)
        require(len({row['target_model'], row['judge_model'], row['reporting_model']}) == 3,
                'Target, judge and reporter identities must be distinct')
        groups[(row['generation'], row['candidate_id'])].append(row)
    require({g for g, _ in groups} >= {0, 1, 2}, 'Candidate observations missing for a validated generation')
    for rows in groups.values():
        require(sorted(r['behavior_index'] for r in rows) == list(range(50))
                and [r['behavior'] for r in sorted(rows, key=lambda r: r['behavior_index'])] == expected,
                'Actual observations do not use the exact ordered 50-behavior seed panel')
    return dict(status='passed', sampling=config['sampling'], reporting_audits=audits)


if __name__ == '__main__':
    # Called only after the runner's second driver process has committed gen 2.
    import os
    from src.run_status import atomic_json
    root = Path(__file__).resolve().parents[1]
    config = read(root / 'experiment.json')
    require(os.environ.get('SLURM_JOB_ID'), 'Continuation receipt requires its live Slurm allocation')
    from src import checkpointing
    run = root / config['run_id']
    commits = [checkpointing.inspect(run / 'ckpt', run / 'global', f'checkpoint_gen_{g}.pkl')[2] for g in (1, 2)]
    atomic_json(run / 'continuation-passed.json', dict(run_id=config['run_id'],
        config_sha256=config['config_sha256'], restored_generation=1, completed_generation=2,
        checkpoint_sha256=[c['checkpoint_sha256'] for c in commits], job_id=os.environ['SLURM_JOB_ID']))
