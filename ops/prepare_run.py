"""Prepare isolated files and ports; never submit a job or load weights."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PORTS = {'gpt-oss-20b':21000, 'gpt-oss-120b':22000,
         'mistral-7b':23000, 'devstral-small-2-24b':24000}


def prepare(target, destination, run_id, seed, port_base=None):
    destination = Path(destination).expanduser().resolve()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', run_id):
        raise ValueError('RUN_ID must contain only letters, numbers, dots, underscores or hyphens')
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError('Keep run checkouts outside the source checkout')
    manifest = json.loads((ROOT/'ops/data/shared50/sample.json').read_text())
    selected = ROOT/'ops/data/shared50/selected.csv'
    if manifest['count'] != 50 or hashlib.sha256(selected.read_bytes()).hexdigest() != manifest['selected_sha256']:
        raise ValueError('Frozen 50-behavior sample failed verification')
    port_base = port_base if port_base is not None else PORTS[target]
    if not 1024 <= port_base <= 65530:
        raise ValueError('Five consecutive unprivileged ports are required')
    paths = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    destination.mkdir(parents=True,exist_ok=False)
    for name in filter(None,paths):
        if name.startswith('ops/'):
            continue
        src, dst = ROOT/name, destination/name
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True,exist_ok=True)
        if src.is_symlink():
            dst.symlink_to(src.readlink())
        else:
            shutil.copy2(src,dst)
    for name in ('common.env','prepare_run.py','freeze_behaviors.py','pipeline_probe.py','validation.sbatch',
                 'auxiliary-models.json','launch_checks.py','gated_validation.sbatch','gated_full.sbatch',
                 'full_payload.sh',
                 'ICE-REPORT.md','BARIS-HANDOFF.txt'):
        src=ROOT/'ops'/name
        if src.exists():
            (destination/'ops').mkdir(exist_ok=True)
            shutil.copy2(src,destination/'ops'/name)
    shutil.copytree(ROOT/'ops/data',destination/'ops/data')
    (destination/'.venv').symlink_to('/storage/ice1/8/3/pablo3/llmge-redsea/.venv',target_is_directory=True)
    env = {'RUN_ID':run_id, 'LLMGE_RUN_ID':run_id, 'LLMGE_EVOLUTION_SEED':str(seed),
           'JB_RUN_TARGET_PROFILE':target, 'JB_RESUME_MODE':'new',
           'JB_CACHE_DIR':str(destination/'cache'),
           'XDG_CACHE_HOME':str(destination/'runtime-cache/xdg'),
           'HF_HOME':str(destination/'runtime-cache/huggingface'),
           'TORCH_HOME':str(destination/'runtime-cache/torch'),
           'TORCHINDUCTOR_CACHE_DIR':str(destination/'runtime-cache/inductor'),
           'TRITON_CACHE_DIR':str(destination/'runtime-cache/triton'),
           'VLLM_CACHE_ROOT':str(destination/'runtime-cache/vllm'),
           'FLASHINFER_WORKSPACE_BASE':str(destination/'runtime-cache/flashinfer'),
           'CUDA_CACHE_PATH':str(destination/'runtime-cache/cuda'),
           'JB_GENE_RESULTS_DIR':str(destination/'results'),
           'JB_SMOKE_RESULTS':str(destination/'serving-smoke'),
           'JB_TARGET_VLLM_PORT':str(port_base), 'JB_JUDGE_VLLM_PORT':str(port_base+1),
           'JB_REPORTING_VLLM_PORT':str(port_base+2), 'JB_MUTATOR_VLLM_PORT':str(port_base+3),
           'LLMGE_PORT':str(port_base+4),
           'JB_SMOKE_PORT':str(port_base+500), 'JB_SMOKE_PROXY_PORT':str(port_base+501)}
    identity = {'target':target, 'driver_evolution_seed':seed, 'sample_sha256':manifest['selected_sha256'],
                'common_env_sha256':hashlib.sha256((ROOT/'ops/common.env').read_bytes()).hexdigest(),
                'source_files':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                for p in paths if p and (ROOT/p).is_file()
                                and (p.startswith(('src/','templates/','sota/Jailbreak/','config/'))
                                     or p in ('run_improved.py','run_jailbreak.sbatch','run_overnight.sbatch','server.py'))}}
    for name in ('pipeline_probe.py', 'launch_checks.py', 'validation.sbatch', 'gated_validation.sbatch',
                 'gated_full.sbatch', 'full_payload.sh', 'auxiliary-models.json'):
        path = ROOT / 'ops' / name
        if path.exists():
            identity['source_files']['ops/' + name] = hashlib.sha256(path.read_bytes()).hexdigest()
    env['LLMGE_EXPERIMENT_CONFIG_SHA256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    (destination/'ops/experiment-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    lines=['# Source this from the isolated checkout.', 'source ops/common.env']
    lines += [f'export {k}={shlex.quote(v)}' for k,v in env.items()]
    (destination/'ops/run.env').write_text('\n'.join(lines)+'\n')
    print(destination)
    print('source ops/run.env')
    print('No job submitted. Check model files, account/QoS, prior smoke and validation first.')
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',choices=tuple(PORTS),required=True)
    parser.add_argument('--destination',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--evolution-seed',type=int,default=0)
    parser.add_argument('--port-base',type=int)
    a=parser.parse_args()
    prepare(a.target,a.destination,a.run_id,a.evolution_seed,a.port_base)
