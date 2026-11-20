"""Freeze a separate 120B memory smoke, validation and gated full experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.prepare_corrected import prepare
from src.model_serving import profile
from ops.serving_memory import require_context


def write_config(folder,config,env):
    config.pop('config_sha256',None)
    config['config_sha256']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    env=re.sub(r'^export LLMGE_EXPERIMENT_CONFIG_SHA256=.*$',
               'export LLMGE_EXPERIMENT_CONFIG_SHA256='+config['config_sha256'],env,flags=re.M)
    (folder/'experiment.json').write_text(json.dumps(config,indent=2)+'\n')
    (folder/'ops/run.env').write_text(env)


def prepare_memory(destination,stamp,cpu_report,context_audit):
    audit_path=Path(context_audit);audit=json.loads(audit_path.read_text())
    item=profile('gpt-oss-120b')
    if (audit['status']!='passed' or audit['tokenizer_revision']!=item['revision']
            or audit['output_tokens']!=512 or audit['max_model_len']!=item['max_model_len']
            or audit['dataset_sha256']!=hashlib.sha256((ROOT/'ops/data/shared50/selected.csv').read_bytes()).hexdigest()):
        raise ValueError('Pinned tokenizer/context audit is missing or differs')
    require_context(audit['max_prompt_tokens'],512,item['max_model_len'])
    smoke=Path(destination).resolve()/f'120b-corrected-{stamp}-serving-smoke'
    if smoke.exists():raise ValueError('Fresh smoke directory already exists')
    validation,full=[Path(p) for p in prepare(destination,stamp,cpu_report,targets=['120b'],port_offset=2000)]
    shutil.copytree(validation,smoke,symlinks=True)
    config=json.loads((smoke/'experiment.json').read_text())
    original_id=config['run_id'];config['run_id']=original_id.replace('_validation_seed0','_serving_smoke_seed0')
    config.update(purpose='serving_memory_smoke',wall_time='00:45:00')
    env=(smoke/'ops/run.env').read_text().replace(str(validation),str(smoke)).replace(original_id,config['run_id'])
    env=env.replace('JB_MAX_HOURS=2','JB_MAX_HOURS=0.75').replace('JB_VALIDATE_CONTINUATION=1','JB_VALIDATE_CONTINUATION=0')
    for old,new in [(25320,25340),(25321,25341),(25322,25342),(25323,25343),(25324,25344)]:
        env=env.replace('='+str(old)+'\n','='+str(new)+'\n')
    env+='export JB_SMOKE_ONLY=1\nexport JB_120B_MEMORY_PROBE=1\n'
    write_config(smoke,config,env)
    for folder in [validation,full]:
        config=json.loads((folder/'experiment.json').read_text())
        config['serving_memory_smoke_directory']=str(smoke)
        config['context_audit_sha256']=hashlib.sha256(audit_path.read_bytes()).hexdigest()
        (folder/'serving-context-audit.json').write_bytes(audit_path.read_bytes())
        write_config(folder,config,(folder/'ops/run.env').read_text())
    return dict(smoke_directory=str(smoke),validation_directory=str(validation),full_directory=str(full))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',required=True);parser.add_argument('--stamp',required=True)
    parser.add_argument('--cpu-report',required=True);parser.add_argument('--context-audit',required=True)
    args=parser.parse_args()
    print(json.dumps(prepare_memory(args.destination,args.stamp,args.cpu_report,args.context_audit),indent=2))
