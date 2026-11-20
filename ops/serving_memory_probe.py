"""120B serving-memory smoke. Resource-stress outputs never enter fitness scoring."""
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.resource_sampler import check_telemetry
from ops.serving_memory import kv_upper_bound,require_context,stress_messages,GIB
from src import checkpointing,model_serving


def check_stress_response(data, expected_prompt, output_tokens, context):
    choice=data['choices'][0];usage=data['usage']
    if (choice.get('finish_reason')!='length' or usage.get('prompt_tokens')!=expected_prompt
            or usage.get('completion_tokens')!=output_tokens):
        raise ValueError('Resource stress did not exercise the full requested token budget')
    require_context(usage['prompt_tokens'],usage['completion_tokens'],context)
    return dict(usage=usage,finish_reason=choice['finish_reason'],
                intentionally_incomplete=True,valid_experimental_answer=False)


def run():
    import requests
    from transformers import AutoTokenizer
    config=json.loads((ROOT/'experiment.json').read_text())
    if config.get('purpose')!='serving_memory_smoke' or config['target']!='gpt-oss-120b':
        raise ValueError('Resource stress is permitted only in its dedicated 120B smoke run')
    profile=config['model_profile'];context=profile['max_model_len'];output=512
    model=Path(model_serving.model_path(config['target']))
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    url=os.environ['JB_TARGET_VLLM_URL'].rstrip('/')+'/chat/completions'
    cases=[stress_messages(tokenizer,context-output,str(i)) for i in range(profile['max_num_seqs'])]
    def query(case):
        messages,count=case
        response=requests.post(url,json=dict(model='target',messages=messages,max_tokens=output,
                               min_tokens=output,ignore_eos=True,temperature=0.0),timeout=240)
        response.raise_for_status()
        return check_stress_response(response.json(),count,output,context)
    started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=profile['max_num_seqs']) as executor:
        results=list(executor.map(query,cases))
    model_config=json.loads((model/'config.json').read_text())
    report=dict(status='passed',job_id=os.environ['SLURM_JOB_ID'],target=config['target'],
                code_revision=config['code_revision'],target_tp=config['resources']['target_tp'],
                max_model_len=context,max_num_seqs=profile['max_num_seqs'],output_tokens=output,
                kv_cache_memory_bytes_per_gpu=8*GIB,
                conservative_all_layers_kv_bytes_per_gpu=kv_upper_bound(model_config,context,len(cases),2),
                resource_stress_only=True,latency_seconds=time.monotonic()-started,responses=results)
    checkpointing.atomic_json(ROOT/os.environ['RUN_ID']/'serving-memory-stress.json',report)
    print('120B SERVING MEMORY STRESS PASSED',flush=True)


def completed(config):
    run=ROOT/config['run_id']
    probe=json.loads((run/'pipeline-probe.json').read_text())
    stress=json.loads((run/'serving-memory-stress.json').read_text())
    if not (probe['status']=='passed' and probe['target_budget']==512
            and probe['target_budget_check']['finish_reason']=='stop'
            and probe['target_budget_check'].get('text','').strip()
            and stress['status']=='passed' and stress['job_id']==os.environ['SLURM_JOB_ID']
            and stress['code_revision']==config['code_revision']
            and len(stress['responses'])==8 and stress['kv_cache_memory_bytes_per_gpu']==8*GIB):
        raise ValueError('120B serving/inference acceptance failed')
    result=dict(status='passed',purpose='serving_memory_smoke',target=config['target'],
                job_id=os.environ['SLURM_JOB_ID'],code_revision=config['code_revision'],
                protocol_sha256=config['protocol_sha256'],resources=config['resources'],stress=stress)
    log=(ROOT/'sota/Jailbreak/target_vllm.out').read_text()
    for marker in ["'kv_cache_memory_bytes': 8589934592",'as specified by kv_cache_memory_bytes config',
                   'quantization=mxfp4','max_seq_len=4096','tensor_parallel_size=2']:
        if marker not in log:
            raise ValueError('Target runtime did not confirm setting: '+marker)
    result['runtime_evidence']=[line for line in log.splitlines() if any(marker in line for marker in [
        'non-default args','Model loading took','as specified by kv_cache_memory_bytes config',
        'GPU KV cache size','Maximum concurrency'])]
    result.update(check_telemetry(run/'gpu-resources.jsonl',5,os.environ['SLURM_JOB_ID'],now=time.time()))
    checkpointing.atomic_json(ROOT/'serving-memory-smoke-passed.json',result)
    print('120B SERVING MEMORY SMOKE CHECKS PASSED',flush=True)


def require_smoke(config):
    folder=Path(config['serving_memory_smoke_directory'])
    result=json.loads((folder/'serving-memory-smoke-passed.json').read_text())
    smoke_config=json.loads((folder/'experiment.json').read_text())
    state=json.loads((folder/smoke_config['run_id']/'status.json').read_text())
    if not (result['status']=='passed' and result['purpose']=='serving_memory_smoke'
            and result['code_revision']==smoke_config['code_revision']
            and result['protocol_sha256']==smoke_config['protocol_sha256']
            and result['target']==config['target'] and result['resources']==config['resources']
            and state['state']=='COMPLETED' and state['returncode']==0
            and state['job_id']==result['job_id']):
        raise ValueError('Matching completed serving-memory smoke is required')
    for path,digest in smoke_config['source_files'].items():
        if checkpointing.digest(folder/path)!=digest:
            raise ValueError('Serving smoke source changed: '+path)
    if (result['code_revision'],result['protocol_sha256']) != (config['code_revision'],config['protocol_sha256']):
        verify_reporting_compatibility(smoke_config,config)
        telemetry=check_telemetry(folder/smoke_config['run_id']/'gpu-resources.jsonl',5,result['job_id'],
                                  now=datetime.datetime.fromisoformat(state['finished_at']).timestamp())
        if telemetry['peak_gpu_used_mib'] != result['peak_gpu_used_mib']:
            raise ValueError('Compatible smoke telemetry differs')


def verify_reporting_compatibility(smoke_config,config):
    """Reuse only the existing serving smoke across reviewed reporting changes."""
    base='674ea6a32f11fa92354c969d09f70f34d2de0a00'
    if smoke_config['code_revision'] != base or config.get('serving_smoke_compatible_revision') != base:
        raise ValueError('Matching completed serving-memory smoke is required')
    for key in ['target','resources','model_profile','auxiliary_models','dataset_sha256']:
        if smoke_config[key] != config[key]:
            raise ValueError('Serving smoke compatibility differs: '+key)
    old_protocol={k:v for k,v in smoke_config['protocol'].items() if k!='source_sha256'}
    new_protocol={k:v for k,v in config['protocol'].items() if k!='source_sha256'}
    if old_protocol != new_protocol:
        raise ValueError('Scientific protocol differs from serving smoke')
    allowed={'sota/Jailbreak/reporting_inputs.py','sota/Jailbreak/audit.py','sota/Jailbreak/judge.py',
             'sota/Jailbreak/eval.py','sota/Jailbreak/diagnose_malformed.py',
             'ops/corrected_checks.py','ops/cpu_validate.py','ops/serving_memory_probe.py',
             'ops/prepare_reporting_followups.py','tests/sota/test_jailbreak_phase1.py',
             'tests/test_reporting_audit.py','tests/test_120b_serving_memory.py'}
    old,new=smoke_config['source_files'],config['source_files']
    changed={p for p in old.keys()|new.keys() if old.get(p)!=new.get(p)}
    if changed-allowed:
        raise ValueError('Changes exceed reporting-only compatibility: '+repr(sorted(changed-allowed)))
    return sorted(changed)
