"""Serving capacity/provenance contracts; no weights or scheduler launches."""
import copy
import json
from pathlib import Path
import pytest
from src import model_serving
from ops.serving_memory import GIB,audit_context,kv_upper_bound,require_context,stress_messages,literal_traits
from ops.serving_memory_probe import check_stress_response,require_smoke


def test_120b_explicit_kv_budget_preserves_native_weights_and_capacity():
    p=model_serving.profile('gpt-oss-120b');cmd=model_serving.command('gpt-oss-120b',tp=2,enforce_eager=True)
    assert p['revision']=='b5c939de8f754692c1647ca79fbf85e8c1e70f8a'
    assert cmd[cmd.index('--kv-cache-memory-bytes')+1]==str(8*GIB)
    assert cmd[cmd.index('--tensor-parallel-size')+1]=='2'
    assert cmd[cmd.index('--max-model-len')+1]=='4096'
    assert cmd[cmd.index('--max-num-seqs')+1]=='8'
    assert '--quantization' not in cmd and '--kv-cache-dtype' not in cmd
    assert '--enforce-eager' in cmd


def test_20b_profile_and_command_have_no_memory_override():
    cmd=model_serving.command('gpt-oss-20b')
    assert '--kv-cache-memory-bytes' not in cmd
    assert cmd[cmd.index('--gpu-memory-utilization')+1]=='0.9'
    assert model_serving.profile('gpt-oss-20b')['revision']=='6cee5e81ee83917806bbde320786a8fb61efebee'


def test_all_layer_kv_upper_bound_has_large_headroom():
    config=dict(num_hidden_layers=36,num_key_value_heads=8,head_dim=64)
    assert kv_upper_bound(config,4096,8,2)==int(1.125*GIB)
    assert kv_upper_bound(config,4096,8,2)<8*GIB/4
    with pytest.raises(ValueError): kv_upper_bound(config,4096,8,3)


@pytest.mark.parametrize('prompt,output,context,valid',[(3584,512,4096,True),(3585,512,4096,False),
                                                       (4096,512,4096,False),(0,512,4096,False)])
def test_context_reserves_entire_output_budget(prompt,output,context,valid):
    if valid:require_context(prompt,output,context)
    else:
        with pytest.raises(ValueError):require_context(prompt,output,context)


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs==dict(tokenize=True,add_generation_prompt=True)
        return [1]*(10+sum(len(m['content'].split()) for m in messages))


def test_stress_messages_cover_input_budget_without_truncation():
    messages,count=stress_messages(Tokenizer(),3584,'test')
    assert 3576<=count<=3584
    assert 'Explain photosynthesis' in messages[0]['content']
    assert 'Resource test test' in messages[0]['content']
    require_context(count,512,4096)


def test_context_audit_parses_literals_and_includes_full_chat_overhead(tmp_path):
    gene=tmp_path/'gene.py';gene.write_text('trait_context = "prefix"\nraise RuntimeError("must not execute")\n')
    def renderer(traits,behavior):
        return [{'role':'system','content':traits['trait_context']},{'role':'user','content':behavior}],None
    result=audit_context(Tokenizer(),renderer,[gene,gene],['one two','three'],512,4096)
    assert result['unique_trait_sets']==1 and result['rendered_prompts']==2
    assert result['max_prompt_tokens']==13 and result['required_context']==525
    assert len(result['source_files'])==2
    with pytest.raises(ValueError,match='exceeds context'):
        audit_context(Tokenizer(),renderer,[gene],['x '*3584],512,4096)


def test_context_audit_requires_observed_prompts():
    with pytest.raises(ValueError,match='No rendered'):
        audit_context(Tokenizer(),lambda a,b:None,[],[],512,4096)


def test_context_audit_includes_every_system_and_user_trait():
    names=['language','persona','context','style','format','turns','instruction','decoding']
    traits=literal_traits('\n'.join(f'trait_{n} = {n!r}' for n in names))
    assert len(traits)==8 and traits['trait_context']=='context' and traits['trait_instruction']=='instruction'
    with pytest.raises(ValueError,match='Nonliteral'):
        literal_traits('trait_context = str(123)')


def stress_response():
    return dict(choices=[dict(finish_reason='length')],usage=dict(prompt_tokens=3584,completion_tokens=512,total_tokens=4096))


def test_intentionally_truncated_stress_output_is_never_a_valid_experimental_answer():
    checked=check_stress_response(stress_response(),3584,512,4096)
    assert checked['intentionally_incomplete'] is True
    assert checked['valid_experimental_answer'] is False


@pytest.mark.parametrize('field,value',[('finish_reason','stop'),('prompt_tokens',3585),('completion_tokens',511)])
def test_smoke_rejects_inadequate_or_overflowing_stress(field,value):
    data=stress_response()
    if field=='finish_reason':data['choices'][0][field]=value
    else:data['usage'][field]=value
    with pytest.raises(ValueError):check_stress_response(data,3584,512,4096)


def smoke_fixture(tmp_path):
    cfg=dict(serving_memory_smoke_directory=str(tmp_path),code_revision='rev',protocol_sha256='protocol',
             target='gpt-oss-120b',resources=dict(gpus=5,target_tp=2))
    result=dict(status='passed',purpose='serving_memory_smoke',job_id='111',**{k:cfg[k] for k in ['code_revision','protocol_sha256','target','resources']})
    (tmp_path/'serving-memory-smoke-passed.json').write_text(json.dumps(result))
    source=tmp_path/'probe.py';source.write_text('source')
    import hashlib
    (tmp_path/'experiment.json').write_text(json.dumps(dict(run_id='smoke',code_revision=cfg['code_revision'],
        protocol_sha256=cfg['protocol_sha256'],source_files={'probe.py':hashlib.sha256(b'source').hexdigest()})))
    (tmp_path/'smoke').mkdir()
    (tmp_path/'smoke/status.json').write_text(json.dumps(dict(state='COMPLETED',returncode=0,job_id='111')))
    return cfg,result


def test_completed_matching_smoke_is_required_before_validation_and_full(tmp_path):
    cfg,result=smoke_fixture(tmp_path);require_smoke(cfg)
    result['code_revision']='other';(tmp_path/'serving-memory-smoke-passed.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='completed serving-memory smoke'):require_smoke(cfg)


def test_failed_smoke_cannot_release_experiments(tmp_path):
    cfg,result=smoke_fixture(tmp_path)
    (tmp_path/'smoke/status.json').write_text(json.dumps(dict(state='FAILED',returncode=1,job_id='111')))
    with pytest.raises(ValueError):require_smoke(cfg)


def test_changed_smoke_source_cannot_release_experiments(tmp_path):
    cfg,result=smoke_fixture(tmp_path);(tmp_path/'probe.py').write_text('changed')
    with pytest.raises(ValueError,match='source changed'):require_smoke(cfg)


def test_memory_preparation_isolates_smoke_validation_full_without_20b(tmp_path):
    import hashlib,subprocess
    from ops.prepare_120b_memory import prepare_memory
    root=Path(__file__).resolve().parents[1]
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    cpu=tmp_path/'cpu.json';cpu.write_text(json.dumps(dict(status='passed',code_revision=revision)))
    audit=tmp_path/'audit.json';audit.write_text(json.dumps(dict(status='passed',
        tokenizer_revision=model_serving.profile('gpt-oss-120b')['revision'],output_tokens=512,
        max_model_len=4096,max_prompt_tokens=452,
        dataset_sha256=hashlib.sha256((root/'ops/data/shared50/selected.csv').read_bytes()).hexdigest())))
    plan=prepare_memory(tmp_path/'runs','test-memory',cpu,audit)
    assert len(set(plan.values()))==3
    ports=set()
    for purpose,folder in plan.items():
        folder=Path(folder);cfg=json.loads((folder/'experiment.json').read_text())
        assert cfg['resources']['gpus']==5 and cfg['resources']['target_tp']==2
        assert cfg['model_profile']['vllm_args']==['--kv-cache-memory-bytes',str(8*GIB)]
        env=(folder/'ops/run.env').read_text()
        assert str(folder/'cache') in env
        if purpose=='smoke_directory':
            assert cfg['purpose']=='serving_memory_smoke' and cfg['wall_time']=='00:45:00'
            assert 'JB_SMOKE_ONLY=1' in env and 'JB_120B_MEMORY_PROBE=1' in env
        else:
            assert cfg['serving_memory_smoke_directory']==plan['smoke_directory']
            assert (folder/'serving-context-audit.json').read_bytes()==audit.read_bytes()
        for line in env.splitlines():
            if '_PORT=' in line:
                port=int(line.split('=')[1]);assert port not in ports;ports.add(port)
    assert not list((tmp_path/'runs').glob('20b*'))


def test_context_audit_integrity_cannot_be_ignored(tmp_path):
    import hashlib
    from ops.serving_memory import require_context_audit
    audit=dict(status='passed',dataset_sha256='data',tokenizer_revision='pin',output_tokens=512,
               max_model_len=4096,max_prompt_tokens=452)
    path=tmp_path/'serving-context-audit.json';path.write_text(json.dumps(audit))
    cfg=dict(context_audit_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),dataset_sha256='data',
             model_profile=dict(revision='pin',max_model_len=4096))
    require_context_audit(cfg,tmp_path)
    path.write_text('{}')
    with pytest.raises(ValueError,match='audit changed'):require_context_audit(cfg,tmp_path)
