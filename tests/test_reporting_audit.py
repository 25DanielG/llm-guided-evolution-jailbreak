"""Regression for the live generation-2 null-answer audit failure."""
import ast
import copy
import datetime
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sota/Jailbreak'))
audit=importlib.import_module('audit')
evaluation=importlib.import_module('eval')
judge=importlib.import_module('judge')
from reporting_inputs import exclusion_reasons,select_eligible,validate_audits


def row(gene='a',response='answer',finish='stop',score=0.0,index=0,generation=2,error=None):
    return dict(run_id='mock',generation=generation,candidate_id=gene,behavior_index=index,
                behavior='Explain a benign topic',target_response=response,target_finish_reason=finish,
                target_error_type=error,in_loop=dict(score=score,parse_ok=not error),target_usage={},verification=None)


@pytest.fixture
def audit_run(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'__file__',str(tmp_path/'audit.py'))
    monkeypatch.setattr(evaluation,'SCRIPT_DIR',tmp_path)
    folder=tmp_path/'results/mock';folder.mkdir(parents=True)
    path=folder/'candidates.jsonl'
    monkeypatch.setattr(audit.jb_config,'audit_sample_size',lambda:3)
    monkeypatch.setattr(audit.vllm_client,'assert_model',Mock())
    reporter=Mock(return_value=dict(success=False,parse_ok=True))
    monkeypatch.setattr(audit.judge,'reporting_score',reporter)
    return folder,path,reporter


def write_rows(path,rows):
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def test_zero_score_tie_uses_valid_later_response_from_same_gene():
    rows=[row(response=None,finish='length',error='missing_final_answer'),row(index=1),row('b')]
    selected=audit.select_top_candidates(rows,2,3)
    assert [(r['candidate_id'],r['behavior_index']) for r in selected]==[('a',1),('b',0)]
    assert rows[0]['target_response'] is None  # raw scientific records are retained


@pytest.mark.parametrize('response,finish,error',[(None,'stop',None),('','stop',None),('  ','stop',None),
    ('partial','length',None),('answer',None,None),('answer','stop','missing_final_answer'),(123,'stop',None)])
def test_invalid_responses_excluded_before_ranking(response,finish,error):
    rows=[row('bad',response,finish,1.0,error=error),row('good')]
    chosen,summary=select_eligible(rows,3,2)
    assert [r['candidate_id'] for r in chosen]==['good']
    assert summary['exclusions'][0]['candidate_id']=='bad'
    assert summary['exclusions'][0]['reasons']


@pytest.mark.parametrize('count',[0,1,2,3])
def test_shortfall_is_explicit_without_fabricated_results(audit_run,count):
    folder,path,reporter=audit_run
    rows=[row(f'good{i}') for i in range(count)]+[row('bad',None,'length',error='missing_final_answer')]
    write_rows(path,rows)
    result=audit.audit('mock',2)
    summary=json.loads((folder/'audit_generation_02.json').read_text())
    assert len(result)==reporter.call_count==count
    assert summary['completed']==count and summary['requested']==3
    assert summary['status']==('complete' if count==3 else 'insufficient_eligible_candidates')
    assert summary['exclusions'][0]['record_index']==count+1
    assert validate_audits(folder,'mock',[2])[0]['completed']==count
    if count==0:audit.vllm_client.assert_model.assert_not_called()


@pytest.mark.parametrize('value',[None,'','  ',42])
@pytest.mark.parametrize('field',['behavior','response'])
def test_reporter_rejects_invalid_inputs_before_string_operations(monkeypatch,value,field):
    completion=Mock();monkeypatch.setattr(judge.vllm_client,'completion',completion)
    args=dict(behavior='request',response='answer');args[field]=value
    with pytest.raises(ValueError,match='Invalid reporter input'):judge.reporting_score(**args)
    completion.assert_not_called()


def test_reporter_failure_keeps_completed_rows_and_stays_failed(audit_run):
    folder,path,reporter=audit_run;write_rows(path,[row('a'),row('b'),row('c')])
    reporter.side_effect=[dict(success=False,parse_ok=True),dict(success=False,parse_ok=False,error='offline')]
    with pytest.raises(RuntimeError,match='invalid result'):audit.audit('mock',2)
    assert len((folder/'audits.jsonl').read_text().splitlines())==1
    assert json.loads((folder/'audit_generation_02.json').read_text())['status']=='reporter_error'
    with pytest.raises(ValueError):validate_audits(folder,'mock',[2])


def test_elite_export_applies_same_validity_rules_and_records_exclusions(audit_run):
    folder,path,_=audit_run
    write_rows(path,[row('a',None,'length',1,index=0),row('a',index=1),row('b','partial','length'),row('c')])
    output=evaluation.export_elites(path)
    selected=[json.loads(line) for line in output.read_text().splitlines()]
    assert [(r['candidate_id'],r['behavior_index']) for r in selected]==[('a',1),('c',0)]
    summary=json.loads((folder/'elites_selection.json').read_text())
    assert len(summary['exclusions'])==2 and summary['completed']==2


def test_fresh_verification_rejects_truncated_output_without_reporter_call(monkeypatch):
    monkeypatch.setattr(evaluation.vllm_client,'chat',lambda *a,**k:dict(text='partial',finish_reason='length'))
    reporter=Mock();monkeypatch.setattr(evaluation.judge,'reporting_score',reporter)
    result=evaluation.verify_record(dict(behavior='request',messages=[]),'unused','target')
    assert result['successes']==0 and not result['verified_success']
    assert len(result['trials'])==3 and all(t['reporting']['parse_ok'] is False for t in result['trials'])
    reporter.assert_not_called()


def test_acceptance_rejects_missing_or_fabricated_audit_records(audit_run):
    folder,path,_=audit_run;write_rows(path,[row('a'),row('b'),row('c')]);audit.audit('mock',2)
    output=folder/'audits.jsonl';records=output.read_text().splitlines();output.write_text('\n'.join(records[:-1])+'\n')
    with pytest.raises(ValueError,match='incomplete'):validate_audits(folder,'mock',[2])


def test_serving_smoke_compatibility_allows_only_reporting_changes():
    from ops.serving_memory_probe import verify_reporting_compatibility
    old=dict(code_revision='674ea6a32f11fa92354c969d09f70f34d2de0a00',target='gpt-oss-120b',resources={'gpus':5},
             model_profile={'kv':8},auxiliary_models={},dataset_sha256='data',protocol={'seed':0,'source_sha256':'old'},
             source_files={'run_improved.py':'same','ops/resource_sampler.py':'same','sota/Jailbreak/audit.py':'old'})
    new=copy.deepcopy(old);new['code_revision']='new';new['serving_smoke_compatible_revision']=old['code_revision']
    new['source_files']['sota/Jailbreak/audit.py']='new'
    assert verify_reporting_compatibility(old,new)==['sota/Jailbreak/audit.py']
    for key in ['run_improved.py','ops/resource_sampler.py','config/served_models.json']:
        changed=copy.deepcopy(new);changed['source_files'][key]='changed'
        with pytest.raises(ValueError,match='exceed reporting-only'):verify_reporting_compatibility(old,changed)
    changed=copy.deepcopy(new);changed['model_profile']['kv']=4
    with pytest.raises(ValueError,match='compatibility differs'):verify_reporting_compatibility(old,changed)


@pytest.mark.parametrize('remove_audit',[False,True])
def test_mocked_generation2_audit_checkpoint_restore_and_final_acceptance(tmp_path,monkeypatch,remove_audit):
    """Execute the actual driver's finalization block; mock only external services.

    Uses synthetic CPU telemetry, never an actual GPU allocation or live report.
    """
    import time
    import run_improved as evo
    from ops import corrected_checks as checks
    from src.fitness_state import snapshot_history
    from src.map_elites import MapElitesArchive
    run_id='mock';run=tmp_path/run_id;jb=tmp_path/'sota/Jailbreak';variants=jb/'models/llmge_models'
    variants.mkdir(parents=True);(tmp_path/'results').mkdir();(jb/'results'/run_id).mkdir(parents=True)
    for key,value in dict(RUN_ID=run_id,SOTA_ROOT=str(jb),VARIANT_DIR=str(variants),
        TRAIN_FILE=str(jb/'eval.py'),start_population_size=8,population_size=8,
        GLOBAL_DATA={},GLOBAL_DATA_HIST={},GLOBAL_DATA_ANCESTRY={},
        MAP_ELITES_ARCHIVE=MapElitesArchive(),hof=evo.tools.ParetoFront(),population=[],
        _selection_method='map_elites',args=SimpleNamespace(checkpoints=str(run/'ckpt'),global_path=str(run/'global'))).items():
        monkeypatch.setattr(evo,key,value,raising=False)
    monkeypatch.setattr(evo,'box_print',lambda *a,**k:None)
    monkeypatch.setattr(evo,'print_scores',lambda *a,**k:None)
    monkeypatch.setenv('JB_RESUME_MODE','new');monkeypatch.delenv('JB_STATUS_FILE',raising=False)
    monkeypatch.setenv('JB_GENE_RESULTS_DIR',str(tmp_path/'results'))
    monkeypatch.setenv('SLURM_JOB_ID','mock-cpu-only')
    monkeypatch.setattr(audit,'__file__',str(jb/'audit.py'))
    monkeypatch.setattr(audit.jb_config,'audit_sample_size',lambda:3)
    monkeypatch.setattr(audit.vllm_client,'assert_model',Mock())
    monkeypatch.setattr(audit.judge,'reporting_score',Mock(return_value=dict(success=False,parse_ok=True)))
    path=jb/'results'/run_id/'candidates.jsonl';all_rows=[]
    for i in range(8):
        gene=f'g{i}';ind=evo.creator.Individual([gene]);ind.fitness.values=(0.0,.1+i*.01,200.0)
        evo.population.append(ind)
        evo.GLOBAL_DATA[gene]=dict(status='completed',sub_flag=True,fitness=ind.fitness.values)
        evo.GLOBAL_DATA_ANCESTRY[gene]={}
        (variants/f'prompt_{gene}.py').write_text('trait_format = \'<format type="freeform">{ASK}</format>\'\ntrait_turns = \'<turns></turns>\'\n')
        (tmp_path/'results'/f'{gene}_results.json').write_text(json.dumps(dict(tier_scores={},behaviors=[],rendered_strategy_sha256=gene)))
    snapshot_history(evo.GLOBAL_DATA,evo.GLOBAL_DATA_HIST)
    # Preserve the exact production ordering from audit -> checkpoint -> metrics.
    tree=ast.parse((ROOT/'run_improved.py').read_text())
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='gen'
              and any(isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and getattr(x.value.func,'id','')=='write_generation_metrics' for x in n.body))
    start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
               and getattr(n.value.func,'id','')=='snapshot_history' and n.lineno>1700)
    stop=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
              and getattr(n.value.func,'id','')=='write_generation_metrics')
    code=compile(ast.Module(body=loop.body[start:stop+1],type_ignores=[]),'run_improved.py','exec')
    def subprocess_audit(command,**kwargs):
        assert command[:3]==['uv','run','python'] and command[3]==str(jb/'audit.py') and kwargs['check'] is True
        audit.audit(command[5],int(command[7]));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(evo.subprocess,'run',subprocess_audit)
    for generation in [0,1,2]:
        all_rows.extend(row(f'g{i}',generation=generation) for i in range(8))
        if generation==2:all_rows.insert(len(all_rows)-8,row('g0',None,'length',generation=2,error='missing_final_answer'))
        write_rows(path,all_rows)
        if generation==0:
            evo.update_map_elites_archive(evo.population);evo.refresh_hall_of_fame(evo.hof,evo.population)
            evo.save_checkpoint(0,run/'ckpt',run/'global');evo.write_generation_metrics(0,evo.population)
        else:
            monkeypatch.setattr(evo,'gen',generation,raising=False)
            monkeypatch.setattr(evo,'offspring',copy.deepcopy(evo.population),raising=False)
            exec(code,evo.__dict__)
        if generation==1:
            before=[tuple(ind.fitness.values) for ind in evo.population]
            monkeypatch.setenv('JB_RESUME_MODE','required')
            restored,next_gen,data=evo.load_checkpoint(run/'ckpt',global_path=run/'global')
            assert next_gen==2 and [tuple(ind.fitness.values) for ind in restored['population']]==before
            for key in ['GLOBAL_DATA','GLOBAL_DATA_HIST','GLOBAL_DATA_ANCESTRY']:
                monkeypatch.setattr(evo,key,data[key])
            monkeypatch.setattr(evo,'population',restored['population'])
            monkeypatch.setattr(evo,'MAP_ELITES_ARCHIVE',restored['map_elites_archive'])
            monkeypatch.setattr(evo,'hof',restored['hof'])
    assert (run/'ckpt/checkpoint_gen_2.json').is_file() and (run/'global/global_gen_2.pkl').is_file()
    assert json.loads((tmp_path/'results'/run_id/'generation_02.json').read_text())['invalid_fitness_count']==0
    summary=json.loads((jb/'results'/run_id/'audit_generation_02.json').read_text())
    assert summary['completed']==3 and len(summary['exclusions'])==1
    # Real final acceptance with an explicitly synthetic, complete four-GPU fixture.
    now=time.time();devices=[dict(cuda_ordinal=i,nvml_index=str(i),uuid=f'GPU-mock-{i}',pci_bus_id=f'0000:{i+32:02x}:00.0') for i in range(4)]
    allocation=dict(schema_version=2,job_id='mock-cpu-only',hostname='synthetic-cpu-test',devices=devices)
    (run/'gpu-resources.allocation.json').write_text(json.dumps(allocation))
    samples=[]
    for offset in [15,10,5]:
        gpus=[dict(index=d['nvml_index'],uuid=d['uuid'],pci_bus_id=d['pci_bus_id'],name='NVIDIA H100',total_mib=81559,used_mib=70000) for d in devices]
        samples.append(dict(schema_version=2,job_id='mock-cpu-only',hostname='synthetic-cpu-test',returncode=0,error='',
            time=datetime.datetime.fromtimestamp(now-offset,datetime.timezone.utc).isoformat(),gpus=gpus))
    write_rows(run/'gpu-resources.jsonl',samples)
    (run/'pipeline-probe.json').write_text(json.dumps(dict(status='passed',target_budget_check=dict(finish_reason='stop'))))
    monkeypatch.setattr(checks,'ROOT',tmp_path)
    config=dict(phase='validation',run_id=run_id,code_revision='mock-cpu-test',target='gpt-oss-20b',
                protocol_sha256='mock',resources=dict(gpus=4,target_tp=1))
    if remove_audit:
        (jb/'results'/run_id/'audit_generation_02.json').unlink()
        with pytest.raises(FileNotFoundError):checks.completed(config)
        assert not (tmp_path/'validation-passed.json').exists()
    else:
        checks.completed(config)
        passed=json.loads((tmp_path/'validation-passed.json').read_text())
        assert passed['status']=='passed' and passed['final_generation']==2
        assert passed['job_id']=='mock-cpu-only' and passed['reporting_audits'][0]['completed']==3
