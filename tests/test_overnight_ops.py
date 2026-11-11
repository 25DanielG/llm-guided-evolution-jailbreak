"""CPU recovery and operator tests; mocks here do not verify real GPU inference."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import run_improved as evolution
from src import checkpointing, run_status, serving_smoke, stage_checkpoint, review_snapshot
from src.map_elites import MapElitesArchive, PromptDescriptor

pytestmark = pytest.mark.mock_inference
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def strict_state(tmp_path, monkeypatch):
    old_random, old_numpy = random.getstate(), np.random.get_state()
    monkeypatch.setenv('JB_RESUME_MODE', 'new')
    monkeypatch.delenv('JB_STATUS_FILE', raising=False)
    monkeypatch.setattr(evolution, 'RUN_ID', 'resume-test')
    monkeypatch.setattr(evolution, 'llm_model', 'gpt-oss-20b', raising=False)
    individual = evolution.creator.Individual(['gene'])
    individual.fitness.values = tuple(float(i+1) for i in range(len(evolution.FITNESS_WEIGHTS)))
    archive = MapElitesArchive()
    archive.add('gene', PromptDescriptor('freeform','zero'), (0.8,0.5,-100),
                fitness=individual.fitness.values, individual=individual)
    hof = evolution.tools.ParetoFront()
    hof.update([individual])
    for name, value in {'population':[individual], 'hof':hof, 'MAP_ELITES_ARCHIVE':archive,
                        'GLOBAL_DATA':{'gene':{'status':'completed'}},
                        'GLOBAL_DATA_HIST':{'gene':{'score':0.8}},
                        'GLOBAL_DATA_ANCESTRY':{'gene':{'GENES':['seed']}}}.items():
        monkeypatch.setattr(evolution, name, value, raising=False)
    random.seed(43)
    np.random.seed(71)
    yield tmp_path/'ckpt', tmp_path/'global'
    random.setstate(old_random)
    np.random.set_state(old_numpy)


def test_real_checkpoint_resume_restores_deap_archive_globals_and_rng(strict_state, monkeypatch):
    ckpt, glob = strict_state
    evolution.save_checkpoint(1, ckpt, glob)
    expected = (random.random(), np.random.random())
    random.seed(999)
    np.random.seed(999)
    monkeypatch.setenv('JB_RESUME_MODE','required')
    state, generation, globals_data = evolution.load_checkpoint(ckpt, global_path=glob)
    assert generation == 2
    assert (random.random(), np.random.random()) == expected
    assert state['population'][0].fitness.values == evolution.population[0].fitness.values
    assert state['hof'][0] == evolution.hof[0]
    assert len(state['map_elites_archive']) == 1
    assert globals_data['GLOBAL_DATA_ANCESTRY']['gene']['GENES'] == ['seed']
    # Advance from the restored state, commit it, and resume again. No seed creation or inference.
    evolution.save_checkpoint(generation, ckpt, glob, checkpoint_data=state)
    _, next_generation, _ = evolution.load_checkpoint(ckpt, global_path=glob)
    assert next_generation == 3


def test_strict_resume_missing_or_mismatched_checkpoint_never_reseeds(strict_state, monkeypatch):
    ckpt, glob = strict_state
    monkeypatch.setenv('JB_RESUME_MODE','required')
    with pytest.raises(ValueError, match='refusing to restart'):
        evolution.load_checkpoint(ckpt, global_path=glob)
    evolution.save_checkpoint(1, ckpt, glob)
    monkeypatch.setattr(evolution,'RUN_ID','other-run')
    with pytest.raises(ValueError, match='identity differs'):
        evolution.load_checkpoint(ckpt, global_path=glob)


def test_interrupted_commit_ignores_uncommitted_generation(strict_state, monkeypatch):
    ckpt, glob = strict_state
    evolution.save_checkpoint(1, ckpt, glob)
    original = checkpointing.atomic_json
    def fail_marker(path, data):
        if Path(path).name == 'checkpoint_gen_2.json':
            raise OSError('fixture interrupted commit')
        return original(path,data)
    monkeypatch.setattr(checkpointing,'atomic_json',fail_marker)
    with pytest.raises(OSError,match='interrupted'):
        evolution.save_checkpoint(2, ckpt, glob)
    monkeypatch.setenv('JB_RESUME_MODE','required')
    _, generation, _ = evolution.load_checkpoint(ckpt, global_path=glob)
    assert generation == 2  # committed generation 1; never pairs it with global generation 2


def test_corrupt_checkpoint_fails_instead_of_falling_back(strict_state, monkeypatch):
    ckpt, glob = strict_state
    evolution.save_checkpoint(1, ckpt, glob)
    evolution.save_checkpoint(2, ckpt, glob)
    (glob/'global_gen_2.pkl').write_bytes(b'corrupt')
    monkeypatch.setenv('JB_RESUME_MODE','required')
    with pytest.raises(ValueError,match='checksum mismatch'):
        evolution.load_checkpoint(ckpt,global_path=glob)
    state, generation, _ = evolution.load_checkpoint(ckpt,global_path=glob,checkpoint_file='checkpoint_gen_1.pkl')
    assert generation == 2
    with pytest.raises(ValueError,match='refusing overwrite'):
        evolution.save_checkpoint(2,ckpt,glob,checkpoint_data=state)
    recovery=ckpt.parent/'recovery'
    recovery.mkdir()
    for path in (ckpt/'checkpoint_gen_2.pkl',ckpt/'checkpoint_gen_2.json',glob/'global_gen_2.pkl'):
        path.rename(recovery/path.name)
    evolution.save_checkpoint(2,ckpt,glob,checkpoint_data=state)
    assert evolution.load_checkpoint(ckpt,global_path=glob)[1] == 3


def test_new_mode_cannot_overwrite_previous_run(strict_state):
    ckpt, glob = strict_state
    evolution.save_checkpoint(0,ckpt,glob)
    with pytest.raises(ValueError,match='New run already'):
        evolution.load_checkpoint(ckpt,global_path=glob)
    with pytest.raises(ValueError,match='refusing overwrite'):
        evolution.save_checkpoint(0,ckpt,glob)


@pytest.mark.parametrize('code,state,reason', [(0,'COMPLETED','completed'),(75,'PAUSED','checkpoint_pause'),(3,'FAILED','command_failed')])
def test_persistent_supervisor_preserves_progress_and_exit(tmp_path, code, state, reason):
    status = tmp_path/'run'/'status.json'
    child = ('from src.run_status import progress; '
             'progress(8, "trusted-checkpoint.pkl"); '
             f'print("fixture exit {code}"); raise SystemExit({code})')
    result = run_status.supervise(status,'fixture',[sys.executable,'-c',child],interval=60,max_hours=1)
    data = json.loads(status.read_text())
    schema = json.loads((ROOT/'docs/run-status.schema.json').read_text())
    assert set(schema['required']) <= set(data)
    assert result == code
    assert data['state'] == state and data['exit_reason'] == reason
    assert data['latest_completed_generation'] == 8
    assert Path(data['console_log']).exists()
    assert (data['exception'] is not None) == (code == 3)


def test_supervisor_hard_deadline_is_bounded_and_does_not_retry(tmp_path):
    result = run_status.supervise(tmp_path/'status.json','deadline',
        [sys.executable,'-c','import time; time.sleep(600)'],interval=60,max_hours=0.00005)
    data = json.loads((tmp_path/'status.json').read_text())
    assert result != 0
    assert data['exit_reason'] == 'signal_or_deadline'
    assert len(list((tmp_path/'attempts').iterdir())) == 1


def test_status_distinguishes_slow_activity_and_suspected_stall():
    now = datetime.now(timezone.utc)
    data = {'state':'RUNNING','last_progress_at':(now-timedelta(hours=4)).isoformat(),
            'last_activity_at':(now-timedelta(seconds=30)).isoformat()}
    assert run_status.health(data,now) == 'slow_but_active'
    data['last_activity_at'] = data['last_progress_at']
    assert run_status.health(data,now) == 'suspected_stall_inspect_logs'
    data['state'] = 'PAUSED'
    assert run_status.health(data,now) == 'paused'


def test_accounting_exit_reason_and_unreachable_are_distinct(monkeypatch):
    monkeypatch.setattr(run_status.subprocess,'run',Mock(return_value=SimpleNamespace(
        returncode=0,stdout='42|TIMEOUT|0:15|TimeLimit|\n42.batch|CANCELLED|0:15||2G\n',stderr='')))
    assert run_status.scheduler_state('42')['job_state'] == 'TIMEOUT'
    monkeypatch.setattr(run_status.subprocess,'run',Mock(side_effect=OSError('DNS unavailable')))
    data = run_status.scheduler_state('42')
    assert 'job_state' not in data and 'DNS unavailable' in data['error']


def test_staging_verifies_every_file_before_manifest(tmp_path):
    content = b'fixture weights'
    (tmp_path/'weights.safetensors').write_bytes(content)
    file = SimpleNamespace(rfilename='weights.safetensors',size=len(content),
                           lfs=SimpleNamespace(sha256=hashlib.sha256(content).hexdigest()))
    stage_checkpoint.verify_files(tmp_path,[file])
    (tmp_path/'weights.safetensors').write_bytes(b'truncated')
    with pytest.raises(ValueError,match='truncated'):
        stage_checkpoint.verify_files(tmp_path,[file])
    assert not (tmp_path/'deployment.json').exists()
    file.rfilename = '../escape'
    with pytest.raises(ValueError,match='Unsafe'):
        stage_checkpoint.verify_files(tmp_path,[file])


def test_native_staging_excludes_duplicate_formats():
    files = [SimpleNamespace(rfilename=n) for n in ['original/a.safetensors','model-00001.safetensors',
             'model.safetensors.index.json','consolidated.safetensors','params.json','tekken.json']]
    assert [f.rfilename for f in stage_checkpoint.selected_files(files,True)] == [
        'consolidated.safetensors','params.json','tekken.json']


@pytest.mark.parametrize('complete',[True,False])
def test_staging_promotes_only_complete_revision_verified_download(tmp_path,monkeypatch,complete):
    import huggingface_hub as hub
    item = stage_checkpoint.profile('gpt-oss-20b')
    contents = {'config.json':b'{}','tokenizer_config.json':b'{}','model.safetensors':b'fixture'}
    files = [SimpleNamespace(rfilename=n,size=len(b),lfs=None,
              blob_id=hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()) for n,b in contents.items()]
    info = SimpleNamespace(sha=item['revision'],siblings=files)
    monkeypatch.setattr(hub,'HfApi',lambda: SimpleNamespace(model_info=Mock(return_value=info)))
    def download(*args,**kwargs):
        p = Path(kwargs['local_dir'])
        p.mkdir(parents=True)
        for n,b in contents.items():
            if complete or n != 'model.safetensors': (p/n).write_bytes(b)
    monkeypatch.setattr(hub,'snapshot_download',download)
    if complete:
        destination = stage_checkpoint.stage('gpt-oss-20b',tmp_path,1)
        assert json.loads((destination/'deployment.json').read_text())['revision'] == item['revision']
        with pytest.raises(ValueError,match='Destination exists'):
            stage_checkpoint.stage('gpt-oss-20b',tmp_path,1)
    else:
        with pytest.raises(ValueError,match='Missing/truncated'):
            stage_checkpoint.stage('gpt-oss-20b',tmp_path,1)
        assert not list(tmp_path.rglob('deployment.json'))


@pytest.mark.parametrize('model,reasoning,fp8,answer,proxy_answer,passes',[
    ('gpt-oss-20b','fixture analysis',False,'391','READY',True),
    ('gpt-oss-20b',None,False,'391','READY',False),
    ('devstral-small-2-24b',None,True,'391','READY',True),
    ('devstral-small-2-24b',None,False,'391','READY',False),
    ('gpt-oss-20b','fixture analysis',False,'400','READY',False),
    ('gpt-oss-20b','fixture analysis',False,'391','NOT_READY',False)])
def test_mock_smoke_orchestration_evidence_and_cleanup(tmp_path,monkeypatch,model,reasoning,fp8,answer,proxy_answer,passes):
    import torch
    import types
    item = serving_smoke.profile(model)
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    for name in ('config.json','tokenizer_config.json','tekken.json'):
        (checkpoint/name).write_text('{}')
    (checkpoint/'params.json').write_text(json.dumps({'quantization':{'qformat':'fp8_e4m3'}}))
    (checkpoint/'consolidated.safetensors').touch()
    (checkpoint/'deployment.json').write_text(json.dumps(item))
    monkeypatch.setenv('SLURM_JOB_ID','fake-allocation')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','fake-gpu')
    monkeypatch.setitem(sys.modules,'vllm',types.ModuleType('vllm'))
    monkeypatch.setattr(torch,'__version__','2.10.0')
    monkeypatch.setattr(torch.version,'cuda','12.9')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda n:(9,0))
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda n:SimpleNamespace(uuid='GPU-fixture'))
    monkeypatch.setattr(serving_smoke.importlib.metadata,'version',lambda n:'0.17.0' if n=='vllm' else 'fixture')
    monkeypatch.setattr(serving_smoke,'model_path',lambda n:str(checkpoint))
    monkeypatch.setattr(serving_smoke,'gpu_sample',lambda uuids:{'at':run_status.utc(),
        'gpus':[{'uuid':'fixture-gpu','used_mib':100,'total_mib':80000}]})
    monkeypatch.setattr(serving_smoke.subprocess,'run',Mock(return_value=SimpleNamespace(stdout='fixture nvidia-smi')))
    created = []
    class FakeProcess:
        pid = 123456789
        def __init__(self,args,**kwargs):
            self.args,self.env,self.returncode=args,kwargs['env'],None
            kwargs['stdout'].write('quantization=fp8\n' if fp8 else 'fixture backend\n')
            kwargs['stdout'].flush()
            created.append(self)
        def wait(self,timeout=None): self.returncode=0; return 0
    kills = []
    monkeypatch.setattr(serving_smoke.os,'killpg',lambda pid,sig:kills.append((pid,sig)))
    monkeypatch.setattr(serving_smoke.subprocess,'Popen',FakeProcess)
    monkeypatch.setattr(serving_smoke,'wait_ready',lambda *a:{'data':[{'id':item['model_id']}]})
    def reply(url,**kwargs):
        data = {'generated_text':proxy_answer} if url.endswith('/generate') else {'choices':[{
            'message':{'content':answer,'reasoning':reasoning},'finish_reason':'stop'}]}
        return SimpleNamespace(raise_for_status=lambda:None,json=lambda:data)
    monkeypatch.setattr(serving_smoke.requests,'post',reply)
    result = serving_smoke.run(model,tmp_path/'result',sys.executable)
    assert result['inference_verified'] is passes
    assert result['status'] == ('passed' if passes else 'failed')
    assert kills and all(p.returncode==0 for p in created)
    if passes:
        assert created[1].env['JB_MUTATOR_PROFILE']==model
        assert result['peak_observed_gpu_used_mib']['fixture-gpu']==100
    # All values above are fixtures; no real inference or CUDA verification occurred.


def test_gpu_sampling_uses_allocated_uuids_not_node_wide_indices(monkeypatch):
    query=Mock(return_value=SimpleNamespace(stdout='0, GPU-a, H100, 575.0, 80000, 100\n'))
    monkeypatch.setattr(serving_smoke.subprocess,'run',query)
    sample=serving_smoke.gpu_sample(['GPU-a'])
    assert query.call_args.args[0][-2:]==['--id','GPU-a']
    assert sample['gpus'][0]['used_mib']==100


def test_observer_has_explicit_bounds(tmp_path,monkeypatch):
    status = tmp_path/'status.json'
    status.write_text(json.dumps({'job_id':'42'}))
    query = Mock(return_value={'job_state':'COMPLETED'})
    monkeypatch.setattr(run_status,'scheduler_state',query)
    monkeypatch.setattr(sys,'argv',['run_status','observe','--status',str(status),'--checks','2'])
    run_status.main()
    query.assert_called_once()
    assert json.loads((tmp_path/'scheduler-observation.json').read_text())['job_state']=='COMPLETED'


def test_review_export_is_isolated_read_only_and_has_no_git_metadata(tmp_path):
    destination=tmp_path/'review'
    try:
        sha=review_snapshot.create(destination)
        assert json.loads((destination/'REVIEW-SNAPSHOT.json').read_text())['commit']==sha
        assert not (destination/'.git').exists()
        assert (destination.stat().st_mode & 0o222)==0
        with pytest.raises(ValueError,match='must be new'):
            review_snapshot.create(destination)
    finally:
        if destination.exists():
            destination.chmod(0o700)
            for path in destination.rglob('*'):
                if not path.is_symlink(): path.chmod(0o700 if path.is_dir() else 0o600)


def test_smoke_without_allocation_records_failure_without_loading(tmp_path,monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    load = Mock(side_effect=AssertionError('must not launch'))
    monkeypatch.setattr(serving_smoke.subprocess,'Popen',load)
    result = serving_smoke.run('gpt-oss-20b',tmp_path/'smoke',sys.executable)
    assert result['status'] == 'failed' and not result['inference_verified']
    assert 'Slurm GPU allocation' in result['error']
    load.assert_not_called()
    assert (tmp_path/'smoke/result.json').exists()


def test_overnight_launcher_requires_explicit_resume_mode(tmp_path):
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(ROOT), SLURM_JOB_ID='fixture',RUN_ID=str(tmp_path/'fresh'),
               JB_EXPERIMENT_PYTHON=sys.executable,JB_RESUME_MODE='auto')
    result = subprocess.run(['bash','run_overnight.sbatch'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode == 2 and 'auto is forbidden' in result.stderr


def test_stop_warning_is_a_cooperative_checkpoint_pause(tmp_path):
    # Actual supervisor receives a Slurm-style warning; the child observes the marker.
    child = ('import os,signal,time; from pathlib import Path; '
             'os.kill(os.getppid(),signal.SIGUSR1); '
             'p=Path(os.environ["JB_STOP_FILE"]); '
             'deadline=time.monotonic()+5; '
             '\nwhile not p.exists() and time.monotonic()<deadline: time.sleep(.01)\n'
             'raise SystemExit(75 if p.exists() else 4)')
    assert run_status.supervise(tmp_path/'status.json','warning',[sys.executable,'-c',child],interval=60,max_hours=1) == 75
    assert json.loads((tmp_path/'status.json').read_text())['exit_reason'] == 'checkpoint_pause'


def test_runtime_setup_never_mutates_existing_environment(tmp_path):
    target=tmp_path/'existing'
    target.mkdir()
    marker=target/'production.txt'
    marker.write_text('keep')
    env=dict(os.environ,JB_SERVING_ENV=str(target),JB_BOOTSTRAP_PYTHON=sys.executable)
    result=subprocess.run(['bash','scripts/serving/setup_runtime.sh'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==2 and 'refusing to modify' in result.stderr
    assert marker.read_text()=='keep'


def test_duplicate_supervisor_refuses_to_launch_another_attempt(tmp_path,monkeypatch):
    import fcntl
    launch=Mock(side_effect=AssertionError('must not start another run'))
    monkeypatch.setattr(run_status.subprocess,'Popen',launch)
    with (tmp_path/'.supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            run_status.supervise(tmp_path/'status.json','existing',[sys.executable,'-c','pass'])
    launch.assert_not_called()
