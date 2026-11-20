import importlib.util
import json
import subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('corrected_launch', ROOT / 'ops/launch_corrected.py')
launch = importlib.util.module_from_spec(spec); spec.loader.exec_module(launch)
prep_spec = importlib.util.spec_from_file_location('corrected_prepare', ROOT / 'ops/prepare_corrected.py')
prepare = importlib.util.module_from_spec(prep_spec); prep_spec.loader.exec_module(prepare)


def config(target='gpt-oss-20b', phase='full'):
    return dict(target=target, phase=phase, run_id=target + '_' + phase,
                account='math', partition='ice-gpu', qos='coe-ice',
                resources=dict(gpus=4 if target.endswith('20b') and not target.endswith('120b') else 5,
                               cpus=24, memory='192G'))


def test_each_model_only_depends_on_its_own_validation():
    a = launch.submission_command(Path('/tmp/a'), config(), '111')
    b = launch.submission_command(Path('/tmp/b'), config('gpt-oss-120b'), '222')
    assert '--dependency=afterok:111' in a
    assert '--dependency=afterok:222' in b
    assert not any('111' in arg for arg in b)
    assert '--gpus=5' in b and '--gpus=4' in a
    assert '--kill-on-invalid-dep=yes' in a


def test_full_submission_without_validation_is_rejected(tmp_path):
    (tmp_path / 'experiment.json').write_text(json.dumps(config()))
    with pytest.raises(ValueError, match='own afterok'):
        launch.submit(tmp_path)


def test_unconfirmed_submission_is_not_retried(tmp_path):
    cfg = config(phase='validation')
    (tmp_path / 'experiment.json').write_text(json.dumps(cfg))
    (tmp_path / 'submission.json').write_text(json.dumps({'state': 'unconfirmed_timeout'}))
    with pytest.raises(RuntimeError, match='unresolved'):
        launch.submit(tmp_path)


def test_preparation_isolates_all_four_checkouts_and_freezes_same_dataset(tmp_path):
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    report = tmp_path / 'cpu.json'
    report.write_text(json.dumps({'status':'passed', 'code_revision':revision}))
    folders = [Path(p) for p in prepare.prepare(tmp_path / 'runs', 'test', report)]
    assert len(folders) == len(set(folders)) == 4
    configs = [json.loads((p / 'experiment.json').read_text()) for p in folders]
    assert len({c['dataset_sha256'] for c in configs}) == 1
    assert len({c['protocol_sha256'] for c in configs}) == 1
    assert len({c['run_id'] for c in configs}) == 4
    ports = set()
    for folder, cfg in zip(folders, configs):
        assert not (folder / cfg['run_id']).exists()
        assert not (folder / 'results').exists()
        assert (folder / 'ops/data/shared50/selected.csv').is_file()
        env = (folder / 'ops/run.env').read_text()
        for line in env.splitlines():
            if '_PORT=' in line or line.startswith('export LLMGE_PORT='):
                port = int(line.split('=')[1]); assert port not in ports; ports.add(port)
        assert str(folder / 'cache') in env
        assert '--quantization' not in cfg['model_profile']['vllm_args']


def test_sbatch_rejection_retains_exact_command_receipt(tmp_path, monkeypatch):
    cfg = config(phase='validation')
    (tmp_path / 'experiment.json').write_text(json.dumps(cfg))
    monkeypatch.setattr(launch, 'validate_checkpoint', lambda *a: None)
    monkeypatch.setattr(launch.subprocess, 'run', lambda *a, **kw:
                        subprocess.CompletedProcess(a[0], 1, '', 'scheduler unavailable'))
    with pytest.raises(RuntimeError, match='scheduler unavailable'):
        launch.submit(tmp_path)
    receipt = json.loads((tmp_path / 'submission.json').read_text())
    assert receipt['state'] == 'rejected'
    assert receipt['command'][0] == 'sbatch' and '--account=math' in receipt['command']


def test_preparation_can_replace_only_20b_with_fresh_ports(tmp_path):
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    report = tmp_path / 'cpu.json'
    report.write_text(json.dumps({'status':'passed', 'code_revision':revision}))
    folders = [Path(p) for p in prepare.prepare(tmp_path / 'runs', 'r2', report,
                                               targets=['20b'], port_offset=1000)]
    assert len(folders) == 2
    assert all(json.loads((p/'experiment.json').read_text())['target']=='gpt-oss-20b' for p in folders)
    assert not list((tmp_path/'runs').glob('120b*'))
    assert 'JB_TARGET_VLLM_PORT=24120' in (folders[0]/'ops/run.env').read_text()
    assert 'JB_TARGET_VLLM_PORT=24100' in (folders[1]/'ops/run.env').read_text()
