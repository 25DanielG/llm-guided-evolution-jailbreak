"""Regression coverage for CUDA/Slurm/NVML identity and fail-closed monitoring."""
import copy
import ctypes
import datetime
import json
import subprocess
import sys
from pathlib import Path

import pytest
from ops import resource_sampler as sampler


def inventory(n=8):
    return [dict(index=str(i), uuid=f'GPU-test-{i}', pci_bus_id=f'00000000:{i+32:02X}:00.0',
                 name='NVIDIA H100 80GB HBM3', total_mib=81559, used_mib=70000+i)
            for i in range(n)]


def fixture(tmp_path, ordinals=(0, 2, 4, 6)):
    gpus = [inventory()[i] for i in ordinals]
    allocation = dict(schema_version=2, job_id='123', hostname='node',
                      devices=sampler.resolve_devices([g['pci_bus_id'] for g in gpus], inventory(), len(gpus)))
    path = tmp_path/'gpu-resources.jsonl'
    path.with_suffix('.allocation.json').write_text(json.dumps(allocation))
    rows = [dict(schema_version=2, job_id='123', hostname='node', returncode=0, error='',
                 time=f'2026-11-14T16:00:{i*5:02d}+00:00', gpus=copy.deepcopy(gpus)) for i in range(3)]
    return path, allocation, rows


def write(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_normalizes_cuda_and_nvml_pci_domain_padding_and_case():
    assert sampler.pci_id('0000:AF:00.0') == sampler.pci_id('00000000:af:00.0')


@pytest.mark.parametrize('value', ['0,1', 'GPU-test', '0000:12:00.9', '', '12:00'])
def test_rejects_invalid_device_ids(value):
    with pytest.raises(ValueError, match='PCI bus ID'):
        sampler.pci_id(value)


def test_resolves_noncontiguous_physical_ids_in_cuda_order(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_GPUS', '0,2,4,6')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    visible = ['0000:26:00.0', '0000:20:00.0', '0000:24:00.0', '0000:22:00.0']
    devices = sampler.resolve_devices(visible, inventory(), 4)
    assert [g['nvml_index'] for g in devices] == ['6', '0', '4', '2']
    assert [g['cuda_ordinal'] for g in devices] == [0,1,2,3]


@pytest.mark.parametrize('visible,expected,message', [
    (['0000:20:00.0'], 4, 'count differs'),
    (['0000:ff:00.0'], 1, 'missing from NVML'),
    (['0000:20:00.0']*2, 2, 'Duplicate CUDA'),
])
def test_allocation_resolution_fails_closed(visible, expected, message):
    with pytest.raises(ValueError, match=message):
        sampler.resolve_devices(visible, inventory(), expected)


def test_rejects_ambiguous_nvml_identity():
    with pytest.raises(ValueError, match='Ambiguous'):
        sampler.resolve_devices(['0000:20:00.0'], inventory()+inventory()[:1], 1)


def test_collection_queries_uuids_and_parses_quoted_names(monkeypatch):
    def run(command, **kw):
        assert '--id=GPU-test-2,GPU-test-6' in command
        assert not any(arg.startswith('--id=0,') for arg in command)
        return subprocess.CompletedProcess(command, 0,
            '2, GPU-test-2, 00000000:22:00.0, "NVIDIA H100, 80GB", 81559, 2000\n'
            '6, GPU-test-6, 00000000:26:00.0, NVIDIA H100, 81559, 3000\n', '')
    monkeypatch.setattr(sampler.subprocess, 'run', run)
    result = sampler.query_gpus(['GPU-test-2','GPU-test-6'])
    assert len(result) == 2 and result[0]['name'] == 'NVIDIA H100, 80GB'


@pytest.mark.parametrize('rc,stderr', [(1,'no GPU'), (0,'warning')])
def test_collection_reports_command_failure(monkeypatch, rc, stderr):
    monkeypatch.setattr(sampler.subprocess, 'run', lambda *a,**k: subprocess.CompletedProcess(a[0],rc,'',stderr))
    with pytest.raises(RuntimeError, match='query failed'):
        sampler.query_gpus(['GPU-test-1'])


@pytest.mark.parametrize('data', ['', '0, GPU-x, invalid, H100, 80, 2',
                                 '0,GPU-x,0000:20:00.0,H100,80,81',
                                 '0,GPU-x,0000:20:00.0,H100,N/A,1',
                                 '0,GPU-x,0000:20:00.0,H100,80'])
def test_rejects_malformed_collection(data):
    with pytest.raises(ValueError):
        sampler.parse_inventory(data)


def test_cuda_runtime_enumerates_visible_ordinals_without_allocating_tensors(monkeypatch):
    class Function:
        def __init__(self, impl): self.impl=impl
        def __call__(self,*args): return self.impl(*args)
    def count(pointer): pointer._obj.value=2; return 0
    def pci(buffer,length,ordinal): buffer.value=f'0000:{32+ordinal:02x}:00.0'.encode(); return 0
    class Runtime:
        cudaGetDeviceCount=Function(count)
        cudaDeviceGetPCIBusId=Function(pci)
    monkeypatch.setattr(sampler.ctypes,'CDLL',lambda path:Runtime())
    assert sampler.cuda_pci_ids() == ['0000:20:00.0','0000:21:00.0']


def test_complete_four_and_five_gpu_samples(tmp_path):
    for indices in [(0,2,4,6), (0,1,3,5,7)]:
        path, allocation, rows=fixture(tmp_path, indices); write(path, rows)
        result=sampler.check_telemetry(path,len(indices),'123')
        assert len(result['peak_gpu_used_mib']) == len(indices)
        assert result['all_gpu_active_samples'] == 3


def test_original_two_of_four_regression_is_rejected(tmp_path):
    path,allocation,rows=fixture(tmp_path)
    for row in rows: row['gpus']=row['gpus'][:2]
    write(path,rows)
    with pytest.raises(ValueError,match='Incomplete GPU coverage'):
        sampler.check_telemetry(path,4,'123')


def test_union_coverage_cannot_mask_incomplete_individual_samples(tmp_path):
    path,allocation,rows=fixture(tmp_path)
    rows[0]['gpus']=rows[0]['gpus'][:2]
    rows[1]['gpus']=rows[1]['gpus'][2:]
    write(path,rows)
    with pytest.raises(ValueError,match='Incomplete GPU coverage'):
        sampler.check_telemetry(path,4,'123')


@pytest.mark.parametrize('kind', ['extra','duplicate','wrong_uuid','wrong_pci','wrong_job','failed',
                                  'negative_memory','non_h100','not_integer','missing_field'])
def test_invalid_sample_is_never_silently_skipped(tmp_path,kind):
    path,allocation,rows=fixture(tmp_path)
    row=rows[1]
    if kind=='extra': row['gpus'].append(inventory()[1])
    elif kind=='duplicate': row['gpus'][1]=copy.deepcopy(row['gpus'][0])
    elif kind=='wrong_uuid': row['gpus'][0]['uuid']='GPU-wrong'
    elif kind=='wrong_pci': row['gpus'][0]['pci_bus_id']='0000:ff:00.0'
    elif kind=='wrong_job': row['job_id']='999'
    elif kind=='failed': row.update(returncode=1,error='query failed')
    elif kind=='negative_memory': row['gpus'][0]['used_mib']=-1
    elif kind=='non_h100': row['gpus'][0]['name']='A100'
    elif kind=='not_integer': row['gpus'][0]['used_mib']='70000'
    elif kind=='missing_field': del row['gpus'][0]['uuid']
    write(path,rows)
    with pytest.raises(ValueError,match='line 2'):
        sampler.check_telemetry(path,4,'123')


def test_malformed_json_is_not_filtered(tmp_path):
    path,allocation,rows=fixture(tmp_path); write(path,rows)
    with path.open('a') as stream: stream.write('{bad\n')
    with pytest.raises(ValueError,match='line 4'):
        sampler.check_telemetry(path,4,'123')


@pytest.mark.parametrize('mode',['only_one','inactive','stale','gap','reverse','foreign_job'])
def test_rejects_incomplete_sampling_and_allocation(tmp_path,mode):
    path,allocation,rows=fixture(tmp_path)
    now=None; job='123'
    if mode=='only_one': rows=rows[:1]
    elif mode=='inactive':
        for row in rows: row['gpus'][0]['used_mib']=0
    elif mode=='stale': now=datetime.datetime.fromisoformat(rows[-1]['time']).timestamp()+60
    elif mode=='gap': rows[1]['time']='2026-11-14T16:00:40+00:00'
    elif mode=='reverse': rows[1]['time']=rows[0]['time']
    elif mode=='foreign_job': job='999'
    write(path,rows)
    with pytest.raises(ValueError): sampler.check_telemetry(path,4,job,now=now)


def test_sampler_persists_full_identity_and_first_sample(tmp_path,monkeypatch):
    path=tmp_path/'gpu-resources.jsonl'
    monkeypatch.setenv('SLURM_JOB_ID','123'); monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1,2,3')
    monkeypatch.setenv('SLURM_JOB_GPUS','0,2,4,6')
    monkeypatch.setattr(sys,'argv',['resource_sampler.py',str(path),'--expected','4'])
    monkeypatch.setattr(sampler,'cuda_pci_ids',lambda:[inventory()[i]['pci_bus_id'] for i in [0,2,4,6]])
    calls=[]
    def query(uuids=None):
        calls.append(uuids)
        return inventory() if uuids is None else [g for g in inventory() if g['uuid'] in uuids]
    monkeypatch.setattr(sampler,'query_gpus',query)
    monkeypatch.setattr(sampler,'legacy_filter_evidence',lambda:{'diagnostic_only':True})
    def stop(seconds): raise KeyboardInterrupt()
    monkeypatch.setattr(sampler.time,'sleep',stop)
    with pytest.raises(KeyboardInterrupt): sampler.main()
    allocation=json.loads(path.with_suffix('.allocation.json').read_text())
    assert allocation['environment']['SLURM_JOB_GPUS']=='0,2,4,6'
    assert calls[1]==['GPU-test-0','GPU-test-2','GPU-test-4','GPU-test-6']
    sampler.validate_sample(json.loads(path.read_text()),allocation)
    with pytest.raises(FileExistsError): sampler.main()


def test_legacy_filter_is_diagnostic_only_and_retains_raw_slurm_ids(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_GPUS','0,2,4,6')
    def run(command,**kwargs):
        assert '--id=0,2,4,6' in command
        return subprocess.CompletedProcess(command,0,
            '0,GPU-test-0,0000:20:00.0,H100,81559,1000\n'
            '2,GPU-test-2,0000:22:00.0,H100,81559,1000\n','')
    monkeypatch.setattr(sampler.subprocess,'run',run)
    assert sampler.legacy_filter_evidence()['uuids']==['GPU-test-0','GPU-test-2']
