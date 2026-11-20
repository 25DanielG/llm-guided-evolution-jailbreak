"""CPU preparation checks; no model inference or Slurm submissions."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest
from ops.freeze_behaviors import freeze

ROOT=Path(__file__).resolve().parents[1]


def test_short_source_fails_instead_of_silently_running_thirty(tmp_path):
    with pytest.raises(ValueError,match='Need 50 distinct behaviors; found 30'):
        freeze(ROOT/'sota/Jailbreak/behaviors/curated.csv',tmp_path/'sample')
    assert not (tmp_path/'sample').exists()


def test_frozen_sample_matches_repository_draw_and_retains_ids(tmp_path):
    source=tmp_path/'input.csv'
    rows=[{'id':f'b{i}', 'behavior':f'Explain benign topic {i}.'} for i in range(100)]
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,['id','behavior']);w.writeheader();w.writerows(rows)
    result=freeze(source,tmp_path/'sample',50,0)
    selected=list(csv.DictReader((tmp_path/'sample/selected.csv').open()))
    assert selected==random.Random(0).sample(rows,50)
    assert result['ids']==[r['id'] for r in selected]
    assert hashlib.sha256((tmp_path/'sample/selected.csv').read_bytes()).hexdigest()==result['selected_sha256']
    assert result['evaluation_settings']['JB_FULL_BEHAVIORS']=='1'


def test_existing_tiers_are_sampled_and_preserved(tmp_path):
    source=tmp_path/'tiered.csv'
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,['id','behavior','tier']);w.writeheader()
        w.writerows({'id':f'{t}{i}','behavior':f'Benign {t} topic {i}.','tier':t}
                    for t in ('easy','medium','hard') for i in range(30))
    result=freeze(source,tmp_path/'sample')
    rows=list(csv.DictReader((tmp_path/'sample/selected.csv').open()))
    assert result['sampling']=='stratified'
    assert {t:sum(r['tier']==t for r in rows) for t in ('easy','medium','hard')}=={'easy':16,'medium':17,'hard':17}


def test_launcher_uses_custom_ports_and_existing_experiment_python(tmp_path):
    prefix=(ROOT/'run_jailbreak.sbatch').read_text().split('# target vLLM (weaker model)')[0]
    script='module() { return 0; };\n'+prefix
    script+='\nprintf "TARGET=%s\\nJUDGE=%s\\nPYTHON=%s\\nREPORTER=%s\\n" "${TARGET_COMMAND[*]}" "${JUDGE_COMMAND[*]}" "${EXPERIMENT_COMMAND[*]}" "$REPORTING_MODEL"\n'
    env={k:v for k,v in os.environ.items() if not k.startswith(('JB_','LLMGE_'))}
    env.update(SLURM_SUBMIT_DIR=str(ROOT),SLURM_JOB_ID='fixture',TMPDIR=str(tmp_path),
               CUDA_VISIBLE_DEVICES='a,b,c,d',JB_EXPERIMENT_PYTHON=sys.executable,
               JB_TARGET_VLLM_PORT='21000',JB_JUDGE_VLLM_PORT='21001',
               JB_RUN_REPORTING_MODEL='/local/verified-reporter')
    proc=subprocess.run(['bash','-c',script],env=env,text=True,capture_output=True,timeout=15)
    assert proc.returncode==0,proc.stderr
    assert '--port 21000' in proc.stdout
    assert '--port 21001' in proc.stdout
    assert f'PYTHON={sys.executable}' in proc.stdout
    assert 'REPORTER=/local/verified-reporter' in proc.stdout
