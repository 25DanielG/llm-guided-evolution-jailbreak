"""Fresh repaired 20B pair plus 120B stages reusing the untouched serving smoke."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.prepare_corrected import prepare
from ops.prepare_120b_memory import write_config
from ops.serving_memory import require_context_audit
from ops.serving_memory_probe import verify_reporting_compatibility


def prepare_followups(destination,stamp,cpu_report,smoke_directory,context_audit):
    smoke=Path(smoke_directory).resolve()
    smoke_config=json.loads((smoke/'experiment.json').read_text())
    for path,digest in smoke_config['source_files'].items():
        if hashlib.sha256((smoke/path).read_bytes()).hexdigest()!=digest:
            raise ValueError('Existing serving smoke was modified: '+path)
    folders=[Path(p) for p in prepare(destination,stamp,cpu_report,port_offset=4000)]
    audit_path=Path(context_audit)
    for folder in folders[2:]:
        config=json.loads((folder/'experiment.json').read_text())
        config.update(serving_memory_smoke_directory=str(smoke),
                      serving_smoke_compatible_revision=smoke_config['code_revision'],
                      context_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest())
        verify_reporting_compatibility(smoke_config,config)
        (folder/'serving-context-audit.json').write_bytes(audit_path.read_bytes())
        require_context_audit(config,folder)
        write_config(folder,config,(folder/'ops/run.env').read_text())
    return dict(validation_20b=str(folders[0]),full_20b=str(folders[1]),
                validation_120b=str(folders[2]),full_120b=str(folders[3]),
                existing_120b_smoke=str(smoke))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['destination','stamp','cpu-report','smoke-directory','context-audit']:
        p.add_argument('--'+arg,required=True)
    a=p.parse_args()
    print(json.dumps(prepare_followups(a.destination,a.stamp,a.cpu_report,a.smoke_directory,a.context_audit),indent=2))
