"""Run CPU-only regression checks and bind their receipt to an exact revision."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TESTS = ['tests/test_scientific_repair.py', 'tests/test_reporting_audit.py', 'tests/test_120b_serving_memory.py', 'tests/test_gpu_telemetry.py', 'tests/test_fitness_bookkeeping.py', 'tests/test_response_reliability.py',
         'tests/test_corrected_launch.py', 'tests/test_map_elites_archive.py',
         'tests/test_map_elites_loop.py', 'tests/test_map_elites_descriptors.py',
         'tests/test_shared_model_serving.py', 'tests/test_overnight_ops.py',
         'tests/operators/test_offspring_propagation.py', 'tests/sota/test_jailbreak_phase1.py',
         'tests/test_jailbreak_run_controls.py', 'tests/test_jailbreak_turn_depth_mutation.py',
         'tests/test_jailbreak_result_paths.py', 'tests/test_ci_policy.py']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, default=ROOT / 'records')
    records = parser.parse_args().output_directory.resolve()
    records.mkdir(parents=True, exist_ok=True)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit source before binding CPU validation to a revision')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    command = [sys.executable, '-B', '-m', 'pytest', '-q', *TESTS, '--tb=short',
               '--junitxml=' + str(records / 'cpu.xml')]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                            env=dict(os.environ, USE_TF='0', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'))
    (records / 'cpu.txt').write_text(result.stdout + result.stderr)
    report = dict(status='passed' if result.returncode == 0 else 'failed',
                  returncode=result.returncode, code_revision=revision,
                  command=command, python=sys.version, packages={})
    if (records / 'cpu.xml').exists():
        report['suites'] = [s.attrib for s in ET.parse(records / 'cpu.xml').iter('testsuite')]
    for name in ['deap', 'numpy', 'torch', 'transformers', 'pytest', 'requests']:
        report['packages'][name] = importlib.metadata.version(name)
    serving = '/storage/ice1/8/3/pablo3/vllm-0.17-py312-f88321e/bin/python'
    script = "import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ['vllm','torch','transformers']}))"
    versions = subprocess.run([serving, '-B', '-c', script], capture_output=True, text=True, timeout=30)
    report['serving_packages'] = json.loads(versions.stdout) if versions.returncode == 0 else {'error': versions.stderr}
    (records / 'cpu.json').write_text(json.dumps(report, indent=2) + '\n')
    print(result.stdout, end=''); print(result.stderr, end='')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
