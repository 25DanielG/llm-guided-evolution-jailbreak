"""Exercise CUDA initialization order and service failures without model loads."""
from contextlib import contextmanager
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.experiments.mutator_service import initialize_cuda


def test_cuda_initializes_each_context_before_querying_memory():
    active, initialized, queried = [0], set(), []

    @contextmanager
    def device(index):
        old, active[0] = active[0], index
        try:
            yield
        finally:
            active[0] = old

    def empty(size, device):
        index = int(device.split(':')[1])
        assert active[0] == index
        initialized.add(index)

    def memory(index):
        assert index in initialized
        assert active[0] == index
        queried.append(index)
        return 80, 100

    cuda = SimpleNamespace(device_count=lambda: 2, is_available=lambda: True,
                           current_device=lambda: active[0], device=device,
                           synchronize=lambda index: None, mem_get_info=memory,
                           set_device=lambda index: active.__setitem__(0, index))
    initialize_cuda(SimpleNamespace(cuda=cuda, empty=empty))
    assert queried == [0, 1]
    assert active == [0]


def test_cuda_unavailable_fails_before_loading_model():
    cuda = SimpleNamespace(device_count=lambda: 0, is_available=lambda: False)
    with pytest.raises(RuntimeError, match='accessible CUDA'):
        initialize_cuda(SimpleNamespace(cuda=cuda))


@pytest.mark.parametrize('mode,expected_attempts,returncode', [
    ('recover', 2, 0), ('fail', 3, 1), ('hang', 3, 1),
])
def test_service_retries_until_ready_or_bounded_failure(tmp_path, mode, expected_attempts, returncode):
    worker = tmp_path / 'worker.py'
    worker.write_text('''import os, pathlib, time
counter = pathlib.Path(os.environ['COUNTER'])
attempt = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(attempt))
print('loading model', flush=True)
time.sleep(0.05)
if os.environ['MODE'] == 'recover' and attempt > 1:
    pathlib.Path(os.environ['READY']).touch()
    time.sleep(30)
elif os.environ['MODE'] == 'hang':
    time.sleep(30)
else:
    raise SystemExit(1)
''')
    script = '''set -euo pipefail
source "$HELPERS"
curl() { [[ -f "$READY" ]]; }
pid=$(start_service test "$LOG" "$PYTHON" "$WORKER")
trap 'kill -KILL -- "-$pid" 2>/dev/null || true' EXIT
pid=$(wait_or_die unused "$pid" test "$LOG" "$PYTHON" "$WORKER")
kill -0 "$pid"
'''
    env = dict(os.environ, HELPERS=str(ROOT / 'scripts/coverage_services.sh'),
               KERNEL_CACHE_BASE=str(tmp_path / 'kernels'), PYTHON=sys.executable,
               WORKER=str(worker), COUNTER=str(tmp_path / 'counter'),
               READY=str(tmp_path / 'ready'), LOG=str(tmp_path / 'service.log'), MODE=mode,
               COVERAGE_STARTUP_TIMEOUT='1' if mode == 'hang' else '10',
               COVERAGE_READINESS_POLL='0.02', COVERAGE_RETRY_DELAY='0.02')
    result = subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == returncode, result.stderr
    assert (tmp_path / 'counter').read_text() == str(expected_attempts)
    assert (tmp_path / 'service.log.attempt1').exists()
