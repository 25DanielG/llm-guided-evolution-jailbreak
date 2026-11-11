import atexit
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Import PORT from constants
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest
import requests
import httpx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HOSTNAME_FILE = ROOT / "hostname.log"
PORT = int(os.getenv("LLMGE_PORT", "8137"))
READY_TIMEOUT = int(os.getenv("LLM_SERVER_READY_TIMEOUT", "3600"))
READY_CHECK_INTERVAL = int(os.getenv("LLM_SERVER_READY_CHECK_INTERVAL", "10"))

_local_server_process = None


def _server_url(hostname):
    return f"http://{hostname}:{PORT}/"


def _hostname_from_file():
    try:
        hostname = HOSTNAME_FILE.read_text().strip()
    except OSError:
        return None
    return hostname or None


def _server_is_ready(hostname):
    if not hostname:
        return False
    try:
        with urllib.request.urlopen(_server_url(hostname), timeout=5) as response:
            return 200 <= response.status < 300
    except (TimeoutError, urllib.error.URLError, OSError):
        return False


def _wait_for_server(timeout=READY_TIMEOUT):
    start_time = time.time()
    last_hostname = None

    while time.time() - start_time <= timeout:
        if _local_server_process is not None and _local_server_process.poll() is not None:
            raise RuntimeError(
                "Local LLM test server exited before becoming ready "
                f"(exit code {_local_server_process.returncode})."
            )

        hostname = _hostname_from_file()
        if hostname:
            last_hostname = hostname
        if _server_is_ready(hostname):
            print(f"LLM test server is ready at {_server_url(hostname)}", flush=True)
            return True

        elapsed = round(time.time() - start_time)
        print(
            f"Waiting for LLM test server ({elapsed}s/{timeout}s). "
            f"Host file: {HOSTNAME_FILE}",
            flush=True,
        )
        time.sleep(READY_CHECK_INTERVAL)

    if last_hostname:
        print(f"Timed out waiting for LLM test server at {_server_url(last_hostname)}", flush=True)
    else:
        print(f"Timed out waiting for LLM test server; {HOSTNAME_FILE} was not written", flush=True)
    return False


def _start_slurm_server():
    print("Submitting LLM test server job with sbatch server.sh", flush=True)
    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            "--export=ALL,SUBMIT_ISLAND_CONTROLLER=0",
            "server.sh",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        errorMessage = "Failed to submit LLM server job with sbatch.\n"+\
        f"stdout:\n{result.stdout}\n"+\
        f"stderr:\n{result.stderr}"
        print(errorMessage)
        raise RuntimeError(
            errorMessage
        )
    job_id = result.stdout.strip()
    print(f"Submitted LLM test server job {job_id}", flush=True)


def _start_local_server():
    global _local_server_process

    hostname = socket.gethostname()
    HOSTNAME_FILE.write_text(f"{hostname}\n")

    env = os.environ.copy()
    env["SERVER_HOSTNAME"] = hostname
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env.setdefault("UV_CACHE_DIR", f"/tmp/uv-cache-llmge-pytest-{os.getpid()}")

    print(f"Starting local LLM test server on {hostname}:{PORT}", flush=True)
    _local_server_process = subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "server:app",
            "--host",
            hostname,
            "--port",
            str(PORT),
            "--workers",
            "1",
        ],
        cwd=ROOT,
        env=env,
    )
    atexit.register(_stop_local_server)


def _stop_local_server():
    if _local_server_process is None or _local_server_process.poll() is not None:
        return
    _local_server_process.terminate()
    try:
        _local_server_process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _local_server_process.kill()
        _local_server_process.wait()


def pytest_sessionstart(session):
    # Legacy server startup remains available only for explicitly requested live tests.
    if not session.config.getoption("--run-live-inference") or os.getenv("LLMGE_AUTO_START_SERVER", "0") != "1":
        return
    hostname = _hostname_from_file()
    if _server_is_ready(hostname):
        print(f"Using existing LLM test server at {_server_url(hostname)}", flush=True)
        return

    if shutil.which("sbatch") and os.getenv("LLMGE_TEST_SERVER_MODE", "slurm") != "local":
        _start_slurm_server()
    else:
        _start_local_server()

    if not _wait_for_server():
        raise RuntimeError(
            "LLM test server did not become ready. "
            "Set LLMGE_AUTO_START_SERVER=0 to skip auto-starting it."
        )


def pytest_addoption(parser):
    for marker in ("live_inference", "gpu", "dataset"):
        parser.addoption("--run-" + marker.replace("_", "-"), action="store_true",
                         help="Explicitly enable tests marked " + marker)


def pytest_collection_modifyitems(config, items):
    selected, deselected = [], []
    for item in items:
        disabled = any(item.get_closest_marker(marker) and not config.getoption("--run-" + marker.replace("_", "-"))
                       for marker in ("live_inference", "gpu", "dataset"))
        (deselected if disabled else selected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture(autouse=True)
def ordinary_test_isolation(request, monkeypatch):
    """No service calls, weights or Slurm from an ordinary test; mocks override these guards."""
    if any(request.node.get_closest_marker(m) for m in ("live_inference", "gpu", "dataset")):
        return
    def forbidden(*args, **kwargs):
        pytest.fail("Ordinary test attempted external inference/network; mock the boundary or mark live_inference")
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(httpx.Client, "request", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "request", forbidden)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("LLM_SERVER_READY_TIMEOUT", "0")
    monkeypatch.setenv("LLM_SERVER_READY_CHECK_INTERVAL", "0")
    # Pretrained loading (including cached weights) is not a unit test.
    import transformers
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", forbidden)
    popen = subprocess.Popen
    def guarded_popen(args, *positional, **kwargs):
        executable = args[0] if isinstance(args, (list, tuple)) else str(args).split()[0]
        if Path(str(executable)).name in {"sbatch", "salloc", "srun", "vllm", "hf", "huggingface-cli"}:
            pytest.fail("Ordinary test attempted a scheduler/model launch")
        return popen(args, *positional, **kwargs)
    monkeypatch.setattr(subprocess, "Popen", guarded_popen)


@pytest.fixture(scope="session")
def live_llm():
    """Real inference preflight, never a mock and never an implicit allocation."""
    hostname = _hostname_from_file()
    if not hostname:
        pytest.fail("Live inference requires hostname.log from an authorized running service")
    url = f"http://{hostname}:{PORT}"
    try:
        ready = requests.get(url + "/", timeout=5)
        ready.raise_for_status()
        response = requests.post(url + "/generate", json={"prompt": "Reply with one short greeting.",
                                 "max_new_tokens": 128, "temperature": 0.1, "top_p": 0.9}, timeout=30)
        response.raise_for_status()
        assert response.json().get("generated_text", "").strip(), "No generated text"
    except (requests.RequestException, ValueError, AssertionError) as err:
        pytest.fail(f"Explicit live inference check failed at {url}: {err}")
    return url


@pytest.fixture
def titanic_eval(tmp_path):
    """Run the real CPU evaluator/data in a fresh sandbox; no production path changes."""
    sandbox = tmp_path / "Titanic"
    (sandbox / "data").mkdir(parents=True)
    shutil.copy2(ROOT / "sota/Titanic/eval.py", sandbox / "eval.py")
    shutil.copy2(ROOT / "sota/Titanic/data/processed_train.csv", sandbox / "data/processed_train.csv")
    def evaluate(model_name, variant_dir):
        return subprocess.run([sys.executable, str(sandbox / "eval.py"), "--model", model_name,
                               "--variant_dir", str(variant_dir)], cwd=sandbox,
                              capture_output=True, text=True, timeout=180)
    return sandbox, evaluate


@pytest.fixture
def mock_codegen(monkeypatch):
    """Mock only inference; keep real selection, extraction, validation and assembly."""
    from unittest.mock import Mock
    import llm_utils
    def configure(code):
        generator = Mock(return_value=f"```python\n{code}\n```" if code is not None else None)
        monkeypatch.setattr(llm_utils, "get_llm_code_generator", lambda model: (generator, None))
        return generator
    return configure


@pytest.fixture
def operators():
    import llm_mutation
    import src.llm_crossover as crossover
    return llm_mutation, crossover
