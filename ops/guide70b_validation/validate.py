"""Run only the live serving probes for the isolated VIP validation."""
from contextlib import ExitStack
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

import requests

from stage_models import CHECKPOINTS, MARGIN_BYTES, check_local_disk, emit, inventory

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sota/Jailbreak"))
from ops import resource_sampler
from src import model_serving
import guide_fixtures

FATAL = re.compile(
    r"CUDA out of memory|torch\.(?:cuda\.)?OutOfMemoryError|Engine core initialization failed|"
    r"EngineCore encountered a fatal error|WorkerProc failed to start|WorkerProc.*\b(?:died|exited)\b|"
    r"Application startup failed|Address already in use",
    re.IGNORECASE,
)
WORKER_PID = re.compile(r"\((?:Worker(?:_TP\d+)?|EngineCore_DP\d+) pid=(\d+)\)")
SHARD_PROGRESS = re.compile(r"Loading safetensors checkpoint shards:.*?\|\s*(\d+)/(\d+)")
LOAD_SECONDS = re.compile(r"Loading weights took ([\d.]+) seconds")


def command_env(env):
    keys = {
        "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS",
        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "TMPDIR",
        "XDG_CACHE_HOME", "PYTHONPATH", "LD_LIBRARY_PATH", "PATH", "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE", "NO_PROXY", "no_proxy",
    }
    return {k: v for k, v in sorted(env.items())
            if (k in keys or k.startswith(("JB_", "VIP_", "VLLM_")))
            and not re.search(r"KEY|TOKEN(?!S)|PASSWORD|SECRET|AUTH", k)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cgroup_memory():
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            leaf = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
            for folder in (leaf, *leaf.parents):
                try:
                    maximum = (folder / "memory.max").read_text().strip()
                    if maximum != "max":
                        result = {"path": str(folder), "memory.max": maximum}
                        for key in ("memory.current", "memory.peak", "memory.events"):
                            try:
                                result[key] = (folder / key).read_text().strip()
                            except OSError:
                                result[key] = None
                        return result
                except OSError:
                    continue
            break
    return {"unavailable": "no readable limited cgroup v2 memory telemetry"}


def allocation():
    inventory_gpus = resource_sampler.query_gpus()
    devices = resource_sampler.resolve_devices(resource_sampler.cuda_pci_ids(), inventory_gpus, 5)
    by_uuid = {gpu["uuid"]: gpu for gpu in inventory_gpus}
    resolved = [{**by_uuid[item["uuid"]], **item} for item in devices]
    if any("H200" not in gpu["name"] or gpu["total_mib"] < 80000 for gpu in resolved):
        raise ValueError("Allocation contains an unsuitable GPU")
    return resolved


def job_scratch(required_bytes):
    errors = []
    for candidate in dict.fromkeys((os.getenv("TMPDIR"), "/tmp")):
        if candidate:
            try:
                check_local_disk(Path(candidate), required_bytes)
                return candidate
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                errors.append(str(error))
    raise RuntimeError("No verified job-local scratch: " + "; ".join(errors))


class Service:
    def __init__(self, name, command, mask, url, model, output):
        self.name, self.url, self.model = name, url, model
        self.path = output / f"{name}-launch-1.log"
        self.progress = None
        self.port_error = None
        self.load_seconds = None
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": mask}
        self.started = time.monotonic()
        with self.path.open("xb") as log:
            self.proc = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
            )
        emit("service_launch", service=name, process_group=self.proc.pid, command=command,
             environment=command_env(env), log=str(self.path))

    def inspect(self):
        status = self.proc.poll()
        raw = self.path.read_bytes().decode(errors="replace")
        if status is not None:
            raise RuntimeError(f"{self.name} process exited with status {status}. See {self.path}")
        if failure := FATAL.search(raw):
            raise RuntimeError(f"{self.name} reported {failure.group()}. See {self.path}")
        for pid in {int(value) for value in WORKER_PID.findall(raw)}:
            try:
                state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].strip().split()[0]
                if state == "Z" or os.getpgid(pid) != self.proc.pid:
                    raise ProcessLookupError(pid)
            except (FileNotFoundError, ProcessLookupError) as error:
                raise RuntimeError(f"{self.name} worker {pid} exited or left its process group. See {self.path}") from error
        progress = SHARD_PROGRESS.findall(raw)
        if progress and progress[-1] != self.progress:
            self.progress = progress[-1]
            emit("weights_progress", service=self.name, shards=list(map(int, self.progress)),
                 since_launch_seconds=round(time.monotonic() - self.started, 3))
        if timings := LOAD_SECONDS.findall(raw):
            self.load_seconds = float(timings[-1])

    def check_endpoint(self, session, request_timeout):
        try:
            response = session.get(self.url, timeout=request_timeout)
        except requests.RequestException as error:
            self.port_error = str(error)
            return False
        if response.status_code != 200:
            self.port_error = f"HTTP {response.status_code}"
            return False
        if self.model is not None:
            models = [entry["id"] for entry in response.json().get("data", [])]
            if models != [self.model]:
                raise RuntimeError(f"{self.name} reports models {models!r}, expected {[self.model]!r}")
        return True


def monitor_readiness(service, live_services, timeout, sample):
    deadline = service.started + timeout
    next_sample = 0.0
    with requests.Session() as client:
        client.trust_env = False
        while (left := deadline - time.monotonic()) > 0:
            for active in live_services:
                active.inspect()
            if time.monotonic() >= next_sample:
                sample(service.name)
                next_sample = time.monotonic() + 30
            left = deadline - time.monotonic()
            if left > 0 and service.check_endpoint(client, min(2, left)):
                service.inspect()
                elapsed = round(time.monotonic() - service.started, 3)
                emit("endpoint_ready", service=service.name, model=service.model,
                     since_launch_seconds=elapsed, weight_load_seconds=service.load_seconds)
                return {"ready_seconds": elapsed, "weight_load_seconds": service.load_seconds}
            time.sleep(max(0.0, min(3, deadline - time.monotonic())))
    raise TimeoutError(f"{service.name} did not reach {service.url} within {timeout}s of launch. "
                       f"Last HTTP observation: {service.port_error}. See {service.path}")


def group_signal(group, sig):
    try:
        os.killpg(group, sig)
        return True
    except ProcessLookupError:
        return False


def cleanup(services):
    for service in services:
        group_signal(service.proc.pid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        for service in services:
            service.proc.poll()
        if not any(group_signal(service.proc.pid, 0) for service in services):
            break
        time.sleep(0.2)
    for service in services:
        forced = group_signal(service.proc.pid, signal.SIGKILL)
        try:
            status = service.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            status = "leader did not reap"
        emit("service_cleanup", service=service.name, forced=forced, leader_status=status)


def vllm_command(role, path, port):
    binary = os.environ["JB_VLLM_BIN"]
    if role == "target":
        model_serving.validate_checkpoint("gpt-oss-20b", path)
        command = model_serving.command(
            "gpt-oss-20b", path=path, served_name="target", host="127.0.0.1", port=port,
        )
    else:
        name = "judge" if role == "judge" else CHECKPOINTS[role][0]
        command = [
            binary, "serve", path, "--served-model-name", name,
            "--host", "127.0.0.1", "--port", str(port), "--gpu-memory-utilization", "0.9",
        ]
        if role == "judge":
            command += ["--dtype", "bfloat16", "--tensor-parallel-size", "2",
                        "--max-model-len", "4096", "--enforce-eager"]
    return command + ["--load-format", "safetensors", "--safetensors-load-strategy", "lazy"]


def run_validation(output, services, models, masks, sample, sockets):
    ports = {role: sock.getsockname()[1] for role, sock in sockets.items()}
    names = {"judge": "judge", "target": "target", "reporter": CHECKPOINTS["reporter"][0]}
    bases = {role: f"http://127.0.0.1:{ports[role]}/v1" for role in names}
    for role, prefix in (("judge", "JUDGE"), ("target", "TARGET"), ("reporter", "REPORTING")):
        os.environ[f"JB_{prefix}_VLLM_URL"] = bases[role]
        os.environ[f"JB_{prefix}_SERVED_NAME"] = names[role]
        os.environ[f"JB_{prefix}_MODEL_PATH"] = models[role]["source"]
    os.environ.update(JB_MUTATOR_MODEL_PATH=models["mutator"]["staged_path"],
                      SERVER_HOSTNAME="127.0.0.1", LLMGE_PORT=str(ports["mutator"]))
    os.environ.pop("JB_MUTATOR_VLLM_URL", None)

    def start(role):
        if role == "mutator":
            command = [os.environ["JB_EXPERIMENT_PYTHON"], "-B", "-m", "uvicorn", "server:app",
                       "--host", "127.0.0.1", "--port", str(ports[role])]
            url = f"http://127.0.0.1:{ports[role]}/"
        else:
            command = vllm_command(role, models[role]["staged_path"], ports[role])
            url = bases[role] + "/models"
        sockets[role].close()
        service = Service(role, command, masks[role], url, names.get(role), output)
        services.append(service)
        return service

    judge = start("judge")
    readiness = {
        "judge": monitor_readiness(judge, services, float(os.environ["JB_SERVER_READY_TIMEOUT"]), sample)
    }
    # Score these fixtures before concurrent auxiliary activity can affect the timing.
    fixture_error = None
    try:
        guide_result = guide_fixtures.run(output / "guide-fixtures")
    except Exception as error:
        fixture_error = error
        guide_result = {"status": "failed", "error": str(error)}
    emit("guide_fixture_result", **guide_result)
    auxiliary = [start(role) for role in ("target", "reporter", "mutator")]
    for service in auxiliary:
        readiness[service.name] = monitor_readiness(
            service, services, float(os.environ["VIP_AUX_READY_TIMEOUT"]), sample,
        )
    args = [os.environ["JB_EXPERIMENT_PYTHON"], "-B", str(HERE / "pipeline_acceptance.py")]
    emit("pipeline_probe_start", command=args, environment=command_env(os.environ))
    with (output / "pipeline-probe.log").open("xb") as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    record = output / "pipeline-probe.json"
    pipeline = json.loads(record.read_text()) if record.is_file() else {}
    if result.returncode or pipeline.get("status") != "passed":
        raise RuntimeError(f"Existing benign pipeline probe failed ({result.returncode}). See {log.name}")
    if fixture_error is not None:
        raise RuntimeError("Live guide fixtures failed. See the emitted fixture result.") from fixture_error
    return {"status": "passed", "readiness": readiness, "guide": guide_result,
            "pipeline_result": str(record)}


def main(output):
    if importlib.metadata.version("vllm") != "0.17.0":
        raise RuntimeError("Expected the existing vLLM 0.17.0 serving environment")
    devices = allocation()
    uuids = [gpu["uuid"] for gpu in devices]
    mask = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(mask) != 5 or len(set(mask)) != 5:
        raise ValueError(f"Expected five CUDA mask entries: {mask}")
    # Use UUIDs to avoid conflating CUDA ordinals and Slurm's node GPU indices.
    masks = {"target": uuids[0], "judge": ",".join(uuids[1:3]),
             "reporter": uuids[3], "mutator": uuids[4]}
    write_json(output / "allocation.json", {"gpu_inventory": devices, "service_masks": masks})

    def sample(phase):
        observation = {"phase": phase, "memory": cgroup_memory()}
        try:
            observation["allocated_gpus"] = resource_sampler.query_gpus(uuids)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            observation["gpu_telemetry_error"] = str(error)
        emit("resource_observation", **observation)

    sources = inventory()
    planned = json.loads((ROOT.parent / "evidence/checkpoint-plan.json").read_text())
    for role, source in sources.items():
        for key in ("model_id", "source", "index_sha256", "sizes"):
            if source[key] != planned[role][key]:
                raise RuntimeError(f"{role} checkpoint {key} differs from preflight")
    required = sum(entry["bytes"] for entry in sources.values()) + MARGIN_BYTES
    job_id = os.environ["SLURM_JOB_ID"]
    services = []
    with tempfile.TemporaryDirectory(prefix=f"vip-models-{job_id}-", dir=job_scratch(required)) as temp:
        stage = Path(temp)
        os.environ["XDG_CACHE_HOME"] = str(stage / "runtime-cache")
        os.environ["JB_CACHE_DIR"] = str(stage / "scoring-cache")
        manifest = output / "staging.json"
        write_json(output / "runtime.json", {
            "vllm": importlib.metadata.version("vllm"), "python": sys.executable,
            "source_checkpoints": sources, "environment": command_env(os.environ),
        })
        sample("before_staging")
        stage_args = [sys.executable, "-B", str(HERE / "stage_models.py"), str(stage), str(manifest)]
        with (output / "staging.log").open("xb") as log:
            subprocess.run(stage_args, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=float(os.environ["VIP_STAGE_TIMEOUT"]))
        models = json.loads(manifest.read_text())["models"]
        sample("after_staging")
        try:
            with ExitStack() as reserved:
                sockets = {role: reserved.enter_context(socket.socket()) for role in CHECKPOINTS}
                for sock in sockets.values():
                    sock.bind(("127.0.0.1", 0))
                summary = run_validation(output, services, models, masks, sample, sockets)
        finally:
            cleanup(services)
            sample("after_cleanup")
    write_json(output / "result.json", {**summary, "slurm_job_id": job_id})
    emit("validation_passed", slurm_job_id=job_id)


def interrupted(signum, _frame):
    raise SystemExit(f"Validation interrupted by signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main(Path(sys.argv[1]).resolve())
