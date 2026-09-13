"""Runtime config for the Jailbreak eval path"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

TARGET_HOST_FILE = os.path.join(HERE, "target_host.log")
JUDGE_HOST_FILE = os.path.join(HERE, "judge_host.log")

def _read_host_file(path):
    try:
        with open(path, "r") as f:
            line = f.readline().strip()
            return line or None
    except OSError:
        return None

def _resolve_base_url(env_url, env_port, port_default, host_file):
    """Return an OpenAI-compatible base URL like http://<host>:<port>/v1."""
    full = os.getenv(env_url)
    if full:
        return full.rstrip("/")
    host_port = _read_host_file(host_file)
    if host_port:
        if ":" not in host_port:
            host_port = f"{host_port}:{os.getenv(env_port, str(port_default))}"
        return f"http://{host_port}/v1"
    host = os.getenv("JB_VLLM_HOST", "localhost")
    port = os.getenv(env_port, str(port_default))
    return f"http://{host}:{port}/v1"

def target_base_url():
    return _resolve_base_url("JB_TARGET_VLLM_URL", "JB_TARGET_VLLM_PORT", 8001, TARGET_HOST_FILE)

def judge_base_url():
    return _resolve_base_url("JB_JUDGE_VLLM_URL", "JB_JUDGE_VLLM_PORT", 8002, JUDGE_HOST_FILE)

def target_model_name():
    return os.getenv("JB_TARGET_SERVED_NAME", "target")

def judge_model_name():
    return os.getenv("JB_JUDGE_SERVED_NAME", "judge")

def judge_mode():
    # guard: Llama-Guard classifier
    # llm: open instruct model + safety rubric.
    return os.getenv("JB_JUDGE_MODE", "llm").lower()

def behaviors_path():
    return os.getenv("JB_BEHAVIORS_PATH", os.path.join(HERE, "behaviors", "curated.csv"))

def n_behaviors_per_eval():
    return int(os.getenv("JB_N_BEHAVIORS_PER_EVAL", "4"))

def behavior_seed():
    return int(os.getenv("JB_BEHAVIOR_SEED", "0"))

def use_full_behaviors():
    return os.getenv("JB_FULL_BEHAVIORS", "0").lower() in ("1", "true", "yes")

def cache_dir():
    return os.getenv("JB_CACHE_DIR", os.path.join(os.path.expanduser("~"), "scratch", "jb_cache"))

def results_dir():
    return os.getenv("JB_RESULTS_DIR", os.path.join(HERE, "results"))

def target_max_new_tokens():
    return int(os.getenv("JB_TARGET_MAX_NEW_TOKENS", "512"))

def target_temperature():
    return float(os.getenv("JB_TARGET_TEMPERATURE", "0.0"))

def request_timeout():
    return float(os.getenv("JB_REQUEST_TIMEOUT", "120"))

def max_concurrency():
    return int(os.getenv("JB_MAX_CONCURRENCY", "2"))

def api_key():
    value = os.getenv("JB_VLLM_API_KEY", "").strip()
    if not value:
        raise RuntimeError("JB_VLLM_API_KEY must be set")
    return value

def judge_rubric_version():
    return os.getenv("JB_JUDGE_RUBRIC_VERSION", "2026-09-13-v1")

def judge_parse_retries():
    return int(os.getenv("JB_JUDGE_PARSE_RETRIES", "2"))

def server_ready_timeout():
    return float(os.getenv("JB_SERVER_READY_TIMEOUT", "1200"))
