"""Optional vLLM transport behind the existing server.py /generate interface."""
import os
from pathlib import Path
import time
import requests
from src.chat_response import parse_chat_response
from src.model_serving import endpoint_profile


def settings():
    return os.environ["JB_MUTATOR_VLLM_URL"].rstrip("/"), endpoint_profile(os.environ["JB_MUTATOR_PROFILE"])


def auth_options():
    if path := os.environ.get("JB_MUTATOR_API_KEY_FILE"):
        key = Path(path).read_text().strip()
        if not key:
            raise ValueError("The configured mutator key file is empty")
        return {"headers": {"Authorization": "Bearer " + key}, "allow_redirects": False,
                "proxies": {"http": "", "https": "", "all": ""}}
    return {}


def assert_ready():
    url, item = settings()
    response = requests.get(f"{url}/models", timeout=10, **auth_options())
    response.raise_for_status()
    expected = item.get("served_model_name", item["model_id"])
    if expected not in {m["id"] for m in response.json()["data"]}:
        raise RuntimeError("Mutator vLLM model identity differs from JB_MUTATOR_PROFILE")


def generate(request):
    url, item = settings()
    payload = {"model": item.get("served_model_name", item["model_id"]),
               "messages": [{"role": "user", "content": request["prompt"]}],
               "max_tokens": request["max_new_tokens"], "top_p": request["top_p"],
               "temperature": request["temperature"]}
    if item.get("reasoning_effort_parameter"):
        effort = os.getenv("JB_MUTATOR_REASONING_EFFORT", "medium")
        if effort not in ("low", "medium", "high"):
            raise ValueError("JB_MUTATOR_REASONING_EFFORT must be low, medium, or high")
        payload[item["reasoning_effort_parameter"]] = effort
    started = time.monotonic()
    response = requests.post(f"{url}/chat/completions", json=payload,
                             timeout=float(os.getenv("JB_REQUEST_TIMEOUT", "120")), **auth_options())
    response.raise_for_status()
    result = parse_chat_response(response.json())
    return {"generated_text": result["text"], "response_time_sec": time.monotonic() - started,
            "batch_size": 1, "usage": result["usage"], "finish_reason": result["finish_reason"]}
