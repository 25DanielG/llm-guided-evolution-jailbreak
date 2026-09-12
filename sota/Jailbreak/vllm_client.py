"""Client for the vLLM servers. vLLM is self-hosted, no API key needed. Plain requests
with a thread pool.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import requests

API_KEY = "EMPTY"

def wait_ready(base_url, timeout=1200, check_interval=10):
    """Block until the server answers GET /models, or timeout."""
    models_url = f"{base_url}/models"
    start = time.time()
    last_err = None
    while time.time() - start <= timeout:
        try:
            resp = requests.get(models_url, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=5)
            if resp.status_code == 200:
                return True
        except requests.exceptions.RequestException as err:
            last_err = err
        elapsed = round(time.time() - start)
        print(f"  waiting for vLLM at {base_url} ({elapsed}s/{int(timeout)}s)", flush=True)
        time.sleep(check_interval)
    print(f"  vLLM not ready at {base_url} after {int(timeout)}s (last: {last_err})", flush=True)
    return False

def chat(base_url, model, messages, max_tokens=512, temperature=0.0, timeout=120, retries=2):
    """One chat completion. Returns {"text", "prompt_tokens"}.
    Raises RuntimeError if the server is unreachable after retries (infra failure,
    distinct from the model producing a refusal).
    """
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage", {}) or {}
                return {"text": text, "prompt_tokens": int(usage.get("prompt_tokens", 0))}
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.exceptions.RequestException as err:
            last_err = str(err)
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"vLLM request to {url} failed after {retries + 1} tries: {last_err}")

def chat_batch(base_url, model, messages_list, max_tokens=512, temperature=0.0,
               timeout=120, max_concurrency=16):
    """Run many chat completions concurrently, preserving input order."""
    results = [None] * len(messages_list)

    def _one(idx_messages):
        idx, messages = idx_messages
        return idx, chat(base_url, model, messages, max_tokens=max_tokens,
                         temperature=temperature, timeout=timeout)

    with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as pool:
        for idx, result in pool.map(_one, list(enumerate(messages_list))):
            results[idx] = result
    return results
