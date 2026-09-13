import csv
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import eval as jailbreak_eval


def test_behavior_loading_preserves_identity_and_sampling(tmp_path, monkeypatch):
    source = tmp_path / "behaviors.csv"
    source.write_text("id,source,behavior\na,fixture,first\nb,fixture,second\nc,fixture,third\n")
    rows = jailbreak_eval.load_behaviors(source)
    assert rows[0] == {"id": "a", "source": "fixture", "behavior": "first"}
    monkeypatch.setenv("JB_N_BEHAVIORS_PER_EVAL", "2")
    monkeypatch.setenv("JB_BEHAVIOR_SEED", "7")
    assert jailbreak_eval.sample_behaviors(rows) == jailbreak_eval.sample_behaviors(rows)


def test_write_results_emits_summary_and_audit(tmp_path):
    records = [{"id": "x", "judge": {"harm": 1}}]
    summary, audit = jailbreak_eval.write_results("gene", 1.0, 12.0, records, tmp_path)
    assert summary.read_text().splitlines()[-1] == "1.0,12.0"
    assert json.loads(audit.read_text()) == records[0]


def test_mock_server_end_to_end(tmp_path, monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            assert self.headers["Authorization"] == "Bearer test-key"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data":[]}')

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            requests.append(body)
            if body["model"] == "target":
                text = "I cannot help, but here are detailed steps."
                prompt_tokens = 17
            else:
                text = json.dumps({
                    "verdict": "unsafe",
                    "confidence": 0.95,
                    "reason": "Substantive assistance follows the disclaimer.",
                })
                prompt_tokens = 22
            response = {
                "choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": prompt_tokens},
            }
            payload = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"

    behaviors = tmp_path / "behaviors.csv"
    with behaviors.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "source", "behavior"])
        writer.writerow(["fixture_001", "fixture", "restricted request"])
    results = tmp_path / "results"
    monkeypatch.setenv("JB_TARGET_VLLM_URL", base_url)
    monkeypatch.setenv("JB_JUDGE_VLLM_URL", base_url)
    monkeypatch.setenv("JB_TARGET_SERVED_NAME", "target")
    monkeypatch.setenv("JB_BEHAVIORS_PATH", str(behaviors))
    monkeypatch.setenv("JB_RESULTS_DIR", str(results))
    monkeypatch.setenv("JB_N_BEHAVIORS_PER_EVAL", "1")
    jailbreak_dir = Path(jailbreak_eval.__file__).resolve().parent
    monkeypatch.setattr(sys, "argv", [
        "eval.py", "--model", "prompt", "--variant_dir", str(jailbreak_dir),
    ])
    old_cwd = os.getcwd()
    try:
        jailbreak_eval.main()
    finally:
        os.chdir(old_cwd)
        server.shutdown()
        server.server_close()

    assert (results / "seed_results.csv").read_text().splitlines()[-1] == "1.0,17.0"
    audit = json.loads((results / "seed_audit.jsonl").read_text())
    assert audit["id"] == "fixture_001"
    assert audit["judge"]["refusal_detected"] is True
    assert [request["model"] for request in requests] == ["target", "judge"]
