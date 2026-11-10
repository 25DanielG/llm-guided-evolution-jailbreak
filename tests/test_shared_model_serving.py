"""CPU checks only: unittest avoids the GPU autostart in tests/conftest.py."""
import ast
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from src import model_serving, mutator_proxy
from src.chat_response import parse_chat_response

ROOT = Path(__file__).resolve().parents[1]


def response(content="answer", reasoning=None, finish="stop"):
    return {"choices": [{"message": {"content": content, "reasoning_content": reasoning},
                         "finish_reason": finish, "logprobs": {"content": []}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3}}


def load_functions(filename, names, namespace):
    # Load endpoint/router functions without importing torch or model weights.
    tree = ast.parse((ROOT / filename).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    return namespace


class SharedModelTests(unittest.TestCase):
    def test_combined_run_gpu_groups_without_launching_services(self):
        # Execute the actual setup prefix, ending BEFORE the first server function.
        script = (ROOT / "run_jailbreak.sbatch").read_text().split("# Retry a service's own startup")[0]
        script = "module() { return 0; }\n" + script
        script += '\nprintf "GROUPS=%s|%s|%s|%s\\n" "$TARGET_DEV" "$JUDGE_DEV" "$REPORTING_DEV" "$MUTATOR_DEV"\n'
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items() if not k.startswith("JB_")}
            env.update(SLURM_SUBMIT_DIR=str(ROOT), SLURM_JOB_ID="cpu-test", TMPDIR=tmp,
                       CUDA_VISIBLE_DEVICES="GPU-a,GPU-b,GPU-c,GPU-d")
            legacy = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
            self.assertEqual(legacy.returncode, 0, legacy.stderr)
            self.assertIn("GROUPS=GPU-a|GPU-b|GPU-c|GPU-d", legacy.stdout)
            env.update(JB_RUN_TARGET_PROFILE="gpt-oss-120b", JB_RUN_JUDGE_PROFILE="llama-3.3-70b",
                       JB_RUN_MUTATOR_PROFILE="devstral-small-2-24b",
                       JB_VLLM_BIN="/fake-serving-env/bin/vllm", JB_CUDA_MODULE="fake-cuda-module",
                       CUDA_VISIBLE_DEVICES="GPU-a,GPU-b,GPU-c,GPU-d,GPU-e")
            profiles = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
            self.assertEqual(profiles.returncode, 0, profiles.stderr)
            self.assertIn("GROUPS=GPU-a|GPU-b,GPU-c|GPU-d|GPU-e", profiles.stdout)
            env["CUDA_VISIBLE_DEVICES"] = "GPU-a,GPU-b"
            insufficient = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
            self.assertNotEqual(insufficient.returncode, 0)
            self.assertIn("expected 5 GPUs", insufficient.stderr)

    def test_discovery_and_legacy_profiles(self):
        models = model_serving.registry()["models"]
        for name in ("gpt-oss-20b", "gpt-oss-120b", "devstral-small-2-24b"):
            self.assertEqual(model_serving.profile(name)["status"], "configured")
            self.assertEqual(model_serving.profile(models[name]["model_id"]), models[name])
        self.assertEqual(model_serving.profile("mistral-7b")["model_id"], "mistralai/Mistral-7B-Instruct-v0.2")
        cmd = model_serving.command("llama-3.3-70b")
        self.assertEqual(cmd[cmd.index("--tensor-parallel-size") + 1], "2")
        self.assertIn("bfloat16", cmd)
        with self.assertRaises(ValueError):
            model_serving.profile("misspelled-model")

    def test_native_quantization_and_chat_template(self):
        for name in ("gpt-oss-20b", "gpt-oss-120b"):
            cmd = model_serving.command(name)
            self.assertNotIn("--quantization", cmd)
            self.assertNotIn("--dtype", cmd)
        cmd = model_serving.command("devstral-small-2-24b")
        for flag in ("--tokenizer-mode", "--config-format", "--load-format"):
            self.assertEqual(cmd[cmd.index(flag) + 1], "mistral")

    def test_checkpoint_provenance_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text("{}")
            (path / "model.safetensors").touch()
            with self.assertRaisesRegex(ValueError, "deployment.json"):
                model_serving.validate_checkpoint("gpt-oss-20b", path)
            item = model_serving.profile("gpt-oss-20b")
            (path / "deployment.json").write_text(json.dumps(item))
            model_serving.validate_checkpoint("gpt-oss-20b", path)
            with self.assertRaisesRegex(ValueError, "does not match"):
                model_serving.validate_checkpoint("gpt-oss-120b", path)

    def test_cli_never_starts_on_login_node(self):
        env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
        run = subprocess.run([sys.executable, "-m", "src.model_serving", "serve", "--model", "gpt-oss-20b"],
                             cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Slurm GPU allocation", run.stderr)

    def test_final_answer_does_not_include_reasoning(self):
        parsed = parse_chat_response(response("final code", "private reasoning"))
        self.assertEqual(parsed["text"], "final code")
        self.assertEqual(parsed["reasoning"], "private reasoning")
        self.assertEqual(parsed["usage"]["prompt_tokens"], 10)
        self.assertEqual(parsed["logprobs"], {"content": []})
        self.assertEqual(parse_chat_response(response())["text"], "answer")
        for data in (response(None, "reasoning", "length"), response("", None, "tool_calls"),
                     response("<|start|>assistant<|channel|>analysis"), {"choices": []}):
            with self.assertRaises(RuntimeError):
                parse_chat_response(data)

    def test_proxy_preserves_generate_contract_and_sampling(self):
        env = {"JB_MUTATOR_PROFILE": "gpt-oss-20b", "JB_MUTATOR_VLLM_URL": "http://fake/v1",
               "JB_MUTATOR_REASONING_EFFORT": "low"}
        reply = Mock()
        reply.json.return_value = response("print(1)", "reasoning")
        with patch.dict(os.environ, env), patch.object(mutator_proxy.requests, "post", return_value=reply) as post:
            result = mutator_proxy.generate({"prompt": "write code", "max_new_tokens": 2048,
                                             "temperature": 0.15, "top_p": 0.9})
            payload = post.call_args.kwargs["json"]
        self.assertEqual(result["generated_text"], "print(1)")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "write code"}])
        self.assertEqual(payload["max_tokens"], 2048)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["top_p"], 0.9)
        self.assertIn("response_time_sec", result)

    def test_proxy_identity_mismatch_fails(self):
        reply = Mock()
        reply.json.return_value = {"data": [{"id": "some-other-model"}]}
        with patch.dict(os.environ, {"JB_MUTATOR_PROFILE": "gpt-oss-120b", "JB_MUTATOR_VLLM_URL": "http://fake/v1"}), \
             patch.object(mutator_proxy.requests, "get", return_value=reply):
            with self.assertRaises(RuntimeError):
                mutator_proxy.assert_ready()

    def test_actual_server_endpoint_routes_proxy_without_loading_transformers(self):
        async def inline_to_thread(func, *args):
            return func(*args)
        request = types.SimpleNamespace(prompt="hello", max_new_tokens=256, temperature=0.1, top_p=0.9)
        ns = load_functions("server.py", ["generate_text", "preload_model"],
                            {"asyncio": asyncio, "os": os, "LLMRequest": object,
                             "HTTPException": RuntimeError, "LLMModel": Mock(side_effect=AssertionError("GPU load"))})
        with patch.object(asyncio, "to_thread", inline_to_thread), \
             patch.dict(os.environ, {"JB_MUTATOR_VLLM_URL": "http://fake/v1"}), \
             patch.object(mutator_proxy, "generate", return_value={"generated_text": "hi"}), \
             patch.object(mutator_proxy, "assert_ready") as ready:
            asyncio.run(ns["preload_model"]())
            result = asyncio.run(ns["generate_text"](request))
        ready.assert_called_once()
        self.assertEqual(result["generated_text"], "hi")
        ns["LLMModel"].assert_not_called()

    def test_router_accepts_registered_and_legacy_models(self):
        legacy = Mock()
        ns = load_functions("src/llm_utils.py", ["get_llm_code_generator"],
                            {"os": os, "LOCAL_LLM": True, "LLM_MIXTRAL": "mixtral", "LLM_LLAMA3": "llama3",
                             "LLM_DEEPSEEK": "deepseek", "submit_mixtral_local": legacy, "llm_code_qc_hf": Mock()})
        with patch.dict(os.environ, {"JB_MUTATOR_PROFILE": "gpt-oss-20b", "JB_MUTATOR_VLLM_URL": "http://fake/v1"}):
            self.assertIs(ns["get_llm_code_generator"]("gpt-oss-20b")[0], legacy)
            with self.assertRaises(ValueError):
                ns["get_llm_code_generator"]("gpt-oss-120b")
        self.assertIs(ns["get_llm_code_generator"]("llama3")[0], legacy)
        self.assertIs(ns["get_llm_code_generator"]("mixtral")[0], legacy)

    def test_mutator_token_budget_and_return_gen(self):
        requests = Mock()
        requests.post.return_value.status_code = 200
        requests.post.return_value.json.return_value = {"generated_text": "answer"}
        ns = load_functions("src/llm_utils.py", ["submit_mixtral_local"],
                            {"os": os, "requests": requests, "PORT": 8137,
                             "get_llm_server_hostname": lambda: "compute-host"})
        with patch.dict(os.environ, {"JB_MUTATOR_MAX_NEW_TOKENS": "2048"}):
            text, generator = ns["submit_mixtral_local"]("hello", return_gen=True)
            self.assertEqual(requests.post.call_args.kwargs["json"]["max_new_tokens"], 2048)
            ns["submit_mixtral_local"]("hello", max_new_tokens=512)
            self.assertEqual(requests.post.call_args.kwargs["json"]["max_new_tokens"], 512)
        self.assertEqual(text, "answer")
        self.assertIs(generator, ns["submit_mixtral_local"])

    def test_jailbreak_client_preserves_logprobs_and_final_channel(self):
        spec = importlib.util.spec_from_file_location("shared_test_client", ROOT / "sota/Jailbreak/vllm_client.py")
        client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client)
        reply = Mock(status_code=200)
        reply.json.return_value = response("answer", "reasoning")
        with patch.object(client.requests, "post", return_value=reply):
            result = client.chat("http://fake/v1", "target", [{"role": "user", "content": "hello"}])
        self.assertEqual(result["text"], "answer")
        self.assertEqual(result["prompt_tokens"], 10)
        self.assertEqual(result["logprobs"], {"content": []})


if __name__ == "__main__":
    unittest.main()
