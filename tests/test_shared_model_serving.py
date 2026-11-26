"""Deterministic CPU contracts with fake inference; no deployment verification."""
import ast
import asyncio
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import pytest

from src import model_serving, mutator_proxy, serving_smoke
from src.chat_response import parse_chat_response

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.mock_inference


def response(content="answer", reasoning=None, finish="stop"):
    return {"choices": [{"message": {"content": content, "reasoning": reasoning},
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
    def test_submission_allows_h100_h200_and_forwards_tp(self):
        script = ROOT / "serve_model.sbatch"
        self.assertIn("#SBATCH -C H100|H200", script.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fake_python = tmp / "python3"
            # A shebang can't hold a path with spaces, so wrap the interpreter in sh.
            fake_python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -c '
                                   '"import json, sys; print(json.dumps(sys.argv[1:]))" "$@"\n')
            fake_python.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if not k.startswith("JB_")}
            env.update(PATH=f'{tmp}:{env["PATH"]}', SLURM_SUBMIT_DIR=str(ROOT),
                       JB_SERVE_MODEL="gpt-oss-120b", JB_VLLM_BIN="/fake/bin/vllm",
                       JB_CUDA_MODULE="fake-cuda", JB_SERVE_TP="2")
            run = subprocess.run(["bash", "-c", 'module() { return 0; }; source "$1"', "test", str(script)],
                                 env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            args = json.loads(run.stdout.splitlines()[-1])
            self.assertEqual(args[args.index("--tp") + 1], "2")
            self.assertEqual(args[args.index("--model") + 1], "gpt-oss-120b")
            self.assertNotIn("--max-len", args)  # Registry defaults remain authoritative.
            env["JB_SERVE_MAX_LEN"] = "50000"
            override = subprocess.run(["bash", "-c", 'module() { return 0; }; source "$1"',
                                       "test", str(script)], env=env, capture_output=True,
                                      text=True, timeout=10)
            self.assertEqual(override.returncode, 0, override.stderr)
            override_args = json.loads(override.stdout.splitlines()[-1])
            self.assertEqual(override_args[override_args.index("--max-len") + 1], "50000")

    def test_serving_wrapper_rejects_unbound_before_generic_runtime_setup(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("JB_")}
        env.update(SLURM_SUBMIT_DIR=str(ROOT), JB_SERVE_MODEL="inkling-2-mini-unbound-v21",
                   REAL_PYTHON=sys.executable)
        script = 'python3() { "$REAL_PYTHON" "$@"; }; module() { exit 91; }; source "$1"'
        run = subprocess.run(["bash", "-c", script, "test", str(ROOT / "serve_model.sbatch")],
                             env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertIn("qualified guarded PACE Inkling launcher", run.stderr)
        self.assertNotIn("Endpoint:", run.stdout)

    def test_generic_wrappers_reject_guarded_raw_model_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text('{"model_type":"inkling2_loop"}')
            link = Path(temp) / "local-alias"
            link.symlink_to(checkpoint, target_is_directory=True)
            env = {k: v for k, v in os.environ.items() if not k.startswith("JB_")}
            env.update(SLURM_SUBMIT_DIR=str(ROOT), TMPDIR=temp, REAL_PYTHON=sys.executable)
            prefix = 'python3() { "$REAL_PYTHON" "$@"; }; module() { exit 91; }; source "$1"'
            cases = [
                ("run_jailbreak.sbatch", {"JB_RUN_MUTATOR_PROFILE": "inkling-2-mini-unbound-v21"}),
                ("run_jailbreak.sbatch", {"JB_RUN_TARGET_MODEL": "nokeepai/Inkling-2-Mini-Unbound"}),
                ("run_jailbreak.sbatch", {"JB_RUN_JUDGE_MODEL": str(link)}),
                ("sota/Jailbreak/serve_vllm.sh", {"JB_REPORTING_MODEL_PATH": link.name,
                                                "JB_TARGET_MODEL_PATH": str(Path(temp) / "missing")}),
            ]
            for script, settings in cases:
                with self.subTest(script=script, settings=settings):
                    run = subprocess.run(["bash", "-c", prefix, "test", str(ROOT / script)],
                                         cwd=temp, env=dict(env, **settings), capture_output=True,
                                         text=True, timeout=10)
                    self.assertEqual(run.returncode, 2, run.stderr)
                    self.assertIn("qualified guarded PACE Inkling launcher", run.stderr)

    def test_profile_tp_override_matches_gpu_groups_and_launch_commands(self):
        # Run actual command construction, stopping before any server is started.
        script = (ROOT / "run_jailbreak.sbatch").read_text().split("# target vLLM (weaker model)")[0]
        # Staging verification alone is mocked; registry lookup and grouping are real.
        script = ('module() { return 0; }; python3() { '
                  'if [[ "$*" == "-m src.model_serving check "* ]]; then return 0; fi; '
                  '"$REAL_PYTHON" "$@"; };\n') + script
        script += '\nprintf "TARGET=%s\\n" "${TARGET_COMMAND[*]}"; printf "JUDGE=%s\\n" "${JUDGE_COMMAND[*]}"\n'
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items() if not k.startswith("JB_")}
            env.update(REAL_PYTHON=sys.executable, SLURM_SUBMIT_DIR=str(ROOT), SLURM_JOB_ID="cpu-test",
                       TMPDIR=tmp, CUDA_VISIBLE_DEVICES="a,b,c,d,e,f,g", JB_CUDA_MODULE="fake-cuda",
                       JB_VLLM_BIN="/fake/bin/vllm", JB_RUN_TARGET_PROFILE="gpt-oss-120b",
                       JB_RUN_JUDGE_PROFILE="gpt-oss-20b", JB_RUN_MUTATOR_PROFILE="gpt-oss-20b",
                       JB_RUN_TARGET_TP="2", JB_RUN_JUDGE_TP="2", JB_RUN_MUTATOR_TP="2")
            run = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("target=a,b judge=c,d reporting=e mutator=f,g", run.stdout)
            for role in ("TARGET", "JUDGE"):
                line = next(line for line in run.stdout.splitlines() if line.startswith(role + "="))
                self.assertIn("--tp 2", line)
                self.assertNotIn("--max-len", line)
            env["JB_TARGET_MAX_MODEL_LEN"] = "50000"
            explicit = subprocess.run(["bash", "-c", script], env=env, capture_output=True,
                                      text=True, timeout=15)
            self.assertEqual(explicit.returncode, 0, explicit.stderr)
            target_line = next(line for line in explicit.stdout.splitlines() if line.startswith("TARGET="))
            self.assertIn("--max-len 50000", target_line)
            env["JB_RUN_TARGET_TP"] = "0"
            invalid = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("positive integers", invalid.stderr)

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

    def test_unbound_profile_path_identity_and_budget(self):
        item = model_serving.profile("inkling-2-mini-unbound-v21")
        self.assertEqual(item["revision"], "ef74b44edd2851b8e8d221a7313f36c419590430")
        self.assertEqual(item["default_max_tokens"], 8192)
        self.assertLessEqual(item["default_max_tokens"], item["max_output_tokens"])
        self.assertLess(item["max_output_tokens"], item["max_model_len"])
        self.assertEqual(item["status"], "configured")
        self.assertIn("CPU tested", item["generic_launcher_validation"])
        cmd = model_serving.command("inkling-2-mini-unbound-v21", served_name="unbound-v21")
        self.assertEqual(cmd[cmd.index("--served-model-name") + 1], "unbound-v21")
        self.assertEqual(item["served_model_name"], "inkling-unbound-v21")
        default_cmd = model_serving.command("inkling-2-mini-unbound-v21")
        self.assertEqual(default_cmd[default_cmd.index("--served-model-name") + 1], item["served_model_name"])
        self.assertIn("inkling", cmd)
        with patch.dict(os.environ, {"JB_UNBOUND_CHECKPOINT": "/approved/unbound"}):
            self.assertEqual(model_serving.model_path("inkling-2-mini-unbound-v21"), "/approved/unbound")

    def test_profile_specific_smoke_runtime_contract(self):
        unbound = model_serving.profile("inkling-2-mini-unbound-v21")
        runtime = unbound["smoke_runtime"]
        serving_smoke.validate_runtime(unbound, runtime["python"], runtime["vllm"],
                                       runtime["torch"], runtime["cuda"])
        self.assertEqual(runtime["torch"], "2.14.0+cu132")
        self.assertEqual(runtime["cuda"], "13.2")
        for field, value in (("python", "3.12.16"), ("vllm", "0.17.0"),
                             ("torch", "2.10.0+cu129"), ("cuda", "12.9")):
            actual = {"python": runtime["python"], "vllm": runtime["vllm"],
                      "torch": runtime["torch"], "cuda": runtime["cuda"]}
            actual[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, f"{field}: expected"):
                serving_smoke.validate_runtime(unbound, actual["python"], actual["vllm"],
                                               actual["torch"], actual["cuda"])
        gpt_oss = model_serving.profile("gpt-oss-20b")
        serving_smoke.validate_runtime(gpt_oss, "3.12.15", "0.17.0", "2.10.0+cu129", "12.9")
        with self.assertRaisesRegex(RuntimeError, "vllm: expected 0.17.0"):
            serving_smoke.validate_runtime(gpt_oss, "3.12.15", "0.34.1", "2.10.0+cu129", "12.9")
        with self.assertRaisesRegex(RuntimeError, "qualified guarded PACE Inkling launcher"):
            serving_smoke.require_supported_smoke_path("inkling-2-mini-unbound-v21", unbound)
        serving_smoke.require_supported_smoke_path("gpt-oss-20b", gpt_oss)

    def test_unbound_smoke_fails_before_checkpoint_or_server_launch(self):
        import torch
        item = model_serving.profile("inkling-2-mini-unbound-v21")
        runtime = item["smoke_runtime"]
        fake_vllm = types.ModuleType("vllm")
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict(os.environ, {"SLURM_JOB_ID": "cpu-fixture", "CUDA_VISIBLE_DEVICES": "fixture"}), \
             patch.dict(sys.modules, {"vllm": fake_vllm}), \
             patch.object(serving_smoke.platform, "python_version", return_value=runtime["python"]), \
             patch.object(torch, "__version__", runtime["torch"]), \
             patch.object(torch.version, "cuda", runtime["cuda"]), \
             patch.object(serving_smoke.importlib.metadata, "version", return_value=runtime["vllm"]), \
             patch.object(serving_smoke, "validate_checkpoint", side_effect=AssertionError("hash should not run")), \
             patch.object(serving_smoke.subprocess, "Popen") as popen:
            result = serving_smoke.run("inkling-2-mini-unbound-v21", Path(temp) / "smoke", sys.executable)
        self.assertEqual(result["status"], "failed")
        self.assertIn("qualified guarded PACE Inkling launcher", result["error"])
        popen.assert_not_called()

    def test_unbound_direct_serve_rejects_before_checkpoint_or_exec(self):
        for model in ("inkling-2-mini-unbound-v21", "nokeepai/Inkling-2-Mini-Unbound"):
            for env in ({}, {"SLURM_JOB_ID": "cpu-fixture", "CUDA_VISIBLE_DEVICES": "fixture"}):
                argv = ["model_serving", "serve", "--model", model]
                with self.subTest(model=model, env=env), patch.object(sys, "argv", argv), \
                     patch.dict(os.environ, env, clear=True), redirect_stderr(io.StringIO()) as stderr, \
                     patch.object(model_serving, "model_path") as path, \
                     patch.object(model_serving, "validate_checkpoint") as verify, \
                     patch.object(model_serving.os, "execvp") as execute:
                    with self.assertRaises(SystemExit) as error:
                        model_serving.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn("qualified guarded PACE Inkling launcher", stderr.getvalue())
                path.assert_not_called()
                verify.assert_not_called()
                execute.assert_not_called()

    def test_guarded_command_inspection_is_safe_as_shell_input(self):
        argv = ["model_serving", "command", "--model", "inkling-2-mini-unbound-v21",
                "--model-path", "/checkpoint\nexit 73\n#"]
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()) as stdout, \
             patch.dict(os.environ, {"JB_VLLM_BIN": "exit"}, clear=True), \
             patch.object(model_serving, "validate_checkpoint") as verify, \
             patch.object(model_serving.os, "execvp") as execute:
            model_serving.main()
        output = stdout.getvalue()
        self.assertIn("Inspection only", output)
        self.assertIn("qualified guarded PACE Inkling launcher", output)
        self.assertIn("exit serve", output)
        self.assertTrue(all(line.startswith("# ") for line in output.splitlines()))
        shell = subprocess.run(["bash"], input=output, text=True, capture_output=True, timeout=10)
        self.assertEqual(shell.returncode, 0, shell.stderr)
        verify.assert_not_called()
        execute.assert_not_called()

    def test_generic_profiles_still_verify_and_exec_expected_command(self):
        models = ("gpt-oss-20b", "gpt-oss-120b", "llama-3.3-70b", "mistral-7b", "devstral-small-2-24b")
        env = {"SLURM_JOB_ID": "cpu-fixture", "CUDA_VISIBLE_DEVICES": "a,b", "JB_VLLM_BIN": "/fake/vllm"}
        for model in models:
            path = "/fixture/checkpoint"
            argv = ["model_serving", "serve", "--model", model, "--model-path", path]
            with self.subTest(model=model), patch.object(sys, "argv", argv), \
                 patch.dict(os.environ, env, clear=True), redirect_stdout(io.StringIO()), \
                 patch.object(model_serving, "validate_checkpoint") as verify, \
                 patch.object(model_serving.os, "execvp") as execute:
                expected = model_serving.command(model, path)
                model_serving.main()
            verify.assert_called_once_with(model, path)
            execute.assert_called_once_with(expected[0], expected)

    def test_unbound_checkpoint_sha256_integrity(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text('{"model_type":"inkling2_loop"}')
            payload = b"test checkpoint weights"
            (path / "model.safetensors").write_bytes(payload)
            manifest_path = ROOT / "config/unbound-v21-checkpoint.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"] = {"model.safetensors": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}
            item = model_serving.profile("inkling-2-mini-unbound-v21")
            manifest["repo_id"] = item["model_id"]
            manifest["revision"] = item["revision"]
            with patch.object(model_serving, "REGISTRY", path / "served_models.json"):
                (path / "served_models.json").write_text(json.dumps({"models": {"inkling-2-mini-unbound-v21": item}}))
                (path / "unbound-v21-checkpoint.json").write_text(json.dumps(manifest))
                with patch.dict(os.environ, {"JB_UNBOUND_CHECKPOINT": str(path)}):
                    (path / "other.safetensors").touch()
                    (path / "model.safetensors").unlink()
                    with self.assertRaisesRegex(ValueError, "Checkpoint file missing"):
                        model_serving.validate_checkpoint("inkling-2-mini-unbound-v21", path)
                    (path / "model.safetensors").write_bytes(payload[:-1])
                    with self.assertRaisesRegex(ValueError, "size mismatch"):
                        model_serving.validate_checkpoint("inkling-2-mini-unbound-v21", path)
                    (path / "model.safetensors").write_bytes(payload)
                    model_serving.validate_checkpoint("inkling-2-mini-unbound-v21", path)
                    with patch.object(Path, "read_bytes", side_effect=AssertionError("unbounded read")), \
                         patch.object(__import__("hashlib"), "file_digest", wraps=__import__("hashlib").file_digest) as digest:
                        model_serving.validate_checkpoint("inkling-2-mini-unbound-v21", path)
                    digest.assert_called_once()
                    (path / "model.safetensors").write_bytes(b"test checkpoint weighTS")
                    with pytest.raises(ValueError, match="SHA-256 mismatch"):
                        model_serving.validate_checkpoint("inkling-2-mini-unbound-v21", path)
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

    def test_native_mistral_staging_needs_native_config_tokenizer_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            item = model_serving.profile("devstral-small-2-24b")
            (path / "deployment.json").write_text(json.dumps(item))
            (path / "config.json").write_text("{}")
            (path / "model-00001.safetensors").touch()
            with self.assertRaisesRegex(ValueError, "Checkpoint not staged"):
                model_serving.validate_checkpoint("devstral-small-2-24b", path)
            (path / "params.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "tekken.json"):
                model_serving.validate_checkpoint("devstral-small-2-24b", path)
            (path / "tekken.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "No safetensors"):
                model_serving.validate_checkpoint("devstral-small-2-24b", path)
            (path / "consolidated.safetensors").touch()
            model_serving.validate_checkpoint("devstral-small-2-24b", path)

    def test_team_storage_root_override_is_used_by_commands(self):
        with patch.dict(os.environ, {"JB_SHARED_MODEL_ROOT": "/team/models"}):
            path = "/team/models/openai/gpt-oss-20b"
            self.assertEqual(model_serving.model_path("gpt-oss-20b"), path)
            self.assertEqual(model_serving.command("gpt-oss-20b")[2], path)

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
        partial = parse_chat_response(response("partial answer", "reasoning", "length"))
        self.assertEqual(partial["text"], "partial answer")
        self.assertEqual(partial["finish_reason"], "length")
        for data in (response(None, "reasoning", "length"), response("", None, "tool_calls"),
                     response("<|start|>assistant<|channel|>analysis"), {"choices": []}):
            with self.assertRaises(RuntimeError):
                parse_chat_response(data)

    def test_reasoning_fields_and_invalid_answers(self):
        old = response("final")
        old["choices"][0]["message"]["reasoning_content"] = "legacy reasoning"
        self.assertEqual(parse_chat_response(old)["reasoning"], "legacy reasoning")
        current = response("final", "current reasoning")
        current["choices"][0]["message"]["reasoning_content"] = None
        self.assertEqual(parse_chat_response(current)["reasoning"], "current reasoning")
        tool = response("Calling a tool", finish="tool_calls")
        tool["choices"][0]["message"]["tool_calls"] = [{"function": {"name": "test"}}]
        for invalid in (response(""), response("  \n"), response(None, "analysis", "length"),
                        response("final", {"bad": "shape"}), tool, {"choices": [{"message": None}]}):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                parse_chat_response(invalid)

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

    def test_unbound_reasoning_does_not_send_gpt_oss_effort_parameter(self):
        reply = Mock()
        reply.json.return_value = response("final", "reasoning")
        env = {"JB_MUTATOR_PROFILE": "inkling-2-mini-unbound-v21",
               "JB_MUTATOR_VLLM_URL": "http://fake/v1", "JB_MUTATOR_REASONING_EFFORT": "high"}
        with patch.dict(os.environ, env), patch.object(mutator_proxy.requests, "post", return_value=reply) as post:
            result = mutator_proxy.generate({"prompt": "question", "max_new_tokens": 8192,
                                             "temperature": 0.1, "top_p": 0.9})
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "inkling-unbound-v21")
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(payload["max_tokens"], 8192)
        self.assertEqual(result["generated_text"], "final")

    def test_unbound_readiness_accepts_served_alias_only(self):
        url = "http://fake/v1"
        response_models = Mock()
        response_models.json.return_value = {"data": [{"id": "inkling-unbound-v21"}]}
        with patch.dict(os.environ, {"JB_MUTATOR_PROFILE": "inkling-2-mini-unbound-v21",
                                    "JB_MUTATOR_VLLM_URL": url}), \
             patch.object(mutator_proxy.requests, "get", return_value=response_models):
            mutator_proxy.assert_ready()
        response_models.json.return_value = {"data": [{"id": "nokeepai/Inkling-2-Mini-Unbound"}]}
        with patch.dict(os.environ, {"JB_MUTATOR_PROFILE": "inkling-2-mini-unbound-v21",
                                    "JB_MUTATOR_VLLM_URL": url}), \
             patch.object(mutator_proxy.requests, "get", return_value=response_models), \
             self.assertRaisesRegex(RuntimeError, "identity differs"):
            mutator_proxy.assert_ready()

    def test_pace_endpoint_reuses_generate_without_legacy_launcher(self):
        profile_id = "RedHatAI/Muse-Glimmer-30B-FP8-block"
        with self.assertRaises(ValueError):
            model_serving.command("muse")
        with tempfile.TemporaryDirectory() as temp:
            key_file = Path(temp) / "api-key"
            key_file.write_text("unit-test-placeholder")
            reply = Mock()
            reply.json.return_value = response("391", "parsed reasoning")
            env = {"JB_MUTATOR_PROFILE": profile_id, "JB_MUTATOR_VLLM_URL": "http://fake/v1",
                   "JB_MUTATOR_API_KEY_FILE": str(key_file)}
            with patch.dict(os.environ, env), patch.object(mutator_proxy.requests, "post", return_value=reply) as post:
                result = mutator_proxy.generate({"prompt": "17 times 23", "max_new_tokens": 32,
                                                 "temperature": 0, "top_p": 1})
        self.assertEqual(result["generated_text"], "391")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "muse")
        self.assertNotIn("reasoning_effort", post.call_args.kwargs["json"])
        self.assertEqual(post.call_args.kwargs["headers"], {"Authorization": "Bearer unit-test-placeholder"})
        self.assertFalse(post.call_args.kwargs["allow_redirects"])
        self.assertEqual(post.call_args.kwargs["proxies"], {"http": "", "https": "", "all": ""})

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
        ns = load_functions("src/llm_utils.py", ["resolve_mutator_max_new_tokens", "submit_mixtral_local"],
                            {"os": os, "requests": requests, "PORT": 8137,
                             "get_llm_server_hostname": lambda: "compute-host",
                             "endpoint_profile": model_serving.endpoint_profile})
        with patch.dict(os.environ, {"JB_MUTATOR_MAX_NEW_TOKENS": "2048"}):
            text, generator = ns["submit_mixtral_local"]("hello", return_gen=True)
            self.assertEqual(requests.post.call_args.kwargs["json"]["max_new_tokens"], 2048)
            ns["submit_mixtral_local"]("hello", max_new_tokens=512)
            self.assertEqual(requests.post.call_args.kwargs["json"]["max_new_tokens"], 512)
        self.assertEqual(text, "answer")
        self.assertIs(generator, ns["submit_mixtral_local"])

    def test_mutator_profile_generation_budget_precedence_and_validation(self):
        ns = load_functions("src/llm_utils.py", ["resolve_mutator_max_new_tokens"],
                            {"os": os, "endpoint_profile": model_serving.endpoint_profile})
        resolve = ns["resolve_mutator_max_new_tokens"]
        with patch.dict(os.environ, {"JB_MUTATOR_PROFILE": "inkling-2-mini-unbound-v21"}, clear=True):
            self.assertEqual(resolve(), 8192)
            self.assertEqual(resolve(1024), 1024)
        with patch.dict(os.environ, {"JB_MUTATOR_PROFILE": "inkling-2-mini-unbound-v21",
                                    "JB_MUTATOR_MAX_NEW_TOKENS": "4096"}, clear=True):
            self.assertEqual(resolve(), 4096)
            self.assertEqual(resolve(2048), 2048)  # Explicit caller argument wins.
            with self.assertRaisesRegex(ValueError, "exceeds.*32768"):
                resolve(32769)
        with patch.dict(os.environ, {"JB_MUTATOR_PROFILE": "gpt-oss-20b"}, clear=True):
            self.assertEqual(resolve(), 850)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve(), 850)
        with patch.dict(os.environ, {"JB_MUTATOR_PROFILE": "inkling-2-mini-unbound-v21",
                                    "JB_MUTATOR_MAX_NEW_TOKENS": "invalid"}, clear=True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                resolve()
        for invalid in (0, -1, "bad"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "positive integer"):
                resolve(invalid)

    def test_unbound_default_budget_reaches_generate_proxy(self):
        requests = Mock()
        requests.post.return_value.status_code = 200
        requests.post.return_value.json.return_value = {"generated_text": "answer"}
        ns = load_functions("src/llm_utils.py", ["resolve_mutator_max_new_tokens", "submit_mixtral_local"],
                            {"os": os, "requests": requests, "PORT": 8137,
                             "get_llm_server_hostname": lambda: "compute-host",
                             "endpoint_profile": model_serving.endpoint_profile})
        with patch.dict(os.environ, {"JB_MUTATOR_PROFILE": "inkling-2-mini-unbound-v21"}, clear=True):
            ns["submit_mixtral_local"]("hello")
        self.assertEqual(requests.post.call_args.kwargs["json"]["max_new_tokens"], 8192)

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
