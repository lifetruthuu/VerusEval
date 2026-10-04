import json
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generation import run_verusage_completion as completion

VERUSAGE = ROOT / "baselines" / "verus-proof-synthesis" / "verusage"
if str(VERUSAGE) not in sys.path:
    sys.path.insert(0, str(VERUSAGE))

import spec_generation


class VerusageCompletionTests(unittest.TestCase):
    def test_single_round_smoke_run_is_accepted_without_api_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ['run_verusage_completion.py', '--model', 'deepseek-chat',
                    '--dataset-root', str(ROOT / 'data/generation'), '--output-root', directory,
                    '--repair-rounds', '1', '--limit', '1', '--dry-run']
            with patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(completion.main(), 0)
            self.assertEqual(json.loads(output.getvalue())['missing'], 1)

    def test_zero_repair_rounds_is_rejected(self):
        with patch.object(sys, 'argv', ['run_verusage_completion.py', '--model', 'deepseek-chat',
                                      '--repair-rounds', '0', '--dry-run']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                completion.main()
        self.assertEqual(raised.exception.code, 2)

    def test_zero_errors_counts_as_verification_success(self):
        self.assertTrue(completion.verification_succeeded(0, 0))
        self.assertTrue(completion.verification_succeeded(3, 0))
        self.assertFalse(completion.verification_succeeded(3, 1))

    def test_discover_tasks_enumerates_dataset_and_skips_completed_work(self):
        models = [completion.parse_model("deepseek-chat"), completion.parse_model("qwen-coder")]
        shots = ["zero-shot", "few-shot"]
        dataset_root = ROOT / "data" / "generation"

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            tasks = completion.discover_tasks(dataset_root, output_root, shots, models)

            counts = {}
            for task in tasks:
                key = (task.shot, task.model.label)
                counts[key] = counts.get(key, 0) + 1
            self.assertEqual(
                counts,
                {(shot, model.label): 762 for shot in shots for model in models},
            )

            written = tasks[0]
            result = completion.result_path(
                output_root, written.shot, written.model.label, written.task_name, "verified"
            )
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text("verified", encoding="utf-8")

            listed = tasks[1]
            (output_root / "manifest.csv").write_text(
                "shot,model,task_name\n"
                f"{listed.shot},{listed.model.label},{listed.task_name}\n",
                encoding="utf-8",
            )

            remaining = completion.discover_tasks(dataset_root, output_root, shots, models)

        self.assertEqual(len(remaining), len(tasks) - 2)
        self.assertTrue(
            {
                (written.shot, written.model.label, written.task_name),
                (listed.shot, listed.model.label, listed.task_name),
            }.isdisjoint(
                {(task.shot, task.model.label, task.task_name) for task in remaining}
            )
        )

    def test_model_label_can_differ_from_api_model(self):
        model = completion.parse_model("qwen-coder=custom-provider-model")
        self.assertEqual(model.label, "qwen-coder")
        self.assertEqual(model.api_model, "custom-provider-model")

    def test_standard_labels_use_requested_dashscope_models(self):
        self.assertEqual(
            completion.parse_model("qwen-coder").api_model,
            "qwen3-coder-480b-a35b-instruct",
        )
        self.assertEqual(
            completion.parse_model("deepseek-chat").api_model,
            "deepseek-v4-flash",
        )

    def test_each_standard_model_uses_its_own_api_key(self):
        models = [completion.parse_model("qwen-coder"), completion.parse_model("deepseek-chat")]
        with patch.dict(
            completion.os.environ,
            {
                "VERUSAGE_QWEN_CODER_API_KEY": "qwen-secret",
                "VERUSAGE_DEEPSEEK_API_KEY": "deepseek-secret",
            },
            clear=True,
        ):
            keys = completion.resolve_api_keys(models, None)
        self.assertEqual(
            keys,
            {"qwen-coder": "qwen-secret", "deepseek-chat": "deepseek-secret"},
        )

    def test_runtime_config_does_not_persist_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            completion.write_runtime_config(
                path,
                completion.parse_model("deepseek-chat"),
                "https://example.invalid/v1",
                "/path/to/verus",
            )
            config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["aoai_api_key"], [])
        self.assertTrue(config["use_openai"])

    def test_default_path_generates_specs_from_raw_before_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.rs"
            raw_path.write_text("raw executable program", encoding="utf-8")
            task = completion.CompletionTask(
                shot="zero-shot",
                model=completion.parse_model("model"),
                dataset="dataset",
                task_name="task",
                raw_path=raw_path,
            )
            args = SimpleNamespace(
                work_root=root / "work",
                verus_path="verus",
                output_root=root / "output",
                repair_rounds=5,
                temperature=1.0,
                api_keys={"model": "secret"},
                task_timeout=60,
            )
            commands = []

            def fake_run(command, **_kwargs):
                commands.append(command)
                output = Path(command[command.index("--output") + 1])
                output.write_text("generated and repaired", encoding="utf-8")
                return SimpleNamespace(returncode=233)

            with (
                patch.object(completion, "verus_score", return_value=(0, 1, "error")),
                patch.object(completion.subprocess, "run", side_effect=fake_run),
            ):
                state = completion.run_task(
                    task,
                    args,
                    root / "runtime.json",
                    {},
                    {},
                    {},
                )

        self.assertEqual(state["status"], "complete")
        self.assertIn("--generate-specs", commands[0])
        self.assertIn("--spec-repair", commands[0])
        self.assertNotIn("--spec-exemplars-json", commands[0])
        self.assertEqual(commands[0][commands[0].index("--input") + 1], str(raw_path))
        self.assertEqual(commands[0][commands[0].index("--repair") + 1], "5")

    def test_claimed_task_skips_output_completed_by_another_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = completion.CompletionTask(
                shot="zero-shot",
                model=completion.parse_model("model"),
                dataset="dataset",
                task_name="task",
                raw_path=root / "raw.rs",
            )
            args = SimpleNamespace(
                work_root=root / "work",
                output_root=root / "output",
            )
            target = completion.result_path(
                args.output_root, "zero-shot", "model", "task", "verified"
            )
            target.parent.mkdir(parents=True)
            target.write_text("complete", encoding="utf-8")

            with patch.object(
                completion,
                "run_task",
                side_effect=AssertionError("completed task must not run again"),
            ):
                state = completion.run_task_claimed(task, args, root / "config.json", {}, {}, {})

        self.assertEqual(state["status"], "skipped")


class VerusageSpecGenerationTests(unittest.TestCase):
    def test_generation_uses_auto_verus_prompt_and_five_examples(self):
        calls = []

        class FakeLLM:
            def infer_llm(self, *args, **kwargs):
                calls.append((args, kwargs))
                return ["```rust\nverus! { fn f() {} }\n```"]

        config = SimpleNamespace(aoai_generation_model="model", max_token=20000)
        examples = [
            {"input": f"input {index}", "output": f"output {index}"}
            for index in range(5)
        ]
        with (
            patch.object(spec_generation.GlobalConfig, "get_config", return_value=config),
            patch.object(spec_generation.GlobalConfig, "get_llm", return_value=FakeLLM()),
        ):
            code = spec_generation.generate_spec_program("target", examples, 1.0)

        args, _kwargs = calls[0]
        self.assertEqual(code, "verus! { fn f() {} }")
        self.assertEqual(len(args[2]), 5)
        self.assertIn("lacks formal specifications", args[3])
        self.assertIn("meaningful preconditions", args[3])

    def test_generation_rejects_verification_bypass(self):
        class FakeLLM:
            def infer_llm(self, *args, **kwargs):
                return ["```rust\nverus! { proof fn p() { assume(true); } }\n```"]

        config = SimpleNamespace(aoai_generation_model="model", max_token=20000)
        with (
            patch.object(spec_generation.GlobalConfig, "get_config", return_value=config),
            patch.object(spec_generation.GlobalConfig, "get_llm", return_value=FakeLLM()),
        ):
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                spec_generation.generate_spec_program("target", [], 1.0)


if __name__ == "__main__":
    unittest.main()
