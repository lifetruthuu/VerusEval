"""Check runnable configurations and generation using a local model service."""
import http.server
import contextlib
import csv
from collections import Counter
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

from scripts.generation import run_dataset

ROOT = Path(__file__).resolve().parents[1]
VERUS = shutil.which(os.environ.get("VERUS_PATH", "verus"))
RAW = "use vstd::prelude::*;\nverus! { fn identity(x: u64) -> (result: u64) { x } }\nfn main() {}\n"
GENERATED = """use vstd::prelude::*;
verus! {
fn identity(x: u64) -> (result: u64)
    ensures result == x,
{
    proof { assert(x == x); }
    x
}
}
fn main() {}
"""


class MatrixTests(unittest.TestCase):
    def test_matrix_covers_paper_population(self):
        config = json.loads((ROOT / "baselines/generation.json").read_text())
        runs = run_dataset.select_runs(config, "all", None, "all")
        self.assertEqual(len(runs), 18)
        self.assertEqual(sum(r["released_programs"] for r in runs), 13716)
        self.assertEqual(len(run_dataset.select_runs(config, "autoverus", "gpt-4o", "all")), 2)
        with self.assertRaisesRegex(ValueError, "No configuration"):
            run_dataset.select_runs(config, "autoverus", "llama", "all")

    @unittest.skipUnless((ROOT / "data/generated/manifest.csv").is_file(), "release data is required")
    def test_configuration_counts_match_released_programs(self):
        config = json.loads((ROOT / "baselines/generation.json").read_text())
        with (ROOT / "data/generated/manifest.csv").open() as stream:
            actual = Counter((r["workflow"].lower(), r["model"], r["shot"])
                             for r in csv.DictReader(stream))
        expected = {(r["baseline"], r["model"], r["shot"]): r["released_programs"] for r in config["runs"]}
        self.assertEqual(actual, expected)

    def test_generation_cannot_write_into_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    run_dataset.main(["--dataset-root", directory,
                                      "--output-root", directory, "--dry-run"])
            self.assertEqual(raised.exception.code, 1)
            self.assertIn("separate from the generation inputs", error.getvalue())


@unittest.skipUnless(VERUS, "pinned Verus is needed for generation integration")
class GenerationIntegrationTests(unittest.TestCase):
    def test_all_adapters_generate_contracts_and_proofs_in_both_shot_modes(self):
        requests = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(payload)
                content = "```rust\n" + GENERATED + "```"
                if "You are the Judge" in str(payload["messages"]):
                    content = "<result>True</result>"
                response = json.dumps({
                    "id": "local-generation", "object": "chat.completion", "created": 0,
                    "model": "offline-model", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    "choices": [{"index": i, "message": {"role": "assistant", "content": content},
                                 "finish_reason": "stop"} for i in range(payload.get("n", 1))],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset = root / "dataset"
                mapping = {}
                def task_name(index):
                    return "HumanEval-Verus_task_1" if index == 0 else f"VeriCoding_VT{index:04}"
                for index in range(762):
                    task = task_name(index)
                    subset = "HumanEval-Verus" if index == 0 else "VeriCoding"
                    for kind, code in (("X_code", RAW), ("Y", GENERATED)):
                        path = dataset / kind / subset / (task + ".rs")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(code)
                    mapping[task] = [task_name((index + step) % 762) for step in range(1, 6)]
                (dataset / "knn_similar.json").write_text(json.dumps(mapping))
                environment = os.environ.copy()
                environment.update(OPENAI_API_KEY="local-test-only", BASELINE_API_KEY="local-test-only")
                for baseline, model in (("alphaverus", "llama"), ("autoverus", "gpt-4o"),
                                        ("verusage", "qwen-coder"), ("starverus", "gpt-4o")):
                    for shot, messages in (("zero-shot", 2), ("few-shot", 12)):
                        with self.subTest(baseline=baseline, shot=shot):
                            requests.clear()
                            output = root / baseline / shot
                            command = [sys.executable, str(ROOT / "scripts/generation/run_dataset.py"),
                                       "--baseline", baseline, "--model", model, "--api-model", "offline-model",
                                       "--shot", shot, "--benchmark", "HumanEval-Verus", "--verus-path", VERUS,
                                       "--dataset-root", str(dataset), "--output-root", str(output),
                                       "--base-url", f"http://127.0.0.1:{server.server_port}/v1"]
                            result = subprocess.run(command, env=environment, capture_output=True,
                                                    text=True, timeout=180)
                            diagnostics = result.stdout + result.stderr
                            for log in output.rglob("*.log"):
                                diagnostics += log.read_text(errors="replace")[-4000:]
                            self.assertEqual(result.returncode, 0, diagnostics)
                            self.assertTrue(requests, diagnostics)
                            self.assertEqual(len(requests[0]["messages"]), messages)
                            prompt = str(requests[0]["messages"]).lower()
                            for term in ("requires", "ensures", "proof"):
                                self.assertIn(term, prompt)
                            generated = output / baseline / model / shot
                            pattern = ("results/*/*.rs" if baseline in ("alphaverus", "autoverus")
                                       else "**/correct.rs" if baseline == "starverus" else "**/verified/verusage/*.rs")
                            candidates = [p for p in generated.glob(pattern)
                                          if "ensures" in p.read_text() and "proof {" in p.read_text()]
                            self.assertTrue(candidates, diagnostics)
                            check = subprocess.run([VERUS, str(candidates[0]), "--crate-name", "smoke"],
                                                   capture_output=True, text=True)
                            self.assertEqual(check.returncode, 0, check.stderr)
                            manifest = json.loads((output / "generation.json").read_text())
                            self.assertEqual(manifest["status"], "completed")
                            self.assertEqual(manifest["runs"][0]["tasks"], 1)
                            self.assertNotIn("local-test-only", json.dumps(manifest))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
