"""Exercise the adapter with local model responses and the pinned verifier."""
import contextlib
import http.server
import io
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.generation import run_starverus as runner

VERUS = shutil.which(os.environ.get('VERUS_PATH', 'verus'))
PROGRAM = '''use vstd::prelude::*;
verus! {
fn identity(x: u64) -> (result: u64)
    ensures result == x,
{ x }
}
fn main() {}
'''


@unittest.skipUnless(VERUS, 'pinned Verus is needed for StarVerus offline integration')
class StarVerusTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / 'dataset'
        self.output = self.root / 'output'
        self.task = 'HumanEval-Verus_task_1'
        for kind in ['X_code', 'Y']:
            directory = self.dataset / kind / 'HumanEval-Verus'
            directory.mkdir(parents=True)
            (directory / f'{self.task}.rs').write_text(PROGRAM)
        (self.dataset / 'knn_similar.json').write_text(json.dumps({self.task: [self.task] * 5}))
        # Dataset-wide pairing is checked separately with the released 762 tasks.
        task = SimpleNamespace(subset='HumanEval-Verus')
        self.loader = patch.object(runner, 'load_dataset', return_value=([task], {}, {}))
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.requests = []
        requests = self.requests

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(payload)
                prompt = payload['messages'][-1]['content']
                if 'You are the Judge' in prompt:
                    text = '<result>False</result><DIAGNOSIS>The postcondition is off by one.</DIAGNOSIS>'
                elif 'You are the Aligner' in prompt:
                    text = f'```rust\n{PROGRAM}```'
                else:
                    incorrect = PROGRAM.replace('result == x,', 'result == x + 1,')
                    text = f'```rust\n{incorrect}```'
                response = json.dumps({'id': 'offline', 'object': 'chat.completion',
                    'created': 0, 'model': 'offline-model',
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text},
                                 'finish_reason': 'stop'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.args = ['--dataset-root', str(self.dataset), '--output-root', str(self.output),
                     '--model', 'offline-model', '--verus-path', VERUS, '--candidates', '1',
                     '--base-url', f'http://127.0.0.1:{self.server.server_port}/v1']

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_dry_run_imports_without_model_calls_or_output_files(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            runner.main(self.args + ['--dry-run'])
        self.assertEqual(json.loads(out.getvalue())['tasks'], 1)
        self.assertEqual(self.requests, [])
        self.assertFalse(self.output.exists())

    def test_few_shot_generation_uses_upstream_and_real_verus(self):
        self.check_generation('few-shot', 12)

    def test_zero_shot_generation_uses_upstream_and_real_verus(self):
        self.check_generation('zero-shot', 2)

    def check_generation(self, shot, expected_messages):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-test-key',
                                     'STARVERUS_RUN_CONFIG': '/nonexistent/stale-config.yaml'}):
            runner.main(self.args + ['--shot', shot])
        results = list(self.output.rglob('correct.rs'))
        self.assertEqual(len(results), 1)
        check = subprocess.run([VERUS, str(results[0])], capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stderr)
        log = next((self.output / 'generation').rglob('*.log')).read_text()
        selected = re.search(r'\[TASK VERIFIED\].*selected=(.*)', log)
        self.assertIsNotNone(selected, log)
        final_path = Path(selected.group(1))
        self.assertTrue(final_path.is_relative_to(self.output / 'repair'))
        self.assertEqual(final_path.read_text(), results[0].read_text())
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(len(self.requests[0]['messages']), expected_messages)
        self.assertTrue(any('You are the Judge' in r['messages'][-1]['content']
                            for r in self.requests))
        self.assertFalse(list(self.output.rglob('*.yaml')))
        for path in self.output.rglob('*'):
            if path.is_file():
                self.assertNotIn(b'offline-test-key', path.read_bytes())

    def test_output_cannot_overwrite_inputs(self):
        with contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as raised:
                runner.main(self.args + ['--output-root', str(self.dataset), '--dry-run'])
        self.assertEqual(raised.exception.code, 1)
        self.assertIn('outside the generation inputs', error.getvalue())

    def test_wrong_verus_version_is_rejected_before_generation(self):
        with patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(stdout='new version', stderr='')):
            with contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    runner.main(self.args + ['--dry-run'])
        self.assertEqual(raised.exception.code, 1)
        self.assertIn(runner.VERUS_VERSION, error.getvalue())
        self.assertEqual(self.requests, [])


if __name__ == '__main__':
    unittest.main()
