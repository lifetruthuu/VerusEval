"""Release integrity and CLI failures must remain explicit and fail closed."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.validate import compare_results, safe_path, validate, validate_references, verify_hash

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_reference_catalog_requires_real_files_and_matching_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            relative = 'data/references/Example/task.rs'
            path = root / relative
            path.parent.mkdir(parents=True)
            raw = b'fn main() {}'
            catalog = [{'reference_path': relative,
                        'reference_sha256': hashlib.sha256(raw).hexdigest()}]
            with self.assertRaisesRegex(ValueError, 'Missing data or code file'):
                validate_references(root, catalog)
            path.write_bytes(raw)
            self.assertEqual(validate_references(root, catalog), 1)
            with self.assertRaisesRegex(ValueError, 'one-to-one'):
                validate_references(root, catalog + catalog)
            path.write_bytes(b'changed reference')
            with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
                validate_references(root, catalog)

    def test_missing_data_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Missing data'):
                validate(Path(folder))

    def test_mutated_bytes_fail_hash_check(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'record.json'
            raw = b'{"score": 0.75, "status": "unknown"}'
            path.write_bytes(raw)
            expected = hashlib.sha256(raw).hexdigest()
            verify_hash(path, expected)
            path.write_bytes(raw.replace(b'0.75', b'1.00'))
            with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
                verify_hash(path, expected)

    def test_manifest_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'release'
            root.mkdir()
            secret = Path(folder) / 'private'
            secret.write_text('not for release')
            (root / 'linked').symlink_to(secret)
            for path in ('../private', str(secret), 'linked'):
                with self.assertRaises(ValueError):
                    safe_path(root, path)

    def test_reproduction_rejects_unsupported_research_question(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/reproduce.py'), '--rq', '5'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)

    def test_numerical_comparison_detects_changed_state(self):
        from scripts import validate as module
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            work, frozen = root / 'work', root / 'frozen'
            name = 'RQ4/results/directions.csv'
            for base in (work / 'RQs', frozen):
                path = base / name
                path.parent.mkdir(parents=True)
                path.write_text('key,artifacts,share\nL1,610,0.162\n')
            from unittest.mock import patch
            with patch.dict(module.TABLES, {'4': [name]}):
                self.assertEqual(compare_results(work, frozen, {'4'}), [name])
                (work / 'RQs' / name).write_text('key,artifacts,share\nL1,611,0.162\n')
                with self.assertRaisesRegex(ValueError, 'Changed result'):
                    compare_results(work, frozen, {'4'})


if __name__ == '__main__':
    unittest.main()
