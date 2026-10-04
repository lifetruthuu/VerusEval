import sys
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generation import run_spec_baselines


class AlphaResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "alpha_verus_utils", ROOT / "baselines/alphaverus/inference/verus_utils.py")
        cls.utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.utils)

    def test_empty_and_truncated_responses_cannot_become_verified_stubs(self):
        for content, reason in [("", "length"), (None, "stop"), ("", "stop"),
                                ("verus! { fn main() {} }", "stop"),
                                ("verus! { fn f() ensures true {", "length")]:
            with self.subTest(content=content, reason=reason):
                with self.assertRaises(ValueError):
                    self.utils.extract_spec_program(content, reason)

    def test_complete_response_preserves_program_with_or_without_fence(self):
        program = 'use vstd::prelude::*;\nverus! { fn f(x: u64) -> (r: u64) ensures r == x { x } }'
        for response in [program, '```rust\n' + program + '\n```']:
            self.assertEqual(self.utils.extract_spec_program(response, 'stop'), program)


class AlphaCandidateTests(unittest.TestCase):
    def make_candidate(self, root: Path, name: str) -> Path:
        path = root / "generation" / "dumps" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("verus! {}", encoding="utf-8")
        return path

    def test_fresh_verus_scores_override_misleading_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            invalid = self.make_candidate(
                task_dir,
                "verified_prog=task_0_0_invalid_0.rs",
            )
            partial = self.make_candidate(
                task_dir,
                "verified_prog=task_2_1_partial_0.rs",
            )
            scores = {
                invalid.name: (0, 2, "compile errors"),
                partial.name: (2, 1, "verification error"),
            }

            with patch.object(
                run_spec_baselines,
                "verus_score",
                side_effect=lambda path, _verus: scores[path.name],
            ):
                selected, verified, errors = run_spec_baselines.alpha_candidate(
                    task_dir,
                    "task",
                    "verus",
                )

            self.assertEqual(selected, partial)
            self.assertEqual((verified, errors), (2, 1))

    def test_verified_candidate_is_preferred(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            partial = self.make_candidate(
                task_dir,
                "verified_prog=task_9_1_partial_0.rs",
            )
            complete = self.make_candidate(
                task_dir,
                "verified_prog=task_1_0_complete_0.rs",
            )
            scores = {
                partial.name: (9, 1, "verification error"),
                complete.name: (1, 0, "verified"),
            }

            with patch.object(
                run_spec_baselines,
                "verus_score",
                side_effect=lambda path, _verus: scores[path.name],
            ):
                selected, verified, errors = run_spec_baselines.alpha_candidate(
                    task_dir,
                    "task",
                    "verus",
                )

            self.assertEqual(selected, complete)
            self.assertEqual((verified, errors), (1, 0))

    def test_zero_error_verification_success_skips_treefinement(self):
        self.assertTrue(run_spec_baselines.verification_succeeded(0, 0))
        self.assertFalse(run_spec_baselines.verification_succeeded(3, 1))
        self.assertTrue(run_spec_baselines.verification_succeeded(1, 0))

    def test_zero_zero_result_skips_treefinement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.rs"
            input_path.write_text("verus! {}", encoding="utf-8")
            selected = root / "selected.rs"
            selected.write_text("broken", encoding="utf-8")
            final_path = root / "result.rs"
            task_dir = root / "task"
            task_dir.mkdir()
            task = run_spec_baselines.DatasetTask(
                task_id="task",
                subset="subset",
                input_path=input_path,
                reference_path=root / "reference.rs",
                shot_ids=(),
            )
            args = SimpleNamespace(
                alpha_model="model",
                alpha_base_url="https://example.invalid/v1",
                temperature=1.0,
                task_timeout=60,
                verus_path="verus",
                skip_alpha_treefinement=False,
                alpha_tree_width=3,
                alpha_repair_rounds=3,
            )
            commands = []

            with (
                patch.object(
                    run_spec_baselines,
                    "run_logged",
                    side_effect=lambda command, *_args: commands.append(command),
                ),
                patch.object(
                    run_spec_baselines,
                    "alpha_candidate",
                    return_value=(selected, 0, 0),
                ),
                patch.object(
                    run_spec_baselines,
                    "verus_score",
                    return_value=(0, 1, "compile error"),
                ),
                patch.object(run_spec_baselines, "ALPHA_DIR", root / "inference"),
            ):
                result = run_spec_baselines.run_alpha(task, task_dir, final_path, args)

            self.assertEqual(len(commands), 1)
            self.assertEqual(result["selected_stage"], "inference")


if __name__ == "__main__":
    unittest.main()
