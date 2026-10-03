#!/usr/bin/env python3
from __future__ import annotations

import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path

from metrics_rebuild.metrics.trivial_spec_ratio import (
    _component_detection_state,
    metric_trivial_spec_ratio,
)
from metrics_rebuild.share.lemma_implication import _verus_bin, set_lemma_verus_binary
from metrics_rebuild.share.proof_probe import (
    probe_postcondition_truth_for_path,
    probe_precondition_falsity_for_path,
)


def _verus_available() -> bool:
    verus_bin = _verus_bin()
    return shutil.which(verus_bin) is not None or Path(verus_bin).is_file()


VERUS_AVAILABLE = _verus_available()


class TestTrivialComponentDetectionState(unittest.TestCase):
    def test_component_detection_states(self) -> None:
        self.assertEqual(
            _component_detection_state({"status": "partial", "score": 1.0}),
            "detected",
        )
        self.assertEqual(
            _component_detection_state({"status": "ok", "score": 0.0}),
            "not_detected",
        )
        self.assertEqual(
            _component_detection_state({"status": "partial", "score": 0.0}),
            "undetermined",
        )
        self.assertEqual(
            _component_detection_state({"status": "not_available", "score": None}),
            "undetermined",
        )


def _write_verus_file(tmpdir: Path, name: str, function_source: str) -> str:
    path = tmpdir / name
    path.write_text(
        textwrap.dedent(
            f"""\
            use vstd::prelude::*;

            verus! {{

            {function_source.rstrip()}

            fn main() {{}}

            }} // verus!
            """
        ),
        encoding="utf-8",
    )
    return str(path)


@unittest.skipUnless(VERUS_AVAILABLE, "Verus not available")
class TestTrivialProofProbesWithVerus(unittest.TestCase):
    def test_precondition_always_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "prefalse.rs",
                """
                fn f(x: i32)
                    requires
                        x > 0,
                        x < 0,
                {
                }
                """,
            )
            result = probe_precondition_falsity_for_path(path)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["score"], 1.0)
            self.assertEqual(result["always_false_functions"], 1)
            self.assertEqual(result["functions"][0]["outcome"], "always_false")

    def test_precondition_normal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "preok.rs",
                """
                fn f(x: i32)
                    requires
                        x > 0,
                {
                }
                """,
            )
            result = probe_precondition_falsity_for_path(path)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["score"], 0.0)
            self.assertEqual(result["always_false_functions"], 0)
            self.assertEqual(result["functions"][0]["outcome"], "not_trivial")

    def test_no_requires_defaults_to_true_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "norequires.rs",
                """
                fn f(x: i32) {
                }
                """,
            )
            result = probe_precondition_falsity_for_path(path)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["score"], 0.0)
            self.assertEqual(result["checked_functions"], 1)
            self.assertEqual(result["functions"][0]["outcome"], "no_requires_default_true")
            self.assertFalse(result["functions"][0]["always_false"])
            self.assertTrue(result["functions"][0]["counted"])

    def test_postcondition_always_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "posttrue.rs",
                """
                fn g(x: i32) -> (r: i32)
                    ensures
                        r == r,
                {
                    x
                }
                """,
            )
            result = probe_postcondition_truth_for_path(path)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["score"], 1.0)
            self.assertEqual(result["always_true_functions"], 1)
            self.assertEqual(result["functions"][0]["outcome"], "always_true")

    def test_no_ensures_defaults_to_true_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "noensures.rs",
                """
                fn g(x: i32) -> (r: i32) {
                    x
                }
                """,
            )
            result = probe_postcondition_truth_for_path(path)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["score"], 1.0)
            self.assertEqual(result["checked_functions"], 1)
            self.assertEqual(result["functions"][0]["outcome"], "no_ensures_default_true")
            self.assertTrue(result["functions"][0]["always_true"])
            self.assertTrue(result["functions"][0]["counted"])

    def test_postcondition_nontrivial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "postreal.rs",
                """
                fn g(x: i32) -> (r: i32)
                    ensures
                        r == x + 1,
                {
                    x
                }
                """,
            )
            result = probe_postcondition_truth_for_path(path)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["score"], 0.0)
            self.assertEqual(result["always_true_functions"], 0)
            self.assertEqual(result["functions"][0]["outcome"], "not_trivial")

    def test_postcondition_checked_as_whole_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "postmixed.rs",
                """
                fn h(x: i32) -> (r: i32)
                    ensures
                        r == r,
                        r >= x,
                {
                    x
                }
                """,
            )
            result = probe_postcondition_truth_for_path(path)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["score"], 0.0)
            self.assertEqual(result["always_true_functions"], 0)
            self.assertEqual(result["functions"][0]["outcome"], "not_trivial")

    def test_postcondition_old_mut_nontrivial_is_probed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "oldmut.rs",
                """
                fn update(data: &mut Vec<i32>)
                    ensures
                        old(data).len() == data.len(),
                {
                }
                """,
            )
            result = probe_postcondition_truth_for_path(path)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["score"], 0.0)
            self.assertEqual(result["checked_functions"], 1)
            self.assertEqual(result["unresolved"], 0)
            self.assertEqual(result["skipped_old_mut"], 0)
            self.assertEqual(result["functions"][0]["outcome"], "not_trivial")
            self.assertFalse(result["functions"][0]["holds"])

    def test_postcondition_old_mut_trivial_is_probed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "oldmut_true.rs",
                """
                fn update(data: &mut Vec<i32>)
                    ensures
                        old (data).len() == old(data).len(),
                {
                }
                """,
            )
            result = probe_postcondition_truth_for_path(path)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["score"], 1.0)
            self.assertEqual(result["skipped_old_mut"], 0)
            self.assertEqual(result["functions"][0]["outcome"], "always_true")
            self.assertTrue(result["functions"][0]["always_true"])
            self.assertTrue(result["functions"][0]["holds"])

    def test_postcondition_mut_without_old_is_probed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "mut_no_old.rs",
                """
                fn update(data: &mut Vec<i32>)
                    ensures
                        data.len() == data.len(),
                {
                }
                """,
            )
            result = probe_postcondition_truth_for_path(path)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["score"], 1.0)
            self.assertEqual(result["skipped_old_mut"], 0)
            self.assertEqual(result["functions"][0]["outcome"], "always_true")
            self.assertTrue(result["functions"][0]["holds"])

    def test_metric_trivial_spec_ratio_exposes_new_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            generated = _write_verus_file(
                tmpdir,
                "eval.rs",
                """
                fn f(x: i32)
                    requires
                        x > 0,
                        x < 0,
                {
                }
                """,
            )
            ground = _write_verus_file(
                tmpdir,
                "ref.rs",
                """
                fn f(x: i32)
                    requires
                        x > 0,
                {
                }
                """,
            )
            result = metric_trivial_spec_ratio(generated, ground)
            self.assertIn("precondition_falsity", result["generated"]["details"])
            self.assertIn("postcondition_truth", result["generated"]["details"])
            self.assertIn("retained_unused", result["generated"]["details"])
            self.assertGreater(result["generated"]["score"], result["ground"]["score"])


class TestTrivialProofProbesUnavailable(unittest.TestCase):
    def test_missing_verus_binary_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "missing_verus.rs",
                """
                fn f(x: i32)
                    requires
                        x > 0,
                {
                }
                """,
            )
            set_lemma_verus_binary("/definitely/not/a/verus/binary")
            try:
                result = probe_precondition_falsity_for_path(path)
            finally:
                set_lemma_verus_binary(None)
            self.assertIn(result["status"], {"unavailable", "partial"})
            self.assertEqual(result["checked_functions"], 1)
            self.assertEqual(result["unresolved"], 1)


if __name__ == "__main__":
    unittest.main()
