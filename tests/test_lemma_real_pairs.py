from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from metrics_rebuild.share.functions import strength_contexts_for_path
from metrics_rebuild.share.lemma_implication import _verus_bin
from metrics_rebuild.share.semantic_strength import semantic_strength_comparison


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "tests/fixtures/lemma_implication"


def _context(path: Path, function_name: str) -> dict:
    return next(
        item
        for item in strength_contexts_for_path(str(path))
        if item["function"] == function_name
    )


def _verus_available() -> bool:
    verus = _verus_bin()
    return shutil.which(verus) is not None or Path(verus).is_file()


class TestLemmaFixturePairs(unittest.TestCase):
    def _compare(self, stem: str) -> dict:
        if not _verus_available():
            self.skipTest("Verus is unavailable")
        generated_path = FIXTURE_ROOT / f"{stem}_generated.rs"
        reference_path = FIXTURE_ROOT / f"{stem}_reference.rs"
        return semantic_strength_comparison(
            [_context(generated_path, "target")],
            [_context(reference_path, "target")],
            lemma_reference_path=str(reference_path),
            generated_rs_path=str(generated_path),
        )

    def test_parameter_and_return_renames_are_equivalent(self) -> None:
        result = self._compare("renamed_signature")

        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["classification"], "equivalent", result)
        self.assertEqual(result["coverage"], 1.0)

    def test_identical_pred_text_uses_each_files_helper_semantics(self) -> None:
        result = self._compare("conflicting_helper")

        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["classification"], "incomparable", result)
        requires = result["functions"][0]["requires"]
        self.assertEqual(requires["generated_implies_ground"]["status"], "invalid")
        self.assertEqual(requires["ground_implies_generated"]["status"], "invalid")

    def test_parameter_count_mismatch_is_unknown(self) -> None:
        generated_path = FIXTURE_ROOT / "signature_mismatch_generated.rs"
        reference_path = FIXTURE_ROOT / "signature_mismatch_reference.rs"
        with patch("metrics_rebuild.share.lemma_implication._verify_lemma") as verify:
            result = semantic_strength_comparison(
                [_context(generated_path, "target")],
                [_context(reference_path, "target")],
                lemma_reference_path=str(reference_path),
                generated_rs_path=str(generated_path),
            )

        verify.assert_not_called()
        self.assertEqual(result["status"], "partial", result)
        self.assertEqual(result["classification"], "unknown", result)
        self.assertEqual(result["coverage"], 0.0)
        details = result["functions"][0]["requires"]["generated_implies_ground"]
        self.assertEqual(details["reason"], "target_signature_mismatch")
        self.assertEqual(details["category"], "parameter_count")

    def test_mutable_vec_uses_current_and_old_snapshots(self) -> None:
        result = self._compare("mutable_snapshot")

        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["classification"], "equivalent", result)
        self.assertEqual(result["coverage"], 1.0)


class TestLemmaDatasetPairs(unittest.TestCase):
    def _compare(
        self,
        generated_path: Path,
        reference_path: Path,
        function_name: str,
    ) -> dict:
        if not generated_path.is_file():
            self.skipTest(f"generated sample is unavailable: {generated_path}")
        if not reference_path.is_file():
            self.skipTest(f"reference sample is unavailable: {reference_path}")
        if not _verus_available():
            self.skipTest("Verus is unavailable")
        return semantic_strength_comparison(
            [_context(generated_path, function_name)],
            [_context(reference_path, function_name)],
            lemma_reference_path=str(reference_path),
            generated_rs_path=str(generated_path),
        )

    def test_vh0050_return_rename_is_equivalent(self) -> None:
        generated = (
            PROJECT_ROOT
            / "data/generated/starverus_by_model/zero-shot/gpt-4o/verified/starverus"
            / "starverus_VeriCoding_VH0050_vericoded_gpt-4o_zero-shot_verified.rs"
        )
        reference = (
            PROJECT_ROOT
            / "data/references/VeriCoding/VeriCoding_VH0050_vericoded.rs"
        )
        result = self._compare(generated, reference, "to_lower_exec")

        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["classification"], "equivalent", result)

    def test_mbpp_733_return_rename_is_generated_stronger(self) -> None:
        generated = (
            PROJECT_ROOT
            / "data/generated/starverus_by_model/zero-shot/deepseek-reasoner/verified/starverus"
            / "starverus_VerusBench_MBPP_task_id_733_deepseek-reasoner_zero-shot_verified.rs"
        )
        reference = (
            PROJECT_ROOT
            / "data/references/VerusBench/VerusBench_MBPP_task_id_733.rs"
        )
        result = self._compare(generated, reference, "find_first_occurrence")

        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["classification"], "generated_stronger_or_equal", result)

    def test_vt0043_generated_spec_method_dependency_is_moved(self) -> None:
        generated = (
            PROJECT_ROOT
            / "data/generated/autoverus_few_shot_verified/autoverus"
            / "autoverus_VeriCoding_VT0043_vericoded_gpt-4o_few-shot_verified.rs"
        )
        reference = (
            PROJECT_ROOT
            / "data/references/VeriCoding/VeriCoding_VT0043_vericoded.rs"
        )
        result = self._compare(generated, reference, "broadcast")

        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["classification"], "generated_stronger_or_equal", result)
        implication = result["functions"][0]["ensures"]["generated_implies_ground"]
        self.assertGreaterEqual(
            implication["support_context_summary"]["copied_spec_functions"],
            1,
            implication,
        )


if __name__ == "__main__":
    unittest.main()
