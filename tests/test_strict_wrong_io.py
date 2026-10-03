"""Regression coverage for admitted-input wrong-output rejection."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from metrics_rebuild.share.io_cases import score_io_cases, strict_wrong_output_decision


class TestStrictWrongIO(unittest.TestCase):
    def test_input_rejection_cannot_pass_wrong_output_check(self):
        evaluation = {"accepted": False, "reason": "contract_rejected"}
        self.assertFalse(strict_wrong_output_decision(evaluation, False)["success"])
        self.assertTrue(strict_wrong_output_decision(evaluation, True)["success"])
        self.assertIsNone(strict_wrong_output_decision(evaluation, None)["success"])

    def test_proved_pair_acceptance_fails_even_if_separate_preproof_unknown(self):
        evaluation = {"accepted": True, "reason": "accepted"}
        self.assertFalse(strict_wrong_output_decision(evaluation, None)["success"])

    def test_unknown_output_is_not_turned_into_a_rejection(self):
        evaluation = {"accepted": None, "reason": "timeout"}
        result = strict_wrong_output_decision(evaluation, True)
        self.assertIsNone(result["success"])
        self.assertEqual(result["reason"], "timeout")
        self.assertFalse(strict_wrong_output_decision(evaluation, False)["success"])

    def test_score_requires_admission_and_records_evidence(self):
        context = {
            "function": "target", "parameters": [{"name": "x", "type": "i32"}],
            "returns": [{"name": "r", "type": "i32"}],
            "requires": [{"text": "x > 10"}], "ensures": [{"text": "r == x"}],
        }
        suite = {"function": "target", "target": {**context, "return_type": "i32"},
                 "cases": [{"id": f"n{i}", "kind": "negative", "inputs": {"x": i},
                            "mutated_output": {"r": -1}, "status": "validated"}
                           for i in range(3)]}
        pair = {f"sc_{i}": {"accepted": False, "reason": "contract_rejected"}
                for i in range(3)}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "gen.rs"
            source.write_text("verus! { fn target(x: i32) -> (r: i32) requires x > 10 ensures r == x { x } }")
            with patch("metrics_rebuild.share.io_cases.strength_contexts_for_path", return_value=[context]), \
                 patch("metrics_rebuild.share.io_cases.batch_verus_contract_decide_detailed", return_value=pair), \
                 patch("metrics_rebuild.share.io_cases.batch_verus_requires_decide",
                       return_value={"sc_0": False, "sc_1": True, "sc_2": None}):
                result = score_io_cases(str(source), suite, "negative")
        self.assertEqual((result["passed"], result["failed"], result["unknown"]), (1, 1, 1))
        self.assertEqual(result["score"], 1 / 3)
        self.assertEqual(result["details"][0]["evaluation"]["reason"], "wrong_input_rejected")
        self.assertEqual(result["check_policy"], "admitted_input_and_rejected_output_v1")


    def test_admission_reason_only_when_output_rejection_is_proved(self):
        rejected = strict_wrong_output_decision({"accepted": False, "reason": "contract_rejected"}, None)
        self.assertIsNone(rejected["success"])
        self.assertEqual(rejected["reason"], "input_admission_unresolved")
        preflight = strict_wrong_output_decision({"accepted": None, "reason": "signature_type_mismatch"}, None)
        self.assertIsNone(preflight["success"])
        self.assertEqual(preflight["reason"], "signature_type_mismatch")

    def test_signature_types_ignore_layout_lifetimes_and_shared_vec_views(self):
        from metrics_rebuild.share.io_cases import _normalized_io_type as norm

        self.assertEqual(norm("(\n    usize,\n    usize,\n)"), norm("(usize, usize)"))
        self.assertEqual(norm("&'static str"), norm("&str"))
        self.assertEqual(norm("&Vec<i32>"), norm("&[i32]"))
        self.assertNotEqual(norm("(usize,)"), norm("(usize)"))
        self.assertNotEqual(norm("&mut Vec<i32>"), norm("&[i32]"))
        self.assertNotEqual(norm("&[int]"), norm("&[i32]"))
        self.assertNotEqual(norm("i8"), norm("i32"))


if __name__ == "__main__":
    unittest.main()
