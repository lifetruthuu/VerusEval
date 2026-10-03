from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from metrics_rebuild.metrics._clause_implication import compute_clause_implication_pair
from metrics_rebuild.metrics.llm import intent_consistency
from metrics_rebuild.metrics.llm import spec_code_intent_consistency
from metrics_rebuild.metrics.llm.common import llm_judge_prompt
from metrics_rebuild.metrics.proportion_at_least_gt import metric_proportion_at_least_gt
from metrics_rebuild.metrics.verus_textual_similarity import (
    metric_verus_textual_similarity_spec_only,
)
from metrics_rebuild.share.functions import (
    authoritative_target_for_path,
    resolve_pair_target,
)


REFERENCE = """
verus! {
fn helper(x: int) -> (r: int)
    requires x > 100,
    ensures r == x,
{ x }

// <vc-spec>
fn target(x: int) -> (r: int)
    requires x >= 0,
    ensures r == x + 1,
// </vc-spec>
// <vc-code>
{ x + 1 }
// </vc-code>
}
"""


GENERATED_SAME_TARGET_DIFFERENT_HELPER = """
verus! {
fn helper(x: int) -> (r: int)
    requires x < -100,
    ensures r == 0,
{ 0 }

// <vc-spec>
fn target(x: int) -> (r: int)
    requires x >= 0,
    ensures r == x + 1,
// </vc-spec>
// <vc-code>
{ x + 1 }
// </vc-code>
}
"""


class PairTargetResolutionTest(unittest.TestCase):
    def _write_pair(self, generated_text: str, reference_text: str = REFERENCE):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        generated = root / "gen.rs"
        reference = root / "ref.rs"
        generated.write_text(generated_text, encoding="utf-8")
        reference.write_text(reference_text, encoding="utf-8")
        return tmp, generated, reference

    def test_vc_spec_target_wins_over_helper(self):
        tmp, generated, reference = self._write_pair(GENERATED_SAME_TARGET_DIFFERENT_HELPER)
        with tmp:
            alignment = resolve_pair_target(str(generated), str(reference))
        self.assertEqual(alignment["status"], "ok")
        self.assertEqual(alignment["function"], "target")
        self.assertEqual(alignment["selection"], "vc_spec_marker")

    def test_catalog_manual_target_agrees_with_last_contracted_exec(self):
        project_root = Path(__file__).resolve().parents[1]
        reference = (
            project_root
            / "data/references"
            / "HumanEval-Verus"
            / "HumanEval-Verus_task_22.rs"
        )
        catalog_target = authoritative_target_for_path(str(reference))
        source_fallback = authoritative_target_for_path(
            str(reference), use_target_catalog=False
        )
        self.assertIsNotNone(catalog_target)
        self.assertIsNotNone(source_fallback)
        self.assertEqual(catalog_target["function"], "string_sequence_impl")
        self.assertEqual(catalog_target["selection"], "manual_source_audit")
        self.assertEqual(source_fallback["function"], "string_sequence_impl")
        self.assertEqual(source_fallback["selection"], "last_contracted_exec")

    def test_last_contracted_exec_wins_over_earlier_helper(self):
        reference_text = """
verus! {
fn helper(x: int) -> (r: int)
    requires x > 100,
    ensures r == x,
{ x }

fn target(x: int) -> (r: int)
    requires x >= 0,
    ensures r == x + 1,
{ x + 1 }
}
fn main() { let _ = target(0); }
"""
        tmp, generated, reference = self._write_pair(reference_text, reference_text)
        with tmp:
            target = authoritative_target_for_path(str(reference), use_target_catalog=False)
        self.assertIsNotNone(target)
        self.assertEqual(target["function"], "target")
        self.assertEqual(target["selection"], "last_contracted_exec")

    def test_last_exec_used_when_no_contracts(self):
        reference_text = """
verus! {
fn helper(x: int) -> (r: int) { x }
fn target(x: int) -> (r: int) { x + 1 }
}
fn main() { let _ = target(0); }
"""
        tmp, generated, reference = self._write_pair(reference_text, reference_text)
        with tmp:
            target = authoritative_target_for_path(str(reference), use_target_catalog=False)
        self.assertIsNotNone(target)
        self.assertEqual(target["function"], "target")
        self.assertEqual(target["selection"], "last_exec")

    def test_last_exec_skips_trailing_main_inside_verus_block(self):
        reference_text = """
verus! {
fn helper(x: int) -> (r: int)
    requires x > 0,
    ensures r == x,
{ x }

fn target(x: int) -> (r: int)
    requires x >= 0,
    ensures r == x + 1,
{ x + 1 }

fn main() {
    let _ = target(0);
}
}
"""
        tmp, generated, reference = self._write_pair(reference_text, reference_text)
        with tmp:
            target = authoritative_target_for_path(str(reference), use_target_catalog=False)
        self.assertIsNotNone(target)
        self.assertEqual(target["function"], "target")
        self.assertNotEqual(target["function"], "main")
        self.assertEqual(target["selection"], "last_contracted_exec")

    def test_generated_marker_mismatch_is_diagnostic_only(self):
        generated_text = GENERATED_SAME_TARGET_DIFFERENT_HELPER.replace(
            "// <vc-spec>\nfn target", "// <vc-spec>\nfn marked_other",
        ).replace(
            "// </vc-code>\n}",
            "// </vc-code>\nfn target(x: int) -> (r: int)\n    requires x >= 0,\n    ensures r == x + 1,\n{ x + 1 }\n}",
        )
        tmp, generated, reference = self._write_pair(generated_text)
        with tmp:
            alignment = resolve_pair_target(str(generated), str(reference))
        self.assertEqual(alignment["status"], "ok")
        self.assertIn("generated_marker_points_to_different_function", alignment["diagnostics"])

    def test_signature_mismatch_is_diagnostic_only(self):
        generated_text = GENERATED_SAME_TARGET_DIFFERENT_HELPER.replace(
            "fn target(x: int)", "fn target(x: nat)",
        )
        tmp, generated, reference = self._write_pair(generated_text)
        with tmp:
            alignment = resolve_pair_target(str(generated), str(reference))
        self.assertEqual(alignment["status"], "ok")
        self.assertFalse(alignment["signature_match"])
        self.assertIn("target_signature_differs", alignment["diagnostics"])

    def test_text_similarity_remains_file_scoped(self):
        tmp, generated, reference = self._write_pair(GENERATED_SAME_TARGET_DIFFERENT_HELPER)
        with tmp:
            result = metric_verus_textual_similarity_spec_only(str(generated), str(reference))
        self.assertNotIn("target_alignment", result)
        self.assertNotEqual(result["score"], 1.0)

    def test_text_similarity_does_not_require_matching_target(self):
        generated_text = GENERATED_SAME_TARGET_DIFFERENT_HELPER.replace(
            "fn target(x: int)", "fn another_target(x: int)",
        )
        tmp, generated, reference = self._write_pair(generated_text)
        with tmp:
            result = metric_verus_textual_similarity_spec_only(str(generated), str(reference))
        self.assertNotEqual(result["status"], "target_mismatch")
        self.assertNotIn("target_alignment", result)

    def test_clause_metric_includes_all_exec_functions(self):
        tmp, generated, reference = self._write_pair(GENERATED_SAME_TARGET_DIFFERENT_HELPER)
        with tmp:
            result = compute_clause_implication_pair(
                str(generated),
                str(reference),
                source_side="reference",
                clause_kind="requires",
                implication_direction="generated_implies_reference_clause",
                metric_kind="fixture_completeness",
                note="fixture",
            )
        self.assertNotIn("target_alignment", result)
        self.assertEqual(len(result["generated"]["functions"]), 2)
        self.assertEqual(
            {item["function"] for item in result["generated"]["functions"]},
            {"helper", "target"},
        )

    def test_gt_proportion_includes_all_exec_functions(self):
        tmp, generated, reference = self._write_pair(GENERATED_SAME_TARGET_DIFFERENT_HELPER)
        with tmp:
            result = metric_proportion_at_least_gt(str(generated), str(reference))
        self.assertNotIn("target_alignment", result)
        self.assertEqual(result["total"], 2)
        self.assertEqual({item["function"] for item in result["functions"]}, {"helper", "target"})

    def test_llm_prompt_contains_only_target_contract(self):
        tmp, generated, reference = self._write_pair(GENERATED_SAME_TARGET_DIFFERENT_HELPER)
        with tmp:
            messages = llm_judge_prompt(str(generated), str(reference), "target")
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["target_function"], "target")
        self.assertIn("fn target", payload["generated"]["contracts"])
        self.assertNotIn("fn helper", payload["generated"]["contracts"])

    def test_llm_judge_target_mismatch_skips_llm_and_scores_zero(self):
        generated_text = GENERATED_SAME_TARGET_DIFFERENT_HELPER.replace(
            "fn target(x: int)", "fn another_target(x: int)",
        )
        frontend = {"success": True, "outcome_status": "verified"}
        tmp, generated, reference = self._write_pair(generated_text)
        with tmp, \
             patch.object(intent_consistency, "verus_frontend_run_to_dict", return_value=frontend), \
             patch.object(intent_consistency, "call_llm_json") as call_llm:
            result = intent_consistency.metric_llm_as_judge_intent_consistency(
                str(generated), str(reference),
            )
        self.assertEqual(result["status"], "target_mismatch")
        self.assertEqual(result["score"], 0.0)
        call_llm.assert_not_called()

    def test_spec_code_judge_does_not_require_matching_gen_ref_target(self):
        generated_text = GENERATED_SAME_TARGET_DIFFERENT_HELPER.replace(
            "fn target(x: int)", "fn another_target(x: int)",
        )
        frontend = {"success": True, "outcome_status": "verified"}
        judgments = [
            {"status": "ok", "score": 0.6, "llm": {"status": "ok"}, "cached": False},
            {"status": "ok", "score": 0.9, "llm": {"status": "ok"}, "cached": False},
        ]
        tmp, generated, reference = self._write_pair(generated_text)
        with tmp, \
             patch.object(spec_code_intent_consistency, "verus_frontend_run_to_dict", return_value=frontend), \
             patch.object(spec_code_intent_consistency, "_llm_spec_code_alignment_judgment", side_effect=judgments):
            result = spec_code_intent_consistency.metric_llm_as_judge_spec_code_intent_consistency(
                str(generated), str(reference),
            )
        self.assertEqual(result["score"], 0.6)
        self.assertAlmostEqual(result["delta"], -0.3)
        self.assertNotIn("target_alignment", result)


if __name__ == "__main__":
    unittest.main()
