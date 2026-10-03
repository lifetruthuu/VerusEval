from __future__ import annotations

import unittest
from unittest.mock import patch

from metrics_rebuild.metrics.proportion_at_least_gt import _core_gt_proportion_metric
from metrics_rebuild.share.lemma_shortcuts import safe_trivial_implication
from metrics_rebuild.share.semantic_strength import semantic_strength_comparison
from metrics_rebuild.share.smt import clause_implication_details_for_context


def _context(name: str, clause: str) -> dict:
    return {
        "function": name,
        "parameters": [{"name": "x", "type": "int"}],
        "returns": [],
        "requires": [{"kind": "requires", "text": clause, "normalized": clause}],
        "ensures": [{"kind": "ensures", "text": clause, "normalized": clause}],
    }


class LemmaMetricScoringTest(unittest.TestCase):
    def test_identical_nontrivial_text_is_not_a_safe_shortcut(self) -> None:
        clause = [{"text": "pred(x)", "normalized": "pred(x)"}]
        self.assertIsNone(safe_trivial_implication(clause, clause))
        self.assertTrue(safe_trivial_implication(clause, [])["holds"])
        self.assertTrue(
            safe_trivial_implication(clause, [{"text": "true", "normalized": "true"}])["holds"]
        )

    def test_clause_unknown_is_excluded_from_score_denominator(self) -> None:
        candidate = _context("f", "x > 0")
        reference = _context("f", "x >= 0")
        reference["requires"].append(
            {"kind": "requires", "text": "x < 10", "normalized": "x < 10"}
        )
        with patch(
            "metrics_rebuild.share.smt.lemma_implication_check",
            side_effect=(
                {"holds": True, "status": "valid"},
                {"holds": None, "status": "unknown"},
            ),
        ):
            result = clause_implication_details_for_context(
                candidate=candidate,
                reference=reference,
                reference_rs_path="ref.rs",
                source_side="reference",
                clause_kind="requires",
                implication_direction="generated_implies_reference_clause",
            )

        self.assertEqual(result["status"], "partial")
        self.assertEqual((result["passed"], result["failed"], result["unknown"]), (1, 0, 1))
        self.assertEqual(result["determined"], 1)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["score"], 1.0)

    def test_clause_all_unknown_has_no_score(self) -> None:
        with patch(
            "metrics_rebuild.share.smt.lemma_implication_check",
            return_value={"holds": None, "status": "unknown"},
        ):
            result = clause_implication_details_for_context(
                candidate=_context("f", "x > 0"),
                reference=_context("f", "x >= 0"),
                reference_rs_path="ref.rs",
                source_side="reference",
                clause_kind="requires",
                implication_direction="generated_implies_reference_clause",
            )

        self.assertIsNone(result["score"])
        self.assertEqual(result["determined"], 0)
        self.assertEqual(result["coverage"], 0.0)

    def test_semantic_strength_uses_determined_function_denominator(self) -> None:
        generated = [_context("f", "x > 0"), _context("g", "x > 0")]
        reference = [_context("f", "x >= 0"), _context("g", "x >= 0")]
        results = (
            {
                "function": "f",
                "status": "ok",
                "contract_relation": "generated_refines_reference",
                "generated_refines_ground": True,
                "ground_refines_generated": False,
            },
            {
                "function": "g",
                "status": "partial",
                "contract_relation": "unknown",
                "generated_refines_ground": None,
                "ground_refines_generated": None,
            },
        )
        with patch("metrics_rebuild.share.semantic_strength._function_strength", side_effect=results):
            result = semantic_strength_comparison(
                generated,
                reference,
                lemma_reference_path="ref.rs",
                generated_rs_path="gen.rs",
            )

        self.assertEqual(result["status"], "partial")
        self.assertEqual((result["passed"], result["failed"], result["unknown"]), (1, 0, 1))
        self.assertEqual(result["determined"], 1)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["score"], 1.0)
        self.assertNotIn("gen_implies_ground_smt", result)
        self.assertNotIn("ground_implies_gen_smt", result)
        self.assertNotIn("lemma_fallback_checks", result)

    def test_missing_generated_function_is_a_determined_failure(self) -> None:
        result = semantic_strength_comparison(
            [],
            [_context("f", "x >= 0")],
            lemma_reference_path="ref.rs",
            generated_rs_path="gen.rs",
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual((result["passed"], result["failed"], result["unknown"]), (0, 1, 0))
        self.assertEqual(result["determined"], 1)
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["functions"][0]["ground_refines_generated"])

    def test_timeout_seconds_reaches_every_lemma_call(self) -> None:
        with patch(
            "metrics_rebuild.share.semantic_strength._lemma_check",
            return_value={"holds": False, "status": "invalid"},
        ) as lemma_check:
            semantic_strength_comparison(
                [_context("f", "x > 0")],
                [_context("f", "x >= 0")],
                timeout_seconds=7,
                lemma_reference_path="ref.rs",
                generated_rs_path="gen.rs",
            )

        self.assertEqual(lemma_check.call_count, 4)
        self.assertEqual({call.kwargs["timeout_seconds"] for call in lemma_check.call_args_list}, {7})

    def test_gt_proportion_uses_determined_denominator(self) -> None:
        strength = {
            "status": "partial",
            "engine": "verus_lemma",
            "method": "verus_lemma_implication",
            "functions": [
                {"function": "f", "generated_refines_ground": True},
                {"function": "g", "generated_refines_ground": None},
            ],
        }
        with patch(
            "metrics_rebuild.metrics.proportion_at_least_gt._strength_classification",
            return_value=strength,
        ):
            result = _core_gt_proportion_metric("gen.rs", "ref.rs", direction="at_least_gt")

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["determined"], 1)
        self.assertEqual(result["coverage"], 0.5)

    def test_gt_proportion_all_unknown_has_no_score(self) -> None:
        strength = {
            "status": "partial",
            "functions": [{"function": "f", "generated_refines_ground": None}],
        }
        with patch(
            "metrics_rebuild.metrics.proportion_at_least_gt._strength_classification",
            return_value=strength,
        ):
            result = _core_gt_proportion_metric("gen.rs", "ref.rs", direction="at_least_gt")

        self.assertIsNone(result["score"])
        self.assertEqual(result["determined"], 0)
        self.assertEqual(result["coverage"], 0.0)


if __name__ == "__main__":
    unittest.main()
