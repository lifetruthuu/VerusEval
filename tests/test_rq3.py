"""Regression tests for RQ3 semantic selection and evidence/denominator boundaries."""
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "RQs/RQ3/scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_rq3_natural import profile
from analyze_rq3_variants import choose_confirmed, classify_metric
from rq3_io_common import edit_valid, witness_matches
from rq3_mutations import OPERATORS, generate, relation_edits


class RQ3Tests(unittest.TestCase):
    def test_eight_operators_preserve_body_and_opposite_contract(self):
        source = "use vstd::prelude::*; verus! { fn f(x: i32) -> (r: i32) requires x > 0, x < 10, ensures r > 0, r < 10, { 5 } }"
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "base.rs"
            mutant = Path(folder) / "mutant.rs"
            base.write_text(source)
            _, groups = generate(base, "f", "test", budget=3)
            self.assertEqual(set(groups), set(OPERATORS))
            for operator, group in groups.items():
                self.assertTrue(group["candidates"], operator)
                for candidate in group["candidates"]:
                    mutant.write_text(candidate["text"])
                    self.assertTrue(edit_valid(base, mutant, "f", candidate["direction"]), operator)

    def test_remove_does_not_delete_last_clause(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "base.rs"
            path.write_text("verus! { fn f(x:i32)->(r:i32) requires x>0, ensures r>0, {x} }")
            _, groups = generate(path, "f", "test")
            self.assertFalse(groups["PRE_REMOVE"]["candidates"])
            self.assertFalse(groups["POST_REMOVE"]["candidates"])

    def test_relation_direction_does_not_rewrite_negative_positions(self):
        for text in ("!(x > 0)", "x > 0 ==> y < 3", "forall|i:int| 0 <= i ==> a[i] > 0"):
            self.assertEqual(list(relation_edits(text, False)), [])
        self.assertEqual(list(relation_edits("x < 3", False)), ["x <= 3"])
        self.assertEqual(list(relation_edits("x >= 0", True)), ["x > 0", "x == 0"])

    def test_unknown_is_never_a_difference_witness(self):
        self.assertFalse(witness_matches("pre_strengthening", True, None))
        self.assertTrue(witness_matches("pre_weakening", False, True))
        self.assertFalse(witness_matches("post_weakening", False, True, False, True))
        self.assertTrue(witness_matches("post_weakening", True, True, False, True))

    def test_equivalent_edits_are_excluded_and_only_first_confirmed_retained(self):
        rows = [{"mutant_id": str(i), "base_key": "b", "operator": "PRE_ADD", "candidate_rank": i} for i in range(1, 4)]
        statuses = {"1": {"status": "equivalent"}, "2": {"status": "confirmed"}, "3": {"status": "confirmed"}}
        self.assertEqual([r["mutant_id"] for r in choose_confirmed(rows, statuses)], ["2"])

    def test_diagnostics_unknown_and_behavior_have_different_states(self):
        def node(success, **ev):
            return {"details": [{"id": "c", "success": success, "evaluation": ev}]}
        self.assertEqual(classify_metric(node(None), "negative")["state"], "unresolved")
        self.assertEqual(classify_metric(node(None, reason="contract_expression_undefined"), "negative")["state"], "diagnostic_only")
        self.assertEqual(classify_metric(node(False, requires_accepted=None), "invalid")["state"], "diagnostic_only")
        self.assertEqual(classify_metric(node(False, requires_accepted=True), "invalid")["state"], "behavior_detected")
        self.assertEqual(classify_metric(node(True), "negative")["state"], "missed")
        self.assertEqual(classify_metric({}, "negative")["state"], "unresolved")

    def test_natural_available_case_scope_is_not_complete_three_kind_scope(self):
        outcome = {k: "valid" for k in ("pre_ref_to_gen", "post_gen_to_ref", "pre_gen_to_ref", "post_ref_to_gen")}
        outcome.update({f"io_{kind}_{s}": "0" for kind in ("correct", "wrong", "invalid") for s in ("passed", "failed", "unresolved")})
        outcome["io_correct_passed"] = "5"
        label = dict.fromkeys(("sample_id", "task_id", "workflow", "model", "shot"), "test")
        result = profile(outcome, label)
        self.assertEqual(result["io_state"], "passed")
        self.assertEqual(result["available_io_kinds"], 1)
        outcome["io_wrong_unresolved"] = "1"
        self.assertEqual(profile(outcome, label)["io_state"], "unresolved")
        outcome["io_wrong_unresolved"] = outcome["io_correct_passed"] = "0"
        self.assertEqual(profile(outcome, label)["io_state"], "unavailable")


if __name__ == "__main__":
    unittest.main()
