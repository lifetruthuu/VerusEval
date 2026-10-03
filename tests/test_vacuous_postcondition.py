#!/usr/bin/env python3
"""Regression tests for vacuous_postcondition detection (syntactic tier)."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics_rebuild.share.logic_utils import (
    is_simple_tautology,
    is_syntactic_tautology,
)
from metrics_rebuild.share.triviality import (
    ensures_true_flags,
    vacuous_postcondition_flags_for_path,
)


class TestIsSyntacticTautology(unittest.TestCase):
    def test_literal_true(self):
        self.assertEqual(is_syntactic_tautology("true"), "literal_true")
        self.assertEqual(is_syntactic_tautology("(true)"), "literal_true")

    def test_self_comparison(self):
        self.assertEqual(is_syntactic_tautology("a == a"), "self_comparison")
        self.assertEqual(is_syntactic_tautology("a.len() == a.len()"), "self_comparison")
        self.assertEqual(is_syntactic_tautology("old(x) == old(x)"), "self_comparison")
        self.assertEqual(is_syntactic_tautology("result == result"), "self_comparison")
        self.assertEqual(is_syntactic_tautology("x <= x"), "self_comparison")

    def test_arithmetic_identity(self):
        self.assertEqual(is_syntactic_tautology("x + 0 == x"), "arithmetic_identity")
        self.assertEqual(is_syntactic_tautology("0 + x == x"), "arithmetic_identity")
        self.assertEqual(is_syntactic_tautology("x - 0 == x"), "arithmetic_identity")
        self.assertEqual(is_syntactic_tautology("x * 1 == x"), "arithmetic_identity")
        self.assertEqual(is_syntactic_tautology("1 * x == x"), "arithmetic_identity")
        self.assertEqual(is_syntactic_tautology("x / 1 == x"), "arithmetic_identity")
        self.assertEqual(is_syntactic_tautology("x - x == 0"), "arithmetic_identity")
        self.assertEqual(is_syntactic_tautology("x * 0 == 0"), "arithmetic_identity")
        self.assertEqual(is_syntactic_tautology("(x + 0) == x"), "arithmetic_identity")

    def test_implication_self(self):
        self.assertEqual(
            is_syntactic_tautology("(n > 0) ==> (n > 0)"), "implication_self"
        )

    def test_low_information_implication(self):
        self.assertEqual(
            is_syntactic_tautology("P ==> true"), "low_information_implication"
        )
        self.assertEqual(
            is_syntactic_tautology("false ==> P"), "low_information_implication"
        )
        self.assertEqual(
            is_syntactic_tautology("n > 0 ==> (a == a)"), "low_information_implication"
        )

    def test_quantifier_trivial_body(self):
        self.assertEqual(
            is_syntactic_tautology("forall|i: int| true"), "quantifier_trivial_body"
        )
        self.assertEqual(
            is_syntactic_tautology("forall|i: int| i == i"), "quantifier_trivial_body"
        )
        self.assertEqual(
            is_syntactic_tautology("exists|i: int| true"), "quantifier_trivial_body"
        )

    def test_compound_tautology(self):
        self.assertEqual(
            is_syntactic_tautology("true && true"), "compound_tautology"
        )
        self.assertEqual(
            is_syntactic_tautology("result == result && true"), "compound_tautology"
        )
        self.assertEqual(is_syntactic_tautology("true || P"), "compound_tautology")
        self.assertEqual(is_syntactic_tautology("P || true"), "compound_tautology")
        self.assertEqual(
            is_syntactic_tautology("if c then true else true"), "compound_tautology"
        )

    def test_non_tautologies_return_none(self):
        for expr in [
            "result == exists|i: int| 0 <= i < arr.len() && arr[i] == k",
            "n > 0",
            "a[i] == k",
            "0 <= index <= arr.len()",
            "result >= 0",
            "old(x) == x",
            "forall|i: int| i < n",
            "exists|i: int| i > 0",
            "n > 0 ==> n > 1",
            "P && Q",
            "P || Q",
            "if c then true else false",
            "a == b",
            "x + 1 == x",
        ]:
            self.assertIsNone(is_syntactic_tautology(expr), msg=expr)


class TestIsSimpleTautologyBackwardCompat(unittest.TestCase):
    def test_returns_bool(self):
        self.assertIsInstance(is_simple_tautology("true"), bool)
        self.assertIsInstance(is_simple_tautology("n > 0"), bool)

    def test_legacy_positives(self):
        for expr in ["true", "a == a", "x <= x", "P ==> true", "false ==> P", "P ==> (a == a)"]:
            self.assertTrue(is_simple_tautology(expr), msg=expr)

    def test_legacy_negatives(self):
        for expr in ["n > 0", "a == b", "result == x + 1"]:
            self.assertFalse(is_simple_tautology(expr), msg=expr)

    def test_new_patterns_also_flagged_by_legacy_bool(self):
        for expr in [
            "a.len() == a.len()",
            "x + 0 == x",
            "(n > 0) ==> (n > 0)",
            "forall|i: int| true",
            "result == result && true",
            "old(x) == old(x)",
        ]:
            self.assertTrue(is_simple_tautology(expr), msg=expr)


class TestVacuousPostconditionFlagsForPath(unittest.TestCase):
    def test_spec_fn_ensures_attributed_to_function(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.rs"
            path.write_text(
                "use vstd::prelude::*;\n\nverus! {\n\nspec fn f() -> bool\n    ensures\n        true,\n{}\n\n} // verus!\n"
            )
            result = vacuous_postcondition_flags_for_path(str(path))
            self.assertEqual(result["score"], 1.0)
            self.assertEqual(result["flags"][0]["reason"], "literal_true")
            self.assertEqual(result["flags"][0]["function"], "f")

    def test_recommends_scanned(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rec.rs"
            path.write_text(
                "use vstd::prelude::*;\n\nverus! {\n\nfn f(n: i32) -> (result: bool)\n    recommends\n        true,\n{\n    n > 0\n}\n\n} // verus!\n"
            )
            result = vacuous_postcondition_flags_for_path(str(path))
            self.assertEqual(result["score"], 1.0, msg="recommends should be scanned")
            self.assertEqual(result["flags"][0]["kind"], "recommends")


class TestEnsuresTrueFlagsLegacyApi(unittest.TestCase):
    def test_legacy_shape_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.rs"
            path.write_text(
                "use vstd::prelude::*;\n\nverus! {\n\nfn f(a: &Vec<i32>) -> (result: usize)\n    ensures\n        a.len() == a.len(),\n{\n    a.len()\n}\n\n} // verus!\n"
            )
            from metrics_rebuild.share.clauses import extract_clauses

            clauses = extract_clauses(str(path))
            result = ensures_true_flags(clauses)
            self.assertIn("score", result)
            self.assertIn("flag_count", result)
            self.assertIn("flags", result)
            self.assertEqual(result["score"], 1.0)
            self.assertEqual(result["flag_count"], 1)
            self.assertEqual(result["flags"][0]["clause"], "a . len ( ) == a . len ( )")
            self.assertEqual(result["flags"][0]["reason"], "self_comparison")


if __name__ == "__main__":
    unittest.main()
