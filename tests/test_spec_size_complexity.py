"""Tests for the Pratt-parser-based spec size/complexity metric and its core."""

import os
import tempfile
import unittest

from metrics_rebuild.metrics.spec_size_complexity import (
    metric_spec_size_complexity,
    metric_spec_size_complexity_proof,
    metric_spec_size_complexity_spec_only,
)
from metrics_rebuild.share import spec_ast


def analyze(expr):
    stats, triggers, ok = spec_ast.analyze_expression(expr)
    return stats, triggers, ok


class TestSpecAstParser(unittest.TestCase):
    def test_flat_conjunction_is_shallow(self):
        # A run of the same connective is one logical level, not n-1.
        self.assertEqual(analyze("a && b && c && d")[0].logic_depth, 1)
        self.assertEqual(analyze("a ==> b ==> c")[0].logic_depth, 1)

    def test_changing_connective_adds_depth(self):
        self.assertEqual(analyze("(a || b) && (c || d)")[0].logic_depth, 2)
        self.assertEqual(analyze("a && (b ==> c)")[0].logic_depth, 2)

    def test_calls_and_indexing_do_not_add_logical_depth(self):
        self.assertEqual(analyze("f(g(h(i(x)))) == 0")[0].logic_depth, 0)
        self.assertEqual(analyze("v@.subrange(0, k as int).len() == k")[0].logic_depth, 0)

    def test_quantifier_alternation_depth(self):
        self.assertEqual(analyze("forall|i: int| a[i] >= 0")[0].alt_depth, 1)
        # Same-kind nesting is still one block (no alternation).
        self.assertEqual(analyze("forall|i| a[i] > 0 && forall|j| b[j] > 0")[0].alt_depth, 1)
        # forall/exists alternation is the complexity cliff.
        self.assertEqual(analyze("forall|i| exists|j| a[i] == b[j]")[0].alt_depth, 2)
        self.assertEqual(analyze("forall|i| exists|j| forall|k| p(i, j, k)")[0].alt_depth, 3)

    def test_quantifier_count(self):
        self.assertEqual(analyze("forall|i| exists|j| a[i] == b[j]")[0].n_quant, 2)

    def test_vocabulary_tracks_operators_and_identifiers(self):
        stats, _triggers, _ok = analyze("forall|i| f(i) && g(x)")
        self.assertIn("forall", stats.operators)
        self.assertIn("&&", stats.operators)
        self.assertIn("call()", stats.operators)
        self.assertIn("i", stats.variables)
        self.assertIn("x", stats.variables)

    def test_trigger_counting(self):
        self.assertEqual(analyze("forall|i: int| #![trigger f(i)] f(i) > 0")[1], 1)
        self.assertEqual(analyze("forall|i| a[#[trigger] i] == 0")[1], 1)
        self.assertEqual(analyze("forall|i| a[i] == 0")[1], 0)

    def test_lifetime_does_not_swallow_brackets(self):
        # Regression: the old scanner treated `'` as a char literal, swallowing
        # everything (incl. brackets) up to the next `'`, yielding depth 0.
        toks = spec_ast.lex("&'a Vec (nested (deep))")
        kinds = [t.kind for t in toks]
        self.assertIn("LIFETIME", kinds)
        # The parenthesised groups after the lifetime are still lexed as parens.
        self.assertEqual(kinds.count("LPAREN"), 2)
        self.assertEqual(kinds.count("RPAREN"), 2)

    def test_char_literal_still_recognized(self):
        toks = spec_ast.lex("c == 'x'")
        self.assertIn("CHAR", [t.kind for t in toks])

    def test_no_double_counting_of_quantifier_leaf(self):
        # `forall` contributes exactly one node (as a QUANT), not also a leaf.
        # forall|i| x  ->  QUANT + IDENT(x) = 2 nodes.
        self.assertEqual(analyze("forall|i| x")[0].nodes, 2)

    def test_parser_is_total_on_garbage(self):
        for junk in ["", "((((", "forall|", "a && && b", "* / % ==>", "#[", "'"]:
            stats, _triggers, _ok = analyze(junk)
            self.assertGreaterEqual(stats.nodes, 0)  # never raises

    def test_non_ascii_letters_do_not_crash_lex(self):
        # Regression: isalpha() matched Cyrillic/Arabic but IDENT_RE is ASCII-only.
        for text in ["a[i] >= a[i] by нор", "برای", "x && б"]:
            toks = spec_ast.lex(text)
            self.assertEqual(toks[-1].kind, "EOF")
            self.assertTrue(any(t.kind == "UNKNOWN" for t in toks[:-1]))
            stats, _triggers, _ok = analyze(text)
            self.assertGreaterEqual(stats.nodes, 0)

    def test_common_collection_macros_are_fully_parsed(self):
        cases = {
            "seq![x, y + 1]": "macro.seq!",
            "seq![x; n]": "macro.seq!",
            "set![x, y]": "macro.set!",
            "map![key => value, other => x + 1]": "macro.map!",
        }
        for expression, macro_operator in cases.items():
            with self.subTest(expression=expression):
                stats, _triggers, ok = analyze(expression)
                self.assertTrue(ok)
                self.assertIn(macro_operator, stats.operators)
                self.assertGreater(stats.nodes, 1)

        map_stats = analyze("map![key => value]")[0]
        self.assertIn("=>", map_stats.operators)
        self.assertIn("key", map_stats.variables)
        self.assertIn("value", map_stats.variables)

    def test_unsupported_macro_tokens_are_not_silently_dropped(self):
        _stats, _triggers, ok = analyze("seq![$x]")
        self.assertFalse(ok)

    def test_prefix_connective_list_matches_infix_spelling(self):
        # Regression: `&&& a &&& b` used to fail parsing and report 1 node,
        # dropping ~90% of the clause structure.
        for prefix, infix in (
            ("&&& contains(m, a, n) &&& upper(m, a, n)", "contains(m, a, n) &&& upper(m, a, n)"),
            ("||| a == 1 ||| a == 2", "a == 1 ||| a == 2"),
        ):
            with self.subTest(prefix=prefix):
                pre, _t, pre_ok = analyze(prefix)
                inf, _t2, inf_ok = analyze(infix)
                self.assertTrue(pre_ok)
                self.assertTrue(inf_ok)
                self.assertEqual(pre.nodes, inf.nodes)
                self.assertEqual(pre.logic_depth, inf.logic_depth)
                self.assertEqual(pre.variables, inf.variables)

    def test_implies_keyword_is_equivalent_to_arrow(self):
        for keyword, arrow in (
            ("a implies b", "a ==> b"),
            ("a implies b implies c", "a ==> b ==> c"),
            ("forall|k: int| 2 <= k < n implies (n % k) != 0",
             "forall|k: int| 2 <= k < n ==> (n % k) != 0"),
        ):
            with self.subTest(keyword=keyword):
                kw, _t, kw_ok = analyze(keyword)
                ar, _t2, ar_ok = analyze(arrow)
                self.assertTrue(kw_ok)
                self.assertTrue(ar_ok)
                self.assertEqual(kw.nodes, ar.nodes)
                self.assertEqual(kw.logic_depth, ar.logic_depth)
                # Normalized to `==>` so the two spellings share a vocabulary entry.
                self.assertIn("==>", kw.operators)
                self.assertNotIn("implies", kw.operators)

    def test_additional_verus_operators_parse(self):
        for expression in (
            "s.to_set() === t",          # extensional equality
            "n when n >= 0",             # decreases ... when
            "n via helper",              # decreases ... via
            "(x =~= y) <== (p && q)",    # reverse implication
        ):
            with self.subTest(expression=expression):
                stats, _triggers, ok = analyze(expression)
                self.assertTrue(ok)
                self.assertGreater(stats.nodes, 1)

    def test_dafny_only_syntax_stays_unparsed(self):
        # `++` is Dafny sequence concatenation, not Verus. Accepting it would
        # silently bless a real defect in a generated spec; it must keep
        # surfacing as a parse failure.
        self.assertFalse(analyze("a@ == b@.take(i) ++ c")[2])
        # The Verus spelling of the same operation parses.
        self.assertTrue(analyze("a@ == b@.take(i) + c")[2])

    def test_reverse_implication_is_distinct_from_comparison(self):
        implication, _t, ok = analyze("a <== b")
        self.assertTrue(ok)
        self.assertEqual(implication.logic_depth, 1)
        self.assertIn("<==", implication.operators)
        # `<=` must still lex as a comparison and stay outside the logic skeleton.
        comparison = analyze("a <= b")[0]
        self.assertEqual(comparison.logic_depth, 0)
        self.assertIn("<=", comparison.operators)

    def test_float_literals_with_suffix_and_exponent(self):
        for expression in ("r[0] == 0.0f32", "r == 1.0e39f32", "r == 2.5e-3f64"):
            with self.subTest(expression=expression):
                self.assertTrue(analyze(expression)[2])
        # Ranges must not be swallowed by the float pattern.
        self.assertEqual([t.value for t in spec_ast.lex("0..5")][:3], ["0", "..", "5"])

    def test_else_if_chain_folds_like_a_match(self):
        # Regression: flat n-way dispatch used to report one level per branch,
        # so a 26-way char map reported logic_depth=26 while the equivalent
        # `match` reported 1.
        chain = "if c == 0 { 1 } else if c == 1 { 2 } else if c == 2 { 3 } else { 4 }"
        match = "match c { 0 => 1, 1 => 2, 2 => 3, _ => 4 }"
        self.assertEqual(analyze(chain)[0].logic_depth, analyze(match)[0].logic_depth)
        self.assertEqual(analyze(chain)[0].logic_depth, 1)
        # Genuine nesting in a branch body still adds a level.
        self.assertEqual(analyze("if a { if b { x } else { y } } else { z }")[0].logic_depth, 2)

    def test_cast_target_type_is_not_a_variable(self):
        stats, _triggers, _ok = analyze("a as int == b as nat")
        self.assertEqual(sorted(stats.variables), ["a", "b"])
        self.assertIn("as", stats.operators)
        # The type role is inherited through GROUP/UNARY wrappers.
        self.assertEqual(sorted(analyze("x as (int)")[0].variables), ["x"])
        self.assertEqual(sorted(analyze("x as &int")[0].variables), ["x"])

    def test_parser_is_total_on_pathologically_deep_input(self):
        # Regression: 500+ chained connectives (legal Verus, and a typical LLM
        # degeneration pattern) used to overflow the recursion limit in either
        # the parse phase or the recursive statistics traversals.
        legal_prefix = "&&& a " * 2000
        stats, _t, ok = analyze(legal_prefix)
        self.assertTrue(ok)
        self.assertEqual(stats.nodes, 3999)  # 2000 leaves + 1999 connectives
        self.assertEqual(stats.logic_depth, 1)

        legal_infix = "a" + " && a" * 2000
        stats, _t, ok = analyze(legal_infix)
        self.assertTrue(ok)
        self.assertEqual(stats.logic_depth, 1)

        # Right-associative chains and deep parens still parse recursively;
        # they must degrade to a parse failure, never raise.
        for pathological in ("a" + " ==> a" * 2000, "(" * 2000 + "a" + ")" * 2000, "&&& " * 3000):
            stats, _t, ok = analyze(pathological)
            self.assertFalse(ok)
            self.assertGreaterEqual(stats.nodes, 0)

    def test_dangling_prefix_connective_is_a_parse_failure(self):
        # A connective without an operand is malformed input, not an empty
        # expression that silently reports status ok.
        for text in ("&&&", "|||", "&&& a &&&", "&&& &&& a"):
            with self.subTest(text=text):
                self.assertFalse(analyze(text)[2])


def _write(tmp, name, body):
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


_SIMPLE = """use vstd::prelude::*;
verus! {
fn max(a: u32, b: u32) -> (r: u32)
    ensures r >= a && r >= b,
{ if a >= b { a } else { b } }
}
"""

_OVER_COMPLEX = """use vstd::prelude::*;
verus! {
fn max(a: u32, b: u32) -> (r: u32)
    requires a < 1000, b < 1000,
    ensures
        r >= a && r >= b,
        (r == a || r == b),
        forall|x: u32| (x == a || x == b) ==> exists|y: u32| y == r && y >= x,
{ if a >= b { a } else { b } }
}
"""

_HEAVY_PROOF = """use vstd::prelude::*;
verus! {
fn max(a: u32, b: u32) -> (r: u32)
    ensures r >= a && r >= b,
{
    assert(a >= 0);
    assert(b >= 0);
    assert(a < 5000);
    if a >= b { a } else { b }
}
"""


class TestSpecSizeComplexityMetric(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.simple = _write(self.tmp, "simple.rs", _SIMPLE)
        self.complex = _write(self.tmp, "complex.rs", _OVER_COMPLEX)
        self.heavy_proof = _write(self.tmp, "heavy_proof.rs", _HEAVY_PROOF)

    def test_identical_files_score_zero(self):
        r = metric_spec_size_complexity(self.simple, self.simple)
        self.assertEqual(r["status"], "ok")
        self.assertIsNone(r["score"])
        self.assertEqual(r["delta"]["node_count"], 0)
        self.assertEqual(r["delta"]["logic_depth"], 0)
        self.assertEqual(r["delta"]["quantifier_count"], 0)
        self.assertEqual(r["delta"]["vocabulary_size"], 0)

    def test_top_level_score_and_status_shape(self):
        r = metric_spec_size_complexity(self.complex, self.simple)
        self.assertNotIn("generated", r)
        self.assertIn("score", r)
        self.assertIn("gen", r)
        self.assertIn("ref", r)
        self.assertIn("delta", r)
        self.assertEqual(r["status"], "ok")

    def test_more_complex_generated_has_larger_raw_counts(self):
        r = metric_spec_size_complexity(self.complex, self.simple)
        self.assertGreater(r["gen"]["node_count"], r["ref"]["node_count"])
        self.assertGreater(r["gen"]["logic_depth"], r["ref"]["logic_depth"])
        self.assertGreater(r["gen"]["quantifier_count"], r["ref"]["quantifier_count"])
        self.assertGreater(r["gen"]["vocabulary_size"], r["ref"]["vocabulary_size"])

    def test_simpler_generated_has_negative_delta(self):
        r = metric_spec_size_complexity(self.simple, self.complex)
        self.assertLess(r["delta"]["node_count"], 0)
        self.assertLessEqual(r["delta"]["logic_depth"], 0)
        self.assertLess(r["delta"]["quantifier_count"], 0)

    def test_proof_asserts_included_in_raw_counts(self):
        r = metric_spec_size_complexity(self.heavy_proof, self.simple)
        self.assertEqual(r["gen"]["spec_clauses"], 4)
        self.assertGreater(r["gen"]["node_count"], r["ref"]["node_count"])
        self.assertEqual(r["gen"]["breakdown"]["contract_clauses"], 1)
        self.assertEqual(r["gen"]["breakdown"]["asserts"], 3)

    def test_missing_generated_file_is_unavailable(self):
        r = metric_spec_size_complexity(os.path.join(self.tmp, "nope.rs"), self.simple)
        self.assertEqual(r["status"], "unavailable")
        self.assertIsNone(r["score"])

    def test_spec_only_scope_ignores_proof_asserts(self):
        r = metric_spec_size_complexity_spec_only(self.heavy_proof, self.simple)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["scope"], "spec_only")
        self.assertEqual(r["gen"]["spec_clauses"], 1)
        self.assertEqual(r["ref"]["spec_clauses"], 1)
        self.assertEqual(r["gen"]["breakdown"]["proof_clauses"], 0)
        self.assertEqual(r["delta"]["quantifier_count"], 0)

    def test_proof_scope_isolated_from_contract(self):
        r = metric_spec_size_complexity_proof(self.heavy_proof, self.simple)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["scope"], "proof")
        self.assertEqual(r["gen"]["spec_clauses"], 3)
        self.assertEqual(r["ref"]["spec_clauses"], 0)
        self.assertEqual(r["gen"]["breakdown"]["contract_clauses"], 0)
        self.assertEqual(r["gen"]["breakdown"]["asserts"], 3)
        self.assertGreater(r["delta"]["node_count"], 0)

    def test_incomplete_clause_marks_metric_partial_and_reports_coverage(self):
        broken = _write(
            self.tmp,
            "broken.rs",
            "verus! { fn f(x: int) requires x && && x { } }",
        )
        r = metric_spec_size_complexity(broken, self.simple)
        self.assertEqual(r["status"], "partial")
        self.assertGreater(r["gen"]["parse_failed_clauses"], 0)
        self.assertLess(r["gen"]["parse_coverage"], 1.0)
        self.assertEqual(
            r["gen"]["parse_ok_clauses"] + r["gen"]["parse_failed_clauses"],
            r["gen"]["spec_clauses"],
        )

    def test_reference_parse_failure_does_not_invalidate_generated_side(self):
        # Regression: status folded both sides together, so a reference clause
        # the parser could not handle made downstream discard a clean generated
        # sample entirely.
        broken_ref = _write(
            self.tmp,
            "broken_ref.rs",
            "verus! { fn f(x: int) requires x && && x { } }",
        )
        r = metric_spec_size_complexity(self.simple, broken_ref)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["gen_parse_failed_clauses"], 0)
        self.assertGreater(r["ref_parse_failed_clauses"], 0)
        self.assertFalse(r["delta_reliable"])
        self.assertGreater(r["gen"]["node_count"], 0)

    def test_spec_fn_body_with_let_statements_is_fully_counted(self):
        # Regression: a body like `let p = f(s); p.0 && p.1` stopped at the
        # first `;`, dropping the tail expression and reporting depth 0.
        multi = _write(
            self.tmp,
            "multi_stmt.rs",
            "verus! { spec fn a(s: int) -> bool { let p = helper(s); p > 0 && p < 9 } }",
        )
        single = _write(
            self.tmp,
            "single_expr.rs",
            "verus! { spec fn a(s: int) -> bool { helper(s) > 0 && helper(s) < 9 } }",
        )
        r = metric_spec_size_complexity_spec_only(multi, single)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["gen"]["parse_failed_clauses"], 0)
        self.assertEqual(r["gen"]["logic_depth"], 1)
        self.assertEqual(r["gen"]["breakdown"]["spec_fn_bodies"], 1)

    def test_single_expression_spec_fn_body_is_unchanged_by_block_wrapping(self):
        # The brace re-wrapping must be a no-op for bodies that already are a
        # single expression, since BLOCK is a transparent node.
        body = _write(
            self.tmp,
            "single_body.rs",
            "verus! { spec fn a(x: int) -> bool { x > 0 && x < 10 } }",
        )
        r = metric_spec_size_complexity_spec_only(body, body)
        bare = spec_ast.analyze_expression("x > 0 && x < 10")[0]
        self.assertEqual(r["gen"]["node_count"], bare.nodes)
        self.assertEqual(r["gen"]["logic_depth"], bare.logic_depth)


if __name__ == "__main__":
    unittest.main()
