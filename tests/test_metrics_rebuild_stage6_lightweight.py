from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import metrics_rebuild.registry as registry  # noqa: E402
from metrics_rebuild.catalog import METRIC_CATALOG  # noqa: E402
from metrics_rebuild.share import config as rebuild_config  # noqa: E402
from metrics_rebuild.cli import main as cli_main  # noqa: E402
from metrics_rebuild.cli.args import build_parser  # noqa: E402
from metrics_rebuild.cli.output import render_json  # noqa: E402
from metrics_rebuild.share import io_cases, llm_client  # noqa: E402
from metrics_rebuild.metrics.verus_textual_similarity import metric_verus_textual_similarity_spec_only  # noqa: E402
from metrics_rebuild.share.clauses import extract_clauses_from_text, spec_tokens, text_similarity_items  # noqa: E402
from metrics_rebuild.share.functions import strength_contexts_for_path  # noqa: E402
from metrics_rebuild.share.contract_eval import (  # noqa: E402
    _contract_check_proof_fn_lines,
    _fixed_sequence_clauses,
    coerce_value_for_type,
    contract_evaluation,
    eval_contract_expr,
    typed_input_payload,
    typed_output_payload,
    verus_literal,
)
from metrics_rebuild.share.mutation import (  # noqa: E402
    generate_all_implementation_mutants,
    generated_self_spec_robustness,
    mutation_kill_rate_for_path,
    mutation_outcome_from_runs,
    sample_mutants_by_family,
    simple_mutation_kill_rate,
)
from metrics_rebuild.share.redundancy import (  # noqa: E402
    extract_removal_candidates_from_text,
    remove_candidate_text,
)
from metrics_rebuild.share.text import (  # noqa: E402
    get_text_metric_scope,
    normalize_text_metric_scope,
    strip_comments,
    text_metric_scope,
    tokens,
)
from metrics_rebuild.share.smt import clause_implication_metric  # noqa: E402
from metrics_rebuild.share.semantic_strength import semantic_strength_comparison  # noqa: E402
from metrics_rebuild.share.verus_runner import VerusRun  # noqa: E402


def _write_fixture(tmpdir: str, name: str, text: str) -> str:
    path = Path(tmpdir) / name
    path.write_text(text, encoding="utf-8")
    return str(path)


class MetricsRebuildStage6LightweightTest(unittest.TestCase):
    def test_compute_all_metrics_uses_native_text_inputs_and_current_notes(self) -> None:
        def metric_stub(generated_rs_path: str, ground_rs_path: str) -> dict:
            return {"status": "ok", "score": 1.0}

        with tempfile.TemporaryDirectory() as tmp:
            generated = _write_fixture(
                tmp,
                "eval.rs",
                "verus! { fn f(x: int) -> (y: int) ensures y >= x { x + 1 } }",
            )
            ground = _write_fixture(
                tmp,
                "ref.rs",
                "verus! { fn f(x: int) -> (y: int) ensures y > x { x + 1 } }",
            )
            with patch.object(registry, "METRIC_FUNCTIONS", (metric_stub,)):
                report = registry.compute_all_metrics(
                    generated,
                    ground,
                    text_scope="spec_only",
                    include_mutation=True,
                )

        self.assertEqual(report["text_metric_scope"], "spec_only")
        self.assertEqual(report["text_metric_inputs"]["token_source"], "spec_items")
        self.assertEqual(list(report["metrics"]), ["stub"])
        self.assertIn("bug_detection_rate", report["metric_catalog"]["extension_metrics"])
        self.assertFalse(any("remaining metrics use legacy adapters" in note for note in report["notes"]))

    def test_cli_parser_accepts_native_text_scope_choices(self) -> None:
        args = build_parser().parse_args(["eval.rs", "ref.rs", "--no-mutation", "--text-scope", "spec_only"])
        self.assertTrue(args.no_mutation)
        self.assertEqual(args.text_scope, "spec_only")

    def test_cli_json_and_registry_error_boundaries_are_stable(self) -> None:
        rendered = render_json({"b": 1, "a": "中文"})
        self.assertEqual(json.loads(rendered), {"a": "中文", "b": 1})
        self.assertIn("中文", rendered)
        self.assertLess(rendered.index('"a"'), rendered.index('"b"'))
        bulky_report = {
            "metrics": {
                "fast": {"status": "ok", "score": 1.0, "stdout": "x", "details": [{"a": 1}]},
                "broken": {"status": "error", "error": "boom", "error_type": "RuntimeError"},
            },
            "text_metric_inputs": {"generated": {"text": "large", "tokens": list(range(25))}},
        }
        compact_rendered = render_json(bulky_report, compact=True)
        self.assertEqual(json.loads(compact_rendered), json.loads(render_json(bulky_report)))
        self.assertEqual(json.loads(compact_rendered), bulky_report)
        self.assertIn("large", compact_rendered)
        self.assertNotIn("\n", compact_rendered)

        def metric_fast(generated_rs_path: str, ground_rs_path: str) -> dict:
            return {"status": "ok", "score": 1.0, "paths": [generated_rs_path, ground_rs_path]}

        def metric_broken(generated_rs_path: str, ground_rs_path: str) -> dict:
            raise RuntimeError("boom")

        fake_inputs = {
            "text_metric_scope": "spec_only",
            "token_source": "spec_items",
            "generated": {"text": "", "tokens": 0},
            "ground": {"text": "", "tokens": 0},
        }
        with patch.object(registry, "METRIC_FUNCTIONS", (metric_fast, registry.metric_mutation_kill_rate, metric_broken)), \
             patch.object(registry, "get_text_metric_inputs", return_value=fake_inputs):
            report = registry.compute_all_metrics("gen.rs", "ref.rs", text_scope="spec_only", include_mutation=False)

        self.assertEqual(list(report["metrics"]), ["fast", "broken"])
        self.assertNotIn("mutation_kill_rate", report["metric_catalog"]["enabled_metrics"])
        self.assertIn("Mutation metric skipped by --no-mutation.", report["notes"])
        self.assertEqual(report["metrics"]["fast"]["paths"], ["gen.rs", "ref.rs"])
        self.assertEqual(report["metrics"]["broken"]["status"], "error")
        self.assertEqual(report["metrics"]["broken"]["error_type"], "RuntimeError")
        self.assertEqual(report["metrics"]["broken"]["error"], "boom")

    def test_cli_main_validates_paths_and_can_fail_on_metric_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generated = _write_fixture(tmp, "eval.rs", "verus! { fn f() {} }")
            ground = _write_fixture(tmp, "ref.rs", "verus! { fn f() {} }")
            report = {
                "metrics": {
                    "fast": {"status": "ok", "stdout": "large"},
                    "broken": {"status": "error", "error": "boom"},
                }
            }
            with patch.object(cli_main, "compute_all_metrics", return_value=report):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = cli_main.main([generated, ground, "--no-mutation", "--compact"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), report)
        self.assertIn("large", stdout.getvalue())
        self.assertNotIn("\n  ", stdout.getvalue())

        with tempfile.TemporaryDirectory() as tmp:
            generated = _write_fixture(tmp, "eval.rs", "verus! { fn f() {} }")
            ground = _write_fixture(tmp, "ref.rs", "verus! { fn f() {} }")
            with patch.object(cli_main, "compute_all_metrics", return_value=report):
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = cli_main.main([generated, ground, "--fail-on-error"])
        self.assertEqual(exit_code, 1)

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            cli_main.main(["/missing/eval.rs", "/missing/ref.rs"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("generated_rs_path does not exist", stderr.getvalue())

    def test_set_verus_binary_updates_native_lemma_helper(self) -> None:
        with patch.object(rebuild_config.verus_runner, "set_verus_binary") as set_runner, \
             patch.object(rebuild_config.lemma_implication, "set_lemma_verus_binary") as set_lemma:
            rebuild_config.set_verus_binary("/tmp/verus")
        set_runner.assert_called_once_with("/tmp/verus")
        set_lemma.assert_called_once_with("/tmp/verus")

    def test_text_scope_helpers_normalize_and_restore(self) -> None:
        self.assertEqual(normalize_text_metric_scope(None), "full")
        self.assertEqual(normalize_text_metric_scope("SPEC"), "spec_only")
        self.assertEqual(normalize_text_metric_scope("specification-only"), "spec_only")
        self.assertEqual(normalize_text_metric_scope("whole-file"), "full")
        with self.assertRaises(ValueError):
            normalize_text_metric_scope("bad")

        original = get_text_metric_scope()
        with text_metric_scope("spec_only"):
            self.assertEqual(get_text_metric_scope(), "spec_only")
        self.assertEqual(get_text_metric_scope(), original)
        self.assertEqual(tokens("x <= y ==> z != 0 // drop"), ["x", "<=", "y", "==>", "z", "!=", "0"])

    def test_strip_comments_handles_lifetimes_and_raw_strings(self) -> None:
        source = """
verus! {
fn borrowed(x: &'static str) {
    let raw = r#"not // a comment and not " #"#;
    // requires false
    /* ensures false */
}
}
"""

        clean = strip_comments(source)
        clauses = extract_clauses_from_text(source)

        self.assertIn("&'static str", clean)
        self.assertIn('r#"not // a comment and not " #"#', clean)
        self.assertNotIn("requires false", clean)
        self.assertNotIn("ensures false", clean)
        self.assertEqual(clauses, [])

    def test_clause_extraction_ignores_delimiters_inside_string_literals(self) -> None:
        source = """
verus! {
fn f() -> (r: bool)
    ensures r == "{",
    ensures r == r#"a,b,{,}"#,
{
    true
}
}
"""

        clauses = extract_clauses_from_text(source)

        self.assertEqual([clause.kind for clause in clauses], ["ensures", "ensures"])
        self.assertEqual(clauses[0].text, 'r == "{"')
        self.assertEqual(clauses[1].text, 'r == r#"a,b,{,}"#')

    def test_strength_context_keeps_if_else_braces_inside_ensures_without_body(self) -> None:
        source = """
verus! {
fn f(a: int, b: int) -> (r: int)
    ensures r == if a >= b { a } else { b },
    ensures r >= a || r >= b
{
    if a >= b { a } else { b }
}
}
"""

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture(tmp, "case.rs", source)
            contexts = strength_contexts_for_path(path)

        self.assertEqual(len(contexts), 1)
        ensures = contexts[0]["ensures"]
        self.assertEqual(
            [clause["text"] for clause in ensures],
            [
                "r == if a >= b { a } else { b }",
                "r >= a || r >= b",
            ],
        )
        self.assertFalse(any("if a >= b { a } else { b }\n}" in clause["text"] for clause in ensures))

    def test_strength_context_ends_signature_after_turbofish_in_ensures(self) -> None:
        source = """
verus! {
fn find(arr: &Vec<i32>, target: i32) -> (index: Option<usize>)
    ensures
        index == None::<usize>
{
    None
}

fn compare(a: int, b: int) -> (r: bool)
    ensures r == (a > b)
{
    a > b
}
}
"""

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture(tmp, "case.rs", source)
            contexts = strength_contexts_for_path(path)

        self.assertEqual([context["function"] for context in contexts], ["find", "compare"])
        self.assertEqual([clause["text"] for clause in contexts[0]["ensures"]], ["index == None::<usize>"])
        self.assertEqual([clause["text"] for clause in contexts[1]["ensures"]], ["r == (a > b)"])

    def test_textual_similarity_empty_spec_scope_is_exact_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generated = _write_fixture(tmp, "eval.rs", "verus! { fn f() {} }")
            ground = _write_fixture(tmp, "ref.rs", "verus! { fn f() {} }")

            result = metric_verus_textual_similarity_spec_only(generated, ground)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["empty_match"])
        self.assertEqual(result["components"]["bleu"]["score"], 1.0)
        self.assertEqual(result["components"]["rouge_l"]["score"], 1.0)
        self.assertEqual(result["components"]["key_spec_match"]["score"], 1.0)

    def test_textual_similarity_one_sided_empty_spec_scope_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = _write_fixture(tmp, "empty.rs", "verus! { fn f() {} }")
            nonempty = _write_fixture(
                tmp, "nonempty.rs", "verus! { fn f() ensures true {} }"
            )

            for generated, ground in ((empty, nonempty), (nonempty, empty)):
                with self.subTest(generated=Path(generated).name):
                    result = metric_verus_textual_similarity_spec_only(
                        generated, ground
                    )
                    self.assertEqual(result["status"], "ok")
                    self.assertEqual(result["score"], 0.0)
                    self.assertEqual(
                        result["components"]["rouge_l"]["score"], 0.0
                    )
                    self.assertEqual(
                        result["components"]["key_spec_match"]["score"], 0.0
                    )

    def test_clause_implication_metric_scores_zero_when_gen_placeholder_cannot_imply_real_clause(self) -> None:
        # gen has no requires -> placeholder "true" antecedent; ref has a real clause.
        # "true" does not syntactically match/subsume "n >= 0", so this now falls through
        # to a real lemma_implication_check() call instead of a hardcoded short-circuit.
        generated_contexts = [{"function": "f", "requires": [], "ensures": [], "parameters": [], "returns": []}]
        reference_contexts = [
            {
                "function": "f",
                "requires": [{"kind": "requires", "text": "n >= 0", "normalized": "n >= 0"}],
                "ensures": [],
                "parameters": [{"name": "n", "type": "int"}],
                "returns": [],
            }
        ]

        with patch("metrics_rebuild.share.smt.lemma_implication_check") as lemma_check:
            lemma_check.return_value = {"holds": False, "status": "invalid"}
            result = clause_implication_metric(
                generated_contexts,
                reference_contexts,
                reference_rs_path="ref.rs",
                source_side="reference",
                clause_kind="requires",
                implication_direction="generated_implies_reference_clause",
                metric_kind="precondition_clause_completeness_rate",
                note="test",
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["passed"], 0)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["unknown"], 0)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["score"], 0.0)
        self.assertTrue(result["implicit_true_contract_clauses"])
        self.assertEqual(result["details"][0]["clause"]["text"], "n >= 0")
        self.assertNotIn("vacuous_true_only", result["functions"][0])
        lemma_check.assert_called_once()
        called_antecedent = lemma_check.call_args.kwargs["antecedent"]
        called_consequent = lemma_check.call_args.kwargs["consequent"]
        self.assertEqual([c["text"] for c in called_antecedent], ["true"])
        self.assertEqual([c["text"] for c in called_consequent], ["n >= 0"])

    def test_clause_implication_metric_scores_one_when_both_sides_empty(self) -> None:
        # Both sides have no requires -> both are the placeholder "true" clause, which
        # hits the syntactic "true_consequent" shortcut after signature binding.
        generated_contexts = [{"function": "f", "requires": [], "ensures": [], "parameters": [], "returns": []}]
        reference_contexts = [{"function": "f", "requires": [], "ensures": [], "parameters": [], "returns": []}]

        with patch(
            "metrics_rebuild.share.smt.lemma_implication_check",
            return_value={"holds": True, "status": "valid", "reason": "true_consequent"},
        ) as lemma_check:
            result = clause_implication_metric(
                generated_contexts,
                reference_contexts,
                reference_rs_path="ref.rs",
                source_side="reference",
                clause_kind="requires",
                implication_direction="generated_implies_reference_clause",
                metric_kind="precondition_clause_completeness_rate",
                note="test",
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["unknown"], 0)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["implicit_true_contract_clauses"])
        self.assertTrue(result["details"][0]["clause"].get("implicit"))
        self.assertEqual(result["details"][0]["clause"]["text"], "true")
        self.assertEqual(result["details"][0]["implication_check"]["reason"], "true_consequent")
        self.assertNotIn("vacuous_true_only", result["functions"][0])
        lemma_check.assert_called_once()

    def test_clause_implication_metric_can_disable_implicit_true_injection(self) -> None:
        generated_contexts = [{"function": "f", "requires": [], "ensures": [], "parameters": [], "returns": []}]
        reference_contexts = [{"function": "f", "requires": [], "ensures": [], "parameters": [], "returns": []}]

        result = clause_implication_metric(
            generated_contexts,
            reference_contexts,
            reference_rs_path="ref.rs",
            source_side="reference",
            clause_kind="requires",
            implication_direction="generated_implies_reference_clause",
            metric_kind="precondition_clause_completeness_rate",
            note="test",
            inject_implicit_true_contract_clauses=False,
        )

        self.assertEqual(result["status"], "not_available")
        self.assertEqual(result["total"], 0)
        self.assertIsNone(result["score"])
        self.assertFalse(result["implicit_true_contract_clauses"])

    def test_clause_implication_metric_real_gen_clauses_skip_vacuous_short_circuit(self) -> None:
        generated_contexts = [
            {
                "function": "f",
                "requires": [{"kind": "requires", "text": "n >= 0", "normalized": "n >= 0"}],
                "ensures": [],
                "parameters": [{"name": "n", "type": "int"}],
                "returns": [],
            }
        ]
        reference_contexts = [
            {
                "function": "f",
                "requires": [{"kind": "requires", "text": "n >= 0", "normalized": "n >= 0"}],
                "ensures": [],
                "parameters": [{"name": "n", "type": "int"}],
                "returns": [],
            }
        ]

        with patch("metrics_rebuild.share.smt.lemma_implication_check") as lemma_check:
            lemma_check.return_value = {"holds": True}
            result = clause_implication_metric(
                generated_contexts,
                reference_contexts,
                reference_rs_path="ref.rs",
                source_side="generated",
                clause_kind="requires",
                implication_direction="reference_implies_generated_clause",
                metric_kind="precondition_clause_reliability_rate",
                note="test",
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["score"], 1.0)
        self.assertNotIn("vacuous_true_only", result["functions"][0])
        self.assertEqual(result["functions"][0]["details"][0]["clause"]["text"], "n >= 0")

    def test_semantic_strength_implicit_true_keeps_empty_contracts_comparable(self) -> None:
        generated_contexts = [{"function": "f", "requires": [], "ensures": [], "parameters": [], "returns": []}]
        reference_contexts = [{"function": "f", "requires": [], "ensures": [], "parameters": [], "returns": []}]

        with patch(
            "metrics_rebuild.share.semantic_strength._lemma_check",
            return_value={"holds": True, "status": "valid", "reason": "true_consequent"},
        ) as lemma_check:
            result = semantic_strength_comparison(
                generated_contexts,
                reference_contexts,
                lemma_reference_path="ref.rs",
                generated_rs_path="gen.rs",
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["classification"], "equivalent")
        self.assertTrue(result["generated_refines_ground"])
        self.assertTrue(result["ground_refines_generated"])
        self.assertTrue(result["implicit_true_contract_clauses"])
        self.assertEqual(result["functions"][0]["requires"]["generated_total"], 1)
        self.assertEqual(result["functions"][0]["ensures"]["ground_total"], 1)
        self.assertEqual(lemma_check.call_count, 4)

    def test_semantic_strength_gen_empty_gt_real_uses_real_lemma_checks(self) -> None:
        # gen has no requires/ensures -> both become placeholder "true" clauses. Two of
        # the four directional checks are trivially true via the syntactic "true
        # consequent" shortcut; the other two ("true" implying a real clause) are not
        # syntactic tautologies, so they now go through a real lemma_implication_check()
        # call instead of being hardcoded to a fixed verdict.
        generated_contexts = [{
            "function": "f",
            "requires": [],
            "ensures": [],
            "parameters": [{"name": "n", "type": "int"}],
            "returns": [{"name": "out", "type": "int"}],
        }]
        reference_contexts = [
            {
                "function": "f",
                "requires": [{"kind": "requires", "text": "n >= 0", "normalized": "n >= 0"}],
                "ensures": [{"kind": "ensures", "text": "result == 2 * n", "normalized": "result == 2 * n"}],
                "parameters": [{"name": "n", "type": "int"}],
                "returns": [{"name": "result", "type": "int"}],
            }
        ]

        def fake_lemma_check(*, check_name: str, consequent: list[dict], **_kwargs) -> dict:
            # "true" cannot imply either real reference clause, so Verus would fail
            # to prove them; a real (mocked) lemma check reports holds=False.
            if all(clause["text"].strip() == "true" for clause in consequent):
                return {"holds": True, "status": "valid", "reason": "true_consequent"}
            self.assertIn(check_name, {"gen_pre_implies_ground_pre", "gen_post_implies_ground_post"})
            return {
                "holds": False,
                "status": "refuted",
                "evidence_kind": "counterexample",
            }

        with patch("metrics_rebuild.share.semantic_strength._lemma_check", side_effect=fake_lemma_check) as lemma_check:
            result = semantic_strength_comparison(
                generated_contexts,
                reference_contexts,
                lemma_reference_path="ref.rs",
                generated_rs_path="gen.rs",
            )

        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["generated_refines_ground"])
        self.assertFalse(result["ground_refines_generated"])
        self.assertEqual(result["classification"], "incomparable")
        self.assertNotIn("vacuous_true_only", result["functions"][0])
        self.assertEqual(result["functions"][0]["contract_relation"], "incomparable")
        self.assertEqual(lemma_check.call_count, 4)

    def test_text_similarity_keeps_proof_function_body(self) -> None:
        source = """
verus! {
proof fn lemma_body(x: int)
    requires x >= 0
    ensures x + 1 > x
{
    let ghost_witness = x + 1;
    assert(ghost_witness > x);
}
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture(tmp, "proof.rs", source)
            proof_items = [
                item for item in text_similarity_items(path)
                if item.kind == "proof_fn"
            ]

        self.assertEqual(len(proof_items), 1)
        self.assertIn("ghost_witness", proof_items[0].text)
        self.assertIn("assert(ghost_witness > x)", proof_items[0].normalized)

    def test_spec_tokens_ignore_operator_spacing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spaced = _write_fixture(
                tmp,
                "spaced.rs",
                "verus! { fn f(x: int) -> (y: int) ensures y == x + 1 { x + 1 } }",
            )
            compact = _write_fixture(
                tmp,
                "compact.rs",
                "verus! { fn f(x: int) -> (y: int) ensures y==x+1 { x + 1 } }",
            )

            self.assertEqual(spec_tokens(spaced), spec_tokens(compact))

    def test_text_similarity_keeps_assert_by_block(self) -> None:
        source = """
verus! {
fn f(x: int) -> (y: int)
    ensures y >= x
{
    assert(y >= x) by {
        let assert_by_unique_token = y - x;
        assert(assert_by_unique_token >= 0);
    }
    y
}
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture(tmp, "assert_by.rs", source)
            items = text_similarity_items(path)
            assert_items = [item for item in items if item.kind == "assert"]
            assert_by_items = [item for item in items if item.kind == "assert_by"]

            self.assertEqual([item.text for item in assert_items], ["y >= x"])
            self.assertEqual(len(assert_by_items), 1)
            self.assertTrue(assert_by_items[0].text.startswith("by {"))
            self.assertNotIn("assert(y >= x)", assert_by_items[0].text)
            self.assertIn("assert_by_unique_token", assert_by_items[0].text)
            self.assertIn("assert_by_unique_token", spec_tokens(path))

    def test_text_similarity_dedupes_assert_by_inside_proof_function(self) -> None:
        source = """
verus! {
proof fn lemma(x: int)
    requires x >= 0
    ensures x + 1 > x
{
    assert(x + 1 > x) by {
        let proof_assert_by_unique_token = x + 1;
        assert(proof_assert_by_unique_token > x);
    }
}
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture(tmp, "proof_assert_by.rs", source)
            items = text_similarity_items(path)
            assert_by_items = [item for item in items if item.kind == "assert_by"]
            proof_items = [item for item in items if item.kind == "proof_fn"]

            self.assertEqual(assert_by_items, [])
            self.assertFalse(any(item.kind == "requires" for item in items))
            self.assertFalse(any(item.kind == "ensures" for item in items))
            self.assertFalse(any(item.kind == "assert" for item in items))
            self.assertEqual(len(proof_items), 1)
            self.assertIn("proof_assert_by_unique_token", proof_items[0].text)
            self.assertIn("proof_assert_by_unique_token", spec_tokens(path))

    def test_llm_json_extraction_and_config_metadata_are_stable(self) -> None:
        fenced = "Here is the result:\n```json\n{\"score\": 0.75, \"items\": [1, 2]}\n```"
        embedded = 'prefix {"reasoning": "brace } in string", "score": 1.0} suffix'
        truncated = '{"reasoning": "mentions sum[0] before truncation"'
        self.assertEqual(llm_client.extract_json_from_llm_text(fenced)["score"], 0.75)
        self.assertEqual(llm_client.extract_json_from_llm_text(embedded)["reasoning"], "brace } in string")
        self.assertIsNone(llm_client.extract_json_from_llm_text(truncated))

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                """
llm:
  api_key: "secret-key"
  base_url: "https://example.test/v1/"
  model_name: "fixture-model"
""",
                encoding="utf-8",
            )
            with patch.object(llm_client, "CONFIG_PATH", config_path):
                old_cache = llm_client._LLM_CONFIG_CACHE
                llm_client._LLM_CONFIG_CACHE = None
                try:
                    config = llm_client.load_llm_config()
                finally:
                    llm_client._LLM_CONFIG_CACHE = old_cache

        metadata = llm_client.public_llm_metadata(config)
        self.assertEqual(config["status"], "ok")
        self.assertEqual(config["base_url"], "https://example.test/v1")
        self.assertNotIn("api_key", metadata)
        self.assertTrue(metadata["api_key_configured"])

        with patch.object(llm_client, "CONFIG_PATH", config_path), patch.dict(
            "os.environ",
            {
                "VERUSEVAL_LLM_API_KEY": "override-secret",
                "VERUSEVAL_LLM_BASE_URL": "https://override.test/v1/",
                "VERUSEVAL_LLM_MODEL_NAME": "override-model",
                "VERUSEVAL_LLM_ENABLE_THINKING": "false",
            },
            clear=False,
        ):
            old_cache = llm_client._LLM_CONFIG_CACHE
            llm_client._LLM_CONFIG_CACHE = None
            try:
                overridden = llm_client.load_llm_config()
            finally:
                llm_client._LLM_CONFIG_CACHE = old_cache
        self.assertEqual(overridden["api_key"], "override-secret")
        self.assertEqual(overridden["base_url"], "https://override.test/v1")
        self.assertEqual(overridden["model_name"], "override-model")
        self.assertIs(overridden["enable_thinking"], False)

    def test_contract_eval_coerces_payloads_and_checks_clauses(self) -> None:
        context = {
            "parameters": [
                {"name": "x", "type": "int"},
                {"name": "flag", "type": "bool"},
                {"name": "xs", "type": "Vec<int>"},
            ],
            "returns": [{"name": "y", "type": "int"}],
            "requires": [
                {"normalized": "x >= 0"},
                {"normalized": "flag ==> xs.len() >= 1"},
            ],
            "ensures": [{"normalized": "y >= x"}],
        }

        self.assertEqual(coerce_value_for_type("7", "int"), (True, 7))
        self.assertEqual(coerce_value_for_type("true", "bool"), (True, True))
        self.assertEqual(typed_input_payload(context, {"x": "3", "flag": True, "xs": [1, 2]}), {"x": 3, "flag": True, "xs": [1, 2]})
        self.assertEqual(typed_output_payload(context, {"y": "4"}), {"y": 4})
        self.assertTrue(eval_contract_expr("forall|i:int| 0 <= i && i < x ==> i < 3", {"x": 3}))
        self.assertEqual(contract_evaluation(context, {"x": 3, "flag": True, "xs": [1]}, {"y": 4}, strict=True)["accepted"], True)
        self.assertEqual(contract_evaluation(context, {"x": -1, "flag": False, "xs": []}, {"y": 0}, strict=True)["reason"], "requires_not_satisfied")

        mutable_context = {
            "parameters": [{"name": "sum", "type": "&mut Vec<i32>"}],
            "returns": [{"name": "sum", "type": "Vec<i32>"}],
            "_mutable_post_state_names": ["sum"],
            "requires": [{"normalized": "old(sum).len() == 1"}],
            "ensures": [{"normalized": "sum[0] == old(sum)[0] + 1"}],
        }
        self.assertTrue(contract_evaluation(mutable_context, {"sum": [2]}, {"sum": [3]})["accepted"])
        self.assertEqual(
            contract_evaluation(mutable_context, {"sum": [2]}, {"sum": [4]})["reason"],
            "ensures_not_satisfied",
        )
        multiplied_context = {
            "requires": [],
            "ensures": [{"normalized": "sum[0] == 2 * N"}],
        }
        self.assertEqual(
            contract_evaluation(multiplied_context, {"sum": [2], "N": 1}, {"sum": [999]})["reason"],
            "ensures_not_satisfied",
        )

    def test_coerce_and_literal_edge_cases(self) -> None:
        # F1: vectors over the cap are dropped, not silently truncated
        self.assertEqual(coerce_value_for_type(list(range(65)), "Vec<u32>"), (False, None))
        self.assertEqual(coerce_value_for_type(list(range(19)), "Vec<u32>"), (True, list(range(19))))
        self.assertEqual(coerce_value_for_type([list(range(13))], "Vec<Vec<u32>>"), (False, None))
        self.assertEqual(coerce_value_for_type([[1, 2] for _ in range(9)], "Vec<Vec<u32>>"), (False, None))
        self.assertEqual(
            coerce_value_for_type([[[1.0, 2.0]]], "Vec<Vec<Vec<f32>>>"),
            (True, [[[1.0, 2.0]]]),
        )
        self.assertEqual(coerce_value_for_type([1, 2, 3], "Vec<u32>"), (True, [1, 2, 3]))
        # F2: Option<T> wraps Some(...) / None
        self.assertEqual(verus_literal(5, "Option<u32>"), "Some(5u32)")
        self.assertEqual(verus_literal(None, "Option<u32>"), "None")
        self.assertEqual(verus_literal(True, "Option<bool>"), "Some(true)")
        self.assertEqual(verus_literal(65, "Option<char>"), "Some('A')")
        # F3: integer bit-width upper/lower bounds are enforced
        self.assertEqual(coerce_value_for_type(100000, "u16"), (False, None))
        self.assertEqual(coerce_value_for_type(5, "u16"), (True, 5))
        self.assertEqual(coerce_value_for_type(-1, "u8"), (False, None))
        self.assertEqual(coerce_value_for_type(-32769, "i16"), (False, None))
        self.assertEqual(coerce_value_for_type(2147483648, "i32"), (False, None))

    def test_io_heuristic_inputs_and_invalid_generation_are_deterministic_without_llm(self) -> None:
        context = {
            "function": "find_value",
            "parameters": [
                {"name": "xs", "type": "Vec<int>"},
                {"name": "needle", "type": "int"},
            ],
            "returns": [{"name": "idx", "type": "int"}],
            "requires": [{"normalized": "needle > 0"}],
            "ensures": [{"normalized": "idx >= -1"}],
        }

        candidates = io_cases.generate_candidate_inputs(context, budget=4)
        self.assertEqual(len(candidates), 4)
        self.assertIn({"xs": [5], "needle": 5}, candidates)
        self.assertIn("duplicate", io_cases.tag_input_values({"xs": [2, 4, 4], "needle": 4}))
        self.assertIn("not_found", io_cases.expected_path_tags(context))

        with patch.object(io_cases, "llm_invalid_candidate_inputs", return_value=([], {"status": "not_available"})):
            invalid_cases, metadata = io_cases.generate_invalid_cases_from_reference_requires(context, budget=2)
        self.assertEqual(metadata["status"], "ok")
        self.assertTrue(invalid_cases)
        self.assertTrue(all(case["kind"] == "invalid" for case in invalid_cases))
        self.assertTrue(all(case["reference_requires_evaluation"]["accepted"] is False for case in invalid_cases))

    def test_mutation_generators_skip_contracts_and_sample_by_family(self) -> None:
        source = """
verus! {
fn calc(x: int, y: int) -> (z: int)
    requires x >= 0,
    ensures z >= x,
{
    let sum = x + y;
    if sum > 0 && y != 0 {
        sum - 1
    } else {
        0
    }
}
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture(tmp, "case.rs", source)
            mutants = generate_all_implementation_mutants(path)

        self.assertTrue(mutants)
        self.assertTrue(any(mutant["family"] == "AOR" for mutant in mutants))
        self.assertTrue(any(mutant["family"] == "ROR" for mutant in mutants))
        self.assertTrue(all("requires" not in mutant.get("source_code_line", "") for mutant in mutants))
        self.assertTrue(all("ensures" not in mutant.get("source_code_line", "") for mutant in mutants))

        sample = sample_mutants_by_family(mutants, 4)
        self.assertLessEqual(len(sample), 4)
        self.assertEqual(len({mutant["id"] for mutant in sample}), len(sample))

        summary = simple_mutation_kill_rate(
            original_success=True,
            original_verification={"success": True},
            mutants_total=5,
            killed_mutants=2,
            survived_mutants=1,
            unknown_mutants=1,
            invalid_mutants=1,
        )
        self.assertEqual(summary["mutation_kill_rate"], 1 / 2)
        self.assertEqual(summary["score"], 1 / 2)
        self.assertEqual(summary["valid_mutants"], 3)
        self.assertEqual(summary["scored_mutants"], 4)
        self.assertNotIn("mutation_verification_pass_rate", summary)
        self.assertNotIn("passed_mutants", summary)
        self.assertNotIn("failed_mutants", summary)

        survived = VerusRun("ok", True, 1, 0, 0, 0.0, "", "", ("verus",))
        failed = VerusRun("ok", False, 0, 1, 1, 0.0, "", "verification failed", ("verus",))
        frontend_failed = VerusRun("ok", False, 0, 1, 1, 0.0, "", "expected an expression", ("verus", "--no-verify"))
        frontend_passed = VerusRun("ok", True, 0, 0, 0, 0.0, "", "", ("verus", "--no-verify"))
        timeout = VerusRun("timeout", None, None, None, None, 0.0, "", "", ("verus",))
        syntax_error = VerusRun("parse_error", None, None, None, 1, 0.0, "", "expected an expression", ("verus",))
        self.assertEqual(mutation_outcome_from_runs(survived), "survived")
        self.assertEqual(mutation_outcome_from_runs(failed, frontend_failed), "invalid_mutant")
        self.assertEqual(mutation_outcome_from_runs(syntax_error, syntax_error), "invalid_mutant")
        self.assertEqual(mutation_outcome_from_runs(failed, frontend_passed), "killed")
        self.assertEqual(mutation_outcome_from_runs(timeout), "unknown")

    def test_mutable_vec_old_values_use_views_after_type_normalization(self) -> None:
        self.assertEqual(
            _fixed_sequence_clauses("__old_x", "&mut Vec<i32>", [1, 2]),
            ["__old_x@.len() == 2", "__old_x@[0] == 1i32", "__old_x@[1] == 2i32"],
        )
        nested = _fixed_sequence_clauses("__old_x", "&'a mut Vec<Vec<i32>>", [[1], [2, 3]])
        self.assertEqual(nested[0], "__old_x@.len() == 2")
        self.assertIn("__old_x@[0]@.len() == 1", nested)
        self.assertIn("__old_x@[1]@[1] == 3i32", nested)

        lines = _contract_check_proof_fn_lines(
            {
                "function": "f",
                "parameters": [{"name": "x", "type": "&'a mut Vec<Vec<i32>>"}],
                "returns": [],
                "requires": [],
                "ensures": [],
                "_mutable_post_state_names": ["x"],
            },
            {"x": [[1], [2, 3]]},
            {},
            "check",
        )
        proof = "\n".join(lines)
        self.assertIn("proof fn check(__old_x: &Vec<Vec<i32>>)", proof)
        self.assertIn("__old_x@.len() == 2", proof)
        self.assertNotIn("__old_x == seq!", proof)

        post_state_lines = _contract_check_proof_fn_lines(
            {
                "function": "f",
                "parameters": [{"name": "x", "type": "&mut Vec<i32>"}],
                "returns": [{"name": "x", "type": "Vec<i32>"}],
                "requires": [],
                "ensures": [
                    {"normalized": "x == old(x)"},
                    {"normalized": "x == old(x)@"},
                ],
                "_mutable_post_state_names": ["x"],
            },
            {"x": [1, 2]},
            {"x": [1, 2]},
            "check",
        )
        self.assertEqual(post_state_lines.count("    assert(x == __old_x@);"), 2)
        self.assertNotIn("__old_x@@", "\n".join(post_state_lines))

    def test_mutation_path_stages_frontend_and_uses_only_valid_denominator(self) -> None:
        verified = VerusRun("ok", True, 1, 0, 0, 0.0, "", "", ("verus",))
        proof_failed = VerusRun("ok", False, 0, 1, 1, 0.0, "", "assertion failed", ("verus",), True)
        syntax_failed = VerusRun("parse_error", None, None, None, 1, 0.0, "", "expected an expression", ("verus",))
        mutants = [
            {
                "id": f"m{index}",
                "text": f"verus! {{ fn f() {{ {index} }} }}",
                "family": "AOR",
                "operator": "AOR",
                "line": 1,
                "column": 1,
                "description": "fixture",
                "original": "+",
                "replacement": "-",
            }
            for index in range(4)
        ]
        runs = [
            verified,
            syntax_failed, syntax_failed,
            proof_failed, verified,
            proof_failed, verified,
            verified,
        ]
        with patch("metrics_rebuild.share.mutation.generate_simple_mutants", return_value=mutants), \
                patch("metrics_rebuild.share.mutation.generate_all_implementation_mutants", return_value=mutants), \
                patch("metrics_rebuild.share.mutation.run_verus", side_effect=runs) as run:
            result = mutation_kill_rate_for_path("fixture.rs")

        self.assertEqual(result["invalid_mutants"], 1)
        self.assertEqual(result["killed_mutants"], 2)
        self.assertEqual(result["survived_mutants"], 1)
        self.assertEqual(result["valid_mutants"], 3)
        self.assertEqual(result["score"], 2 / 3)
        self.assertEqual(
            [call.kwargs["no_verify"] for call in run.call_args_list],
            [False, False, True, False, True, False, True, False],
        )

    def test_mutation_generators_use_body_when_ensures_contains_if_braces(self) -> None:
        source = """
verus! {
fn choose(a: int, b: int) -> (r: int)
    ensures r == if a >= b { a } else { b },
    ensures r >= a || r >= b
{
    if a >= b {
        a + 1
    } else {
        b - 1
    }
}
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture(tmp, "case.rs", source)
            mutants = generate_all_implementation_mutants(path)

        self.assertTrue(mutants)
        source_lines = [mutant.get("source_code_line", "") for mutant in mutants]
        self.assertFalse(any("ensures" in line for line in source_lines))
        self.assertTrue(any("a + 1" in line or "b - 1" in line for line in source_lines))
        self.assertTrue(any(mutant["family"] == "AOR" for mutant in mutants))
        self.assertTrue(any(mutant["family"] in {"ROR", "UOI_UOD"} for mutant in mutants))

    def test_mutation_generators_skip_generic_angle_brackets_but_keep_comparisons(self) -> None:
        source = """
verus! {
fn solve(n: int) -> (r: int)
{
    let mut v: Vec<char> = Vec::new();
    if n < 1000 {
        r = n + 1;
    } else {
        r = n - 1;
    }
    r
}
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture(tmp, "case.rs", source)
            mutants = generate_all_implementation_mutants(path)

        ror_lines = [mutant.get("source_code_line", "") for mutant in mutants if mutant["family"] == "ROR"]
        self.assertTrue(any("if n < 1000" in line for line in ror_lines))
        self.assertFalse(any("Vec<char>" in line for line in ror_lines))

    def test_mutation_generators_skip_nested_generic_angle_brackets(self) -> None:
        source = """
verus! {
fn solve(n: int) -> (r: int)
{
    let maybe: Option<Vec<int>> = Option::None;
    if n > 0 {
        r = n + 1;
    } else {
        r = 0;
    }
    r
}
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture(tmp, "case.rs", source)
            mutants = generate_all_implementation_mutants(path)

        ror_lines = [mutant.get("source_code_line", "") for mutant in mutants if mutant["family"] == "ROR"]
        self.assertTrue(any("if n > 0" in line for line in ror_lines))
        self.assertFalse(any("Option<Vec<int>>" in line for line in ror_lines))

    def test_mutation_boundary_and_svr_keep_vec_bool_distinct_from_bool(self) -> None:
        """Regression: IO had is_bool_type('Vec<bool>')==True; mutation must not conflate them."""
        from metrics_rebuild.share.contract_eval import is_bool_type, normalize_type_key
        from metrics_rebuild.share.mutation import _function_param_type_groups

        self.assertFalse(is_bool_type("Vec<bool>"))
        self.assertNotEqual(normalize_type_key("Vec<bool>"), normalize_type_key("bool"))
        self.assertEqual(normalize_type_key("& Vec< bool >"), normalize_type_key("Vec<bool>"))

        source = """
verus! {
fn logical_or(x1: Vec<bool>, x2: Vec<bool>, flag: bool) -> (result: Vec<bool>)
    requires x1.len() == x2.len(),
{
    let mut r: Vec<bool> = Vec::new();
    let mut i: usize = 0;
    let keep = true;
    while i < x1.len()
        invariant i <= x1.len(),
        decreases x1.len() - i,
    {
        if keep {
            r.push(x1[i] || x2[i]);
        } else {
            r.push(false);
        }
        i = i + 1;
    }
    r
}
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture(tmp, "case.rs", source)
            mutants = generate_all_implementation_mutants(path)
            groups = _function_param_type_groups(path)

        self.assertIn("Vec<bool>", groups)
        self.assertEqual(set(groups["Vec<bool>"]), {"x1", "x2"})
        self.assertNotIn("bool", groups)  # only one bool param → no SVR group

        # BOUNDARY may flip true/false literals, but must not rewrite type annotations.
        boundary_lines = [
            mutant.get("source_code_line") or ""
            for mutant in mutants
            if mutant["family"] == "BOUNDARY"
        ]
        self.assertFalse(any("Vec<true>" in line or "Vec<false>" in line for line in boundary_lines))
        self.assertTrue(
            any(mutant["original"] in {"true", "false"} for mutant in mutants if mutant["family"] == "BOUNDARY")
        )

        # ROR must not turn Vec<bool> angle brackets into comparisons.
        for mutant in mutants:
            if mutant["family"] != "ROR":
                continue
            line = mutant.get("source_code_line") or ""
            self.assertNotIn("Vec<=", line)
            self.assertNotIn("Vec>=", line)

    def test_generated_self_spec_robustness_wraps_flat_mutation_result(self) -> None:
        flat_result = {
            "status": "ok",
            "score": 0.75,
            "mutation_kill_rate": 0.75,
            "mutants_total": 4,
            "killed_mutants": 3,
            "survived_mutants": 1,
            "candidate_mutants_total": 8,
        }
        with patch(
            "metrics_rebuild.share.mutation.mutation_kill_rate_for_path",
            return_value=dict(flat_result),
        ) as mutation_for_path:
            result = generated_self_spec_robustness("gen.rs", "non_verifying_ref.rs")

        mutation_for_path.assert_called_once_with("gen.rs")
        self.assertEqual(set(result), {"generated", "ground", "delta"})
        self.assertIsNone(result["delta"])
        generated = result["generated"]
        self.assertEqual(generated["status"], "ok")
        self.assertEqual(generated["score"], generated["mutation_kill_rate"])
        self.assertEqual(generated["method"], "generated_self_spec_robustness_mutation")
        self.assertEqual(generated["reference_filter"], {"status": "not_used"})
        self.assertIn("does not use a GT/reference oracle filter", generated["note"])
        self.assertEqual(
            result["ground"],
            {
                "status": "not_applicable",
                "score": None,
                "reason": "reference_not_used",
                "path": "non_verifying_ref.rs",
            },
        )
        self.assertNotIn("no_shared_implementation_mutants", str(result))
        self.assertNotIn("shared_candidate_mutants_total", generated)
        self.assertNotIn("reference_killed_mutants", generated)
        self.assertNotIn("reference_filtered_mutants", generated)

    def test_generated_self_spec_robustness_preserves_skipped_and_no_mutants(self) -> None:
        cases = [
            {
                "status": "skipped",
                "score": None,
                "reason": "original_file_does_not_verify",
                "mutants_total": 0,
            },
            {
                "status": "no_mutants",
                "score": None,
                "mutants_total": 0,
            },
        ]
        for flat_result in cases:
            with self.subTest(status=flat_result["status"]):
                with patch(
                    "metrics_rebuild.share.mutation.mutation_kill_rate_for_path",
                    return_value=dict(flat_result),
                ):
                    result = generated_self_spec_robustness("gen.rs", "bad_ref.rs")

                self.assertEqual(result["generated"]["status"], flat_result["status"])
                self.assertEqual(result["generated"]["mutants_total"], 0)
                self.assertEqual(result["ground"]["status"], "not_applicable")
                self.assertNotEqual(
                    result["generated"].get("reason"),
                    "ground_file_does_not_verify_for_mutation_oracle",
                )

    def test_mutation_metric_label_uses_kill_rate_name(self) -> None:
        labels = {entry.metric_id: entry.display_name for entry in METRIC_CATALOG}
        self.assertEqual(labels["mutation_kill_rate"], "变异击杀率")

    def test_redundancy_removal_spans_handle_multi_clause_and_assert_by(self) -> None:
        source = """
verus! {
fn check(x: int)
{
    while x > 0
        invariant x >= 0,
                  x <= 10
        decreases x
    {
        assert(x >= 0) by {
            assert(x == x);
        };
    }
}
}
"""
        clean, candidates = extract_removal_candidates_from_text(source)
        invariant_candidates = [candidate for candidate in candidates if candidate.kind == "invariant"]
        assert_candidates = [candidate for candidate in candidates if candidate.kind == "assert"]

        self.assertGreaterEqual(len(invariant_candidates), 2)
        removed_first_invariant = remove_candidate_text(clean, invariant_candidates[0])
        self.assertNotIn("x >= 0,", removed_first_invariant)
        self.assertIn("x <= 10", removed_first_invariant)

        assert_candidate = next(candidate for candidate in assert_candidates if candidate.normalized == "x >= 0")
        removed_assert = remove_candidate_text(clean, assert_candidate)
        self.assertNotIn("by {\n            assert(x == x);", removed_assert)


if __name__ == "__main__":
    unittest.main()
