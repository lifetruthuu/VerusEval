from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from metrics_rebuild.share.functions import (
    function_generic_parameters_from_header,
    function_parameters_from_header,
    function_return_parameters_from_header,
    function_where_clause_from_header,
    spec_fn_blocks_for_path,
    strength_contexts_for_path,
)
from metrics_rebuild.share import lemma_implication
from metrics_rebuild.share.clauses import split_top_level_commas
from metrics_rebuild.share.lemma_harness import (
    apply_generated_support_renames as _apply_generated_support_renames,
    append_lemma_to_reference as _append_lemma_to_reference,
    collect_generated_support as _collect_generated_support,
)
from metrics_rebuild.share.lemma_binding import (
    bind_function_contexts,
    render_clause_expression,
)
from metrics_rebuild.share.lemma_implication import (
    _clause_syntax_issue,
    _classify_verus_run,
    _cache_key,
    _delimiter_issue,
    _make_lemma,
    _mutable_snapshot_context,
    _verus_bin,
    lemma_implication_check,
)
from metrics_rebuild.share.proof_probe import (
    probe_postcondition_truth_for_path,
    probe_precondition_falsity_for_path,
)
from metrics_rebuild.share.semantic_strength import semantic_strength_comparison
from metrics_rebuild.share.smt import (
    clause_implication_details_for_context,
)


def _write_verus_file(tmpdir: Path, name: str, body: str) -> str:
    path = tmpdir / name
    path.write_text(
        textwrap.dedent(
            f"""\
            use vstd::prelude::*;

            verus! {{
            {body.rstrip()}
            }}

            fn main() {{}}
            """
        ),
        encoding="utf-8",
    )
    return str(path)


def _verus_available() -> bool:
    verus_bin = _verus_bin()
    return shutil.which(verus_bin) is not None or Path(verus_bin).is_file()


def _strength_context(path: str | Path, function_name: str) -> dict:
    return next(
        item
        for item in strength_contexts_for_path(str(path))
        if item["function"] == function_name
    )


class TestLemmaHarnessContext(unittest.TestCase):
    def test_signature_parser_preserves_mut_and_generic_context(self) -> None:
        header = """
        fn update<T: Copy>(mut data: &mut Vec<T>) -> (out: &mut Vec<T>)
            where T: Eq,
            requires old(data).len() > 0,
        """
        self.assertEqual(
            function_parameters_from_header(header),
            [{"name": "data", "type": "&mut Vec<T>"}],
        )
        self.assertEqual(
            function_return_parameters_from_header(header),
            [{"name": "out", "type": "&mut Vec<T>"}],
        )
        self.assertEqual(function_generic_parameters_from_header(header), "<T: Copy>")
        self.assertEqual(function_where_clause_from_header(header), "where T: Eq,")
        self.assertEqual(
            function_return_parameters_from_header(
                "fn call<F>(f: F) where F: Fn() -> int, requires true,"
            ),
            [],
        )

        lemma = _make_lemma(
            "check_update",
            [{"name": "data", "type": "&mut Vec<T>"}],
            ["old(data).len() > 0"],
            ["old(data).len() >= 0"],
            generic_parameters="<T: Copy>",
            where_clause="where T: Eq,",
        )
        self.assertIn("proof fn check_update<T: Copy>(data: &mut Vec<T>)", lemma)
        self.assertIn("where T: Eq,", lemma)

    def test_signature_parser_keeps_commas_inside_nested_generic_types(self) -> None:
        header = """
        fn optimize<T>(e: Exp, s: Map<String, Seq<(T, int)>>)
            -> (out: Result<Map<String, int>, Error>)
            ensures true,
        """
        self.assertEqual(
            function_parameters_from_header(header),
            [
                {"name": "e", "type": "Exp"},
                {"name": "s", "type": "Map<String, Seq<(T, int)>>"},
            ],
        )
        self.assertEqual(
            function_return_parameters_from_header(header),
            [{"name": "out", "type": "Result<Map<String, int>, Error>"}],
        )

    def test_signature_parser_preserves_unnamed_semantic_return_slots(self) -> None:
        self.assertEqual(
            function_return_parameters_from_header("fn value() -> Vec<int> ensures true"),
            [{"name": None, "type": "Vec<int>"}],
        )
        self.assertEqual(
            function_return_parameters_from_header("fn pair() -> (usize, usize,)"),
            [{"name": None, "type": "(usize, usize,)"}],
        )
        self.assertEqual(
            function_return_parameters_from_header("fn singleton() -> (T,)"),
            [{"name": None, "type": "(T,)"}],
        )
        self.assertEqual(function_return_parameters_from_header("fn unit() -> ()"), [])

    def test_signature_parser_normalizes_generated_contract_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "contract_markers.rs",
                """
                fn commented(x: int) -> (result: bool)
                    /* requires */ x >= 0,
                    /* ensures */ result,
                { true }

                fn at_marker(x: int) -> (result: bool)
                    @requires x >= 0,
                    @ensures result,
                { true }

                fn unicode_marker(x: int) -> (result: int)
                    『
                        requires x >= 0,
                        ensures result >= 0,
                    』
                { x }
                """,
            )
            contexts = {
                item["function"]: item
                for item in strength_contexts_for_path(path)
            }

        self.assertEqual(contexts["commented"]["returns"], [{"name": "result", "type": "bool"}])
        self.assertEqual(contexts["at_marker"]["returns"], [{"name": "result", "type": "bool"}])
        self.assertEqual(contexts["unicode_marker"]["returns"], [{"name": "result", "type": "int"}])
        for name in ("commented", "at_marker", "unicode_marker"):
            self.assertEqual(len(contexts[name]["requires"]), 1)
            self.assertEqual(len(contexts[name]["ensures"]), 1)

    def test_real_generated_marker_samples_keep_runtime_return_types(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        cases = [
            (
                project_root
                / "data/generated/alphaverus_zero_shot_unverified/alphaverus/"
                "alphaverus_MBPP-verified_task_2_llama_zero-shot_unverified.rs",
                "contains",
                {"name": "result", "type": "bool"},
            ),
            (
                project_root
                / "data/generated/alphaverus_zero_shot_unverified/alphaverus/"
                "alphaverus_HumanEval-Verus_task_9_llama_zero-shot_unverified.rs",
                "divide_i32_by_u32",
                {"name": "qr", "type": "(i32, u32)"},
            ),
        ]
        for path, function_name, expected_return in cases:
            if not path.is_file():
                self.skipTest(f"generated marker sample is unavailable: {path}")
            context = _strength_context(path, function_name)
            with self.subTest(function=function_name):
                self.assertEqual(context["returns"], [expected_return])
                self.assertTrue(context["requires"])
                self.assertTrue(context["ensures"])

    def test_golden_harness_order_names_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / "ref.rs"
            reference.write_text(
                "use vstd::prelude::*;\n\n"
                "verus! {\nfn target(x: int) { }\n}\n\nfn main() {}\n",
                encoding="utf-8",
            )
            lemma = _make_lemma(
                "lemma_golden",
                [{"name": "x", "type": "int"}],
                ["x > 0"],
                ["x >= 0"],
            )
            harness = _append_lemma_to_reference(
                reference,
                lemma,
                ["spec fn helper() -> bool { true }"],
                ["use vstd::seq::*;", "use vstd::map::*;", "use vstd::seq::*;"],
            )
        self.assertLess(harness.index("use vstd::map::*;"), harness.index("use vstd::seq::*;"))
        self.assertEqual(harness.count("use vstd::seq::*;"), 1)
        self.assertIn("proof fn lemma_golden(x: int)", harness)
        self.assertEqual(
            hashlib.sha256(harness.encode("utf-8")).hexdigest(),
            "5dd7601b66c641f695724e2bd7dd1efc5853d33a4158c2c3debe5b29f17ee0f8",
        )

    def test_pipe_scanner_distinguishes_binders_from_or_expressions(self) -> None:
        text = """
            match kind { A | B => true, _ => false },
            (a | b) == expected,
            forall|i: int, j: int| p(i, j),
            values.map(|x: int, y: int| x + y)
        """
        parts = split_top_level_commas(text)
        self.assertEqual(len(parts), 4)
        self.assertIn("A | B", parts[0])
        self.assertEqual(parts[1], "(a | b) == expected")
        self.assertIn("i: int, j: int", parts[2])
        self.assertIn("x: int, y: int", parts[3])

    def test_lemma_removes_source_trailing_separator_and_rejects_dangling_expr(self) -> None:
        lemma = _make_lemma(
            "check_match",
            [],
            ["match kind { A | B => true, _ => false },"],
            ["true"],
        )
        self.assertIn("match kind { A | B => true, _ => false },\n", lemma)
        self.assertNotIn("},,", lemma)
        self.assertEqual(_clause_syntax_issue("x ==>"), "dangling_==>")
        self.assertIsNone(_clause_syntax_issue("None::<Option<int>>"))
        self.assertIsNone(_clause_syntax_issue("Option<int>"))
        self.assertEqual(_clause_syntax_issue("x >"), "dangling_>")
        self.assertEqual(_clause_syntax_issue("result =="), "dangling_==")
        self.assertEqual(
            _clause_syntax_issue("forall|i: int, j: int p(i, j)"),
            "unclosed_pipe_binder",
        )

    def test_multiline_implication_block_is_kept_in_clause(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "multiline_block.rs",
                """
                fn clip(value: int, min_value: int, max_value: int) -> (out: int)
                    ensures
                        min_value <= max_value ==>
                        {
                            if value < min_value { out == min_value }
                            else if value > max_value { out == max_value }
                            else { out == value }
                        }
                {
                    value
                }
                """,
            )
            context = next(
                item for item in strength_contexts_for_path(path)
                if item["function"] == "clip"
            )
        self.assertEqual(len(context["ensures"]), 1)
        clause = context["ensures"][0]["text"]
        self.assertIn("min_value <= max_value ==>\n", clause)
        self.assertIn("if value < min_value", clause)
        self.assertTrue(clause.rstrip().endswith("}"))

    def test_equality_block_rhs_and_invariant_method_are_not_clause_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "block_rhs.rs",
                """
                struct Holder { value: int }
                impl Holder {
                    spec fn invariant(&self) -> bool { self.value >= 0 }
                }
                fn build(value: int) -> (result: Holder)
                    ensures
                        result.value == {
                            let copied = value;
                            copied
                        },
                        result.invariant(),
                {
                    Holder { value }
                }
                """,
            )
            context = next(
                item for item in strength_contexts_for_path(path)
                if item["function"] == "build"
            )
        self.assertEqual(len(context["ensures"]), 2)
        self.assertIn("let copied = value", context["ensures"][0]["text"])
        self.assertEqual(context["ensures"][1]["text"], "result.invariant()")

    def test_representative_parser_regressions_from_corpus(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        paths = {
            "vd0197": project_root / "data/generated/autoverus_zero_shot_verified/autoverus/autoverus_VeriCoding_VD0197_vericoded_gpt-4o_zero-shot_verified.rs",
            "vt0123": project_root / "data/generated/alphaverus_zero_shot_verified/alphaverus/alphaverus_VeriCoding_VT0123_vericoded_llama_zero-shot_verified.rs",
            "vt0293": project_root / "data/generated/autoverus_few_shot_verified/autoverus/autoverus_VeriCoding_VT0293_vericoded_gpt-4o_few-shot_verified.rs",
            "vs0006": project_root / "data/generated/starverus_by_model/zero-shot/deepseek-chat/verified/starverus/starverus_VeriCoding_VS0006_vericoded_deepseek-chat_zero-shot_verified.rs",
            "va0041": project_root / "data/generated/starverus_by_model/few-shot/deepseek-reasoner/verified/starverus/starverus_VeriCoding_VA0041_vericoded_deepseek-reasoner_few-shot_verified.rs",
        }
        if not all(path.is_file() for path in paths.values()):
            self.skipTest("representative parser samples are unavailable")

        vd0197 = next(
            item for item in strength_contexts_for_path(str(paths["vd0197"]))
            if item["function"] == "optimize_correct"
        )
        self.assertEqual(vd0197["parameters"][1]["type"], "Map<String, int>")

        vt0123 = next(
            item for item in strength_contexts_for_path(str(paths["vt0123"]))
            if item["function"] == "numpy_isdtype"
        )
        self.assertTrue(vt0123["ensures"][0]["text"].rstrip().endswith("}"))
        self.assertFalse(vt0123["ensures"][0]["text"].rstrip().endswith("},"))

        vt0293 = next(
            item for item in strength_contexts_for_path(str(paths["vt0293"]))
            if item["function"] == "clip_elem"
        )
        self.assertIn("min_val <= max_val ==>", vt0293["ensures"][0]["text"])
        self.assertIn("if x < min_val", vt0293["ensures"][0]["text"])

        vs0006 = next(
            item for item in strength_contexts_for_path(str(paths["vs0006"]))
            if item["function"] == "bitwise_or"
        )
        self.assertEqual(len(vs0006["ensures"]), 2)
        self.assertIn("(a[i] | b[i])", vs0006["ensures"][1]["text"])

        va0041_helpers = {
            item["name"]: item["text"]
            for item in spec_fn_blocks_for_path(str(paths["va0041"]))
        }
        self.assertIn("trim_newline_spec", va0041_helpers)
        self.assertIn("strip_whitespace_spec", va0041_helpers)
        self.assertNotIn(
            "spec fn strip_whitespace_spec",
            va0041_helpers["trim_newline_spec"],
        )

    def test_h1_through_h6_representative_corpus_shapes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        va0115_generated = root / (
            "data/generated/alphaverus_few_shot_verified/alphaverus/"
            "alphaverus_VeriCoding_VA0115_vericoded_llama_few-shot_verified.rs"
        )
        va0115_reference = root / (
            "data/references/VeriCoding/VeriCoding_VA0115_vericoded.rs"
        )
        mbpp755_generated = root / (
            "data/generated/starverus_by_model/few-shot/deepseek-chat/verified/starverus/"
            "starverus_VerusBench_MBPP_task_id_755_deepseek-chat_few-shot_verified.rs"
        )
        mbpp755_reference = root / (
            "data/references/VerusBench/VerusBench_MBPP_task_id_755.rs"
        )
        vt0219_generic = root / (
            "data/generated/starverus_by_model/few-shot/deepseek-chat/verified/starverus/"
            "starverus_VeriCoding_VT0219_vericoded_deepseek-chat_few-shot_verified.rs"
        )
        vt0219_control = root / (
            "data/generated/starverus_by_model/zero-shot/llama/verified/starverus/"
            "starverus_VeriCoding_VT0219_vericoded_llama_zero-shot_verified.rs"
        )
        va0673_generated = root / (
            "data/generated/starverus_by_model/few-shot/deepseek-chat/verified/starverus/"
            "starverus_VeriCoding_VA0673_vericoded_deepseek-chat_few-shot_verified.rs"
        )
        va0673_reference = root / (
            "data/references/VeriCoding/VeriCoding_VA0673_vericoded.rs"
        )
        mbpp58_generated = root / (
            "data/generated/starverus_by_model/few-shot/deepseek-reasoner/verified/starverus/"
            "starverus_MBPP-verified_task_58_deepseek-reasoner_few-shot_verified.rs"
        )
        mbpp58_reference = root / (
            "data/references/MBPP-verified/MBPP-verified_task_58.rs"
        )
        paths = (
            va0115_generated,
            va0115_reference,
            mbpp755_generated,
            mbpp755_reference,
            vt0219_generic,
            vt0219_control,
            va0673_generated,
            va0673_reference,
            mbpp58_generated,
            mbpp58_reference,
        )
        if not all(path.is_file() for path in paths):
            self.skipTest("H1-H6 representative corpus samples are unavailable")

        h1_generated = _strength_context(va0115_generated, "get_presidents")
        h1_reference = _strength_context(va0115_reference, "get_presidents")
        self.assertEqual(h1_reference["returns"], [{"name": None, "type": "Vec<&'static str>"}])
        self.assertTrue(bind_function_contexts(h1_reference, h1_generated).ok)

        h2_generated = _strength_context(mbpp755_generated, "second_smallest")
        h2_reference = _strength_context(mbpp755_reference, "second_smallest")
        self.assertTrue(bind_function_contexts(h2_reference, h2_generated).ok)

        h3_context = _strength_context(vt0219_generic, "check_lin_alg_error")
        h3_clause = next(
            item["text"] for item in h3_context["ensures"] if "None::<" in item["text"]
        )
        self.assertIsNone(_clause_syntax_issue(h3_clause))

        h4_context = _strength_context(vt0219_control, "check_lin_alg_error")
        h4_clause = h4_context["ensures"][0]["text"]
        rendered = render_clause_expression(h4_clause)
        self.assertTrue(rendered.startswith("(if condition"), rendered)
        self.assertIn(")\n        && result.is_some()", rendered)

        h5_context = _strength_context(va0673_generated, "solve")
        h5_support = _collect_generated_support(
            va0673_generated,
            va0673_reference,
            generated_texts=[item["text"] for item in h5_context["ensures"]],
            generated_sources=["generated"] * len(h5_context["ensures"]),
        )
        self.assertIsNone(h5_support.issue)
        self.assertEqual(
            h5_support.candidate_vstd_globs,
            ["vstd::seq::*", "vstd::string::*"],
        )

        h6_context = _strength_context(mbpp58_generated, "is_sorted")
        h6_support = _collect_generated_support(
            mbpp58_generated,
            mbpp58_reference,
            generated_texts=[item["text"] for item in h6_context["ensures"]],
            generated_sources=["generated"] * len(h6_context["ensures"]),
        )
        self.assertIsNone(h6_support.issue)

    def test_match_arm_blocks_are_kept_in_extracted_ensures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "match.rs",
                """
                fn choose(flag: bool) -> (result: Option<int>)
                    ensures
                        match result {
                            Some(value) => { value == value },
                            None => { true },
                        }
                {
                    None
                }
                """,
            )
            context = next(item for item in strength_contexts_for_path(path) if item["function"] == "choose")
        self.assertEqual(len(context["ensures"]), 1)
        clause = context["ensures"][0]["text"]
        self.assertIn("Some(value) => { value == value }", clause)
        self.assertIn("None => { true }", clause)
        self.assertTrue(clause.rstrip().endswith("}"))
        self.assertIsNone(_delimiter_issue(clause))
        self.assertEqual(_delimiter_issue("match value { Some(x) => (x > 0)"), "unclosed_{")

    def test_generated_spec_method_keeps_its_impl_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            reference = _write_verus_file(
                tmpdir,
                "ref.rs",
                """
                struct Holder { value: int }
                """,
            )
            generated = _write_verus_file(
                tmpdir,
                "gen.rs",
                """
                struct Holder { value: int }
                impl Holder {
                    spec fn valid(&self) -> bool { self.value >= 0 }
                }
                """,
            )
            support = _collect_generated_support(
                Path(generated),
                Path(reference),
                generated_texts=["holder.valid()"],
                generated_sources=["generated"],
            )
        self.assertIsNone(support.issue)
        self.assertEqual(support.call_renames, {})
        self.assertEqual(len(support.inner_items), 1)
        self.assertIn("impl Holder {", support.inner_items[0])
        self.assertIn("spec fn valid(&self)", support.inner_items[0])

    def test_exec_dependency_distinguishes_member_and_free_contains_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            reference = _write_verus_file(
                tmpdir,
                "ref.rs",
                "fn target(xs: &Vec<int>, value: int) { }",
            )
            generated = _write_verus_file(
                tmpdir,
                "gen.rs",
                """
                fn contains(xs: &Vec<int>, value: int) -> (out: bool) {
                    xs@.contains(value)
                }
                fn target(xs: &Vec<int>, value: int)
                    ensures xs@.contains(value),
                { }
                """,
            )

            member_support = _collect_generated_support(
                Path(generated),
                Path(reference),
                generated_texts=["xs@.contains(value)"],
                generated_sources=["generated"],
            )
            free_support = _collect_generated_support(
                Path(generated),
                Path(reference),
                generated_texts=["contains(xs, value)"],
                generated_sources=["generated"],
            )

        self.assertIsNone(member_support.issue)
        self.assertEqual(
            free_support.issue,
            {
                "reason": "unsupported_context",
                "support_issue": "requires_exec_declaration",
                "detail": "contains",
            },
        )

    def test_support_collector_includes_attribute_spec_and_transitive_constants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            reference = _write_verus_file(
                tmpdir,
                "ref.rs",
                """
                fn target(x: int) -> (out: bool)
                    ensures out == (x <= 10),
                { x <= 10 }
                """,
            )
            generated = _write_verus_file(
                tmpdir,
                "gen.rs",
                """
                spec const LIMIT: int = 10;
                #[verifier::spec]
                pub fn within_limit(x: int) -> bool { x <= LIMIT }
                fn target(x: int) -> (out: bool)
                    ensures out == within_limit(x),
                { x <= 10 }
                """,
            )
            support = _collect_generated_support(
                Path(generated),
                Path(reference),
                generated_texts=["out == within_limit(x)"],
                generated_sources=["generated"],
            )
        self.assertIsNone(support.issue)
        self.assertEqual(support.summary["copied_spec_functions"], 1)
        self.assertEqual(support.summary["copied_constants"], 1)
        rendered = "\n".join(support.inner_items)
        self.assertIn("#[verifier::spec]", rendered)
        self.assertIn("fn within_limit", rendered)
        self.assertNotIn("pub fn within_limit", rendered)
        self.assertIn("spec const LIMIT", rendered)

    def test_support_collector_records_vstd_globs_without_copying_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            reference = _write_verus_file(tmpdir, "ref.rs", "fn target(x: int) { }")
            generated = _write_verus_file(
                tmpdir,
                "gen.rs",
                """
                use vstd::set::*;
                use vstd::multiset::*;
                use vstd::seq::*;
                fn target(x: int) { }
                """,
            )
            support = _collect_generated_support(
                Path(generated),
                Path(reference),
                generated_texts=["Set::<int>::empty().len() == 0"],
                generated_sources=["generated"],
            )
            self.assertIsNone(support.issue)
            self.assertEqual(support.outer_uses, [])
            self.assertEqual(
                support.candidate_vstd_globs,
                ["vstd::multiset::*", "vstd::seq::*", "vstd::set::*"],
            )
            self.assertEqual(support.summary["copied_glob_imports"], 0)
            self.assertEqual(support.summary["candidate_glob_imports"], 3)
            self.assertEqual(support.summary["aliased_imports"], 0)

            unsupported = _write_verus_file(
                tmpdir,
                "unsupported.rs",
                """
                use vstd::map::*;
                fn target(x: int) { }
                """,
            )
            unsupported_support = _collect_generated_support(
                Path(unsupported),
                Path(reference),
                generated_texts=["Map::<int, int>::empty().len() == 0"],
                generated_sources=["generated"],
            )
            self.assertIsNone(unsupported_support.issue)
            self.assertEqual(unsupported_support.candidate_vstd_globs, ["vstd::map::*"])

    def test_support_collector_reuses_alpha_equivalent_reference_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            reference = _write_verus_file(
                tmpdir,
                "ref.rs",
                """
                spec fn spec_sum(n: nat) -> nat
                    decreases n,
                {
                    if n == 0 { 0 } else { n + spec_sum((n - 1) as nat) }
                }
                fn target(x: u32) -> (out: u32)
                    ensures out == spec_sum(x as nat),
                { 0 }
                """,
            )
            generated = _write_verus_file(
                tmpdir,
                "gen.rs",
                """
                spec fn spec_sum(k: nat) -> nat
                    decreases k,
                {
                    if k == 0 { 0 } else { k + spec_sum((k - 1) as nat) }
                }
                fn target(x: u32) -> (out: u32)
                    ensures out == spec_sum(x as nat),
                { 0 }
                """,
            )
            support = _collect_generated_support(
                Path(generated),
                Path(reference),
                generated_texts=["out == spec_sum(x as nat)"],
                generated_sources=["generated"],
            )
        self.assertIsNone(support.issue)
        self.assertEqual(support.summary["copied_spec_functions"], 0)
        self.assertEqual(support.summary["reused_reference_items"], 1)
        self.assertEqual(support.call_renames, {})
        self.assertEqual(support.inner_items, [])

    def test_alpha_reuse_keeps_structural_and_reserved_differences_copied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            different_body_ref = _write_verus_file(
                tmpdir,
                "ref_diff.rs",
                """
                spec fn helper(n: int) -> bool { n >= 0 }
                fn target(x: int) ensures helper(x), { }
                """,
            )
            different_body_gen = _write_verus_file(
                tmpdir,
                "gen_diff.rs",
                """
                spec fn helper(k: int) -> bool { k > 0 }
                fn target(x: int) ensures helper(x), { }
                """,
            )
            different_support = _collect_generated_support(
                Path(different_body_gen),
                Path(different_body_ref),
                generated_texts=["helper(x)"],
                generated_sources=["generated"],
            )
            self.assertIsNone(different_support.issue)
            self.assertEqual(different_support.summary["copied_spec_functions"], 1)
            self.assertIn("helper", different_support.call_renames)

            reserved_ref = _write_verus_file(
                tmpdir,
                "ref_reserved.rs",
                """
                spec const flag: int = 1;
                spec fn helper(n: int) -> bool { n >= 0 }
                fn target(x: int) ensures helper(x), { }
                """,
            )
            reserved_gen = _write_verus_file(
                tmpdir,
                "gen_reserved.rs",
                """
                spec const flag: int = 1;
                spec fn helper(flag: int) -> bool { flag >= 0 }
                fn target(x: int) ensures helper(x), { }
                """,
            )
            reserved_support = _collect_generated_support(
                Path(reserved_gen),
                Path(reserved_ref),
                generated_texts=["helper(x)"],
                generated_sources=["generated"],
            )
            self.assertIsNone(reserved_support.issue)
            self.assertEqual(reserved_support.summary["copied_spec_functions"], 1)
            self.assertIn("helper", reserved_support.call_renames)

    def test_alpha_equivalent_recursive_helper_keeps_identical_clause_valid(self) -> None:
        if not _verus_available():
            self.skipTest("Verus is unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            reference_path = _write_verus_file(
                tmpdir,
                "ref.rs",
                """
                spec fn spec_sum(n: nat) -> nat
                    decreases n,
                {
                    if n == 0 { 0 } else { n + spec_sum((n - 1) as nat) }
                }
                fn target(x: u32) -> (out: u32)
                    ensures out == spec_sum(x as nat),
                { 0 }
                """,
            )
            generated_path = _write_verus_file(
                tmpdir,
                "gen.rs",
                """
                spec fn spec_sum(k: nat) -> nat
                    decreases k,
                {
                    if k == 0 { 0 } else { k + spec_sum((k - 1) as nat) }
                }
                fn target(x: u32) -> (out: u32)
                    ensures out == spec_sum(x as nat),
                { 0 }
                """,
            )
            reference_context = _strength_context(reference_path, "target")
            generated_context = _strength_context(generated_path, "target")
            result = lemma_implication_check(
                reference_rs_path=reference_path,
                reference_context=reference_context,
                generated_context=generated_context,
                function_name="target",
                check_name="alpha_equivalent_helper",
                antecedent=[
                    {**clause, "_lemma_source": "generated"}
                    for clause in generated_context["ensures"]
                ],
                consequent=[
                    {**clause, "_lemma_source": "reference"}
                    for clause in reference_context["ensures"]
                ],
                generated_rs_path=generated_path,
            )
        self.assertEqual(result["status"], "valid", result)
        self.assertTrue(result["holds"], result)

    def test_support_collector_renames_cross_mode_conflicts_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            reference = _write_verus_file(
                tmpdir,
                "ref.rs",
                """
                fn helper(x: i32) -> (out: bool) { x >= 0 }
                fn target(x: int) { }
                """,
            )
            generated = _write_verus_file(
                tmpdir,
                "gen.rs",
                """
                spec fn helper(x: int) -> bool { x >= 0 }
                fn target(x: int)
                    ensures helper(x),
                { }
                """,
            )
            first = _collect_generated_support(
                Path(generated),
                Path(reference),
                generated_texts=["helper(x)"],
                generated_sources=["generated"],
            )
            second = _collect_generated_support(
                Path(generated),
                Path(reference),
                generated_texts=["helper(x)"],
                generated_sources=["generated"],
            )
            first_name = first.call_renames["helper"]
            reference = _write_verus_file(
                tmpdir,
                "ref.rs",
                f"""
                fn helper(x: i32) -> (out: bool) {{ x >= 0 }}
                fn {first_name}(x: i32) -> (out: bool) {{ x >= 0 }}
                fn target(x: int) {{ }}
                """,
            )
            with_prefixed_collision = _collect_generated_support(
                Path(generated),
                Path(reference),
                generated_texts=["helper(x)"],
                generated_sources=["generated"],
            )
        self.assertIsNone(first.issue)
        self.assertEqual(first.call_renames, second.call_renames)
        renamed = first.call_renames["helper"]
        self.assertTrue(renamed.startswith("__sqm_gen_helper_"))
        self.assertIn(f"spec fn {renamed}(", first.inner_items[0])
        self.assertNotEqual(with_prefixed_collision.call_renames["helper"], renamed)

    def test_support_collector_preflights_unsafe_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            reference = _write_verus_file(tmpdir, "ref.rs", "fn target(x: int) { }")

            missing_owner = _write_verus_file(
                tmpdir,
                "missing_owner.rs",
                """
                struct GeneratedOnly { value: int }
                impl GeneratedOnly {
                    spec fn valid(&self) -> bool { self.value >= 0 }
                }
                fn target(item: GeneratedOnly)
                    ensures item.valid(),
                { }
                """,
            )
            owner_support = _collect_generated_support(
                Path(missing_owner),
                Path(reference),
                generated_texts=["item.valid()"],
                generated_sources=["generated"],
                parameter_types=["GeneratedOnly"],
            )
            self.assertEqual(owner_support.issue["support_issue"], "missing_reference_owner")

            unsafe_glob = _write_verus_file(
                tmpdir,
                "glob.rs",
                """
                use crate::generated_helpers::*;
                fn target(x: int)
                    ensures mystery(x),
                { }
                """,
            )
            glob_support = _collect_generated_support(
                Path(unsafe_glob),
                Path(reference),
                generated_texts=["mystery(x)"],
                generated_sources=["generated"],
            )
            self.assertEqual(glob_support.issue["support_issue"], "unsafe_generated_import")

            exec_dependency = _write_verus_file(
                tmpdir,
                "exec.rs",
                """
                fn runtime_helper(x: int) -> (out: bool) { x >= 0 }
                fn target(x: int)
                    ensures runtime_helper(x),
                { }
                """,
            )
            exec_support = _collect_generated_support(
                Path(exec_dependency),
                Path(reference),
                generated_texts=["runtime_helper(x)"],
                generated_sources=["generated"],
            )
            self.assertEqual(exec_support.issue["support_issue"], "requires_exec_declaration")

            ambiguous_import = _write_verus_file(
                tmpdir,
                "ambiguous_import.rs",
                """
                use vstd::math::min;
                spec fn shadowed(min: int, x: int) -> bool { min(x) >= 0 }
                fn target(x: int)
                    ensures shadowed(0, x),
                { }
                """,
            )
            ambiguous_support = _collect_generated_support(
                Path(ambiguous_import),
                Path(reference),
                generated_texts=["shadowed(0, x)"],
                generated_sources=["generated"],
            )
            self.assertEqual(
                ambiguous_support.issue["support_issue"],
                "ambiguous_symbol_rewrite",
            )

            shadow_reference = _write_verus_file(
                tmpdir,
                "shadow_ref.rs",
                "spec const LIMIT: int = 0; fn target(x: int) { }",
            )
            shadow_generated = _write_verus_file(
                tmpdir,
                "shadow_gen.rs",
                "spec const LIMIT: int = 10; fn target(x: int) { }",
            )
            shadow_support = _collect_generated_support(
                Path(shadow_generated),
                Path(shadow_reference),
                generated_texts=["{ let LIMIT = 1; LIMIT == 1 }"],
                generated_sources=["generated"],
            )
            self.assertEqual(
                shadow_support.issue["support_issue"],
                "ambiguous_symbol_rewrite",
            )

    def test_strength_context_keeps_impl_owner_and_generic_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "context.rs",
                """
                struct Holder { value: int }
                impl Holder {
                    spec fn valid(&self) -> bool { self.value >= 0 }
                }
                fn target<T: Copy>(xs: &Vec<T>, holder: &Holder) -> (r: usize)
                    where T: Eq,
                    ensures holder.valid() ==> r == xs@.len(),
                { xs.len() }
                """,
            )
            context = next(item for item in strength_contexts_for_path(path) if item["function"] == "target")
        self.assertIn("impl Holder {", context["spec_preamble"])
        self.assertIn("spec fn valid(&self)", context["spec_preamble"])
        self.assertEqual(context["generic_parameters"], "<T: Copy>")
        self.assertEqual(context["where_clause"], "where T: Eq,")

    def test_vv0029_preamble_does_not_lift_self_method_to_top_level(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "data/generated"
            / "starverus_by_model"
            / "zero-shot"
            / "llama"
            / "unverified"
            / "starverus"
            / "starverus_VeriCoding_VV0029_vericoded_llama_zero-shot_unverified.rs"
        )
        if not path.is_file():
            self.skipTest("VV0029 representative sample is unavailable")
        context = next(
            item for item in strength_contexts_for_path(str(path))
            if item["function"] == "longest_increasing_streak"
        )
        preamble = context["spec_preamble"]
        method_offset = preamble.index("spec fn longest_increasing_streak(&self)")
        owner_offset = preamble.rfind("impl Seq<i32> {", 0, method_offset)
        self.assertGreaterEqual(owner_offset, 0)

    def test_helper_renames_only_touch_generated_clauses(self) -> None:
        support = lemma_implication.GeneratedSupportContext(
            call_renames={"is_odd": "_gen_is_odd", "generic": "_gen_generic"}
        )
        renamed = _apply_generated_support_renames(
            [
                "is_odd(x) && state.is_odd() && generic::<int>(x)",
                "is_odd(y) && generic::<int>(y)",
            ],
            ["generated", "reference"],
            support,
        )
        self.assertEqual(
            renamed,
            [
                "_gen_is_odd(x) && state.is_odd() && _gen_generic::<int>(x)",
                "is_odd(y) && generic::<int>(y)",
            ],
        )

        generic_forms = _apply_generated_support_renames(
            [
                "generic<int>(x) && generic::<Seq<int>>(x)",
                "generic < int > (x) && generic <==> other",
            ],
            ["generated", "generated"],
            support,
        )
        self.assertEqual(
            generic_forms,
            [
                "_gen_generic<int>(x) && _gen_generic::<Seq<int>>(x)",
                "generic < int > (x) && generic <==> other",
            ],
        )

    def test_verus_errors_distinguish_limits_from_compile_errors(self) -> None:
        trigger = _classify_verus_run(
            subprocess.CompletedProcess(
                ["verus"],
                1,
                "",
                "error: Could not automatically infer triggers for this quantifier.",
            ),
            0.1,
        )
        self.assertEqual(trigger["reason"], "verification_unresolved")
        self.assertEqual(trigger["verification_issue"], "trigger_inference")

        compile_error = _classify_verus_run(
            subprocess.CompletedProcess(
                ["verus"],
                1,
                "",
                "error[E0308]: mismatched types",
            ),
            0.1,
        )
        self.assertEqual(compile_error["reason"], "unclassified_verus_failure")
        self.assertEqual(compile_error["diagnostic_codes"], ["E0308"])

    def test_mutable_parameters_use_distinct_pre_and_post_snapshots(self) -> None:
        params, antecedent, consequent = _mutable_snapshot_context(
            [{"name": "data", "type": "&mut Vec<i32>"}],
            ["old(data).len() == 1", "data[0] > 0"],
            ["data[0] == old(data)[0] + 1"],
            antecedent_kinds=["requires", "requires"],
            consequent_kinds=["ensures"],
        )
        self.assertEqual(
            params,
            [
                {"name": "data", "type": "&Vec<i32>"},
                {"name": "__sqm_old_data", "type": "&Vec<i32>"},
            ],
        )
        self.assertEqual(
            antecedent,
            ["__sqm_old_data.len() == 1", "__sqm_old_data[0] > 0"],
        )
        self.assertEqual(consequent, ["data[0] == __sqm_old_data[0] + 1"])

    def test_unknown_checks_are_excluded_from_score_denominator(self) -> None:
        candidate = {
            "function": "f",
            "parameters": [{"name": "x", "type": "int"}],
            "returns": [],
            "requires": [{"kind": "requires", "text": "x > 0", "normalized": "x > 0"}],
        }
        reference = {
            "function": "f",
            "parameters": [{"name": "x", "type": "int"}],
            "returns": [],
            "requires": [{"kind": "requires", "text": "x >= 0", "normalized": "x >= 0"}],
        }
        with patch(
            "metrics_rebuild.share.smt.lemma_implication_check",
            return_value={"holds": None, "status": "tool_error"},
        ):
            result = clause_implication_details_for_context(
                candidate=candidate,
                reference=reference,
                reference_rs_path="ref.rs",
                source_side="reference",
                clause_kind="requires",
                implication_direction="generated_implies_reference_clause",
            )
        self.assertEqual(result["unknown"], 1)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["determined"], 0)
        self.assertEqual(result["coverage"], 0.0)
        self.assertIsNone(result["score"])

    def test_signature_mismatch_is_stopped_before_verus(self) -> None:
        candidate = {
            "function": "f",
            "parameters": [{"name": "x", "type": "usize"}],
            "returns": [],
            "requires": [{"kind": "requires", "text": "x > 0", "normalized": "x > 0"}],
        }
        reference = {
            "function": "f",
            "parameters": [{"name": "x", "type": "int"}],
            "returns": [],
            "requires": [{"kind": "requires", "text": "x >= 0", "normalized": "x >= 0"}],
        }
        result = clause_implication_details_for_context(
            candidate=candidate,
            reference=reference,
            reference_rs_path="ref.rs",
            source_side="reference",
            clause_kind="requires",
            implication_direction="generated_implies_reference_clause",
        )
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["unknown"], 1)
        self.assertEqual(result["determined"], 0)
        self.assertEqual(result["coverage"], 0.0)
        self.assertIsNone(result["score"])
        self.assertEqual(
            result["details"][0]["implication_check"]["reason"],
            "target_signature_mismatch",
        )

    def test_signature_mismatch_does_not_use_cross_file_text_shortcut(self) -> None:
        candidate = {
            "function": "f",
            "parameters": [{"name": "x", "type": "usize"}],
            "returns": [],
            "requires": [{"kind": "requires", "text": "x > 0", "normalized": "x > 0"}],
        }
        reference = {
            "function": "f",
            "parameters": [{"name": "x", "type": "int"}],
            "returns": [],
            "requires": [{"kind": "requires", "text": "x > 0", "normalized": "x > 0"}],
        }
        result = clause_implication_details_for_context(
            candidate=candidate,
            reference=reference,
            reference_rs_path="ref.rs",
            source_side="reference",
            clause_kind="requires",
            implication_direction="generated_implies_reference_clause",
        )
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["unknown"], 1)
        self.assertEqual(result["determined"], 0)
        self.assertEqual(result["coverage"], 0.0)
        self.assertIsNone(result["score"])

    def test_semantic_strength_does_not_use_cross_file_text_shortcut(self) -> None:
        candidate = {
            "function": "f",
            "parameters": [{"name": "x", "type": "usize"}],
            "returns": [],
            "requires": [{"kind": "requires", "text": "x > 0", "normalized": "x > 0"}],
            "ensures": [{"kind": "ensures", "text": "true", "normalized": "true"}],
        }
        reference = {
            "function": "f",
            "parameters": [{"name": "x", "type": "int"}],
            "returns": [],
            "requires": [{"kind": "requires", "text": "x > 0", "normalized": "x > 0"}],
            "ensures": [{"kind": "ensures", "text": "true", "normalized": "true"}],
        }
        result = semantic_strength_comparison(
            [candidate],
            [reference],
            lemma_reference_path="ref.rs",
            generated_rs_path="gen.rs",
        )
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["unknown"], 1)
        self.assertEqual(result["determined"], 0)
        self.assertEqual(result["coverage"], 0.0)
        self.assertIsNone(result["score"])

    def test_different_generic_contexts_stop_before_harness_compilation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            reference_path = _write_verus_file(
                tmpdir,
                "ref.rs",
                """
                fn f(x: int)
                    requires x >= 0,
                {
                }
                """,
            )
            generated_path = _write_verus_file(
                tmpdir,
                "gen.rs",
                """
                fn f<T>(x: int, marker: T)
                    requires x > 0,
                {
                }
                """,
            )
            result = lemma_implication_check(
                reference_rs_path=reference_path,
                generated_rs_path=generated_path,
                reference_context=_strength_context(reference_path, "f"),
                generated_context=_strength_context(generated_path, "f"),
                function_name="f",
                check_name="generic_context_mismatch",
                antecedent=[
                    {
                        "kind": "requires",
                        "text": "x > 0",
                        "_lemma_source": "generated",
                    }
                ],
                consequent=[
                    {
                        "kind": "requires",
                        "text": "x >= 0",
                        "_lemma_source": "reference",
                    }
                ],
            )
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["reason"], "generic_context_mismatch")

    def test_target_self_context_stops_before_top_level_lemma(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = _write_verus_file(
                Path(tmp),
                "method.rs",
                """
                struct Holder { value: int }
                impl Holder {
                    fn check(&self)
                        ensures self.value == self.value,
                    {
                    }
                }
                """,
            )
            result = lemma_implication_check(
                reference_rs_path=reference_path,
                reference_context={"mode": "exec", "parameters": [], "returns": []},
                generated_context={"mode": "exec", "parameters": [], "returns": []},
                function_name="check",
                check_name="self_context",
                antecedent=[],
                consequent=[
                    {
                        "kind": "ensures",
                        "text": "self.value == self.value",
                        "_lemma_source": "reference",
                    }
                ],
            )
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["reason"], "unsupported_context")
        self.assertEqual(result["support_issue"], "associated_self")

    def test_malformed_clause_stops_before_verus_and_keeps_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = _write_verus_file(Path(tmp), "ref.rs", "fn target(x: int) { }")
            with patch("metrics_rebuild.share.lemma_implication.subprocess.run") as run:
                result = lemma_implication_check(
                    reference_rs_path=reference_path,
                    reference_context=_strength_context(reference_path, "target"),
                    generated_context=_strength_context(reference_path, "target"),
                    function_name="target",
                    check_name="malformed_clause",
                    antecedent=[],
                    consequent=[
                        {
                            "kind": "ensures",
                            "text": "x ==>",
                            "_lemma_source": "reference",
                        }
                    ],
                )
        run.assert_not_called()
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["reason"], "malformed_clause")
        self.assertEqual(result["engine"], "lemma_preflight")
        self.assertEqual(result["lemma_harness_version"], "bound-wrapper-hardening-v5")

    def test_same_source_duplicate_clause_reuses_wrapper_for_self_implication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = _write_verus_file(
                Path(tmp),
                "ref.rs",
                """
                fn target(xs: Vec<int>)
                    ensures forall|i: int| 0 <= i < xs@.len() ==> xs@[i] == xs@[i],
                { }
                """,
            )
            context = _strength_context(reference_path, "target")
            calls: list[tuple[list[str], list[str]]] = []

            def fake_verify(**kwargs):
                calls.append((list(kwargs["antecedent_texts"]), list(kwargs["consequent_texts"])))
                return {"holds": True, "status": "valid", "engine": "test"}

            with patch.object(lemma_implication, "_verify_lemma", side_effect=fake_verify):
                result = lemma_implication_check(
                    reference_rs_path=reference_path,
                    reference_context=context,
                    generated_context=context,
                    function_name="target",
                    check_name="self_duplicate",
                    antecedent=[{**context["ensures"][0], "_lemma_source": "reference"}],
                    consequent=[{**context["ensures"][0], "_lemma_source": "reference"}],
                )

        self.assertTrue(result["holds"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], calls[0][1])

    def test_same_text_from_different_sources_keeps_wrappers_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = _write_verus_file(Path(tmp), "ref.rs", "fn target(x: int) { }")
            generated_path = _write_verus_file(Path(tmp), "gen.rs", "fn target(x: int) { }")
            context = _strength_context(reference_path, "target")
            generated_context = _strength_context(generated_path, "target")
            calls: list[tuple[list[str], list[str]]] = []

            def fake_verify(**kwargs):
                calls.append((list(kwargs["antecedent_texts"]), list(kwargs["consequent_texts"])))
                return {"holds": True, "status": "valid", "engine": "test"}

            with patch.object(lemma_implication, "_verify_lemma", side_effect=fake_verify):
                result = lemma_implication_check(
                    reference_rs_path=reference_path,
                    reference_context=context,
                    generated_context=generated_context,
                    function_name="target",
                    check_name="cross_source_duplicate",
                    antecedent=[
                        {
                            "kind": "ensures",
                            "text": "x >= 0",
                            "_lemma_source": "generated",
                        }
                    ],
                    consequent=[
                        {
                            "kind": "ensures",
                            "text": "x >= 0",
                            "_lemma_source": "reference",
                        }
                    ],
                    generated_rs_path=generated_path,
                )

        self.assertTrue(result["holds"])
        self.assertEqual(len(calls), 1)
        self.assertNotEqual(calls[0][0][0], calls[0][1][0])

    def test_vh0095_reference_encode_self_check_is_valid(self) -> None:
        if not _verus_available():
            self.skipTest("Verus is unavailable")
        project_root = Path(__file__).resolve().parents[1]
        reference_path = project_root / "data/references/VeriCoding/VeriCoding_VH0095_vericoded.rs"
        if not reference_path.is_file():
            self.skipTest("VH0095 reference sample is unavailable")
        context = _strength_context(reference_path, "encode")
        result = lemma_implication_check(
            reference_rs_path=str(reference_path),
            reference_context=context,
            generated_context=context,
            function_name="encode",
            check_name="generated_implies_reference_clause_ensures_002",
            antecedent=[{**clause, "_lemma_source": "generated"} for clause in context["ensures"]],
            consequent=[{**context["ensures"][1], "_lemma_source": "reference"}],
        )
        self.assertEqual(result["status"], "valid", result)
        self.assertTrue(result["holds"], result)


class TestLemmaCacheRuntime(unittest.TestCase):
    def setUp(self) -> None:
        lemma_implication._LEMMA_CACHE.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.reference = Path(
            _write_verus_file(root, "ref.rs", "fn target(x: int) { }")
        )
        self.verus = root / "verus"
        self.verus.write_text("fake", encoding="utf-8")
        lemma_implication.set_lemma_verus_binary(str(self.verus))

    def tearDown(self) -> None:
        lemma_implication.set_lemma_verus_binary(None)
        lemma_implication._LEMMA_CACHE.clear()

    def _check(self, timeout_seconds: int = 20, rlimit: float | None = None) -> dict:
        return lemma_implication._verify_lemma(
            reference_path=self.reference,
            function_name="target",
            check_name="cache",
            params=[{"name": "x", "type": "int"}],
            antecedent_texts=["x > 0"],
            consequent_texts=["x >= 0"],
            timeout_seconds=timeout_seconds,
            rlimit=rlimit,
        )

    def test_cache_key_isolated_by_harness_version(self) -> None:
        common = {
            "reference_path": self.reference,
            "function_name": "target",
            "check_name": "cache",
            "params": [{"name": "x", "type": "int"}],
            "generic_parameters": "",
            "where_clause": "",
            "antecedent": ["x > 0"],
            "consequent": ["x >= 0"],
            "runtime_fingerprint": {"configured_path": "verus", "version": "1"},
            "timeout_seconds": 20,
            "rlimit": None,
        }
        v4_key = _cache_key(**common)
        with patch.object(lemma_implication, "LEMMA_HARNESS_VERSION", "bound-wrapper-hardening-v3"):
            v3_key = _cache_key(**common)
        self.assertNotEqual(v3_key, v4_key)

    def test_cache_key_tracks_path_version_and_timeout(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        runtime_a = {"configured_path": "verus-a", "resolved_path": "/a", "version": "1"}
        runtime_b = {"configured_path": "verus-a", "resolved_path": "/a", "version": "2"}
        runtime_c = {"configured_path": "verus-c", "resolved_path": "/c", "version": "2"}
        with patch.object(lemma_implication.subprocess, "run", return_value=completed) as run:
            with patch.object(lemma_implication, "verus_runtime_fingerprint", return_value=runtime_a):
                first = self._check()
                cached = self._check()
                self._check(timeout_seconds=21)
            with patch.object(lemma_implication, "verus_runtime_fingerprint", return_value=runtime_b):
                self._check()
            with patch.object(lemma_implication, "verus_runtime_fingerprint", return_value=runtime_c):
                self._check()

        self.assertTrue(first["holds"])
        self.assertTrue(cached["cached"])
        self.assertEqual(run.call_count, 8)
        self.assertEqual(len(lemma_implication._LEMMA_CACHE), 4)

    def test_cache_key_and_command_track_rlimit(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        runtime = {"configured_path": "verus", "resolved_path": "/verus", "version": "1"}
        with patch.object(lemma_implication.subprocess, "run", return_value=completed) as run, \
                patch.object(lemma_implication, "verus_runtime_fingerprint", return_value=runtime):
            first = self._check(rlimit=10)
            second = self._check(rlimit=300)

        self.assertTrue(first["holds"])
        self.assertTrue(second["holds"])
        self.assertEqual(len(lemma_implication._LEMMA_CACHE), 2)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertTrue(all("--rlimit" in command for command in commands))
        self.assertIn("300", commands[-1])

    def test_cache_key_tracks_source_content_not_only_file_metadata(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        runtime = {"configured_path": "verus", "resolved_path": "/verus", "version": "1"}
        with patch.object(lemma_implication.subprocess, "run", return_value=completed) as run, \
                patch.object(lemma_implication, "verus_runtime_fingerprint", return_value=runtime):
            self._check()
            self._check()
            source = self.reference.read_text(encoding="utf-8")
            self.reference.write_text(source.replace("{ }", "{\n}"), encoding="utf-8")
            self._check()

        self.assertEqual(run.call_count, 4)
        self.assertEqual(len(lemma_implication._LEMMA_CACHE), 2)

    def test_timeout_is_retried_instead_of_cached(self) -> None:
        runtime = {"configured_path": "verus", "resolved_path": "/verus", "version": "1"}
        timeout = subprocess.TimeoutExpired(["verus"], 20)
        with patch.object(lemma_implication, "verus_runtime_fingerprint", return_value=runtime), \
                patch.object(lemma_implication.subprocess, "run", side_effect=timeout) as run:
            first = self._check()
            second = self._check()

        self.assertEqual(first["status"], "unknown")
        self.assertEqual(second["status"], "unknown")
        self.assertEqual(first["reason"], "harness_frontend_timeout")
        self.assertEqual(first["phase"], "harness_frontend")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(lemma_implication._LEMMA_CACHE, {})

    def test_unavailable_runtime_is_retried_instead_of_cached(self) -> None:
        lemma_implication.set_lemma_verus_binary("/missing/verus")
        runtime = {"configured_path": "/missing/verus", "resolved_path": None, "version": None}
        with patch.object(lemma_implication, "verus_runtime_fingerprint", return_value=runtime), \
                patch.object(lemma_implication.shutil, "which", return_value=None) as which:
            first = self._check()
            second = self._check()

        self.assertEqual(first["status"], "unknown")
        self.assertEqual(second["status"], "unknown")
        self.assertEqual(first["reason"], "verus_binary_not_found")
        self.assertEqual(first["phase"], "runtime")
        self.assertEqual(which.call_count, 2)
        self.assertEqual(lemma_implication._LEMMA_CACHE, {})

    def test_frontend_failure_is_unknown_not_invalid(self) -> None:
        runtime = {"configured_path": "verus", "resolved_path": "/verus", "version": "1"}
        compile_failure = subprocess.CompletedProcess(
            [],
            1,
            stdout="",
            stderr="error[E0425]: cannot find value `missing` in this scope",
        )
        with patch.object(
            lemma_implication,
            "verus_runtime_fingerprint",
            return_value=runtime,
        ), patch.object(
            lemma_implication.subprocess,
            "run",
            return_value=compile_failure,
        ) as run:
            result = self._check()

        self.assertEqual(run.call_count, 1)
        self.assertIsNone(result["holds"])
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["reason"], "harness_frontend_failed")
        self.assertEqual(result["phase"], "harness_frontend")
        self.assertEqual(lemma_implication._LEMMA_CACHE, {})

    def test_postcondition_failure_is_the_only_determined_invalid_case(self) -> None:
        runtime = {"configured_path": "verus", "resolved_path": "/verus", "version": "1"}
        proof_failure = subprocess.CompletedProcess(
            [],
            1,
            stdout="",
            stderr="error: postcondition not satisfied",
        )
        with patch.object(
            lemma_implication,
            "verus_runtime_fingerprint",
            return_value=runtime,
        ), patch.object(
            lemma_implication.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess([], 0, stdout="", stderr=""),
                proof_failure,
            ],
        ) as run:
            result = self._check()

        self.assertEqual(run.call_count, 2)
        self.assertFalse(result["holds"])
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(len(lemma_implication._LEMMA_CACHE), 1)

    def test_frontend_selects_only_the_unique_import_that_reduces_unresolved_names(self) -> None:
        runtime = {"configured_path": "verus", "resolved_path": "/verus", "version": "1"}

        def fake_run(command, **_kwargs):
            harness = Path(command[-1]).read_text(encoding="utf-8")
            if "--no-verify" not in command:
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            if "use vstd::seq::*;" in harness:
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="error[E0425]: cannot find value `needed` in this scope",
            )

        support = lemma_implication.GeneratedSupportContext(
            candidate_vstd_globs=["vstd::map::*", "vstd::seq::*"]
        )
        with patch.object(
            lemma_implication, "verus_runtime_fingerprint", return_value=runtime
        ), patch.object(lemma_implication.subprocess, "run", side_effect=fake_run) as run:
            result = lemma_implication._verify_lemma(
                reference_path=self.reference,
                function_name="target",
                check_name="diagnostic_import",
                params=[{"name": "x", "type": "int"}],
                antecedent_texts=["x > 0"],
                consequent_texts=["x >= 0"],
                timeout_seconds=20,
                support_context=support,
            )

        self.assertEqual(result["status"], "valid", result)
        self.assertEqual(result["selected_vstd_globs"], ["vstd::seq::*"])
        self.assertEqual(result["frontend_status"], "passed")
        self.assertEqual(run.call_count, 4)

    def test_frontend_keeps_ambiguous_glob_resolution_unknown(self) -> None:
        runtime = {"configured_path": "verus", "resolved_path": "/verus", "version": "1"}

        def fake_run(command, **_kwargs):
            harness = Path(command[-1]).read_text(encoding="utf-8")
            if "use vstd::map::*;" in harness or "use vstd::seq::*;" in harness:
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="error[E0425]: cannot find value `shared` in this scope",
            )

        support = lemma_implication.GeneratedSupportContext(
            candidate_vstd_globs=["vstd::map::*", "vstd::seq::*"]
        )
        with patch.object(
            lemma_implication, "verus_runtime_fingerprint", return_value=runtime
        ), patch.object(lemma_implication.subprocess, "run", side_effect=fake_run):
            result = lemma_implication._verify_lemma(
                reference_path=self.reference,
                function_name="target",
                check_name="ambiguous_import",
                params=[{"name": "x", "type": "int"}],
                antecedent_texts=["x > 0"],
                consequent_texts=["x >= 0"],
                timeout_seconds=20,
                support_context=support,
            )

        self.assertEqual(result["status"], "unknown", result)
        self.assertEqual(result["support_issue"], "ambiguous_generated_import")
        self.assertEqual(result["frontend_status"], "ambiguous_import")

@unittest.skipUnless(_verus_available(), "Verus not available")
class TestLemmaHarnessContextWithVerus(unittest.TestCase):
    def test_generic_probe_does_not_raise_e0412(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "generic.rs",
                """
                fn impossible<T>(items: &Vec<T>)
                    requires
                        items.len() == 0,
                        items.len() > 0,
                {
                }
                """,
            )
            result = probe_precondition_falsity_for_path(path)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["always_false_functions"], 1)

    def test_generic_where_clause_is_carried_into_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "generic_where.rs",
                """
                fn impossible<T>(item: T)
                    where T: Copy,
                    requires false,
                {
                }
                """,
            )
            result = probe_precondition_falsity_for_path(path)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["always_false_functions"], 1)

    def test_mut_probe_preserves_old_argument_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "mutable.rs",
                """
                fn impossible(data: &mut Vec<i32>)
                    requires
                        old(data).len() == 0,
                        old(data).len() > 0,
                {
                }
                """,
            )
            result = probe_precondition_falsity_for_path(path)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["always_false_functions"], 1)

    def test_match_postcondition_probe_reaches_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_verus_file(
                Path(tmp),
                "match.rs",
                """
                fn choose(flag: bool) -> (result: Option<int>)
                    ensures
                        match result {
                            Some(value) => { value == value },
                            None => { true },
                        }
                {
                    None
                }
                """,
            )
            result = probe_postcondition_truth_for_path(path)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["always_true_functions"], 1)

    def test_generated_spec_method_is_verified_inside_impl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            reference_path = _write_verus_file(
                tmpdir,
                "ref.rs",
                """
                struct Holder { value: int }
                impl Holder {
                    spec fn nonnegative(&self) -> bool { self.value >= 0 }
                }
                fn make(value: int) -> (result: Holder)
                    requires value >= 0,
                    ensures result.nonnegative(),
                {
                    Holder { value }
                }
                """,
            )
            generated_path = _write_verus_file(
                tmpdir,
                "gen.rs",
                """
                struct Holder { value: int }
                impl Holder {
                    pub open spec fn valid(&self) -> bool { self.value >= 0 }
                }
                fn make(value: int) -> (result: Holder)
                    requires value >= 0,
                    ensures result.valid(),
                {
                    Holder { value }
                }
                """,
            )
            generated = next(
                item for item in strength_contexts_for_path(generated_path)
                if item["function"] == "make"
            )
            reference = next(
                item for item in strength_contexts_for_path(reference_path)
                if item["function"] == "make"
            )
            result = lemma_implication_check(
                reference_rs_path=reference_path,
                generated_rs_path=generated_path,
                reference_context=reference,
                generated_context=generated,
                function_name="make",
                check_name="generated_valid_implies_reference_nonnegative",
                antecedent=[
                    {**clause, "_lemma_source": "generated"}
                    for clause in generated["ensures"]
                ],
                consequent=[
                    {**clause, "_lemma_source": "reference"}
                    for clause in reference["ensures"]
                ],
            )
        self.assertEqual(result["status"], "valid", result)
        self.assertTrue(result["holds"])
        self.assertEqual(result["support_context_summary"]["copied_spec_functions"], 1)

    def test_cross_mode_helper_conflict_compiles_after_rename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            reference_path = _write_verus_file(
                tmpdir,
                "ref.rs",
                """
                fn helper(x: i32) -> (out: bool) { x >= 0 }
                fn target(x: int) { }
                """,
            )
            generated_path = _write_verus_file(
                tmpdir,
                "gen.rs",
                """
                open spec fn helper(x: int) -> bool { x >= 0 }
                fn target(x: int)
                    ensures helper(x),
                { }
                """,
            )
            result = lemma_implication_check(
                reference_rs_path=reference_path,
                generated_rs_path=generated_path,
                reference_context=_strength_context(reference_path, "target"),
                generated_context=_strength_context(generated_path, "target"),
                function_name="target",
                check_name="cross_mode_helper",
                antecedent=[
                    {
                        "kind": "ensures",
                        "text": "helper(x)",
                        "_lemma_source": "generated",
                    }
                ],
                consequent=[
                    {
                        "kind": "ensures",
                        "text": "helper(x)",
                        "_lemma_source": "generated",
                    }
                ],
            )
        self.assertEqual(result["status"], "valid", result)
        self.assertEqual(result["support_context_summary"]["renamed_symbols"], 1)

    def test_attribute_spec_and_constant_support_compile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            reference_path = _write_verus_file(tmpdir, "ref.rs", "fn target(x: usize) { }")
            generated_path = _write_verus_file(
                tmpdir,
                "gen.rs",
                """
                spec const LIMIT: int = 10;
                #[verifier::spec]
                fn nonzero(x: usize) -> bool { x > 0 }
                spec fn within_limit(x: int) -> bool { x <= LIMIT }
                fn target(x: usize)
                    ensures nonzero(x) && within_limit(x as int),
                { }
                """,
            )
            result = lemma_implication_check(
                reference_rs_path=reference_path,
                generated_rs_path=generated_path,
                reference_context=_strength_context(reference_path, "target"),
                generated_context=_strength_context(generated_path, "target"),
                function_name="target",
                check_name="attribute_spec_constant",
                antecedent=[
                    {
                        "kind": "ensures",
                        "text": "nonzero(x) && within_limit(x as int)",
                        "_lemma_source": "generated",
                    }
                ],
                consequent=[
                    {
                        "kind": "ensures",
                        "text": "nonzero(x) && within_limit(x as int)",
                        "_lemma_source": "generated",
                    }
                ],
            )
        self.assertEqual(result["status"], "valid", result)
        self.assertEqual(result["support_context_summary"]["copied_spec_functions"], 2)
        self.assertEqual(result["support_context_summary"]["copied_constants"], 1)

    def test_vstd_globs_are_only_selected_when_frontend_needs_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            reference_path = _write_verus_file(tmpdir, "ref.rs", "fn target() { }")
            generated_path = _write_verus_file(
                tmpdir,
                "gen.rs",
                """
                use vstd::set::*;
                use vstd::multiset::*;
                use vstd::seq::*;
                fn target()
                    ensures
                        Set::<int>::empty().len() == 0,
                        Multiset::<int>::empty().len() == 0,
                        Seq::<int>::empty().len() == 0,
                { }
                """,
            )
            result = lemma_implication_check(
                reference_rs_path=reference_path,
                generated_rs_path=generated_path,
                reference_context=_strength_context(reference_path, "target"),
                generated_context=_strength_context(generated_path, "target"),
                function_name="target",
                check_name="whitelisted_vstd_globs",
                antecedent=[
                    {
                        "kind": "ensures",
                        "text": "Set::<int>::empty().len() == 0 && Multiset::<int>::empty().len() == 0 && Seq::<int>::empty().len() == 0",
                        "_lemma_source": "generated",
                    }
                ],
                consequent=[
                    {
                        "kind": "ensures",
                        "text": "Set::<int>::empty().len() == 0 && Multiset::<int>::empty().len() == 0 && Seq::<int>::empty().len() == 0",
                        "_lemma_source": "generated",
                    }
                ],
            )
        self.assertEqual(result["status"], "valid", result)
        self.assertEqual(result["support_context_summary"]["copied_glob_imports"], 0)
        self.assertEqual(result["support_context_summary"]["candidate_glob_imports"], 3)
        self.assertTrue(
            set(result["selected_vstd_globs"])
            <= {"vstd::set::*", "vstd::multiset::*", "vstd::seq::*"}
        )

    def test_va0455_imported_min_is_aliased_away_from_reference_min(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        generated_path = (
            project_root
            / "data/generated/verusage_by_model/zero-shot/deepseek-chat/verified/verusage"
            / "verusage_VeriCoding_VA0455_vericoded_deepseek-chat_zero-shot_verified.rs"
        )
        reference_path = (
            project_root
            / "data/references/VeriCoding/VeriCoding_VA0455_vericoded.rs"
        )
        if not generated_path.is_file() or not reference_path.is_file():
            self.skipTest("VA0455 representative sample is unavailable")
        generated = next(
            item for item in strength_contexts_for_path(str(generated_path))
            if item["function"] == "solve_cookie_distribution"
        )
        reference = next(
            item for item in strength_contexts_for_path(str(reference_path))
            if item["function"] == "solve_cookie_distribution"
        )
        result = lemma_implication_check(
            reference_rs_path=str(reference_path),
            generated_rs_path=str(generated_path),
            reference_context=reference,
            generated_context=generated,
            function_name="solve_cookie_distribution",
            check_name="va0455_import_collision",
            antecedent=[
                {**clause, "_lemma_source": "generated"}
                for clause in generated["ensures"]
            ],
            consequent=[
                {**clause, "_lemma_source": "generated"}
                for clause in generated["ensures"]
            ],
        )
        self.assertEqual(result["status"], "valid", result)
        self.assertEqual(result["support_context_summary"]["aliased_imports"], 1)

    def test_representative_support_context_samples_compile(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        cases = [
            (
                "VT0043",
                project_root / "data/generated/autoverus_few_shot_verified/autoverus/autoverus_VeriCoding_VT0043_vericoded_gpt-4o_few-shot_verified.rs",
                project_root / "data/references/VeriCoding/VeriCoding_VT0043_vericoded.rs",
                "broadcast",
                {"copied_spec_functions": 1},
            ),
            (
                "s4if",
                project_root / "data/generated/autoverus_few_shot_verified/autoverus/autoverus_VerusBench_Diffy_s4if_gpt-4o_few-shot_verified.rs",
                project_root / "data/references/VerusBench/VerusBench_Diffy_s4if.rs",
                "myfun",
                {},
            ),
            (
                "VT0030",
                project_root / "data/generated/autoverus_few_shot_verified/autoverus/autoverus_VeriCoding_VT0030_vericoded_gpt-4o_few-shot_verified.rs",
                project_root / "data/references/VeriCoding/VeriCoding_VT0030_vericoded.rs",
                "ones_like",
                {},
            ),
            (
                "VT0541",
                project_root / "data/generated/starverus_by_model/few-shot/gpt-4o/verified/starverus/starverus_VeriCoding_VT0541_vericoded_gpt-4o_few-shot_verified.rs",
                project_root / "data/references/VeriCoding/VeriCoding_VT0541_vericoded.rs",
                "mt19937",
                {"renamed_symbols": 1},
            ),
            (
                "MBPP6",
                project_root / "data/generated/starverus_by_model/zero-shot/gpt-4o/verified/starverus/starverus_MBPP-verified_task_6_gpt-4o_zero-shot_verified.rs",
                project_root / "data/references/MBPP-verified/MBPP-verified_task_6.rs",
                "is_odd_at_odd_index",
                {"copied_spec_functions": 1},
            ),
            (
                "VA0515",
                project_root / "data/generated/starverus_few_shot_verified/starverus/starverus_VeriCoding_VA0515_vericoded_deepseek-reasoner_few-shot_verified.rs",
                project_root / "data/references/VeriCoding/VeriCoding_VA0515_vericoded.rs",
                "solve",
                {"copied_constants": 2},
            ),
        ]
        for label, generated_path, reference_path, function_name, expected in cases:
            with self.subTest(case=label):
                if not generated_path.is_file() or not reference_path.is_file():
                    self.skipTest(f"{label} representative sample is unavailable")
                generated = next(
                    item for item in strength_contexts_for_path(str(generated_path))
                    if item["function"] == function_name
                )
                reference = next(
                    item for item in strength_contexts_for_path(str(reference_path))
                    if item["function"] == function_name
                )
                result = lemma_implication_check(
                    reference_rs_path=str(reference_path),
                    generated_rs_path=str(generated_path),
                    reference_context=reference,
                    generated_context=generated,
                    function_name=function_name,
                    check_name=f"{label.lower()}_support_context",
                    antecedent=[
                        {**clause, "_lemma_source": "generated"}
                        for clause in generated["ensures"]
                    ],
                    consequent=[
                        {**clause, "_lemma_source": "generated"}
                        for clause in generated["ensures"]
                    ],
                )
                self.assertEqual(result["status"], "valid", result)
                for key, value in expected.items():
                    self.assertEqual(result["support_context_summary"][key], value)

    def test_conflicting_spec_method_signatures_are_renamed_by_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            reference_path = _write_verus_file(
                tmpdir,
                "ref.rs",
                """
                struct Holder { value: int }
                impl Holder {
                    spec fn valid(&self, limit: int) -> bool { self.value >= limit }
                }
                fn make(value: int) -> (result: Holder)
                    requires value >= 0,
                    ensures result.valid(0),
                {
                    Holder { value }
                }
                """,
            )
            generated_path = _write_verus_file(
                tmpdir,
                "gen.rs",
                """
                struct Holder { value: int }
                impl Holder {
                    spec fn valid(&self) -> bool { self.value >= 0 }
                }
                fn make(value: int) -> (result: Holder)
                    requires value >= 0,
                    ensures result.valid(),
                {
                    Holder { value }
                }
                """,
            )
            generated = next(
                item for item in strength_contexts_for_path(generated_path)
                if item["function"] == "make"
            )
            reference = next(
                item for item in strength_contexts_for_path(reference_path)
                if item["function"] == "make"
            )
            result = lemma_implication_check(
                reference_rs_path=reference_path,
                generated_rs_path=generated_path,
                reference_context=reference,
                generated_context=generated,
                function_name="make",
                check_name="generated_valid_implies_reference_valid",
                antecedent=[
                    {**clause, "_lemma_source": "generated"}
                    for clause in generated["ensures"]
                ],
                consequent=[
                    {**clause, "_lemma_source": "reference"}
                    for clause in reference["ensures"]
                ],
            )
        self.assertEqual(result["status"], "valid")
        self.assertTrue(result["holds"])


if __name__ == "__main__":
    unittest.main()
