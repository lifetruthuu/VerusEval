#!/usr/bin/env python3
"""Unit tests for metrics_rebuild.share.io_harness core functions."""
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from metrics_rebuild.share.io_harness import (
    ParamInfo, FuncInfo, extract_function, parse_params, parse_type,
    format_value_rust, parse_results, preferred_io_function_name,
    runtime_type_support_issue,
)
from metrics_rebuild.share.io_cases import generate_requires_satisfying_candidate_inputs
from metrics_rebuild.share.contract_eval import (
    _contract_check_proof_fn_lines,
    _hidden_spec_index_is_out_of_bounds,
    _quantifier_trigger_witness_assertions,
    coerce_value_for_type,
)
from scripts.io.generate_io_tests_llm import (
    RunOutcome,
    Target,
    _find_harness_binary,
    _has_verus_math_runtime_type,
    _invalid_target_info,
    _merge_audit,
    _negative_context,
    _negative_target_info,
    _parse_typed_output_to_dict,
    _positive_target_info,
    _supplement_negatives_in_working,
    _typed_input_key,
    _write_summary,
    build_verus_harness,
    collect_invalid_cases,
    format_typed_output_for_json,
    gen_fmt_expr,
    gen_fmt_fn,
    parse_typed_value,
    process_one,
    resolve_existing_target,
    rewrite_verus_math_types,
)


class TestParseType(unittest.TestCase):
    def test_simple_types(self):
        ty, ref, mut, sl, inner = parse_type("i32")
        self.assertEqual(ty, "i32")
        self.assertFalse(ref)
        self.assertFalse(mut)

    def test_contract_harness_surfaces_concrete_sequence_return_facts(self):
        context = {
            "parameters": [],
            "returns": [{"name": "result", "type": "Vec<i32>"}],
            "requires": [],
            "ensures": [{"text": "forall|i: int| 0 <= i < result.len() ==> result[i] == 5"}],
        }

        lines = _contract_check_proof_fn_lines(
            context,
            {},
            {"result": [5]},
            "__check",
            negate_contract=True,
        )

        self.assertIn("    assert(result.len() == 1);", lines)
        self.assertIn("    assert(result[0] == 5i32);", lines)

    def test_hidden_spec_index_detects_concrete_out_of_bounds_output(self):
        context = {
            "spec_preamble": """
open spec fn is_max(v: &Vec<i32>, idx: usize) -> bool {
    forall|i: usize| i < v.len() ==> v[idx as int] >= v[i as int]
}
""",
            "ensures": [{"text": "is_max(values, result)"}],
        }

        self.assertTrue(
            _hidden_spec_index_is_out_of_bounds(
                context,
                {"inputs": {"values": [5]}, "output": {"result": 1}},
            )
        )
        self.assertFalse(
            _hidden_spec_index_is_out_of_bounds(
                context,
                {"inputs": {"values": [5]}, "output": {"result": 0}},
            )
        )

    def test_quantifier_trigger_witnesses_use_concrete_scalar_values(self):
        context = {
            "ensures": [
                {
                    "text": (
                        "forall|r: int| #[trigger] is_min(r, a, b) ==> result <= r"
                    )
                }
            ]
        }

        assertions = _quantifier_trigger_witness_assertions(
            context,
            {"inputs": {"a": 0, "b": -1}, "output": {"result": 0}},
        )

        self.assertIn("    assert(is_min(-1int, a, b));", assertions)

    def test_ref(self):
        ty, ref, mut, sl, inner = parse_type("&Vec<i32>")
        self.assertTrue(ref)
        self.assertFalse(mut)
        self.assertEqual(inner, "i32")

    def test_mut_ref(self):
        ty, ref, mut, sl, inner = parse_type("&mut Vec<i32>")
        self.assertTrue(ref)
        self.assertTrue(mut)
        self.assertEqual(inner, "i32")
        self.assertIn("mut", ty)

    def test_slice(self):
        ty, ref, mut, sl, inner = parse_type("&[i64]")
        self.assertTrue(ref)
        self.assertTrue(sl)
        self.assertEqual(inner, "i64")


class TestExtractFunction(unittest.TestCase):
    def test_simple_fn(self):
        code = """
use vstd::prelude::*;
verus! {
fn is_greater(arr: &Vec<i32>, number: i32) -> (result: bool) {
    true
}
} // verus!
"""
        func = extract_function(code)
        self.assertIsNotNone(func)
        self.assertEqual(func.name, "is_greater")
        self.assertEqual(func.return_type, "bool")
        self.assertEqual(len(func.params), 2)
        self.assertEqual(func.params[0].name, "arr")
        self.assertEqual(func.params[1].name, "number")

    def test_skip_main(self):
        code = "fn main() {} fn foo(x: i32) -> (r: i32) { x }"
        func = extract_function(code)
        self.assertIsNotNone(func)
        self.assertEqual(func.name, "foo")

    def test_skip_spec_fn(self):
        code = """
verus! {
spec fn helper() -> bool { true }
fn actual(x: i32) -> (r: bool) { true }
}
"""
        func = extract_function(code)
        self.assertIsNotNone(func)
        self.assertEqual(func.name, "actual")

    def test_void_fn(self):
        code = "verus! { pub fn myfun(a: &mut Vec<i32>, N: i32) { } }"
        func = extract_function(code)
        self.assertIsNotNone(func)
        self.assertEqual(func.name, "myfun")
        self.assertTrue(func.params[0].is_mut_ref)

    def test_named_tuple_return_keeps_full_type(self):
        code = """
verus! {
fn pair(x: usize) -> (result: (usize, usize))
    ensures result.0 <= result.1
{
    (x, x + 1)
}
}
"""
        func = extract_function(code)
        self.assertIsNotNone(func)
        self.assertEqual(func.name, "pair")
        self.assertEqual(func.return_name, "result")
        self.assertEqual(func.return_type, "(usize, usize)")

    def test_commented_signature_does_not_shadow_marked_target(self):
        code = """
verus! {
// <vc-spec>
// fn reverse(a: &Vec<char>) -> Vec<char>
fn reverse(a: &Vec<char>) -> (result: Vec<char>)
    ensures result.len() == a.len()
{
    Vec::new()
}
// </vc-spec>
}
"""
        preferred = preferred_io_function_name(code)
        func = extract_function(code, preferred)
        self.assertEqual(preferred, "reverse")
        self.assertIsNotNone(func)
        self.assertEqual(func.return_type, "Vec<char>")

    def test_plain_tuple_return_keeps_full_type(self):
        code = "verus! { fn pair(x: usize) -> (usize, usize) { (x, x + 1) } }"
        func = extract_function(code)
        self.assertIsNotNone(func)
        self.assertEqual(func.return_name, "result")
        self.assertEqual(func.return_type, "(usize, usize)")

    def test_nested_vec_return_keeps_full_type(self):
        code = "verus! { fn matrix(x: i32) -> Vec<Vec<i32>> { vec![vec![x]] } }"
        func = extract_function(code)
        self.assertIsNotNone(func)
        self.assertEqual(func.return_type, "Vec<Vec<i32>>")

    def test_preferred_vc_spec_function_wins_over_helper(self):
        code = """
verus! {
// <vc-helpers>
fn helper(x: i32) -> (r: i32) { x }
// </vc-helpers>
// <vc-spec>
fn target(x: i32) -> (r: i32) ensures r == x
// </vc-spec>
// <vc-code>
{ x }
// </vc-code>
}
"""
        preferred = preferred_io_function_name(code)
        self.assertEqual(preferred, "target")
        self.assertEqual(extract_function(code, preferred).name, "target")

    def test_generic_vc_spec_function_is_extracted(self):
        code = """
verus! {
// <vc-helpers>
proof fn lemma_subrange_full<T>(s: Seq<T>) ensures s.subrange(0, s.len() as int) == s {}
// </vc-helpers>
// <vc-spec>
fn copy<T: Copy>(a: &Vec<T>) -> (result: Vec<T>)
    ensures
        result.len() == a.len(),
        forall|i: int| 0 <= i < a.len() ==> result[i] == a[i],
// </vc-spec>
// <vc-code>
{ a.clone() }
// </vc-code>
}
"""
        preferred = preferred_io_function_name(code)
        self.assertEqual(preferred, "copy")
        func = extract_function(code, preferred)
        self.assertIsNotNone(func)
        self.assertEqual(func.name, "copy")
        self.assertEqual(len(func.params), 1)
        self.assertEqual(func.params[0].name, "a")
        self.assertEqual(func.return_type, "Vec<T>")

    def test_vericoding_generic_targets_resolve(self):
        names = [
            "VeriCoding_VT0007_vericoded",
            "VeriCoding_VT0030_vericoded",
            "VeriCoding_VT0048_vericoded",
            "VeriCoding_VT0052_vericoded",
            "VeriCoding_VT0058_vericoded",
        ]
        source_root = ROOT / "data/references" / "VeriCoding"
        if not source_root.exists():
            self.skipTest("VeriCoding source dataset is unavailable")
        from scripts.io.generate_io_tests_llm import resolve_target

        for name in names:
            path = source_root / f"{name}.rs"
            if not path.exists():
                self.skipTest(f"missing {path}")
            preferred = preferred_io_function_name(path.read_text(encoding="utf-8"))
            self.assertIsNotNone(preferred, name)
            target = resolve_target(str(path))
            self.assertIsNotNone(target, name)
            self.assertEqual(target.func.name, preferred, name)

    def test_all_vericoding_marked_targets_are_selected(self):
        source_root = ROOT / "data/references" / "VeriCoding"
        if not source_root.exists():
            self.skipTest("VeriCoding source dataset is unavailable")
        checked = 0
        mismatches = []
        for path in source_root.glob("*.rs"):
            code = path.read_text(encoding="utf-8")
            preferred = preferred_io_function_name(code)
            if preferred is None:
                continue
            checked += 1
            selected = extract_function(code, preferred)
            if selected is None or selected.name != preferred:
                mismatches.append(path.name)
        self.assertGreaterEqual(checked, 400)
        self.assertEqual(mismatches, [])


class TestFormatValueRust(unittest.TestCase):
    def test_bool(self):
        self.assertEqual(format_value_rust(True, "bool"), "true")
        self.assertEqual(format_value_rust(False, "bool"), "false")

    def test_i32(self):
        self.assertIn("5", format_value_rust(5, "i32"))
        self.assertIn("i32", format_value_rust(5, "i32"))

    def test_i32_min_uses_const(self):
        result = format_value_rust(-2147483648, "i32")
        self.assertIn("MIN", result)

    def test_u32_clamps_negative(self):
        result = format_value_rust(-1, "u32")
        self.assertIn("0", result)

    def test_char_single(self):
        self.assertEqual(format_value_rust('a', "char"), "'a'")

    def test_char_multi_fallback(self):
        result = format_value_rust('abc', "char")
        self.assertEqual(len(result), 3)  # 'x'

    def test_char_int_value(self):
        # LLM/heuristic 生成的 char 值可能是 ASCII 整数
        self.assertEqual(format_value_rust(97, "char"), "'a'")

    def test_vec(self):
        result = format_value_rust([1, 2, 3], "Vec<i32>")
        self.assertIn("vec![", result)
        self.assertIn("1", result)

    def test_verus_math_int_literals_use_verus_suffixes(self):
        self.assertEqual(format_value_rust(5, "int"), "5int")
        self.assertEqual(format_value_rust(-3, "int"), "-3int")
        self.assertEqual(format_value_rust(-1, "nat"), "0nat")
        self.assertEqual(format_value_rust([1, 2], "Vec<int>"), "vec![1int, 2int]")

    def test_string(self):
        result = format_value_rust("hello", "String")
        self.assertIn("String::from", result)

    def test_string_escapes_quote(self):
        # 含 " 的字符串必须转义，否则整个 harness 编译失败
        result = format_value_rust('a"b', "String")
        self.assertEqual(result, 'String::from("a\\"b")')

    def test_string_escapes_backslash_and_newline(self):
        result = format_value_rust("c\\d\ne", "String")
        self.assertEqual(result, 'String::from("c\\\\d\\ne")')

    def test_str_input_is_quoted(self):
        # &str 输入参数必须产出带引号的字面量（此前落到裸文本 fallback）
        self.assertEqual(format_value_rust("hi", "&str"), '"hi"')
        self.assertEqual(format_value_rust('a"b', "&'static str"), '"a\\"b"')

    def test_tuple_input_formats_as_rust_tuple(self):
        self.assertEqual(format_value_rust((1, True), "(i32, bool)"), "(1_i32, true)")
        self.assertEqual(format_value_rust([1], "(i32,)"), "(1_i32,)")

    def test_extended_integer_types_are_suffixed(self):
        self.assertEqual(format_value_rust(5, "i128"), "5_i128")
        self.assertEqual(format_value_rust(5, "isize"), "5_isize")
        self.assertEqual(format_value_rust(5, "u128"), "5_u128")

    def test_f32_literal_is_clamped(self):
        result = format_value_rust(1.0e39, "f32")
        self.assertTrue(result.endswith("_f32"))
        self.assertNotIn("e+39", result)

    def test_option_input_formats_as_rust_option(self):
        self.assertEqual(format_value_rust(3, "Option<usize>"), "Some(3_usize)")
        self.assertEqual(format_value_rust(None, "Option<usize>"), "None")


class TestParseTypedValue(unittest.TestCase):
    def test_vec_bool(self):
        # harness 输出 Rust 的 true/false，ast.literal_eval 无法解析，需递归按 bool 解析
        self.assertEqual(parse_typed_value("[true, false, true]", "Vec<bool>"), [True, False, True])

    def test_vec_int(self):
        self.assertEqual(parse_typed_value("[1, 2, 3]", "Vec<i32>"), [1, 2, 3])

    def test_vec_char_ordinals(self):
        # Vec<char> 的 harness 输出为 unicode ordinals
        self.assertEqual(parse_typed_value("[97, 98]", "Vec<char>"), [97, 98])

    def test_nested_vec_bool(self):
        self.assertEqual(parse_typed_value("[[true], [false]]", "Vec<Vec<bool>>"), [[True], [False]])

    def test_tuple_with_bool(self):
        self.assertEqual(parse_typed_value("[1, true]", "(i32, bool)"), [1, True])

    def test_tuple_arity_mismatch_fails(self):
        from scripts.io.generate_io_tests_llm import _PARSE_FAIL
        self.assertIs(parse_typed_value("[1]", "(i32, bool)"), _PARSE_FAIL)

    def test_float_parse(self):
        self.assertEqual(parse_typed_value("1.5", "f32"), 1.5)


class TestParseResults(unittest.TestCase):
    def test_ok_result(self):
        stdout = "RESULT:0:OK:true\nRESULT:1:OK:42\nRESULT:2:PANIC\n"
        results = parse_results(stdout)
        self.assertEqual(results[0], ("OK", "true"))
        self.assertEqual(results[1], ("OK", "42"))
        self.assertEqual(results[2], ("PANIC", ""))

    def test_empty(self):
        results = parse_results("")
        self.assertEqual(len(results), 0)


class TestVerusNativeOutputParsing(unittest.TestCase):
    def test_existing_target_is_selected_by_input_parameter_names(self):
        code = """
verus! {
fn helper(digit: char) -> (r: bool) ensures r == (digit == '0') { digit == '0' }
fn solve(input: &Vec<char>) -> (r: usize) ensures r <= input.len() { input.len() }
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "multi.rs"
            source.write_text(code, encoding="utf-8")
            target = resolve_existing_target(
                str(source),
                [{"input": {"digit": "0"}, "expected": True, "unexpected": []}],
                "solve",
            )
        self.assertIsNotNone(target)
        self.assertEqual(target.func.name, "helper")

    def test_typed_input_key_normalizes_integer_float_seed(self):
        target = Target(
            func=FuncInfo("pair", [ParamInfo("x", "f32")], "f32", return_name="r"),
            context={
                "function": "pair",
                "parameters": [{"name": "x", "type": "f32"}],
                "returns": [{"name": "r", "type": "f32"}],
                "requires": [],
                "ensures": [],
            },
            raw_code="",
        )
        self.assertEqual(
            _typed_input_key(target, {"x": 0}),
            _typed_input_key(target, {"x": 0.0}),
        )

    def test_verus_math_runtime_type_detection(self):
        self.assertTrue(_has_verus_math_runtime_type(
            FuncInfo("f", [ParamInfo("a", "Vec<int>", inner_type="int")], "int")
        ))
        self.assertTrue(_has_verus_math_runtime_type(
            FuncInfo("f", [ParamInfo("n", "nat")], "usize")
        ))
        self.assertFalse(_has_verus_math_runtime_type(
            FuncInfo("f", [ParamInfo("n", "usize")], "i32")
        ))

    def test_harness_rewrites_verus_math_types_to_i128(self):
        # int/nat 签名任务不再整题跳过：native 副本把 exec 位置改写成 i128/u128 运行。
        func = FuncInfo(
            name="MaxDifference",
            params=[ParamInfo("a", "Vec<int>", inner_type="int")],
            return_type="int",
            return_name="diff",
        )
        target = Target(
            func=func,
            context={
                "function": "MaxDifference",
                "parameters": [{"name": "a", "type": "Vec<int>"}],
                "returns": [{"name": "diff", "type": "int"}],
                "requires": ["a.len() > 1"],
                "ensures": [],
            },
            raw_code="use vstd::prelude::*;\nverus! { fn MaxDifference(a: Vec<int>) -> (diff: int) { let mut m: int = 0; m } }",
        )
        harness = build_verus_harness(target, [{"a": [1, 2]}])
        self.assertIn("fn fmt_typed(v: i128)", harness)
        self.assertIn("fn MaxDifference(a: Vec<i128>) -> (diff: i128)", harness)
        # exec let 标注一并改写
        self.assertIn("let mut m: i128 = 0;", harness)
        self.assertIn("let p_a_0: Vec<i128> = vec![1_i128, 2_i128];", harness)

    def test_math_rewrite_keeps_ghost_positions(self):
        # ghost 代码（spec fn、量词 binder、子句）保持 int/nat：
        # Seq::index/subrange 等内建操作硬性要求 int 下标。
        code = (
            "verus! {\n"
            "spec fn triangle(n: nat) -> nat decreases n { if n == 0 { 0 } else { n + triangle((n - 1) as nat) } }\n"
            "fn minimum(a: Vec<int>) -> (m: int)\n"
            "    requires a.len() > 0,\n"
            "    ensures forall|i: int| 0 <= i && i < a.len() ==> m <= a[i],\n"
            "{\n"
            "    let mut m: int = a[0];\n"
            "    m\n"
            "}\n"
            "}\n"
        )
        rewritten = rewrite_verus_math_types(code)
        self.assertIn("fn minimum(a: Vec<i128>) -> (m: i128)", rewritten)
        self.assertIn("let mut m: i128 = a[0];", rewritten)
        self.assertIn("spec fn triangle(n: nat) -> nat", rewritten)
        self.assertIn("forall|i: int|", rewritten)

    def test_void_mut_output_uses_func_signature_when_context_loses_mut(self):
        func = FuncInfo(
            name="zap_negatives",
            params=[ParamInfo("a", "&mut Vec<i32>", is_ref=True, is_mut_ref=True, inner_type="i32")],
            return_type="()",
        )
        context = {
            "function": "zap_negatives",
            "parameters": [{"name": "a", "type": "&  Vec<i32>"}],
            "returns": [],
        }
        parsed = _parse_typed_output_to_dict(context, "[0, 2]", func)
        self.assertEqual(parsed, {"a": [0, 2]})
        self.assertEqual(format_typed_output_for_json(context, parsed, func), [0, 2])

    def test_runtime_invalid_is_recorded_separately(self):
        target = Target(
            func=FuncInfo("f", [ParamInfo("x", "i32")], "i32"),
            context={"function": "f", "parameters": [{"name": "x", "type": "i32"}], "returns": [], "requires": []},
            raw_code="",
        )
        cases, meta = collect_invalid_cases(target, [], [({"x": 1}, "timeout")], [], 5)
        self.assertEqual(cases, [])
        self.assertEqual(meta["timeout_invalid"], 1)

    def test_adaptive_invalid_targets(self):
        def make_target(param_type, requires):
            return Target(
                func=FuncInfo("f", [ParamInfo("x", param_type)], "bool"),
                context={
                    "function": "f",
                    "parameters": [{"name": "x", "type": param_type}],
                    "returns": [{"name": "r", "type": "bool"}],
                    "requires": [{"normalized": requires}],
                    "ensures": [],
                },
                raw_code="verus! { fn f(x: u64) -> bool { true } }",
            )

        self.assertEqual(_invalid_target_info(make_target("Vec<i32>", "x.len() > 0"), 5)["target"], 1)
        self.assertEqual(_invalid_target_info(make_target("u64", "x >= 2"), 5)["target"], 2)
        self.assertEqual(_invalid_target_info(make_target("u64", "1 < x"), 5)["target"], 2)
        self.assertEqual(_invalid_target_info(make_target("u64", "true"), 5)["target"], 0)

        runtime_bounds = make_target("&str", "x.len() <= i32::MAX")
        runtime_bounds.context["requires"].append({"normalized": "-x.len() >= i32::MIN"})
        self.assertEqual(_invalid_target_info(runtime_bounds, 5)["target"], 0)

    def test_positive_targets_respect_finite_or_unobservable_inputs(self):
        constant = Target(
            func=FuncInfo("constant", [], "i32", return_name="r"),
            context={"returns": [{"name": "r", "type": "i32"}], "ensures": [{"text": "r == 1"}]},
            raw_code="",
        )
        no_contract = Target(
            func=FuncInfo("noop", [], "()"),
            context={"returns": [], "ensures": []},
            raw_code="",
        )
        self.assertEqual(_positive_target_info(constant, 5)["target"], 1)
        self.assertEqual(_positive_target_info(no_contract, 5)["target"], 0)

        unbounded = Target(
            func=FuncInfo("bad", [ParamInfo("xs", "Vec<i32>")], "i32", return_name="r"),
            context={
                "returns": [{"name": "r", "type": "i32"}],
                "requires": [{"text": "forall |k:int| k <= xs[k] <= k + 1"}],
                "ensures": [{"text": "r >= 0"}],
            },
            raw_code="",
        )
        self.assertEqual(_positive_target_info(unbounded, 5)["reason"], "unbounded_sequence_precondition")
        self.assertEqual(_negative_target_info(unbounded, 5)["target"], 0)

    def test_negative_targets_skip_tautologies_and_unit_output(self):
        tautology = Target(
            func=FuncInfo("trace", [ParamInfo("x", "i32")], "i32", return_name="r"),
            context={"returns": [{"name": "r", "type": "i32"}], "ensures": [{"text": "true"}]},
            raw_code="",
        )
        unit = Target(
            func=FuncInfo("noop", [ParamInfo("x", "i32")], "()", return_name="r"),
            context={"returns": [{"name": "r", "type": "()"}], "ensures": [{"text": "r == ()"}]},
            raw_code="",
        )
        self.assertEqual(_negative_target_info(tautology, 5)["reason"], "tautological_ensures")
        self.assertEqual(_negative_target_info(unit, 5)["reason"], "no_observable_output")

    def test_mut_post_state_is_an_observable_negative_output(self):
        target = Target(
            func=FuncInfo(
                "zap",
                [ParamInfo("a", "&mut Vec<i32>", is_ref=True, is_mut_ref=True)],
                "()",
            ),
            context={
                "function": "zap",
                "parameters": [{"name": "a", "type": "&mut Vec<i32>"}],
                "returns": [],
                "requires": [],
                "ensures": [{"text": "a@ == old(a)@"}],
            },
            raw_code="",
        )
        context = _negative_context(target)
        self.assertEqual(context["returns"], [{"name": "a", "type": "Vec<i32>"}])
        self.assertEqual(context["_mutable_post_state_names"], ["a"])
        working = [{"input": {"a": "[1, 2]"}, "expected": [1, 2], "unexpected": []}]
        with patch(
            "scripts.io.generate_io_tests_llm.batch_verus_contract_check_detailed",
            side_effect=lambda _ctx, cases: {
                case["key"]: {"accepted": False, "reason": "verification_failed"}
                for case in cases
            },
        ):
            added, _log = _supplement_negatives_in_working(target, working, 1, 3)
        self.assertEqual(added, 1)
        self.assertTrue(working[0]["unexpected"])

    def test_negative_supplement_uses_sound_local_fallback(self):
        target = Target(
            func=FuncInfo("plus_one", [ParamInfo("x", "i32")], "i32", return_name="r"),
            context={
                "function": "plus_one",
                "parameters": [{"name": "x", "type": "i32"}],
                "returns": [{"name": "r", "type": "i32"}],
                "requires": [],
                "ensures": [{"text": "r == x + 1"}],
            },
            raw_code="",
        )
        working = [{"input": {"x": 1}, "expected": 2, "unexpected": []}]
        with patch(
            "scripts.io.generate_io_tests_llm.batch_verus_contract_check_detailed",
            side_effect=lambda _ctx, cases: {
                case["key"]: {"accepted": None, "reason": "verification_unresolved"}
                for case in cases
            },
        ):
            added, log = _supplement_negatives_in_working(target, working, 1, 3)
        self.assertEqual(added, 1)
        self.assertTrue(working[0]["unexpected"])
        self.assertEqual(log["verification_reasons"]["local_contract_ensures_not_satisfied"], 1)

    def test_negative_supplement_does_not_count_existing_mutation(self):
        target = Target(
            func=FuncInfo("plus_one", [ParamInfo("x", "i32")], "i32", return_name="r"),
            context={
                "function": "plus_one",
                "parameters": [{"name": "x", "type": "i32"}],
                "returns": [{"name": "r", "type": "i32"}],
                "requires": [],
                "ensures": [{"text": "r == x + 1"}],
            },
            raw_code="",
        )
        working = [{"input": {"x": 1}, "expected": 2, "unexpected": [3]}]
        added, _log = _supplement_negatives_in_working(target, working, 1, 3)
        self.assertEqual(added, 1)
        self.assertEqual(len(working[0]["unexpected"]), 2)

    def test_audit_merge_drops_removed_cases(self):
        current = {
            "positive": [{"case_key": "kept", "source": "current"}],
            "negative": [],
            "invalid": [],
        }
        existing = {
            "positive": [{"case_key": "kept", "source": "reviewed"}],
            "negative": [],
            "invalid": [{"case_key": "removed", "oracle": "requires_violation"}],
        }
        merged = _merge_audit(existing, current)
        self.assertEqual(merged["positive"], [{"case_key": "kept", "source": "reviewed"}])
        self.assertEqual(merged["invalid"], [])

    def test_str_reference_types_are_explicitly_supported(self):
        self.assertEqual(coerce_value_for_type("safe", "&str"), (True, "safe"))
        self.assertEqual(coerce_value_for_type("safe", "&'static str"), (True, "safe"))

    def test_flat_character_vectors_cover_realistic_text_inputs(self):
        value = list("abcde,fghijkl,mnopq")
        self.assertEqual(coerce_value_for_type(value, "Vec<char>"), (True, value))

    def test_multi_return_output_parses_rust_bool(self):
        context = {
            "returns": [
                {"name": "left", "type": "i32"},
                {"name": "right", "type": "bool"},
            ]
        }
        self.assertEqual(
            _parse_typed_output_to_dict(context, "[1, true]"),
            {"left": 1, "right": True},
        )

    def test_tuple_fmt_signature_is_valid_when_type_is_complete(self):
        self.assertIn("fn fmt_typed(v: (usize, usize)) -> String", gen_fmt_fn("(usize, usize)"))

    def test_ref_vec_fmt_signature_accepts_reference(self):
        harness = gen_fmt_fn("&Vec<i32>")
        self.assertIn("fn fmt_typed(v: &Vec<i32>) -> String", harness)
        self.assertIn("v.iter()", harness)

    def test_vec_bool_formatter_dereferences_iter_item(self):
        expr = gen_fmt_expr("Vec<bool>", "v")
        self.assertIn("if *x", expr)

    def test_runtime_type_support_issue(self):
        self.assertIsNone(runtime_type_support_issue("i128"))
        self.assertIsNone(runtime_type_support_issue("u128"))
        self.assertIsNone(runtime_type_support_issue("Option<(i32, bool)>"))
        self.assertEqual(runtime_type_support_issue("NumpyOperand"), "custom_runtime_type")
        # 定长数组按元素类型递归判定
        self.assertIsNone(runtime_type_support_issue("[f32; 2]"))
        self.assertIsNone(runtime_type_support_issue("&[bool; 10]"))
        self.assertEqual(runtime_type_support_issue("[NumpyOperand; 2]"), "custom_runtime_type")
        # Seq 是数学类型：exec harness 无法构造其值
        self.assertEqual(runtime_type_support_issue("Seq<i32>"), "verus_math_runtime_type")
        # 单字母泛型参数可单态化运行
        self.assertIsNone(runtime_type_support_issue("Vec<T>"))
        # 文件内注册的 enum/struct 按定义递归判定
        registry = {
            "DType": {"kind": "enum", "variants": ["Int8", "Int16"]},
            "TimeDelta64": {"kind": "struct", "fields": [("value", "i64"), ("unit", "DType")], "tuple": False},
            "Bad": {"kind": "struct", "fields": [("x", "NumpyOperand")], "tuple": False},
            "Matrix": {"kind": "alias", "target": "Vec<Vec<i8>>"},
        }
        self.assertIsNone(runtime_type_support_issue("DType", registry))
        self.assertIsNone(runtime_type_support_issue("TimeDelta64", registry))
        self.assertIsNone(runtime_type_support_issue("Matrix", registry))
        self.assertEqual(runtime_type_support_issue("Bad", registry), "custom_runtime_type")

    def test_requires_driven_candidates(self):
        context = {
            "parameters": [
                {"name": "x", "type": "i32"},
                {"name": "a", "type": "Vec<i32>"},
                {"name": "i", "type": "usize"},
                {"name": "n", "type": "i32"},
            ],
            "requires": [
                "x > 0",
                "i < a.len()",
                "n % 2 == 0",
            ],
        }
        cases = generate_requires_satisfying_candidate_inputs(context, budget=6)
        self.assertTrue(any(case["x"] > 0 for case in cases))
        self.assertTrue(any(isinstance(case["a"], list) and len(case["a"]) > 0 and case["i"] < len(case["a"]) for case in cases))
        self.assertTrue(any(case["n"] % 2 == 0 for case in cases))

    def test_type_predicates_ignore_nested_scalars(self):
        from metrics_rebuild.share.contract_eval import (
            coerce_value_for_type,
            is_bool_type,
            is_char_type,
            is_float_type,
            is_int_like_type,
            is_vec_type,
        )

        self.assertTrue(is_bool_type("bool"))
        self.assertFalse(is_bool_type("Vec<bool>"))
        self.assertFalse(is_bool_type("Option<bool>"))
        self.assertTrue(is_vec_type("Vec<bool>"))
        self.assertTrue(is_int_like_type("i32"))
        self.assertFalse(is_int_like_type("Vec<i32>"))
        self.assertTrue(is_float_type("f32"))
        self.assertFalse(is_float_type("Vec<f32>"))
        self.assertFalse(is_int_like_type("f32"))
        self.assertTrue(is_char_type("char"))
        self.assertFalse(is_char_type("Vec<char>"))
        self.assertTrue(coerce_value_for_type(1.5, "f32")[0])
        self.assertTrue(coerce_value_for_type([1.0, 2.5], "Vec<f32>")[0])

    def test_equal_len_requires_candidates_and_violations(self):
        from metrics_rebuild.share.io_cases import generate_requires_violating_from_positives

        context = {
            "function": "zippy",
            "parameters": [
                {"name": "a", "type": "Vec<i32>"},
                {"name": "b", "type": "Vec<i32>"},
            ],
            "requires": [{"normalized": "a . len ( ) == b . len ( )"}],
            "returns": [],
            "ensures": [],
        }
        sat = generate_requires_satisfying_candidate_inputs(context, budget=6)
        self.assertTrue(sat)
        self.assertTrue(all(len(case["a"]) == len(case["b"]) for case in sat))
        viol = generate_requires_violating_from_positives(
            context, [{"a": [1, 2], "b": [3, 4]}], budget=8,
        )
        self.assertTrue(any(len(case["a"]) != len(case["b"]) for case in viol))

    def test_mutate_output_handles_vec_bool_and_char(self):
        from metrics_rebuild.share.io_cases import mutate_output_values

        vec_ctx = {
            "function": "logical_or",
            "parameters": [
                {"name": "x1", "type": "Vec<bool>"},
                {"name": "x2", "type": "Vec<bool>"},
            ],
            "returns": [{"name": "result", "type": "Vec<bool>"}],
            "requires": [],
            "ensures": [],
        }
        muts = mutate_output_values(
            vec_ctx,
            {"x1": [True, False], "x2": [False, True]},
            {"result": [True, True]},
        )
        self.assertTrue(any(isinstance(m["result"], list) for m, _ in muts))
        self.assertFalse(any(m["result"] is False for m, _ in muts))

        char_ctx = {
            "function": "next_char",
            "parameters": [{"name": "c", "type": "char"}],
            "returns": [{"name": "d", "type": "char"}],
            "requires": [],
            "ensures": [],
        }
        char_muts = mutate_output_values(char_ctx, {"c": ord("a")}, {"d": ord("b")})
        self.assertTrue(char_muts)
        self.assertTrue(any(isinstance(m["d"], int) and m["d"] != ord("b") for m, _ in char_muts))

    def test_mutate_output_recurses_into_tuple_and_nested_vec(self):
        from metrics_rebuild.share.io_cases import mutate_output_values

        tuple_ctx = {"returns": [{"name": "r", "type": "(Vec<i32>, bool)"}]}
        tuple_muts = mutate_output_values(tuple_ctx, {}, {"r": [[1, 2], True]})
        self.assertTrue(any(label.startswith("tuple_0_") for _value, label in tuple_muts))
        self.assertTrue(any(label.startswith("tuple_1_") for _value, label in tuple_muts))

        nested_ctx = {"returns": [{"name": "r", "type": "Vec<Vec<i32>>"}]}
        nested_muts = mutate_output_values(nested_ctx, {}, {"r": [[1, 2], [3]]})
        self.assertTrue(any(label.startswith("nested_vec_first_") for _value, label in nested_muts))

    def test_process_one_marks_unsupported_runtime_type(self):
        # NumpyObject 未在文件内定义（注册表查不到），保持整题 unsupported。
        # 注意：文件内定义的简单 struct/enum 现在是受支持类型（见 type_defs）。
        code = """
verus! {
fn eval_operand(x: NumpyObject) -> (r: i32) { 0 }
}
"""
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "case.rs"
            out = Path(d) / "out"
            src.write_text(code)
            result = process_one(str(src), out, 5, 3, "verus")
            self.assertEqual(result["status"], "unsupported_runtime_type")
            meta = __import__("json").loads((out / "meta.json").read_text())
            self.assertEqual(meta["schema_version"], 2)
            self.assertEqual(meta["category_status"]["positive"]["state"], "blocked")

    def test_summary_refreshes_counts_from_test_json(self):
        import json

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            case_dir = root / "Bench" / "case"
            case_dir.mkdir(parents=True)
            (case_dir / "meta.json").write_text(json.dumps({
                "status": "ok", "positive_count": 1, "negative_count": 0, "invalid_count": 0,
            }))
            (case_dir / "test.json").write_text(json.dumps([
                {"input": {"x": 1}, "expected": 1, "unexpected": [2]},
                {"input": {"x": 0}, "expected": "INVALID_INPUT", "unexpected": []},
            ]))
            (root / "summary.json").write_text(json.dumps({"results": [
                {"name": "case", "status": "partial", "positive": 99},
                {"name": "legacy_no_function", "status": "no_function", "positive": 0,
                 "negative": 0, "invalid": 0},
            ]}))
            _write_summary(root, [], append_existing=True)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(
                [row["name"] for row in summary["results"]],
                ["case", "legacy_no_function"],
            )
            by_name = {row["name"]: row for row in summary["results"]}
            self.assertEqual(by_name["case"]["positive"], 1)
            self.assertEqual(by_name["case"]["negative"], 1)
            self.assertEqual(by_name["case"]["invalid"], 1)
            self.assertEqual(by_name["case"]["status"], "ok")
            self.assertEqual(by_name["legacy_no_function"]["status"], "no_function")
            self.assertEqual(summary["stats"], {"ok": 1, "no_function": 1})

    def test_supplement_summary_rebuilds_untouched_tasks(self):
        import json

        from scripts.io.suite_summary import _rewrite_summary

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name, status, cases in (
                (
                    "changed",
                    "ok",
                    [{"input": {"x": 1}, "expected": 1, "unexpected": [2]}],
                ),
                (
                    "untouched",
                    "partial",
                    [
                        {"input": {"x": 2}, "expected": 2, "unexpected": [3, 4]},
                        {"input": {"x": 0}, "expected": "INVALID_INPUT", "unexpected": []},
                    ],
                ),
            ):
                case_dir = root / "Bench" / name
                case_dir.mkdir(parents=True)
                (case_dir / "meta.json").write_text(json.dumps({
                    "status": status,
                    "positive_count": 99,
                    "negative_count": 99,
                    "invalid_count": 99,
                }))
                (case_dir / "test.json").write_text(json.dumps(cases))

            (root / "summary.json").write_text(json.dumps({"results": [
                {"name": "changed", "status": "ok", "positive": 7, "negative": 7, "invalid": 7},
                {"name": "untouched", "status": "ok", "positive": 8, "negative": 8, "invalid": 8},
                {"name": "legacy_no_function", "status": "no_function", "positive": 0,
                 "negative": 0, "invalid": 0},
            ]}))

            _rewrite_summary(root, {
                "changed": {"name": "changed", "status": "ok", "positive": 5,
                            "negative": 5, "invalid": 5},
            })

            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(
                [row["name"] for row in summary["results"]],
                ["changed", "untouched", "legacy_no_function"],
            )
            by_name = {row["name"]: row for row in summary["results"]}
            self.assertEqual(
                by_name["changed"],
                {"name": "changed", "status": "ok", "positive": 1, "negative": 1, "invalid": 0},
            )
            self.assertEqual(
                by_name["untouched"],
                {"name": "untouched", "status": "partial", "positive": 1, "negative": 2, "invalid": 1},
            )
            self.assertEqual(by_name["legacy_no_function"]["status"], "no_function")
            self.assertEqual(summary["stats"], {"ok": 1, "no_function": 1, "partial": 1})

    def test_offline_loader_uses_meta_function_context(self):
        import json
        from metrics_rebuild.share.io_cases import (
            load_offline_io_suite,
            set_io_suite_dir,
            source_hash,
        )

        helper = {
            "function": "helper", "parameters": [{"name": "x", "type": "i32"}],
            "returns": [{"name": "r", "type": "bool"}], "requires": [], "ensures": [],
            "has_contract": True,
        }
        target = {
            "function": "target", "parameters": [{"name": "x", "type": "i32"}],
            "returns": [{"name": "r", "type": "i32"}], "requires": [], "ensures": [],
            "has_contract": True,
        }
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ref_dir = root / "verified" / "VeriCoding"
            ref_dir.mkdir(parents=True)
            ref = ref_dir / "case.rs"
            ref.write_text("""
verus! {
fn helper(x: i32) -> (r: bool) { true }
// <vc-spec>
fn target(x: i32) -> (r: i32)
// </vc-spec>
// <vc-code>
{ x }
// </vc-code>
}
""")
            suite_dir = root / "suite" / "VeriCoding" / "case"
            suite_dir.mkdir(parents=True)
            (suite_dir / "test.json").write_text(json.dumps([
                {"input": {"x": 1}, "expected": 1, "unexpected": [2]},
            ]))
            (suite_dir / "meta.json").write_text(json.dumps({
                "function": "target", "source_hash": source_hash(str(ref)), "status": "ok",
            }))
            set_io_suite_dir(str(root / "suite"))
            try:
                with patch("metrics_rebuild.share.io_cases.strength_contexts_for_path", return_value=[helper, target]):
                    suite = load_offline_io_suite(str(ref))
            finally:
                set_io_suite_dir(None)
        self.assertIsNotNone(suite)
        self.assertEqual(suite["function"], "target")

    def test_schema_v3_loader_fails_loud_when_case_audit_is_missing(self):
        import json
        from metrics_rebuild.share.io_cases import (
            _audit_case_key,
            load_offline_io_suite,
            set_io_suite_dir,
            source_hash,
        )

        source = """
verus! {
// <vc-spec>
fn target(x: i32) -> (r: i32)
    requires x > 0,
    ensures r == x,
// </vc-spec>
{ x }
}
"""
        positive = {"input": {"x": 1}, "expected": 1, "unexpected": [2]}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reference_dir = root / "verified" / "Bench"
            reference_dir.mkdir(parents=True)
            reference = reference_dir / "case.rs"
            reference.write_text(source)
            suite_dir = root / "suite" / "Bench" / "case"
            suite_dir.mkdir(parents=True)
            (suite_dir / "test.json").write_text(json.dumps([positive]))
            meta = {
                "schema_version": 3,
                "validation_complete": True,
                "function": "target",
                "parameters": [{"name": "x", "type": "i32"}],
                "returns": [{"name": "r", "type": "i32"}],
                "return_type": "i32",
                "source_hash": source_hash(str(reference)),
                "positive_count": 1,
                "negative_count": 1,
                "invalid_count": 0,
                "category_status": {
                    "positive": {"target": 1, "count": 1, "state": "complete"},
                    "negative": {"target": 1, "count": 1, "state": "complete"},
                    "invalid": {"target": 0, "count": 0, "state": "not_applicable"},
                },
                "case_audit": {
                    "positive": [{
                        "case_key": _audit_case_key({"input": positive["input"], "expected": 1}),
                        "state": "validated", "function": "target",
                        "oracle": "reference_execution_exact",
                        "engine": "verus_native_runtime+verus_requires_dual_proof",
                        "requires_verdict": True, "runtime_status": "OK",
                    }],
                    "negative": [{
                        "case_key": _audit_case_key({"input": positive["input"], "unexpected": 2}),
                        "state": "validated", "function": "target",
                        "oracle": "reference_contract_rejection", "engine": "verus_contract_dual_proof",
                        "contract_verdict": False,
                    }],
                    "invalid": [],
                },
            }
            (suite_dir / "meta.json").write_text(json.dumps(meta))
            set_io_suite_dir(str(root / "suite"))
            try:
                self.assertIsNotNone(load_offline_io_suite(str(reference)))
                meta["case_audit"]["positive"] = []
                (suite_dir / "meta.json").write_text(json.dumps(meta))
                self.assertIsNone(load_offline_io_suite(str(reference)))
            finally:
                set_io_suite_dir(None)

    def test_process_one_marks_no_runtime_results(self):
        code = "verus! { fn f(x: i32) -> (r: i32) { x } }"
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "case.rs"
            out = Path(d) / "out"
            src.write_text(code)
            empty = RunOutcome(ok=[], panic=[], runtime_invalid=[], failure_reason="ok",
                               diagnostics={"compile_reason": "ok", "raw_result_count": 0})
            with patch("scripts.io.generate_io_tests_llm.llm_candidate_inputs", return_value=([], {"status": "skipped"})):
                with patch("scripts.io.generate_io_tests_llm.run_verus_compile_batch", return_value=empty):
                    result = process_one(str(src), out, 5, 3, "verus")
            self.assertEqual(result["status"], "no_runtime_results")

    def test_resolve_target_backfills_missing_context_returns(self):
        from scripts.io.generate_io_tests_llm import resolve_target
        code = "verus! { fn eligible_u8(x: u8) -> (r: bool) { x > 0 } }"
        context = {
            "function": "eligible_u8",
            "parameters": [{"name": "x", "type": "u8"}],
            "returns": [],
            "requires": [],
            "ensures": [],
        }
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "case.rs"
            src.write_text(code)
            with patch("scripts.io.generate_io_tests_llm.strength_contexts_for_path", return_value=[context]):
                target = resolve_target(str(src))
        self.assertIsNotNone(target)
        self.assertEqual(target.context["returns"], [{"name": "r", "type": "bool"}])


class TestIOEvaluationV2(unittest.TestCase):
    @staticmethod
    def _suite(*, parameters=None, returns=None, return_type="i32", cases=None, category_status=None):
        parameters = parameters or [{"name": "x", "type": "i32"}]
        returns = returns if returns is not None else [{"name": "r", "type": return_type}]
        return {
            "function": "target",
            "target": {
                "function": "target",
                "parameters": parameters,
                "returns": returns,
                "return_type": return_type,
            },
            "category_status": category_status or {},
            "cases": cases if cases is not None else [{
                "id": "pos_001",
                "kind": "positive",
                "inputs": {"x": 1},
                "output": {"r": 1},
                "status": "validated",
            }],
        }

    def test_missing_exact_generated_target_scores_zero(self):
        from metrics_rebuild.share.io_cases import score_io_cases

        helper = {
            "function": "helper", "parameters": [{"name": "x", "type": "i32"}],
            "returns": [{"name": "r", "type": "i32"}], "requires": [], "ensures": [],
        }
        with tempfile.TemporaryDirectory() as d:
            generated = Path(d) / "gen.rs"
            generated.write_text("verus! { fn helper(x: i32) -> (r: i32) { x } }")
            with patch("metrics_rebuild.share.io_cases.strength_contexts_for_path", return_value=[helper]):
                result = score_io_cases(str(generated), self._suite(), "positive")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["status"], "target_mismatch")
        self.assertEqual(result["reason"], "target_function_not_found")
        self.assertEqual((result["failed"], result["unknown"]), (0, 1))
        self.assertEqual(result["coverage"], 0.0)

    def test_ground_marker_overrides_stale_preferred_metadata(self):
        from metrics_rebuild.share.functions import resolve_pair_target

        code = """
verus! {
fn helper(x: i32) -> (r: i32) ensures r == x { x }
// <vc-spec>
fn solve(x: i32) -> (r: i32) ensures r == x
// </vc-spec>
{ x }
}
"""
        with tempfile.TemporaryDirectory() as d:
            reference = Path(d) / "ref.rs"
            generated = Path(d) / "gen.rs"
            reference.write_text(code)
            generated.write_text(code)
            alignment = resolve_pair_target(
                str(generated), str(reference), preferred_name="helper",
            )
        self.assertEqual(alignment["status"], "ok")
        self.assertEqual(alignment["function"], "solve")
        self.assertIn("preferred_metadata_points_to_different_function", alignment["diagnostics"])

    def test_invalid_accepts_requires_rejection_or_explicit_panic(self):
        from metrics_rebuild.share.io_cases import score_io_cases

        generated_context = {
            "function": "target", "parameters": [{"name": "x", "type": "i32"}],
            "returns": [{"name": "r", "type": "i32"}], "requires": [], "ensures": [],
        }
        suite = self._suite(cases=[
            {"id": "inv_1", "kind": "invalid", "inputs": {"x": 1}, "status": "validated"},
            {"id": "inv_2", "kind": "invalid", "inputs": {"x": 2}, "status": "validated"},
            {"id": "inv_3", "kind": "invalid", "inputs": {"x": 3}, "status": "validated"},
        ])
        with tempfile.TemporaryDirectory() as d:
            generated = Path(d) / "gen.rs"
            generated.write_text("verus! { fn target(x: i32) -> (r: i32) { x } }")
            with patch("metrics_rebuild.share.io_cases.strength_contexts_for_path", return_value=[generated_context]), \
                    patch("metrics_rebuild.share.io_cases.batch_verus_requires_decide", return_value={
                        "sc_0": False, "sc_1": True, "sc_2": True,
                    }), \
                    patch("metrics_rebuild.share.io_cases._run_generated_invalid_inputs", return_value={
                        "sc_1": {"status": "PANIC", "reason": "ok"},
                        "sc_2": {"status": "OK", "reason": "ok"},
                    }):
                result = score_io_cases(str(generated), suite, "invalid")
        self.assertEqual(result["passed"], 2)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["unknown"], 0)

    def test_invalid_timeout_is_unknown_not_passed(self):
        from metrics_rebuild.share.io_cases import score_io_cases

        generated_context = {
            "function": "target", "parameters": [{"name": "x", "type": "i32"}],
            "returns": [{"name": "r", "type": "i32"}], "requires": [], "ensures": [],
        }
        suite = self._suite(cases=[
            {"id": "inv_1", "kind": "invalid", "inputs": {"x": 1}, "status": "validated"},
        ])
        with tempfile.TemporaryDirectory() as d:
            generated = Path(d) / "gen.rs"
            generated.write_text("verus! { fn target(x: i32) -> (r: i32) { x } }")
            with patch("metrics_rebuild.share.io_cases.strength_contexts_for_path", return_value=[generated_context]), \
                    patch("metrics_rebuild.share.io_cases.batch_verus_requires_decide", return_value={"sc_0": True}), \
                    patch("metrics_rebuild.share.io_cases._run_generated_invalid_inputs", return_value={
                        "sc_0": {"status": "TIMEOUT", "reason": "ok"},
                    }):
                result = score_io_cases(str(generated), suite, "invalid")
        self.assertEqual(result["passed"], 0)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["unknown"], 1)

    def test_invalid_compile_unknown_is_unknown_not_passed(self):
        from metrics_rebuild.share.io_cases import score_io_cases

        generated_context = {
            "function": "target", "parameters": [{"name": "x", "type": "i32"}],
            "returns": [{"name": "r", "type": "i32"}], "requires": [], "ensures": [],
        }
        suite = self._suite(cases=[
            {"id": "inv_1", "kind": "invalid", "inputs": {"x": 1}, "status": "validated"},
        ])
        with tempfile.TemporaryDirectory() as d:
            generated = Path(d) / "gen.rs"
            generated.write_text("verus! { fn target(x: i32) -> (r: i32) { x } }")
            with patch("metrics_rebuild.share.io_cases.strength_contexts_for_path", return_value=[generated_context]), \
                    patch("metrics_rebuild.share.io_cases.batch_verus_requires_decide", return_value={"sc_0": None}), \
                    patch("metrics_rebuild.share.io_cases._run_generated_invalid_inputs", return_value={
                        "sc_0": {"status": "UNKNOWN", "reason": "compile_failed"},
                    }):
                result = score_io_cases(str(generated), suite, "invalid")
        self.assertEqual(result["passed"], 0)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["unknown"], 1)

    def test_invalid_requires_unknown_but_runtime_ok_is_failed(self):
        from metrics_rebuild.share.io_cases import score_io_cases

        generated_context = {
            "function": "target", "parameters": [{"name": "x", "type": "i32"}],
            "returns": [{"name": "r", "type": "i32"}], "requires": [], "ensures": [],
        }
        suite = self._suite(cases=[
            {"id": "inv_1", "kind": "invalid", "inputs": {"x": 1}, "status": "validated"},
        ])
        with tempfile.TemporaryDirectory() as d:
            generated = Path(d) / "gen.rs"
            generated.write_text("verus! { fn target(x: i32) -> (r: i32) { x } }")
            with patch("metrics_rebuild.share.io_cases.strength_contexts_for_path", return_value=[generated_context]), \
                    patch("metrics_rebuild.share.io_cases.batch_verus_requires_decide", return_value={"sc_0": None}), \
                    patch("metrics_rebuild.share.io_cases._run_generated_invalid_inputs", return_value={
                        "sc_0": {"status": "OK", "reason": "ok"},
                    }):
                result = score_io_cases(str(generated), suite, "invalid")
        self.assertEqual(result["passed"], 0)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["unknown"], 0)

    def test_generated_target_signature_type_mismatch_is_unknown_without_verus(self):
        from metrics_rebuild.share.io_cases import score_io_cases

        generated_context = {
            "function": "target", "parameters": [{"name": "y", "type": "u64"}],
            "returns": [{"name": "z", "type": "i32"}], "requires": [], "ensures": [],
        }
        with tempfile.TemporaryDirectory() as d:
            generated = Path(d) / "gen.rs"
            generated.write_text("verus! { fn target(y: u64) -> (z: i32) { 0 } }")
            with patch("metrics_rebuild.share.io_cases.strength_contexts_for_path", return_value=[generated_context]), \
                    patch("metrics_rebuild.share.io_cases.batch_verus_contract_decide_detailed") as decide:
                    result = score_io_cases(str(generated), self._suite(), "positive")
        decide.assert_not_called()
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["unknown"], 1)
        self.assertEqual(result["coverage"], 0.0)
        self.assertEqual(result["reason"], "signature_type_mismatch")

    def test_generated_vc_marker_mismatch_is_evaluated(self):
        from metrics_rebuild.share.io_cases import score_io_cases

        generated_context = {
            "function": "target", "parameters": [{"name": "x", "type": "i32"}],
            "returns": [{"name": "r", "type": "i32"}], "requires": [], "ensures": [],
        }
        code = """
verus! {
// <vc-spec>
fn other(x: i32) -> (r: i32) { x }
// </vc-spec>
fn target(x: i32) -> (r: i32) { x }
}
"""
        with tempfile.TemporaryDirectory() as d:
            generated = Path(d) / "gen.rs"
            generated.write_text(code)
            with patch("metrics_rebuild.share.io_cases.strength_contexts_for_path", return_value=[generated_context]):
                with patch("metrics_rebuild.share.io_cases.batch_verus_contract_decide_detailed", return_value={
                    "sc_0": {"accepted": True, "reason": "accepted", "engine": "unit_test"},
                }):
                    result = score_io_cases(str(generated), self._suite(), "positive")
        self.assertEqual(result["score"], 1.0)
        self.assertNotEqual(result["status"], "target_mismatch")

    def test_not_applicable_category_is_not_scored(self):
        from metrics_rebuild.share.io_cases import score_io_cases

        generated_context = {
            "function": "target", "parameters": [{"name": "x", "type": "i32"}],
            "returns": [{"name": "r", "type": "i32"}], "requires": [], "ensures": [],
        }
        suite = self._suite(
            cases=[],
            category_status={
                "negative": {"target": 0, "count": 0, "state": "not_applicable", "reason": "no_ensures"},
            },
        )
        with tempfile.TemporaryDirectory() as d:
            generated = Path(d) / "gen.rs"
            generated.write_text("verus! { fn target(x: i32) -> (r: i32) { x } }")
            with patch("metrics_rebuild.share.io_cases.strength_contexts_for_path", return_value=[generated_context]):
                result = score_io_cases(str(generated), suite, "negative")
        self.assertIsNone(result["score"])
        self.assertEqual(result["reason"], "category_not_applicable")

    def test_validated_cases_override_generation_category_status(self):
        from metrics_rebuild.share.io_cases import score_io_cases

        context = {
            "function": "target", "parameters": [{"name": "x", "type": "i32"}],
            "returns": [{"name": "r", "type": "i32"}], "requires": [], "ensures": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "gen.rs"
            generated.write_text("verus! { fn target(x: i32) -> (r: i32) { x } }")
            for state in ("blocked", "not_applicable"):
                with self.subTest(state=state):
                    suite = self._suite(category_status={
                        "positive": {"state": state, "target": 5, "count": 1},
                    })
                    with patch("metrics_rebuild.share.io_cases.strength_contexts_for_path", return_value=[context]):
                        result = score_io_cases(str(generated), suite, "positive")
                    self.assertEqual(result["total"], 1)
                    self.assertEqual(result["score"], 1.0)

    def test_offline_scalar_parser_preserves_whitespace_chars(self):
        from metrics_rebuild.share.io_cases import _parse_benchmark_value

        self.assertEqual(_parse_benchmark_value(" "), " ")
        self.assertEqual(_parse_benchmark_value("\n"), "\n")

    def test_offline_input_parser_preserves_numeric_and_newline_strings(self):
        from metrics_rebuild.share.io_cases import _parse_benchmark_inputs

        context = {
            "parameters": [
                {"name": "text", "type": "&str"},
                {"name": "ch", "type": "char"},
                {"name": "number", "type": "i32"},
            ],
        }
        parsed = _parse_benchmark_inputs(context, {
            "text": "123", "ch": "\n", "number": "123",
        })
        self.assertEqual(parsed, {"text": "123", "ch": "\n", "number": 123})

    def test_offline_loader_maps_mut_post_state_to_output(self):
        import json
        from metrics_rebuild.share.io_cases import load_offline_io_suite, set_io_suite_dir, source_hash

        parsed_context = {
            "function": "zap", "parameters": [{"name": "a", "type": "& Vec<i32>"}],
            "returns": [], "requires": [], "ensures": [{"text": "a@ == old(a)@"}],
        }
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ref_dir = root / "verified" / "Bench"
            ref_dir.mkdir(parents=True)
            reference = ref_dir / "case.rs"
            reference.write_text("verus! { fn zap(a: &mut Vec<i32>) { } }")
            suite_dir = root / "suite" / "Bench" / "case"
            suite_dir.mkdir(parents=True)
            (suite_dir / "test.json").write_text(json.dumps([{
                "input": {"a": [1, 2]},
                "expected": {"a": [1, 2]},
                "unexpected": [{"a": [2, 1]}],
            }]))
            (suite_dir / "meta.json").write_text(json.dumps({
                "schema_version": 2,
                "function": "zap",
                "parameters": [{"name": "a", "type": "&mut Vec<i32>"}],
                "returns": [],
                "return_type": "()",
                "source_hash": source_hash(str(reference)),
                "category_status": {
                    "positive": {"target": 5, "count": 1, "state": "partial", "reason": None},
                    "negative": {"target": 5, "count": 1, "state": "partial", "reason": None},
                },
            }))
            set_io_suite_dir(str(root / "suite"))
            try:
                with patch("metrics_rebuild.share.io_cases.strength_contexts_for_path", return_value=[parsed_context]):
                    suite = load_offline_io_suite(str(reference))
            finally:
                set_io_suite_dir(None)
        self.assertEqual(suite["target"]["parameters"][0]["type"], "&mut Vec<i32>")
        positive = next(case for case in suite["cases"] if case["kind"] == "positive")
        negative = next(case for case in suite["cases"] if case["kind"] == "negative")
        self.assertEqual(positive["output"], {"a": [1, 2]})
        self.assertEqual(negative["mutated_output"], {"a": [2, 1]})

    def test_score_uses_mut_post_state_as_observable_output(self):
        from metrics_rebuild.share.io_cases import score_io_cases

        generated_context = {
            "function": "target", "parameters": [{"name": "a", "type": "& Vec<i32>"}],
            "returns": [], "requires": [], "ensures": [{"text": "a@ == old(a)@"}],
        }
        suite = self._suite(
            parameters=[{"name": "a", "type": "&mut Vec<i32>"}],
            returns=[],
            return_type="()",
            cases=[{
                "id": "pos_001", "kind": "positive", "inputs": {"a": [1, 2]},
                "output": {"a": [1, 2]}, "status": "validated",
            }],
        )
        observed = {}

        def accept(context, cases):
            observed.update(context)
            return {
                case["key"]: {"accepted": True, "reason": "accepted", "engine": "unit_test"}
                for case in cases
            }

        with tempfile.TemporaryDirectory() as d:
            generated = Path(d) / "gen.rs"
            generated.write_text("verus! { fn target(a: &mut Vec<i32>) { } }")
            with patch("metrics_rebuild.share.io_cases.strength_contexts_for_path", return_value=[generated_context]):
                with patch("metrics_rebuild.share.io_cases.batch_verus_contract_decide_detailed", side_effect=accept):
                    result = score_io_cases(str(generated), suite, "positive")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(observed["returns"], [{"name": "a", "type": "Vec<i32>"}])
        self.assertEqual(observed["_mutable_post_state_names"], ["a"])

    def test_contract_unknown_stays_unknown_and_counts_in_all_case_score(self):
        from metrics_rebuild.share.io_cases import score_io_cases

        generated_context = {
            "function": "target", "parameters": [{"name": "x", "type": "i32"}],
            "returns": [{"name": "r", "type": "i32"}], "requires": [],
            "ensures": [{"text": "r == x"}],
        }
        suite = self._suite(cases=[
            {"id": "pos_1", "kind": "positive", "inputs": {"x": 1}, "output": {"r": 1}, "status": "validated"},
            {"id": "pos_2", "kind": "positive", "inputs": {"x": 2}, "output": {"r": 2}, "status": "validated"},
            {"id": "pos_3", "kind": "positive", "inputs": {"x": 3}, "output": {"r": 3}, "status": "validated"},
        ])
        detailed = {
            "sc_0": {"accepted": True, "reason": "accepted", "engine": "unit_test"},
            "sc_1": {"accepted": None, "reason": "compile_error", "engine": "unit_test"},
            "sc_2": {"accepted": False, "reason": "contract_rejected", "engine": "unit_test"},
        }
        with tempfile.TemporaryDirectory() as d:
            generated = Path(d) / "gen.rs"
            generated.write_text("verus! { fn target(x: i32) -> (r: i32) ensures r == x { x } }")
            with patch("metrics_rebuild.share.io_cases.strength_contexts_for_path", return_value=[generated_context]), \
                    patch("metrics_rebuild.share.io_cases.batch_verus_contract_decide_detailed", return_value=detailed):
                result = score_io_cases(str(generated), suite, "positive")

        self.assertEqual((result["passed"], result["failed"], result["unknown"]), (1, 1, 1))
        self.assertAlmostEqual(result["score"], 1 / 3)
        self.assertAlmostEqual(result["coverage"], 2 / 3)
        self.assertEqual(result["decidable_score"], 0.5)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["unknown_reason_summary"], {"compile_error": 1})
        self.assertIsNone(result["details"][1]["success"])

    def test_missing_postcondition_accepts_every_admitted_pair(self):
        from metrics_rebuild.share.io_cases import score_io_cases

        generated_context = {
            "function": "target", "parameters": [{"name": "x", "type": "i32"}],
            "returns": [{"name": "r", "type": "i32"}], "requires": [], "ensures": [],
        }
        suite = self._suite(cases=[
            {"id": "pos_001", "kind": "positive", "inputs": {"x": 1}, "output": {"r": 1}, "status": "validated"},
            {"id": "neg_001_1", "kind": "negative", "inputs": {"x": 1}, "mutated_output": {"r": 9}, "status": "validated"},
        ])
        with tempfile.TemporaryDirectory() as d:
            generated = Path(d) / "gen.rs"
            generated.write_text("verus! { fn target(x: i32) -> (r: i32) { x } }")
            with patch("metrics_rebuild.share.io_cases.strength_contexts_for_path", return_value=[generated_context]):
                positive = score_io_cases(str(generated), suite, "positive")
                negative = score_io_cases(str(generated), suite, "negative")
        # Without ensures the postcondition is true, so the wrong output is accepted.
        self.assertEqual((positive["passed"], positive["failed"], positive["unknown"]), (1, 0, 0))
        self.assertEqual((negative["passed"], negative["failed"], negative["unknown"]), (0, 1, 0))
        self.assertEqual(negative["details"][0]["evaluation"]["reason"], "wrong_output_accepted")

    def test_return_params_bind_vec_returns_as_pinned_parameters(self):
        from metrics_rebuild.share.contract_eval import _contract_check_proof_fn_lines

        context = {
            "function": "target", "parameters": [{"name": "a", "type": "&Vec<i32>"}],
            "returns": [{"name": "result", "type": "Vec<i32>"}], "requires": [],
            "ensures": [{"kind": "ensures", "text": "same(a, &result)", "normalized": "same(a, &result)"}],
        }
        default = "\n".join(_contract_check_proof_fn_lines(context, {"a": [1, 2]}, {"result": [1, 2]}, "h"))
        pinned = "\n".join(_contract_check_proof_fn_lines(context, {"a": [1, 2]}, {"result": [1, 2]}, "h", return_params=True))
        self.assertIn("let result: Seq<i32>", default)
        self.assertIn("proof fn h(a: &Vec<i32>, result: Vec<i32>)", pinned)
        self.assertIn("result@.len() == 2", pinned)
        self.assertIn("result@[1] == 2", pinned)
        self.assertNotIn("let result", pinned)
        self.assertIn("assert(same(a, &result));", pinned)

    def test_bitwise_contracts_get_bit_vector_facts_for_element_pairs(self):
        from metrics_rebuild.share.contract_eval import _bitwise_ops_in, _contract_check_proof_fn_lines

        self.assertEqual(_bitwise_ops_in(["result[i] == #[trigger] a[i] & #[trigger] b[i]"]), ["&"])
        self.assertEqual(_bitwise_ops_in(["forall|i: int| p(i) && q(i) || f(a, &r)"]), [])
        context = {
            "function": "target", "parameters": [{"name": "a", "type": "Vec<u8>"}, {"name": "b", "type": "Vec<u8>"}],
            "returns": [{"name": "result", "type": "Vec<u8>"}], "requires": [],
            "ensures": [{"text": "forall|i: int| 0 <= i < result.len() ==> result[i] == (a[i] | b[i])"}],
        }
        lines = _contract_check_proof_fn_lines(context, {"a": [1, 6], "b": [2, 3]}, {"result": [3, 7]}, "h")
        self.assertIn("    assert(6u8 | 3u8 == 7u8) by (bit_vector);", lines)
        self.assertIn("    assert(3u8 | 6u8 == 7u8) by (bit_vector);", lines)
        plain = dict(context, ensures=[{"text": "result[0] == a[0]"}])
        self.assertFalse(any("bit_vector" in line for line in _contract_check_proof_fn_lines(plain, {"a": [1], "b": [2]}, {"result": [1]}, "h")))

    def test_finite_forall_keeps_trigger_attribute_after_binders(self):
        from metrics_rebuild.share.contract_eval import _finite_index_forall_assertion_lines

        lines = _finite_index_forall_assertion_lines("forall|i: int| #![trigger b[i]] 0 <= i < b.len() ==> b[i] > 0", {"b": 2})
        self.assertEqual(lines[0], "    assert forall|i: int| #![trigger b[i]] (0 <= i < b.len()) implies (b[i] > 0) by {")

    def test_missing_postcondition_delegates_pair_decision_to_precondition(self):
        from metrics_rebuild.share import contract_eval

        context = {
            "function": "target", "parameters": [{"name": "x", "type": "i32"}],
            "returns": [{"name": "r", "type": "i32"}],
            "requires": [{"kind": "requires", "text": "x > 0", "normalized": "x > 0"}], "ensures": [],
        }
        cases = [{"key": key, "inputs": {"x": 1}, "output": {"r": 1}} for key in "abc"]
        with patch.object(contract_eval, "batch_verus_requires_decide", return_value={"a": True, "b": False, "c": None}):
            details = contract_eval.batch_verus_contract_decide_detailed(context, cases)
        self.assertEqual([details[k]["accepted"] for k in "abc"], [True, False, None])
        self.assertEqual([details[k]["reason"] for k in "abc"], ["accepted", "contract_rejected", "verification_unresolved"])
        self.assertTrue(all(details[k]["empty_postcondition"] for k in "abc"))

    def test_negative_contract_unknown_is_not_recorded_as_failed(self):
        from metrics_rebuild.share.io_cases import score_io_cases

        generated_context = {
            "function": "target", "parameters": [{"name": "x", "type": "i32"}],
            "returns": [{"name": "r", "type": "i32"}], "requires": [],
            "ensures": [{"text": "r == x"}],
        }
        suite = self._suite(cases=[
            {"id": "neg_1", "kind": "negative", "inputs": {"x": 1}, "mutated_output": {"r": 2}, "status": "validated"},
            {"id": "neg_2", "kind": "negative", "inputs": {"x": 2}, "mutated_output": {"r": 3}, "status": "validated"},
            {"id": "neg_3", "kind": "negative", "inputs": {"x": 3}, "mutated_output": {"r": 4}, "status": "validated"},
        ])
        detailed = {
            "sc_0": {"accepted": False, "reason": "contract_rejected", "engine": "unit_test"},
            "sc_1": {"accepted": True, "reason": "accepted", "engine": "unit_test"},
            "sc_2": {"accepted": None, "reason": "timeout", "engine": "unit_test"},
        }
        with tempfile.TemporaryDirectory() as d:
            generated = Path(d) / "gen.rs"
            generated.write_text("verus! { fn target(x: i32) -> (r: i32) ensures r == x { x } }")
            with patch("metrics_rebuild.share.io_cases.strength_contexts_for_path", return_value=[generated_context]), \
                    patch("metrics_rebuild.share.io_cases.batch_verus_contract_decide_detailed", return_value=detailed):
                result = score_io_cases(str(generated), suite, "negative")

        self.assertAlmostEqual(result["score"], 1 / 3)
        self.assertEqual((result["passed"], result["failed"], result["unknown"]), (1, 1, 1))
        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["details"][0]["success"])
        self.assertFalse(result["details"][1]["success"])
        self.assertIsNone(result["details"][2]["success"])


class TestStrictIORevalidation(unittest.TestCase):
    def test_all_known_helper_target_suites_were_rebuilt(self):
        import csv
        import json
        from metrics_rebuild.share.functions import authoritative_target_for_path

        expected = {
            "VA0018": ("days_in_month_exec", "solve"),
            "VA0186": ("good_digit_count_exec", "solve"),
            "VA0323": ("exec_last_occurrence_position", "solve"),
            "VA0453": ("exec_is_power_of_two", "solve"),
            "VA0534": ("exec_turns_to_defeat", "solve"),
            "VA0589": ("compute_moves", "solve"),
            "VD0144": ("find_max_in_prefix", "barrier"),
            "VJ0040": ("four_times", "myfun"),
            "VJ0042": ("times_five", "myfun"),
            "VJ0092": ("contains_z_from", "contains_z"),
            "VJ0094": ("count_uppercase_upto", "count_uppercase"),
            "VJ0104": ("check_match_at", "is_sub_array"),
            "VS0011": ("get_column", "column_stack"),
        }
        io_root = ROOT / "data/io"
        source_root = ROOT / "data/references"
        with (ROOT / "data/evaluation/target_functions.csv").open(newline="") as handle:
            manual_rows = [
                row for row in csv.DictReader(handle)
                if row["selection"] == "manual_source_audit"
            ]
        expected_sources = {
            row["reference_file"].removesuffix(".rs"): source_root / row["benchmark"] / row["reference_file"]
            for row in manual_rows
        }
        expected_sources.update({
            f"VeriCoding_{task}_vericoded": source_root / "VeriCoding" / f"VeriCoding_{task}_vericoded.rs"
            for task in expected
        })
        summary = json.loads((io_root / "summary.json").read_text())
        rebuilt = {
            item["name"]
            for item in summary["results"]
            if item.get("wrong_target") is True
        }
        self.assertEqual(rebuilt, set(expected_sources))

        for task, (old_function, official_function) in expected.items():
            stem = f"VeriCoding_{task}_vericoded"
            meta = json.loads((io_root / "VeriCoding" / stem / "meta.json").read_text())
            source = expected_sources[stem]
            authoritative = authoritative_target_for_path(str(source))
            self.assertEqual(meta["schema_version"], 3)
            self.assertEqual(meta["old_function"], old_function)
            self.assertEqual(meta["function"], official_function)
            self.assertEqual(meta["function"], authoritative["function"])
            self.assertTrue(meta["strict_validation"]["legacy_target_mismatch"])
            self.assertGreater(meta["strict_validation"]["quarantined_count"], 0)
            self.assertEqual(
                meta["strict_validation"]["quarantined_count"],
                meta["strict_validation"]["legacy_case_count"],
            )
            # No legacy case may survive a wrong-target rebuild; both allowed
            # sources re-derive the positive from the authoritative target.
            self.assertLessEqual(
                {entry["source"] for entry in meta["case_audit"]["positive"]},
                {"deterministic_supplement", "gap_supplement"},
            )

        for row in manual_rows:
            stem = row["reference_file"].removesuffix(".rs")
            meta = json.loads((io_root / row["benchmark"] / stem / "meta.json").read_text())
            authoritative = authoritative_target_for_path(str(expected_sources[stem]))
            self.assertEqual(meta["schema_version"], 3)
            self.assertEqual(meta["function"], row["target_function"])
            self.assertEqual(meta["function"], authoritative["function"])
            self.assertTrue(meta["strict_validation"]["legacy_target_mismatch"])
            self.assertEqual(
                meta["strict_validation"]["quarantined_count"],
                meta["strict_validation"]["legacy_case_count"],
            )

    def _run_case(self, *, expected=1, requires=True, runtime_output="1", unexpected=None,
                  contract_accepted=None):
        import json
        from scripts.io import revalidate_io_tests as audit

        source_text = """
verus! {
fn target(x: i32) -> (r: i32)
    ensures r == x,
{ x }
}
"""
        old_case = {"input": {"x": 1}, "expected": expected, "unexpected": unexpected or []}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "verified" / "Bench" / "case.rs"
            source.parent.mkdir(parents=True)
            source.write_text(source_text)
            old_dir = root / "old" / "Bench" / "case"
            old_dir.mkdir(parents=True)
            (old_dir / "meta.json").write_text(json.dumps({"function": "target"}))
            (old_dir / "test.json").write_text(json.dumps([old_case]))
            output_dir = root / "new" / "Bench" / "case"
            quarantine = root / "quarantine" / "Bench" / "case.json"

            def requires_result(_target, inputs):
                return {audit._input_key(item): requires for item in inputs}

            def runtime_result(_target, inputs, _verus_bin):
                return ({
                    audit._input_key(item): {
                        "status": "OK", "output": runtime_output, "reason": "ok",
                    }
                    for item in inputs
                }, {"compile_reason": "ok", "raw_result_count": len(inputs)})

            def contract_result(_context, cases):
                return {
                    case["key"]: {"accepted": contract_accepted, "reason": "unit_test"}
                    for case in cases
                }

            with patch.object(audit, "_requires_verdicts", side_effect=requires_result), \
                    patch.object(audit, "_runtime_verdicts", side_effect=runtime_result), \
                    patch.object(audit, "generate_requires_satisfying_candidate_inputs", return_value=[]), \
                    patch.object(audit, "generate_candidate_inputs", return_value=[]), \
                    patch.object(audit, "generate_boundary_candidate_inputs", return_value=[]), \
                    patch.object(audit, "generate_requires_violating_from_positives", return_value=[]), \
                    patch.object(audit, "batch_verus_contract_decide_detailed", side_effect=contract_result), \
                    patch.object(audit.generate, "_positive_target_info", return_value={"target": 1, "reason": None}), \
                    patch.object(audit.generate, "_negative_target_info", return_value={
                        "target": 1 if unexpected else 0, "reason": None,
                    }), \
                    patch.object(audit.generate, "_invalid_target_info", return_value={
                        "target": 0, "reason": "no_invalid_domain",
                    }):
                audit._audit_task(
                    source, old_dir, output_dir, quarantine,
                    per_kind=1, candidate_budget=1, verus_bin="verus", resume=False,
                )
            return (
                json.loads((output_dir / "test.json").read_text()),
                json.loads((output_dir / "meta.json").read_text()),
                json.loads(quarantine.read_text()) if quarantine.exists() else {"cases": []},
            )

    def test_positive_output_mismatch_is_quarantined(self):
        cases, meta, quarantine = self._run_case(expected=2, runtime_output="1")
        self.assertEqual(cases, [])
        self.assertEqual(meta["positive_count"], 0)
        self.assertIn("positive_output_mismatch", {item["reason"] for item in quarantine["cases"]})

    def test_positive_requires_unknown_is_quarantined(self):
        cases, meta, quarantine = self._run_case(requires=None, runtime_output="1")
        self.assertEqual(cases, [])
        self.assertEqual(meta["positive_count"], 0)
        self.assertIn("positive_not_machine_validated", {item["reason"] for item in quarantine["cases"]})

    def test_negative_not_rejected_is_quarantined(self):
        cases, meta, quarantine = self._run_case(
            unexpected=[2], contract_accepted=True,
        )
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["unexpected"], [])
        self.assertEqual(meta["negative_count"], 0)
        self.assertIn("negative_rejection_not_proven", {item["reason"] for item in quarantine["cases"]})

    def test_negative_decision_sees_mutable_post_state_returns(self):
        """`&mut` outputs must be decided against the observable context.

        The plain target context lists no returns for such functions, so the
        proof harness would bind nothing and every rejection stays unprovable.
        """
        import json
        from scripts.io import revalidate_io_tests as audit

        source_text = """
verus! {
fn target(a: &mut Vec<i32>)
    ensures a[0] == old(a)[0] + 1,
{ }
}
"""
        seen_contexts = []
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "verified" / "Bench" / "case.rs"
            source.parent.mkdir(parents=True)
            source.write_text(source_text)
            old_dir = root / "old" / "Bench" / "case"
            old_dir.mkdir(parents=True)
            (old_dir / "meta.json").write_text(json.dumps({"function": "target"}))
            (old_dir / "test.json").write_text(json.dumps([
                {"input": {"a": [1]}, "expected": [2], "unexpected": [[3]]},
            ]))

            def contract_result(context, cases):
                seen_contexts.append(context)
                return {case["key"]: {"accepted": False, "reason": "unit_test"} for case in cases}

            with patch.object(audit, "_requires_verdicts", side_effect=lambda _t, items: {
                        audit._input_key(item): True for item in items
                    }), \
                    patch.object(audit, "_runtime_verdicts", side_effect=lambda _t, items, _b: ({
                        audit._input_key(item): {"status": "OK", "output": "[2]", "reason": "ok"}
                        for item in items
                    }, {})), \
                    patch.object(audit, "generate_requires_satisfying_candidate_inputs", return_value=[]), \
                    patch.object(audit, "generate_candidate_inputs", return_value=[]), \
                    patch.object(audit, "generate_boundary_candidate_inputs", return_value=[]), \
                    patch.object(audit, "generate_requires_violating_from_positives", return_value=[]), \
                    patch.object(audit, "batch_verus_contract_decide_detailed", side_effect=contract_result):
                audit._audit_task(
                    source, old_dir, root / "new" / "Bench" / "case",
                    root / "quarantine" / "Bench" / "case.json",
                    per_kind=1, candidate_budget=1, verus_bin="verus", resume=False,
                )

        self.assertTrue(seen_contexts)
        self.assertEqual(
            [item.get("name") for item in seen_contexts[0].get("returns") or []], ["a"],
        )


class TestFindHarnessBinary(unittest.TestCase):
    def test_prefers_harness_name(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "harness")
            Path(p).write_text("")
            os.chmod(p, 0o755)
            self.assertEqual(_find_harness_binary(d), p)

    def test_falls_back_to_executable_when_name_differs(self):
        with tempfile.TemporaryDirectory() as d:
            # 源码/中间产物应被跳过
            Path(os.path.join(d, "harness.rs")).write_text("")
            Path(os.path.join(d, "harness.d")).write_text("")
            other = os.path.join(d, "harness_bin")
            Path(other).write_text("")
            os.chmod(other, 0o755)
            self.assertEqual(_find_harness_binary(d), other)

    def test_returns_none_when_no_binary(self):
        with tempfile.TemporaryDirectory() as d:
            Path(os.path.join(d, "harness.rs")).write_text("")
            self.assertIsNone(_find_harness_binary(d))


if __name__ == "__main__":
    unittest.main()
