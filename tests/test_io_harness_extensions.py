#!/usr/bin/env python3
"""IO harness 扩展的回归测试（纯 Python 断言，不依赖 Verus 编译）。

覆盖：
- 证明侧类型支持（&str/String/&mut 标量）与 &mut/old() 的 proof harness 编码;
- 递归 spec fn 的 reveal_with_fuel 注入;
- 非有限浮点与输出侧 Vec cap;
- 字符串候选值池 / String len() 模式 / 字符串输出变异;
- 定长数组、泛型单态化、文件内 struct/enum/alias 注册表。
"""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics_rebuild.share.contract_eval import (
    _contract_case_support_issue,
    _contract_check_proof_fn_lines,
    _requires_check_proof_fn_lines,
    _type_supported_by_harness,
    base_type_name,
    coerce_value_for_type,
    typed_output_payload,
    verus_literal,
)
from metrics_rebuild.share.io_cases import (
    _string_candidate_pool,
    generate_candidate_inputs,
    generate_requires_satisfying_candidate_inputs,
    mutate_output_values,
)
from metrics_rebuild.share.io_harness import (
    array_type_parts,
    format_value_rust,
    parse_type,
    runtime_type_support_issue,
)
from metrics_rebuild.share.type_defs import (
    parse_type_definitions,
    registry_entry,
    render_type_definitions,
    resolve_alias_text,
)
from scripts.io.generate_io_tests_llm import (
    _generic_binder_names,
    _monomorphize_source,
    build_verus_harness,
    parse_typed_value,
    resolve_target,
)


class TestProofHarnessTypes(unittest.TestCase):
    def test_str_and_mut_scalar_supported(self):
        self.assertTrue(_type_supported_by_harness("&str", set()))
        self.assertTrue(_type_supported_by_harness("String", set()))
        self.assertTrue(_type_supported_by_harness("&mut u32", set()))
        self.assertTrue(_type_supported_by_harness("&u64", set()))
        self.assertFalse(_type_supported_by_harness("HashMap<u32, u32>", set()))

    def test_support_issue_accepts_str_param(self):
        context = {
            "parameters": [{"name": "day", "type": "&str"}],
            "returns": [{"name": "result", "type": "i8"}],
            "requires": [{"text": "valid_day(day)"}],
            "ensures": [{"text": "result as int >= 1"}],
        }
        self.assertIsNone(
            _contract_case_support_issue(context, {"day": "SUN"}, {"result": 7}, include_ensures=True)
        )

    def test_requires_harness_binds_mut_vec_as_immutable_ref(self):
        context = {
            "parameters": [
                {"name": "x", "type": "&Vec<u64>"},
                {"name": "y", "type": "&mut Vec<u64>"},
            ],
            "requires": [{"text": "old(y).len() == 0"}],
            "ensures": [],
        }
        lines = _requires_check_proof_fn_lines(context, {"x": [1, 3], "y": []}, "__p")
        text = "\n".join(lines)
        self.assertIn("y: &Vec<u64>", text)
        self.assertNotIn("&mut", text)
        # old(y) 在 requires 中即前置状态，替换为绑定名
        self.assertIn("assert(y.len() == 0);", text)
        self.assertNotIn("old(", text)

    def test_requires_harness_binds_mut_scalar_by_value_and_strips_deref(self):
        context = {
            "parameters": [
                {"name": "n", "type": "u32"},
                {"name": "sum", "type": "&mut u32"},
            ],
            "requires": [{"text": "*old(sum) == 0"}],
            "ensures": [],
        }
        lines = _requires_check_proof_fn_lines(context, {"n": 3, "sum": 0}, "__p")
        text = "\n".join(lines)
        self.assertIn("let sum: u32 = 0u32;", text)
        self.assertIn("assert(sum == 0);", text)

    def test_contract_harness_mut_scalar_post_state(self):
        context = {
            "parameters": [
                {"name": "n", "type": "u32"},
                {"name": "sum", "type": "&mut u32"},
            ],
            "requires": [{"text": "*old(sum) == 0"}],
            "ensures": [{"text": "*sum == n"}],
            "returns": [{"name": "sum", "type": "u32"}],
            "_mutable_post_state_names": ["sum"],
        }
        lines = _contract_check_proof_fn_lines(context, {"n": 3, "sum": 0}, {"sum": 3}, "__p")
        text = "\n".join(lines)
        self.assertIn("let __old_sum: u32 = 0u32;", text)
        self.assertIn("let sum: u32 = 3u32;", text)
        self.assertIn("assert(__old_sum == 0);", text)
        self.assertIn("assert(sum == n);", text)

    def test_string_return_binds_as_str_and_keeps_view(self):
        context = {
            "parameters": [],
            "requires": [],
            "ensures": [{"text": 'result@ == "abc"@'}],
            "returns": [{"name": "result", "type": "String"}],
        }
        lines = _contract_check_proof_fn_lines(context, {}, {"result": "abc"}, "__p")
        text = "\n".join(lines)
        self.assertIn('let result: &str = "abc";', text)
        # &str 自带 view，result@ 原样保留
        self.assertIn('assert(result@ == "abc"@);', text)

    def test_fuel_reveal_injected_for_recursive_spec_fn(self):
        context = {
            "parameters": [{"name": "n", "type": "u32"}],
            "requires": [{"text": "triangle(n as nat) < 100"}],
            "ensures": [],
            "spec_preamble": (
                "spec fn triangle(n: nat) -> nat decreases n "
                "{ if n == 0 { 0 } else { n + triangle((n - 1) as nat) } }"
            ),
        }
        lines = _requires_check_proof_fn_lines(context, {"n": 3}, "__p")
        self.assertIn("    reveal_with_fuel(triangle, 12);", lines)

    def test_base_type_name_unknown_is_empty(self):
        self.assertEqual(base_type_name("Matrix"), "")
        self.assertFalse(coerce_value_for_type(3, "Matrix")[0])


class TestNonFiniteFloatsAndCaps(unittest.TestCase):
    def test_coerce_accepts_non_finite(self):
        self.assertEqual(coerce_value_for_type(math.inf, "f32"), (True, math.inf))
        self.assertEqual(coerce_value_for_type("-inf", "f64"), (True, -math.inf))
        ok, value = coerce_value_for_type(float("nan"), "f32")
        self.assertTrue(ok and math.isnan(value))
        # 有限越界仍拒绝
        self.assertFalse(coerce_value_for_type(1e39, "f32")[0])

    def test_format_value_rust_non_finite(self):
        self.assertEqual(format_value_rust(math.inf, "f32"), "f32::INFINITY")
        self.assertEqual(format_value_rust(-math.inf, "f64"), "f64::NEG_INFINITY")
        self.assertEqual(format_value_rust(float("nan"), "f32"), "f32::NAN")

    def test_parse_typed_value_non_finite(self):
        self.assertEqual(parse_typed_value("inf", "f32"), math.inf)
        self.assertTrue(math.isnan(parse_typed_value("NaN", "f64")))

    def test_verus_literal_floats(self):
        self.assertEqual(verus_literal(0.5, "f32"), "0.5_f32")
        self.assertEqual(verus_literal(math.inf, "f32"), "f32::INFINITY")

    def test_output_cap_relaxed_input_cap_kept(self):
        big = list(range(624))
        self.assertFalse(coerce_value_for_type(big, "Vec<u32>")[0])
        payload = typed_output_payload(
            {"returns": [{"name": "state", "type": "Vec<u32>"}]}, {"state": big},
        )
        self.assertIsNotNone(payload)
        self.assertEqual(len(payload["state"]), 624)


class TestStringCandidatesAndMutations(unittest.TestCase):
    def test_pool_extracts_contract_literals(self):
        context = {
            "parameters": [{"name": "day", "type": "&str"}],
            "requires": [{"text": 'valid_day(day)'}],
            "ensures": [],
            "spec_preamble": 'spec fn valid_day(day: &str) -> bool { day == "SUN" || day == "MON" }',
        }
        pool = _string_candidate_pool(context)
        self.assertIn("SUN", pool)
        self.assertIn("MON", pool)

    def test_pool_builds_char_combinations(self):
        context = {
            "parameters": [{"name": "s", "type": "&str"}],
            "requires": [],
            "ensures": [{"text": "ret <==> spec_bracketing(s@)"}],
            "spec_preamble": "spec fn spec_bracketing(s: Seq<char>) -> bool { s.contains('<') || s.contains('>') }",
        }
        pool = _string_candidate_pool(context)
        self.assertIn("<>", pool)
        self.assertIn("<<>>", pool)

    def test_string_candidates_reach_generator(self):
        context = {
            "function": "f",
            "parameters": [{"name": "s", "type": "&str"}],
            "requires": [],
            "ensures": [],
        }
        cases = generate_candidate_inputs(context, budget=6)
        self.assertTrue(all(isinstance(case["s"], str) for case in cases))

    def test_string_len_requires_pattern(self):
        context = {
            "function": "rsplit",
            "parameters": [
                {"name": "a", "type": "Vec<String>"},
                {"name": "sep", "type": "String"},
            ],
            "requires": [{"normalized": "sep . len ( ) > 0"}],
            "ensures": [],
        }
        cases = generate_requires_satisfying_candidate_inputs(context, budget=8)
        self.assertTrue(any(len(case["sep"]) > 0 for case in cases))
        self.assertTrue(all(isinstance(case["a"], list) for case in cases))

    def test_string_output_mutations(self):
        muts = mutate_output_values(
            {"returns": [{"name": "r", "type": "&'static str"}]}, {}, {"r": "abc"},
        )
        values = [m["r"] for m, _label in muts]
        self.assertIn("", values)
        self.assertIn("abcx", values)
        self.assertNotIn("abc", values)

    def test_vec_string_mutation_appends_string(self):
        muts = mutate_output_values(
            {"returns": [{"name": "r", "type": "Vec<String>"}]}, {}, {"r": ["a", "b"]},
        )
        append = next(value for value, label in muts if label == "vec_append")
        self.assertTrue(all(isinstance(item, str) for item in append["r"]))


class TestArrays(unittest.TestCase):
    def test_array_type_parts(self):
        self.assertEqual(array_type_parts("[bool; 10]"), ("bool", 10))
        self.assertEqual(array_type_parts("&[f64; 2]"), ("f64", 2))
        self.assertIsNone(array_type_parts("&[i64]"))

    def test_parse_type_array_not_slice(self):
        ty, ref, mut, is_slice, inner = parse_type("&[bool; 10]")
        self.assertTrue(ref)
        self.assertFalse(is_slice)
        self.assertEqual(inner, "bool")

    def test_format_array_literal_exact_length(self):
        self.assertEqual(
            format_value_rust([True, False], "[bool; 3]"),
            "[true, false, false]",
        )

    def test_coerce_array_requires_exact_length(self):
        self.assertTrue(coerce_value_for_type([0.0, 1.0], "[f64; 2]")[0])
        self.assertFalse(coerce_value_for_type([0.0], "[f64; 2]")[0])

    def test_parse_array_output(self):
        self.assertEqual(parse_typed_value("[0, -1]", "[i8; 2]"), [0, -1])


class TestTypeRegistry(unittest.TestCase):
    CODE = """
verus! {
pub enum TimeUnit { Year, Month }
#[derive(PartialEq)]
struct TimeDelta64 { value: i64, unit: TimeUnit }
type Matrix = Vec<Vec<i8>>;
enum Message { Text(String), Quit }
}
"""

    def test_parse_type_definitions(self):
        registry = parse_type_definitions(self.CODE)
        self.assertEqual(registry["TimeUnit"], {"kind": "enum", "variants": ["Year", "Month"]})
        self.assertEqual(
            registry["TimeDelta64"],
            {"kind": "struct", "fields": [("value", "i64"), ("unit", "TimeUnit")], "tuple": False},
        )
        self.assertEqual(registry["Matrix"], {"kind": "alias", "target": "Vec<Vec<i8>>"})
        # 带字段的 enum variant 不支持
        self.assertNotIn("Message", registry)

    def test_alias_resolution(self):
        registry = parse_type_definitions(self.CODE)
        self.assertEqual(resolve_alias_text(registry, "Matrix"), "Vec<Vec<i8>>")
        self.assertEqual(resolve_alias_text(registry, "Vec<Matrix>"), "Vec<Vec<Vec<i8>>>")

    def test_format_and_parse_registry_values(self):
        registry = parse_type_definitions(self.CODE)
        self.assertEqual(format_value_rust("Year", "TimeUnit", registry), "TimeUnit::Year")
        self.assertEqual(
            format_value_rust([5, "Month"], "TimeDelta64", registry),
            "TimeDelta64 { value: 5_i64, unit: TimeUnit::Month }",
        )
        self.assertEqual(
            format_value_rust([[1, 2]], "Matrix", registry), "vec![vec![1_i8, 2_i8]]",
        )
        self.assertEqual(parse_typed_value("Year", "TimeUnit", registry), "Year")
        self.assertEqual(parse_typed_value("[5, Month]", "TimeDelta64", registry), [5, "Month"])

    def test_coerce_registry_values(self):
        registry = parse_type_definitions(self.CODE)
        self.assertEqual(coerce_value_for_type("Year", "TimeUnit", registry=registry), (True, "Year"))
        self.assertFalse(coerce_value_for_type("Century", "TimeUnit", registry=registry)[0])
        self.assertEqual(
            coerce_value_for_type([5, "Month"], "TimeDelta64", registry=registry),
            (True, [5, "Month"]),
        )

    def test_verus_literal_registry(self):
        registry = parse_type_definitions(self.CODE)
        self.assertEqual(verus_literal("Year", "TimeUnit", registry), "TimeUnit::Year")
        self.assertEqual(
            verus_literal([5, "Month"], "TimeDelta64", registry),
            "TimeDelta64 { value: 5i64, unit: TimeUnit::Month }",
        )

    def test_render_type_definitions(self):
        registry = parse_type_definitions(self.CODE)
        rendered = render_type_definitions(registry)
        self.assertIn("pub enum TimeUnit { Year, Month }", rendered)
        self.assertIn("pub struct TimeDelta64 { pub value: i64, pub unit: TimeUnit }", rendered)
        self.assertIn("pub type Matrix = Vec<Vec<i8>>;", rendered)

    def test_enum_output_mutates_to_other_variants(self):
        registry = parse_type_definitions(self.CODE)
        muts = mutate_output_values(
            {"returns": [{"name": "r", "type": "TimeUnit"}], "type_registry": registry},
            {},
            {"r": "Year"},
        )
        self.assertEqual([m["r"] for m, _label in muts], ["Month"])

    def test_registry_entry_follows_named_alias(self):
        registry = {
            "A": {"kind": "alias", "target": "B"},
            "B": {"kind": "enum", "variants": ["X"]},
        }
        self.assertEqual(registry_entry(registry, "A"), registry["B"])


class TestGenericMonomorphization(unittest.TestCase):
    CODE = """
verus! {
proof fn seq_push_indexing<T>(s_old: Seq<T>, s_new: Seq<T>, x: T) {}
fn ones_like<T>(a: &Vec<T>) -> (result: Vec<i32>)
    ensures result.len() == a.len(),
{
    proof { seq_push_indexing::<i32>(Seq::empty(), Seq::empty(), 1i32); }
    Vec::new()
}
}
"""

    def test_binder_detection(self):
        self.assertEqual(_generic_binder_names(self.CODE, "ones_like"), ["T"])
        self.assertEqual(_generic_binder_names(self.CODE, "missing"), [])

    def test_monomorphize_source_strips_decls_and_turbofish(self):
        mono = _monomorphize_source(self.CODE, ["T"])
        self.assertIn("fn ones_like(a: &Vec<i32>)", mono)
        self.assertIn("proof fn seq_push_indexing(s_old: Seq<i32>", mono)
        # 被去泛型化函数的 turbofish 调用一并删除
        self.assertIn("seq_push_indexing(Seq::empty(), Seq::empty(), 1i32);", mono)
        self.assertNotIn("::<i32>(Seq::empty()", mono)

    def test_generic_coerce_accepts_i32_values(self):
        self.assertEqual(coerce_value_for_type(5, "T"), (True, 5))
        self.assertEqual(coerce_value_for_type("7", "T"), (True, 7))
        self.assertFalse(coerce_value_for_type(2**40, "T")[0])
        self.assertTrue(coerce_value_for_type([1, 2], "Vec<T>")[0])


class TestEndToEndHarnessText(unittest.TestCase):
    def test_registry_task_harness_builds(self):
        import tempfile

        code = """
use vstd::prelude::*;
verus! {
pub enum TimeUnit { Year, Month }
pub struct TimeDelta64 { pub value: i64, pub unit: TimeUnit }
fn timedelta64(value: i64, unit: TimeUnit) -> (result: TimeDelta64)
    ensures result.value == value,
{
    TimeDelta64 { value, unit }
}
}
fn main() {}
"""
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "case.rs"
            src.write_text(code)
            target = resolve_target(str(src))
            self.assertIsNotNone(target)
            harness = build_verus_harness(target, [{"value": 5, "unit": "Month"}])
        self.assertIn("TimeUnit::Month", harness)
        self.assertIn("fn fmt_typed(v: TimeDelta64)", harness)
        # struct 输出按字段顺序格式化为列表
        self.assertIn('TimeUnit::Year => "Year".to_string()', harness)


if __name__ == "__main__":
    unittest.main()
