"""旧格式基准语料的类型感知解析回归测试。

背景（RQ7 S5 假缺陷根因）：旧格式 test.json 把 Vec<char> 输入存成
"['a', '1']" 这样的 repr 文本，无类型解析会把数字字符 '1' 转成 int 1，
下游按码点渲染成 '\\u{0001}'，导致合法契约拒绝被破坏的正例。
"""

from __future__ import annotations

import unittest

from metrics_rebuild.share.io_cases import (
    _parse_benchmark_inputs,
    _parse_benchmark_value,
    _parse_benchmark_value_typed,
    _return_payload_from_benchmark,
)


class TypedBenchmarkParsingTest(unittest.TestCase):
    def test_vec_char_digit_characters_stay_characters(self):
        self.assertEqual(
            _parse_benchmark_value_typed("['a', '1']", "&Vec<char>"), ["a", "1"]
        )
        self.assertEqual(_parse_benchmark_value_typed("['5']", "Vec<char>"), ["5"])
        self.assertEqual(
            _parse_benchmark_value_typed("['0', '1']", "&[char]"), ["0", "1"]
        )

    def test_new_corpus_char_ordinals_are_preserved(self):
        # 重生成后的语料把 char 存为码点 int，coerce 按码点处理。
        self.assertEqual(_parse_benchmark_value_typed("[97, 48]", "Vec<char>"), [97, 48])

    def test_scalar_char_and_string_keep_text(self):
        self.assertEqual(_parse_benchmark_value_typed("5", "char"), "5")
        self.assertEqual(_parse_benchmark_value_typed("123", "String"), "123")
        self.assertEqual(_parse_benchmark_value_typed("42", "&str"), "42")

    def test_vec_string_numeric_looking_elements_stay_strings(self):
        self.assertEqual(
            _parse_benchmark_value_typed("['abc', '123']", "Vec<String>"),
            ["abc", "123"],
        )

    def test_nested_vec_char(self):
        self.assertEqual(
            _parse_benchmark_value_typed("[['1', 'a'], ['2']]", "Vec<Vec<char>>"),
            [["1", "a"], ["2"]],
        )

    def test_numeric_types_keep_legacy_behavior(self):
        self.assertEqual(_parse_benchmark_value_typed("[1, 2, 3]", "Vec<i32>"), [1, 2, 3])
        self.assertEqual(_parse_benchmark_value_typed("42", "i32"), 42)
        self.assertEqual(_parse_benchmark_value_typed("true", "bool"), True)

    def test_unknown_type_falls_back_to_untyped_parse(self):
        self.assertEqual(
            _parse_benchmark_value_typed("['1']", ""), _parse_benchmark_value("['1']")
        )

    def test_parse_benchmark_inputs_uses_declared_types(self):
        context = {
            "parameters": [
                {"name": "text", "type": "&Vec<char>"},
                {"name": "n", "type": "usize"},
            ]
        }
        parsed = _parse_benchmark_inputs(context, {"text": "['5', 'a']", "n": "3"})
        self.assertEqual(parsed, {"text": ["5", "a"], "n": 3})

    def test_return_payload_uses_declared_single_return_type(self):
        context = {"returns": [{"name": "result", "type": "Vec<char>"}]}
        payload = _return_payload_from_benchmark(context, "['1', '2']")
        self.assertEqual(payload, {"result": ["1", "2"]})


if __name__ == "__main__":
    unittest.main()
