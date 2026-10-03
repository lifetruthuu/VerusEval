"""批量 Verus 结果解析的诊断丢失防护回归测试。

背景：run_verus 曾把 stderr 截断为末尾 32KB，批量 harness 中排在前部的
proof fn 的失败诊断被截掉后，_parse_batch_result 把"行区间内无错误行"
误判为证明成功，导致 wrong_io_reject_rate 等 I/O 指标出现假接受
（RQ1 形式×I/O 交叉检查发现的 12 例评估工件的根因）。

修复由两部分组成，本文件分别锁定：
1. verus_runner 在内存中保留接近完整的 stderr（按行边界截断），持久化仍为
   32KB 尾部摘录；
2. _parse_batch_result 用 Verus stdout 汇总的 errors 计数交叉校验解析到的
   验证失败诊断数，诊断缺失时返回全 None（触发上层二分），而不是把
   "看不到报错"当作证明成功。
"""

from __future__ import annotations

import json
import unittest

from metrics_rebuild.share.contract_eval import (
    _BatchItem,
    _build_batch_harness,
    _parse_batch_result,
)
from metrics_rebuild.share.verus_runner import (
    VerusRun,
    _tail_on_line_boundary,
    verus_run_to_dict,
)


def _diag_line(line_start: int, message: str = "assertion failed") -> str:
    return json.dumps(
        {
            "$message_type": "diagnostic",
            "level": "error",
            "message": message,
            "spans": [{"line_start": line_start}],
        }
    )


def _batch_items() -> list[_BatchItem]:
    items = []
    for index in range(3):
        item = _BatchItem(key=f"case_{index}", fn_name=f"__sqm_{index:04d}")
        item.start_line = 10 + index * 10
        item.end_line = item.start_line + 5
        items.append(item)
    return items


def _failed_run(stderr: str, errors: int) -> VerusRun:
    return VerusRun(
        status="ok",
        success=False,
        verified=0,
        errors=errors,
        returncode=1,
        elapsed_seconds=0.1,
        stdout="",
        stderr=stderr,
        command=("verus",),
    )


class ParseBatchResultGuardTest(unittest.TestCase):
    def test_complete_diagnostics_mark_each_failure(self):
        items = _batch_items()
        stderr = "\n".join(_diag_line(item.start_line + 1) for item in items)
        run = _failed_run(stderr, errors=3)
        self.assertEqual(
            _parse_batch_result(run, items),
            {"case_0": False, "case_1": False, "case_2": False},
        )

    def test_truncated_diagnostics_return_unresolved_not_success(self):
        # 只保留最后一个 proof fn 的诊断，模拟 stderr 截断丢失前部诊断。
        items = _batch_items()
        stderr = _diag_line(items[-1].start_line + 1)
        run = _failed_run(stderr, errors=3)
        self.assertEqual(
            _parse_batch_result(run, items),
            {"case_0": None, "case_1": None, "case_2": None},
        )

    def test_partial_failure_with_matching_count_keeps_success(self):
        # 3 项中 1 项失败且诊断计数与 errors 吻合：其余项仍判证明成功。
        items = _batch_items()
        stderr = _diag_line(items[1].start_line + 2)
        run = _failed_run(stderr, errors=1)
        self.assertEqual(
            _parse_batch_result(run, items),
            {"case_0": True, "case_1": False, "case_2": True},
        )

    def test_missing_errors_count_keeps_legacy_behavior(self):
        # errors=None（stdout 汇总不可用）时无法交叉校验，保持原判定。
        items = _batch_items()
        stderr = _diag_line(items[1].start_line + 2)
        run = _failed_run(stderr, errors=None)
        self.assertEqual(
            _parse_batch_result(run, items),
            {"case_0": True, "case_1": False, "case_2": True},
        )

    def test_full_success_short_circuits(self):
        items = _batch_items()
        run = VerusRun(
            status="ok",
            success=True,
            verified=3,
            errors=0,
            returncode=0,
            elapsed_seconds=0.1,
            stdout="",
            stderr="",
            command=("verus",),
        )
        self.assertEqual(
            _parse_batch_result(run, items),
            {"case_0": True, "case_1": True, "case_2": True},
        )


class BatchHarnessLineRangeTest(unittest.TestCase):
    def test_multiline_elements_keep_ranges_aligned_with_rendered_lines(self):
        # 子句文本原样取自规约时，单个列表元素可内嵌换行；行区间必须按
        # 渲染后的物理行计数，否则后续条目的区间漂移、错误被归属到邻居。
        first = _BatchItem(key="first", fn_name="__sqm_0000")
        first.lines = [
            "proof fn __sqm_0000()",
            "{",
            "    assert(!((a ==>\n        b) &&\n        (c ==>\n        d)));",
            "}",
        ]
        second = _BatchItem(key="second", fn_name="__sqm_0001")
        second.lines = ["proof fn __sqm_0001()", "{", "    assert(true);", "}"]

        harness = _build_batch_harness([first, second])
        rendered = harness.splitlines()

        for item in (first, second):
            block = rendered[item.start_line - 1 : item.end_line]
            self.assertTrue(
                block[0].startswith(f"proof fn {item.fn_name}"),
                f"{item.key} start misaligned: {block[:1]}",
            )
            self.assertEqual(block[-1], "}", f"{item.key} end misaligned")
        # 两个区间无重叠且覆盖各自全部物理行。
        self.assertLess(first.end_line, second.start_line)

    def test_ranges_direct_error_attribution_between_items(self):
        first = _BatchItem(key="first", fn_name="__sqm_0000")
        first.lines = ["proof fn __sqm_0000()", "{", "    assert(x\n        && y);", "}"]
        second = _BatchItem(key="second", fn_name="__sqm_0001")
        second.lines = ["proof fn __sqm_0001()", "{", "    assert(z);", "}"]
        harness = _build_batch_harness([first, second])
        rendered = harness.splitlines()

        # 第二个条目 assert 所在的物理行必须落在其声称的区间内。
        assert_line = next(
            number
            for number, text in enumerate(rendered, start=1)
            if "assert(z)" in text
        )
        self.assertTrue(second.start_line <= assert_line <= second.end_line)
        self.assertFalse(first.start_line <= assert_line <= first.end_line)

        run = VerusRun(
            status="ok",
            success=False,
            verified=1,
            errors=1,
            returncode=1,
            elapsed_seconds=0.1,
            stdout="",
            stderr=_diag_line(assert_line),
            command=("verus",),
        )
        self.assertEqual(
            _parse_batch_result(run, [first, second]),
            {"first": True, "second": False},
        )


class StderrCapTest(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(_tail_on_line_boundary("a\nb", 10), "a\nb")

    def test_long_text_keeps_whole_trailing_lines(self):
        text = "\n".join(f"line_{index:04d}" for index in range(100))
        capped = _tail_on_line_boundary(text, 100)
        self.assertLessEqual(len(capped), 100)
        self.assertTrue(capped.startswith("line_"))
        self.assertTrue(capped.endswith("line_0099"))

    def test_persisted_dict_keeps_32k_excerpt_of_long_stderr(self):
        stderr = "\n".join(_diag_line(index) for index in range(2_000))
        run = _failed_run(stderr, errors=2_000)
        self.assertGreater(len(run.stderr), 32_000)
        persisted = verus_run_to_dict(run)
        self.assertLessEqual(len(persisted["stderr"]), 32_000)
        # 摘录从整行开始，不残留半行 JSON。
        self.assertTrue(persisted["stderr"].startswith("{"))


if __name__ == "__main__":
    unittest.main()
