"""Tests for coerce tightening (#8 tuple arity, #7 char full range),
char-literal invalid-escape behavior (#11), and overflow-as-verification (#5)."""

import unittest

from metrics_rebuild.share.contract_eval import (
    _BatchItem,
    _parse_batch_result,
    _verus_char_literal,
    coerce_value_for_type,
    verus_literal,
)
from metrics_rebuild.share.verus_runner import VerusRun


class CharCoerceTest(unittest.TestCase):
    def test_char_accepts_full_unicode_scalar_range(self) -> None:
        self.assertEqual(coerce_value_for_type(65, "char"), (True, 65))
        self.assertEqual(coerce_value_for_type(0x1F600, "char"), (True, 0x1F600))
        self.assertEqual(coerce_value_for_type(0x10FFFF, "char"), (True, 0x10FFFF))

    def test_char_rejects_surrogates_and_out_of_range(self) -> None:
        self.assertEqual(coerce_value_for_type(0xD800, "char"), (False, None))
        self.assertEqual(coerce_value_for_type(0xDFFF, "char"), (False, None))
        self.assertEqual(coerce_value_for_type(0x110000, "char"), (False, None))
        self.assertEqual(coerce_value_for_type(-1, "char"), (False, None))

    def test_char_and_u8_reject_bool(self) -> None:
        self.assertEqual(coerce_value_for_type(True, "char"), (False, None))
        self.assertEqual(coerce_value_for_type(True, "u8"), (False, None))

    def test_u8_still_capped_at_255(self) -> None:
        self.assertEqual(coerce_value_for_type(200, "u8"), (True, 200))
        self.assertEqual(coerce_value_for_type(300, "u8"), (False, None))
        self.assertEqual(coerce_value_for_type(0x1F600, "u8"), (False, None))


class TupleCoerceTest(unittest.TestCase):
    def test_exact_arity_and_per_element(self) -> None:
        self.assertEqual(coerce_value_for_type([1, 2], "(u32, u32)"), (True, [1, 2]))
        self.assertEqual(coerce_value_for_type([5, True], "(u32, bool)"), (True, [5, True]))
        self.assertEqual(coerce_value_for_type([65, 66], "(u8, char)"), (True, [65, 66]))

    def test_arity_mismatch_rejected(self) -> None:
        self.assertEqual(coerce_value_for_type([1], "(u32, u32)"), (False, None))
        self.assertEqual(coerce_value_for_type([1, 2], "(u32, u32, u32)"), (False, None))
        self.assertEqual(coerce_value_for_type([1, 2, 3], "(u32, u32)"), (False, None))

    def test_element_coerce_failure_rejects(self) -> None:
        # 300 out of u8 range -> whole tuple dropped.
        self.assertEqual(coerce_value_for_type([300, 1], "(u8, u32)"), (False, None))


class CharLiteralTest(unittest.TestCase):
    def test_valid_scalar_emits_unicode_escape(self) -> None:
        self.assertEqual(_verus_char_literal(0x1F600), "'\\u{1f600}'")
        self.assertEqual(verus_literal(0x1F600, "char"), "'\\u{1f600}'")

    def test_invalid_ordinal_emits_uncompilable_escape_not_masked(self) -> None:
        # Out of range / surrogate must NOT mask to a valid char; the escape it
        # emits is rejected by rustc, so the harness fails to compile -> None.
        self.assertEqual(_verus_char_literal(0x110000), "'\\u{110000}'")
        self.assertEqual(_verus_char_literal(0xD800), "'\\u{d800}'")
        self.assertEqual(_verus_char_literal(-1), "'\\u{110000}'")


class OverflowVerificationTest(unittest.TestCase):
    def _run_with_diag(self, message: str) -> VerusRun:
        diag = (
            '{"$message_type":"diagnostic","level":"error","message":"'
            + message
            + '","spans":[{"line_start":3}]}'
        )
        return VerusRun(
            status="ok", success=False, verified=0, errors=1, returncode=1,
            elapsed_seconds=0.0, stdout="", stderr=diag, command=(),
        )

    def _item(self) -> _BatchItem:
        item = _BatchItem(key="k0", fn_name="f")
        item.start_line = 1
        item.end_line = 5
        return item

    def test_arithmetic_overflow_counts_as_verification_failure(self) -> None:
        item = self._item()
        run = self._run_with_diag("possible arithmetic underflow/overflow")
        self.assertEqual(_parse_batch_result(run, [item]), {"k0": False})

    def test_plain_overflow_wording_also_counts(self) -> None:
        item = self._item()
        run = self._run_with_diag("possible arithmetic overflow")
        self.assertEqual(_parse_batch_result(run, [item]), {"k0": False})


if __name__ == "__main__":
    unittest.main()
