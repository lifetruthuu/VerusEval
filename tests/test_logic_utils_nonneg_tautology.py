"""Regression test for the nonneg-tautology false positive: a chained
comparison like `0 <= n < a.len()` must NOT be treated as the tautology
`0 <= a.len()` just because its right side ends with `.len()`."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics_rebuild.share.logic_utils import is_simple_tautology, is_syntactic_tautology


class NonnegTautologyTest(unittest.TestCase):
    def test_chained_comparison_is_not_a_tautology(self) -> None:
        # `n < a.len()` is substantive; the whole clause is not vacuous.
        self.assertFalse(is_simple_tautology("0 <= n < a.len()"))
        self.assertIsNone(is_syntactic_tautology("0 <= n < a.len()"))
        self.assertFalse(is_simple_tautology("a.len() > n >= 0"))

    def test_bare_len_bounds_stay_tautologies(self) -> None:
        # Genuinely vacuous nonneg bounds on an unsigned length must still fire.
        self.assertEqual(is_syntactic_tautology("0 <= a.len()"), "nonneg_tautology")
        self.assertEqual(is_syntactic_tautology("a.len() >= 0"), "nonneg_tautology")
        self.assertEqual(is_syntactic_tautology("0 <= (a.len())"), "nonneg_tautology")

    def test_non_len_nonneg_not_flagged_by_this_check(self) -> None:
        self.assertIsNone(is_syntactic_tautology("0 <= n"))


if __name__ == "__main__":
    unittest.main()
