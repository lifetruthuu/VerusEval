#!/usr/bin/env python3
"""Regression tests for Verus textual similarity preprocessing."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics_rebuild.share.clauses import spec_text_for_text_similarity


class TestVerusTextualSimilarityInputs(unittest.TestCase):
    def _write_temp(self, source: str) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False)
        handle.write(source)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_spec_fn_body_is_included_in_spec_only_text(self):
        path = self._write_temp(
            """
verus! {
pub open spec fn is_prime(n: int) -> bool {
    if n < 2 {
        false
    } else {
        forall|k: int| 2 <= k < n ==> n % k != 0
    }
}

fn prime_length(str: &[char]) -> (result: bool)
    ensures
        result == is_prime(str.len() as int),
{
    true
}
}
"""
        )

        text = spec_text_for_text_similarity(path)

        self.assertIn("spec_fn pub open spec fn is_prime", text)
        self.assertIn("forall|k: int| 2 <= k < n ==> n % k != 0", text)
        self.assertIn("ensures result == is_prime", text)

    def test_proof_fn_item_only_includes_declaration(self):
        path = self._write_temp(
            """
verus! {
proof fn lemma_nonnegative(x: int)
    requires
        x > 0,
    ensures
        x >= 0,
{
    assert(x >= 0);
}
}
"""
        )

        lines = spec_text_for_text_similarity(path).splitlines()

        self.assertEqual(len(lines), 1)
        self.assertIn("proof_fn proof fn lemma_nonnegative(x: int)", lines[0])
        self.assertIn("requires x > 0", lines[0])
        self.assertIn("ensures x >= 0", lines[0])


if __name__ == "__main__":
    unittest.main()
