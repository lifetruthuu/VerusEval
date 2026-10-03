"""Regression tests for spec_size_complexity counting spec fn bodies."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from metrics_rebuild.metrics.spec_size_complexity import (
    metric_spec_size_complexity,
    metric_spec_size_complexity_proof,
    metric_spec_size_complexity_spec_only,
)
from metrics_rebuild.share.functions import spec_fn_blocks_for_path

_PROOF_ASSERT_ONLY = """use vstd::prelude::*;
verus! {
proof fn helper(x: int)
{
    assert(x >= 0);
    assert(x + 1 > x);
    ghost_call(x);
}

proof fn ghost_call(x: int)
{
}
}
"""

_UNINTERP_SPEC = """use vstd::prelude::*;
verus! {
uninterp spec fn helper(x: int) -> bool;

fn foo(x: int) -> (r: int)
    ensures helper(x)
{
    x
}
}
"""

_SPEC_FN_REUSED = """use vstd::prelude::*;
verus! {
spec fn positive(x: int) -> bool {
    x > 0
}

fn foo(x: int) -> (r: int)
    requires positive(x)
    ensures positive(x), positive(x)
{
    x
}

fn bar(y: int) -> (r: int)
    ensures positive(y) ==> positive(y)
{
    y
}
}
"""


class TestSpecSizeComplexitySpecFn(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _write(self, name: str, body: str) -> str:
        path = self.tmp / name
        path.write_text(body, encoding="utf-8")
        return str(path)

    def test_proof_fn_body_is_not_counted_or_double_counted(self):
        path = self._write("proof_assert_only.rs", _PROOF_ASSERT_ONLY)

        proof = metric_spec_size_complexity_proof(path, path)
        full = metric_spec_size_complexity(path, path)

        self.assertEqual(proof["gen"]["spec_clauses"], 2)
        self.assertEqual(proof["gen"]["breakdown"]["asserts"], 2)
        self.assertEqual(proof["gen"]["breakdown"]["spec_fn_bodies"], 0)
        self.assertEqual({item["kind"] for item in proof["gen"]["clauses"]}, {"assert"})

        self.assertEqual(full["gen"]["spec_clauses"], 2)
        self.assertEqual(full["gen"]["breakdown"]["proof_clauses"], 2)
        self.assertEqual(full["gen"]["breakdown"]["spec_fn_bodies"], 0)

    def test_uninterpreted_spec_fn_without_body_is_skipped(self):
        path = self._write("uninterp_spec.rs", _UNINTERP_SPEC)

        spec = metric_spec_size_complexity_spec_only(path, path)
        blocks = spec_fn_blocks_for_path(path)

        self.assertEqual(spec["gen"]["breakdown"]["contract_clauses"], 1)
        self.assertEqual(spec["gen"]["breakdown"]["spec_fn_bodies"], 0)
        self.assertEqual(spec["gen"]["spec_clauses"], 1)
        self.assertEqual(blocks, [{"name": "helper", "text": "uninterp spec fn helper(x: int) -> bool;", "body": None}])

    def test_spec_fn_body_count_is_by_definition_not_by_reference(self):
        path = self._write("spec_reused.rs", _SPEC_FN_REUSED)

        spec = metric_spec_size_complexity_spec_only(path, path)

        self.assertEqual(spec["gen"]["breakdown"]["spec_fn_bodies"], 1)
        self.assertEqual(
            [item.get("name") for item in spec["gen"]["clauses"] if item["kind"] == "spec_fn_body"],
            ["positive"],
        )


if __name__ == "__main__":
    unittest.main()
