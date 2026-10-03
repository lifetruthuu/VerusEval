#!/usr/bin/env python3
"""Tests for triviality detection enhancements: equivalence_self, excluded_middle, nonneg, unsigned_nonneg."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics_rebuild.share.logic_utils import (
    is_syntactic_tautology,
    logic_tokens,
)
from metrics_rebuild.share.triviality import (
    trivial_clause_rate,
    trivial_clause_rate_for_path,
)


def _run() -> None:
    passed = 0
    failed = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS  {label}")
        else:
            failed += 1
            print(f"  FAIL  {label}  {detail}")

    # ── 1. Equivalence self (a <==> a) ───────────────────────────────────

    print("\n=== Equivalence self ===")

    check("x <==> x", is_syntactic_tautology("x <==> x") == "equivalence_self")
    check("(a > 0) <==> (a > 0)", is_syntactic_tautology("(a > 0) <==> (a > 0)") == "equivalence_self")
    check("a <==> b is NOT tautology", is_syntactic_tautology("a <==> b") is None)
    check("a ==> a stays implication_self", is_syntactic_tautology("a ==> a") == "implication_self")

    # ── 2. Tokenizer: <==> as single token ──────────────────────────────

    print("\n=== <==> tokenizer ===")

    tokens = logic_tokens("a <==> b")
    check("<==> tokenized correctly", tokens == ["a", "<==>", "b"], f"got {tokens}")

    # Make sure <= and ==> still work
    check("<= still works", logic_tokens("a <= b") == ["a", "<=", "b"])
    check("==> still works", logic_tokens("a ==> b") == ["a", "==>", "b"])

    # ── 3. Excluded middle (a || !a) ────────────────────────────────────

    print("\n=== Excluded middle ===")

    check("a || !a", is_syntactic_tautology("a || !a") == "excluded_middle")
    check("!a || a", is_syntactic_tautology("!a || a") == "excluded_middle")
    check("(x > 0) || !(x > 0)", is_syntactic_tautology("(x > 0) || !(x > 0)") == "excluded_middle")
    check("a || b || !a (3 parts)", is_syntactic_tautology("a || b || !a") == "excluded_middle")
    check("a || b is NOT excluded middle", is_syntactic_tautology("a || b") is None)
    check("a && !a is NOT excluded middle", is_syntactic_tautology("a && !a") is None)

    # ── 4. Nonneg tautology (len() >= 0) ────────────────────────────────

    print("\n=== Nonneg tautology (syntactic) ===")

    check("v.len() >= 0", is_syntactic_tautology("v.len() >= 0") == "nonneg_tautology")
    check("self.data.len() >= 0", is_syntactic_tautology("self.data.len() >= 0") == "nonneg_tautology")
    check("0 <= v.len()", is_syntactic_tautology("0 <= v.len()") == "nonneg_tautology")
    check("(v.len() >= 0)", is_syntactic_tautology("(v.len() >= 0)") == "nonneg_tautology")
    check("v.len() > 0 is NOT nonneg tautology", is_syntactic_tautology("v.len() > 0") is None)
    check("v.len() >= 1 is NOT nonneg tautology", is_syntactic_tautology("v.len() >= 1") is None)
    check("x >= 0 (no .len) is NOT nonneg", is_syntactic_tautology("x >= 0") is None)

    # ── 5. Composition: nonneg in implication consequent ─────────────────

    print("\n=== Composition with existing checks ===")

    check(
        "P ==> v.len() >= 0 → low_information_implication",
        is_syntactic_tautology("P ==> v.len() >= 0") == "low_information_implication",
    )
    check(
        "forall |x| v.len() >= 0 → quantifier_trivial_body",
        is_syntactic_tautology("forall |x| v.len() >= 0") == "quantifier_trivial_body",
    )
    check(
        "a <==> a && b <==> b → compound (not equivalence_self)",
        # a <==> a is not top level here (split by &&)
        is_syntactic_tautology("a <==> a && b <==> b") is not None,
    )

    # ── 6. Type-aware unsigned nonneg ────────────────────────────────────

    print("\n=== Type-aware unsigned nonneg ===")

    nonneg = {"x", "result", "n"}

    clauses_with_nonneg = [
        {"kind": "ensures", "text": "x >= 0", "normalized": "x >= 0"},
        {"kind": "ensures", "text": "result >= 0", "normalized": "result >= 0"},
        {"kind": "ensures", "text": "0 <= n", "normalized": "0 <= n"},
        {"kind": "ensures", "text": "y >= 0", "normalized": "y >= 0"},  # y not in nonneg
        {"kind": "ensures", "text": "x > 0", "normalized": "x > 0"},   # > 0, not >= 0
    ]

    result = trivial_clause_rate(clauses_with_nonneg, nonneg_names=nonneg)
    check(
        "3 out of 5 unsigned nonneg detected",
        result["trivial_clauses"] == 3,
        f"got {result['trivial_clauses']}",
    )

    flagged_clauses = {f["clause"] for f in result["flags"]}
    check("x >= 0 flagged", "x >= 0" in flagged_clauses)
    check("result >= 0 flagged", "result >= 0" in flagged_clauses)
    check("0 <= n flagged", "0 <= n" in flagged_clauses)
    check("y >= 0 NOT flagged (not in nonneg)", "y >= 0" not in flagged_clauses)
    check("x > 0 NOT flagged (strict >)", "x > 0" not in flagged_clauses)

    flagged_reasons = {f["clause"]: f["reasons"] for f in result["flags"]}
    check(
        "reason is unsigned_nonneg",
        all("unsigned_nonneg" in r for r in flagged_reasons.values()),
        f"got {flagged_reasons}",
    )

    # Without nonneg_names, same clauses should not be flagged
    result_no_types = trivial_clause_rate(clauses_with_nonneg)
    check(
        "without nonneg_names → 0 trivial",
        result_no_types["trivial_clauses"] == 0,
        f"got {result_no_types['trivial_clauses']}",
    )

    # ── 7. trivial_clause_rate_for_path integration ─────────────────────

    print("\n=== trivial_clause_rate_for_path integration ===")

    verus_source = """\
use vstd::prelude::*;

verus! {

pub fn process(v: Vec<u32>, n: u32) -> (result: u32)
    requires
        v.len() > 0,
    ensures
        result >= 0,
        v@.len() >= 0,
{
    n
}

pub fn check_signed(x: i32) -> (ret: i32)
    ensures
        ret >= 0,
{
    if x < 0 { -x } else { x }
}

}
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".rs", delete=False) as f:
        f.write(verus_source)
        tmp_path = f.name

    try:
        path_result = trivial_clause_rate_for_path(tmp_path)
        flagged = {f["clause"] for f in path_result["flags"]}
        reasons_map = {f["clause"]: f["reasons"] for f in path_result["flags"]}

        check(
            "result >= 0 flagged (u32 return)",
            "result >= 0" in flagged,
            f"flagged: {flagged}",
        )
        check(
            "result >= 0 reason is unsigned_nonneg",
            "unsigned_nonneg" in reasons_map.get("result >= 0", []),
            f"reasons: {reasons_map}",
        )
        check(
            "v@.len() >= 0 flagged (syntactic nonneg)",
            any("len" in c and ">= 0" in c for c in flagged),
            f"flagged: {flagged}",
        )
        check(
            "ret >= 0 NOT flagged (i32 return)",
            "ret >= 0" not in flagged,
            f"flagged: {flagged}",
        )
    finally:
        os.unlink(tmp_path)

    # ── 8. Backward compat: existing patterns still detected ────────────

    print("\n=== Backward compatibility ===")

    check("true still literal_true", is_syntactic_tautology("true") == "literal_true")
    check("x == x still self_comparison", is_syntactic_tautology("x == x") == "self_comparison")
    check("P ==> P still implication_self", is_syntactic_tautology("P ==> P") == "implication_self")
    check("x + 0 == x still arithmetic_identity", is_syntactic_tautology("x + 0 == x") == "arithmetic_identity")
    check("false ==> Q still low_info", is_syntactic_tautology("false ==> Q") == "low_information_implication")
    check("forall |x| true still quantifier_trivial", is_syntactic_tautology("forall |x| true") == "quantifier_trivial_body")

    # ── Summary ──────────────────────────────────────────────────────────

    print(f"\n{'='*50}")
    print(f"Total: {passed + failed}  PASS: {passed}  FAIL: {failed}")
    if failed:
        sys.exit(1)
    print("All tests passed.")


if __name__ == "__main__":
    _run()
