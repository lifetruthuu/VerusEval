#!/usr/bin/env python3
"""Tests for _Z3ExprParser enhancements: old(), deref/ref, as, is_empty."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run() -> None:
    try:
        import z3
    except ImportError:
        print("SKIP: z3 not installed")
        return

    from metrics_rebuild.share.precondition_satisfiability import (
        _Z3ExprParser,
        _tokens,
        _z3_expr,
        precondition_satisfiability,
        UnsupportedExpression,
    )

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

    def parse_ok(expr: str, variables: dict | None = None) -> bool:
        if variables is None:
            variables = {}
        try:
            _z3_expr(expr, variables)
            return True
        except UnsupportedExpression:
            return False

    # ── 1. old() with complex expressions ─────────────────────────────────

    print("\n=== old() with complex expressions ===")

    check("old(x) — simple ident still works", parse_ok("old(x) == 0"))

    check("old(*x) — deref inside old", parse_ok("old(*x) == 0"))

    v = {"v": z3.Array("v", z3.IntSort(), z3.IntSort()), "v.len()": z3.Int("v_len")}
    check("old(v).len() — postfix on old", parse_ok("old(v).len() >= 0", dict(v)))

    check("old(self.field) — field inside old", parse_ok("old(self.field) == 0"))

    v2 = {"v": z3.Array("v", z3.IntSort(), z3.IntSort()), "v.len()": z3.Int("v_len")}
    check("old(v.len()) — method inside old", parse_ok("old(v.len()) >= 0", dict(v2)))

    # old(x) and x should be distinct variables
    vars_distinct = {}
    expr_old = _z3_expr("old(x) > 0", vars_distinct)
    expr_cur = _z3_expr("x < 0", vars_distinct)
    solver = z3.Solver()
    solver.add(expr_old)
    solver.add(expr_cur)
    check("old(x) and x are distinct", solver.check() == z3.sat)

    # ── 2. Dereference * and reference & ──────────────────────────────────

    print("\n=== Dereference * and reference & ===")

    check("*x parses", parse_ok("*x == 0"))

    check("**x parses (double deref)", parse_ok("**x == 0"))

    check("&x parses", parse_ok("&x == 0"))

    check("&mut x parses", parse_ok("&mut x == 0"))

    # *x and x should resolve to the same variable
    vars_deref = {}
    e1 = _z3_expr("*x == 5", vars_deref)
    e2 = _z3_expr("x == 5", vars_deref)
    check("*x and x are same variable", z3.eq(e1, e2))

    # In multiplication context: x * *y should work
    check("x * *y parses", parse_ok("x * *y == 0"))

    # ── 3. as with multi-token types ──────────────────────────────────────

    print("\n=== as with multi-token types ===")

    check("as int (baseline)", parse_ok("x as int == 0"))

    check("as u64", parse_ok("x as u64 == 0"))

    check("as Seq<int>", parse_ok("x as Seq<int> == 0"))

    check("as &u32", parse_ok("x as &u32 == 0"))

    check("as &mut u32", parse_ok("x as &mut u32 == 0"))

    check("as Vec<Vec<u32>>", parse_ok("x as Vec<Vec<u32>> == 0"))

    # chained casts
    check("x as u32 as int", parse_ok("x as u32 as int == 0"))

    # ── 4. is_empty() semantic modeling ───────────────────────────────────

    print("\n=== is_empty() semantic modeling ===")

    # is_empty() should be equivalent to len() == 0
    v_arr = {"v": z3.Array("v", z3.IntSort(), z3.IntSort()), "v.len()": z3.Int("v_len")}
    v_arr2 = dict(v_arr)
    e_empty = _z3_expr("v.is_empty()", v_arr)
    e_len_zero = _z3_expr("v.len() == 0", v_arr2)
    solver = z3.Solver()
    solver.add(e_empty != e_len_zero)
    check("is_empty() ↔ len()==0", solver.check() == z3.unsat)

    # is_empty() with len > 0 should be unsat
    v_arr3 = {"v": z3.Array("v", z3.IntSort(), z3.IntSort()), "v.len()": z3.Int("v_len")}
    solver2 = z3.Solver()
    solver2.add(_z3_expr("v.is_empty()", v_arr3))
    solver2.add(_z3_expr("v.len() > 0", v_arr3))
    check("is_empty() && len()>0 is unsat", solver2.check() == z3.unsat)

    # ── 5. Integration: precondition_satisfiability with new features ─────

    print("\n=== Integration tests ===")

    # requires old(*x) == 0 — should be satisfiable
    result = precondition_satisfiability([
        {"kind": "requires", "text": "old(*x) == 0", "normalized": "old(*x) == 0"}
    ])
    check("old(*x) in requires → satisfiable", result["score"] != 0.0)

    # requires *v.len() > 0 — deref on collection
    result2 = precondition_satisfiability([
        {"kind": "requires", "text": "*n > 0", "normalized": "*n > 0"}
    ])
    check("*n > 0 in requires → satisfiable", result2["score"] != 0.0)

    # requires v.is_empty() && v.len() > 0 — should be unsat
    result3 = precondition_satisfiability([
        {
            "function": "test_fn",
            "mode": "exec",
            "parameters": [{"name": "v", "type": "Vec<u32>"}],
            "requires": [
                {"kind": "requires", "text": "v.is_empty()", "normalized": "v.is_empty()"},
                {"kind": "requires", "text": "v.len() > 0", "normalized": "v.len() > 0"},
            ],
        }
    ])
    check(
        "is_empty() && len()>0 → unsat (contradiction detected)",
        result3["score"] == 0.0,
        f"got score={result3.get('score')}",
    )

    # requires x as Seq<int> == y — should not crash
    result4 = precondition_satisfiability([
        {"kind": "requires", "text": "x as Seq<int> == y", "normalized": "x as Seq<int> == y"}
    ])
    check("as Seq<int> does not crash", result4["status"] != "unavailable")

    # ── Summary ───────────────────────────────────────────────────────────

    print(f"\n{'='*50}")
    print(f"Total: {passed + failed}  PASS: {passed}  FAIL: {failed}")
    if failed:
        sys.exit(1)
    print("All tests passed.")


if __name__ == "__main__":
    _run()
