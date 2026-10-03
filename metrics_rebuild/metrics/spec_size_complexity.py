"""规约复杂度 (specification complexity).

This metric reports only raw structural statistics, without normalization or
any single aggregate score. It supports three scopes:

* full      — all spec/proof clauses + user-defined spec fn bodies
* spec_only — caller-visible contract clauses + user-defined spec fn bodies
* proof     — verifier-only scaffolding clauses (proof fn bodies excluded)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from metrics_rebuild.share.clauses import (
    Clause,
    FLAG_CLAUSE_KINDS,
    extract_clauses,
)
from metrics_rebuild.share.functions import spec_fn_blocks_for_path
from metrics_rebuild.share.spec_ast import analyze_expression
from metrics_rebuild.share.text import read_text

# --------------------------------------------------------------------------- #
# Clause scoping: every expression-bearing clause is scored (contract + proof),
# and user-defined spec fn bodies are counted as spec-side units.
# --------------------------------------------------------------------------- #

CONTRACT_KINDS = frozenset(
    {"requires", "ensures", "default_ensures", "returns", "recommends", "opens_invariants"}
)
PROOF_KINDS = frozenset({"assert", "decreases", "invariant", "invariant_except_break"})
SPEC_FN_KIND = "spec_fn_body"
SPEC_SCOPE_KINDS = CONTRACT_KINDS | {SPEC_FN_KIND}
FULL_SCOPE = "full"
SPEC_ONLY_SCOPE = "spec_only"
PROOF_SCOPE = "proof"


@dataclass(frozen=True)
class _FnBodyUnit:
    kind: str
    text: str
    name: str
    normalized: str = ""


def _spec_fn_body_units(path: str) -> list[_FnBodyUnit]:
    units: list[_FnBodyUnit] = []
    for block in spec_fn_blocks_for_path(path):
        body = block.get("body")
        if body:
            # Re-wrap in braces: a spec fn body may be a statement sequence
            # (`let x = ...; expr`), which only the block form parses. A body
            # that is already a single expression is unaffected, because BLOCK
            # is a transparent node.
            units.append(
                _FnBodyUnit(kind=SPEC_FN_KIND, text="{" + body + "}", name=block["name"])
            )
    return units


def _aggregate(clauses: Sequence[Clause | _FnBodyUnit]) -> dict:
    nodes = 0
    depth_max = 0
    n_quant = 0
    parse_ok = 0
    operators: set = set()
    variables: set = set()
    per_clause: List[dict] = []

    for clause in clauses:
        # Clause and _FnBodyUnit deliberately share kind/text; name is optional
        # and only used for spec-fn-body diagnostics in per_clause output.
        stats, _trig, ok = analyze_expression(clause.text)
        nodes += stats.nodes
        depth_max = max(depth_max, stats.logic_depth)
        n_quant += stats.n_quant
        parse_ok += 1 if ok else 0
        operators |= set(stats.operators)
        variables |= set(stats.variables)
        per_clause.append(
            {
                "kind": clause.kind,
                "nodes": stats.nodes,
                "logic_depth": stats.logic_depth,
                "n_quant": stats.n_quant,
                "distinct_operators": sorted(stats.operators),
                "distinct_variables": sorted(stats.variables),
                "parse_ok": ok,
            }
        )
        name = getattr(clause, "name", None)
        if name:
            per_clause[-1]["name"] = name

    operator_count = len(operators)
    variable_count = len(variables)
    clause_count = len(clauses)
    parse_failed = clause_count - parse_ok
    return {
        "clauses": clause_count,
        "nodes": nodes,
        "logic_depth": depth_max,
        "n_quant": n_quant,
        "vocabulary": operator_count + variable_count,
        "distinct_operators_count": operator_count,
        "distinct_variable_names_count": variable_count,
        "distinct_operators": sorted(operators),
        "distinct_variable_names": sorted(variables),
        "parse_ok_clauses": parse_ok,
        "parse_failed_clauses": parse_failed,
        "parse_coverage": parse_ok / clause_count if clause_count else 1.0,
        "per_clause": per_clause,
    }


def _scoped_clauses(path: str, scope: str) -> list[Clause | _FnBodyUnit]:
    clauses = [c for c in extract_clauses(path) if c.kind not in FLAG_CLAUSE_KINDS]
    all_units = clauses + _spec_fn_body_units(path)
    if scope == FULL_SCOPE:
        return all_units
    if scope == SPEC_ONLY_SCOPE:
        return [c for c in all_units if c.kind in SPEC_SCOPE_KINDS]
    if scope == PROOF_SCOPE:
        return [c for c in all_units if c.kind in PROOF_KINDS]
    raise ValueError(f"unsupported_spec_size_complexity_scope:{scope}")


def _analyze_path(path: str, *, scope: str) -> dict:
    clauses = _scoped_clauses(path, scope)
    agg = _aggregate(clauses)

    contract = [c for c in clauses if c.kind in CONTRACT_KINDS]
    proof = [c for c in clauses if c.kind in PROOF_KINDS]
    spec_fn_bodies = sum(1 for c in clauses if c.kind == SPEC_FN_KIND)
    proof_counts: Dict[str, int] = {}
    for clause in proof:
        proof_counts[clause.kind] = proof_counts.get(clause.kind, 0) + 1

    return {
        "scope": scope,
        "spec_clauses": agg["clauses"],
        "node_count": agg["nodes"],
        "logic_depth": agg["logic_depth"],
        "quantifier_count": agg["n_quant"],
        "vocabulary_size": agg["vocabulary"],
        "distinct_operators_count": agg["distinct_operators_count"],
        "distinct_variable_names_count": agg["distinct_variable_names_count"],
        "distinct_operators": agg["distinct_operators"],
        "distinct_variable_names": agg["distinct_variable_names"],
        "parse_ok_clauses": agg["parse_ok_clauses"],
        "parse_failed_clauses": agg["parse_failed_clauses"],
        "parse_coverage": agg["parse_coverage"],
        "breakdown": {
            "contract_clauses": len(contract),
            "proof_clauses": len(proof),
            "spec_fn_bodies": spec_fn_bodies,
            "asserts": proof_counts.get("assert", 0),
            "decreases": proof_counts.get("decreases", 0),
            "loop_invariants": proof_counts.get("invariant", 0)
            + proof_counts.get("invariant_except_break", 0),
        },
        "clauses": agg["per_clause"][:80],
    }


def _metric_spec_size_complexity_for_scope(
    generated_rs_path: str,
    ground_rs_path: str,
    *,
    scope: str,
    metric_kind: str,
    note: str,
) -> dict:
    try:
        read_text(generated_rs_path)
    except OSError as exc:
        return {"status": "unavailable", "score": None, "reason": str(exc)}

    gen = _analyze_path(generated_rs_path, scope=scope)
    ref = _analyze_path(ground_rs_path, scope=scope)
    delta = {
        "node_count": gen["node_count"] - ref["node_count"],
        "logic_depth": gen["logic_depth"] - ref["logic_depth"],
        "quantifier_count": gen["quantifier_count"] - ref["quantifier_count"],
        "vocabulary_size": gen["vocabulary_size"] - ref["vocabulary_size"],
    }

    gen_parse_failed = gen["parse_failed_clauses"]
    ref_parse_failed = ref["parse_failed_clauses"]
    return {
        # Status reflects the generated side only: downstream consumers read
        # `gen.*` and gate on `status == "ok"`, so folding reference-side parse
        # failures in here would discard perfectly usable generated data.
        # `delta_reliable` carries the reference-side caveat instead.
        "status": "partial" if gen_parse_failed else "ok",
        "score": None,
        "metric_kind": metric_kind,
        "scope": scope,
        "gen": gen,
        "ref": ref,
        "delta": delta,
        "delta_reliable": not (gen_parse_failed or ref_parse_failed),
        "parse_failed_clauses": gen_parse_failed + ref_parse_failed,
        "gen_parse_failed_clauses": gen_parse_failed,
        "ref_parse_failed_clauses": ref_parse_failed,
        "method": "pratt_ast_raw_v6",
        "note": note,
    }


def metric_spec_size_complexity(generated_rs_path: str, ground_rs_path: str) -> dict:
    return _metric_spec_size_complexity_for_scope(
        generated_rs_path,
        ground_rs_path,
        scope=FULL_SCOPE,
        metric_kind="spec_size_complexity",
        note=(
            "Full scope: all spec/proof clauses + user-defined spec fn bodies. "
            "只报告 4 个原始结构指标，不做归一化和总分：节点数（非透明节点）、逻辑深度、"
            "量词个数、词汇表大小（不同运算符 + 不同变量名，类型名不算变量）。用户定义的 "
            "spec fn body 按定义点计一次（引用不展开、不递归）；无 body 的 uninterpreted "
            "spec fn 不计。proof fn body 不计入。逻辑深度只在 LOGICAL/QUANT/IF/MATCH 上"
            "加深；连续相同逻辑连接符链算 1 层，else-if 链是扁平分派同样算 1 层。"
        ),
    )


def metric_spec_size_complexity_spec_only(
    generated_rs_path: str,
    ground_rs_path: str,
) -> dict:
    return _metric_spec_size_complexity_for_scope(
        generated_rs_path,
        ground_rs_path,
        scope=SPEC_ONLY_SCOPE,
        metric_kind="spec_size_complexity_spec_only",
        note=(
            "Spec scope: caller-visible contract clauses (requires, ensures, returns, "
            "default_ensures, recommends, opens_invariants) + user-defined spec fn "
            "bodies. 只报告 4 个原始结构指标，不做归一化和总分。spec fn body 按定义点"
            "计一次，引用不展开。"
        ),
    )


def metric_spec_size_complexity_proof(
    generated_rs_path: str,
    ground_rs_path: str,
) -> dict:
    return _metric_spec_size_complexity_for_scope(
        generated_rs_path,
        ground_rs_path,
        scope=PROOF_SCOPE,
        metric_kind="spec_size_complexity_proof",
        note=(
            "Proof scope: verifier-only scaffolding clauses (assert, invariant, "
            "invariant_except_break, decreases). 只报告 4 个原始结构指标，不做归一化和"
            "总分。proof fn body 不计入，避免把语句序列当表达式解析并与内部 assert 双重计数。"
        ),
    )
