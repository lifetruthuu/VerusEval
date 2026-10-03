from __future__ import annotations

import re
from typing import Any, Sequence

from metrics_rebuild.share._shared import (
    IMPLICIT_TRUE_CONTRACT_CLAUSES_DEFAULT,
    materialize_contract_clauses,
)
from metrics_rebuild.share.lemma_implication import (
    DEFAULT_LEMMA_TIMEOUT_SECONDS,
    lemma_implication_check,
)
from metrics_rebuild.share.precondition_satisfiability import (
    UnsupportedExpression,
    precondition_satisfiability as _precondition_satisfiability,
)
from metrics_rebuild.share.semantic_strength import semantic_strength_comparison as _semantic_strength_comparison

def precondition_satisfiability(items: Sequence[Any], timeout_ms: int = 1000) -> dict:
    return _precondition_satisfiability(items, timeout_ms=timeout_ms)


def semantic_strength_comparison(
    generated_contexts: Sequence[Any],
    ground_contexts: Sequence[Any],
    timeout_seconds: int = 20,
    *,
    rlimit: float | None = None,
    lemma_reference_path: str | None = None,
    generated_rs_path: str | None = None,
    **_kwargs: Any,
) -> dict:
    return _semantic_strength_comparison(
        generated_contexts,
        ground_contexts,
        timeout_seconds=timeout_seconds,
        rlimit=rlimit,
        lemma_reference_path=lemma_reference_path,
        generated_rs_path=generated_rs_path,
    )


def _tag_lemma_source(clauses: Sequence[dict], source: str) -> list[dict]:
    return [{**clause, "_lemma_source": source} for clause in clauses]


def _implication_check_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value or "implication")


def clause_implication_details_for_context(
    *,
    candidate: dict | None,
    reference: dict,
    reference_rs_path: str,
    source_side: str,
    clause_kind: str,
    implication_direction: str,
    generated_rs_path: str | None = None,
    timeout_seconds: int = DEFAULT_LEMMA_TIMEOUT_SECONDS,
    inject_implicit_true_contract_clauses: bool = IMPLICIT_TRUE_CONTRACT_CLAUSES_DEFAULT,
) -> dict:
    function_name = reference.get("function")
    if candidate is None:
        source_clauses = materialize_contract_clauses(
            reference.get(clause_kind) or [],
            kind=clause_kind,
            inject_implicit_true=inject_implicit_true_contract_clauses,
        )
        details = [
            {
                "id": f"{function_name or 'target'}_{clause_kind}_{index:03d}",
                "function": function_name,
                "clause": clause,
                "success": False,
                "outcome": "failed",
                "reason": "missing_generated_function",
                "implication_check": None,
            }
            for index, clause in enumerate(source_clauses, 1)
        ]
        return {
            "status": "ok" if source_clauses else "not_available",
            "score": 0.0 if source_clauses else None,
            "passed": 0,
            "failed": len(source_clauses),
            "unknown": 0,
            "total": len(source_clauses),
            "determined": len(source_clauses),
            "coverage": 1.0 if source_clauses else None,
            "function": function_name,
            "details": details,
        }

    gen_clauses = materialize_contract_clauses(
        candidate.get(clause_kind) or [],
        kind=clause_kind,
        inject_implicit_true=inject_implicit_true_contract_clauses,
    )
    ref_clauses = materialize_contract_clauses(
        reference.get(clause_kind) or [],
        kind=clause_kind,
        inject_implicit_true=inject_implicit_true_contract_clauses and source_side == "reference",
    )
    source_clauses = ref_clauses if source_side == "reference" else gen_clauses

    passed = failed = unknown = 0
    details: list[dict] = []
    for index, clause in enumerate(source_clauses, 1):
        if implication_direction == "reference_implies_generated_clause":
            antecedent = _tag_lemma_source(ref_clauses, "reference")
            consequent = _tag_lemma_source([clause], "generated")
            expected = "P_GT -> Pi"
        elif implication_direction == "generated_implies_reference_clause":
            # 前置条件方向：P -> Pi_GT；后置条件方向（可靠率）：Q -> Qi_GT
            # 均不在 antecedent 中加入 requires，直接用 gen_clauses 作为蕴含前件
            antecedent = _tag_lemma_source(gen_clauses, "generated")
            consequent = _tag_lemma_source([clause], "reference")
            expected = "P -> Pi_GT or Q -> Qi_GT"
        elif implication_direction == "reference_post_implies_generated_clause":
            # 后置条件子句完备率：Q_GT -> Qi，不将前置条件加入 antecedent
            antecedent = _tag_lemma_source(ref_clauses, "reference")
            consequent = _tag_lemma_source([clause], "generated")
            expected = "Q_GT -> Qi"
        else:
            raise ValueError(f"unsupported clause implication direction: {implication_direction}")

        check = lemma_implication_check(
            reference_rs_path=reference_rs_path,
            reference_context=reference,
            generated_context=candidate,
            function_name=str(function_name or "target"),
            check_name=f"{_implication_check_name(implication_direction)}_{clause_kind}_{index:03d}",
            antecedent=antecedent,
            consequent=consequent,
            generated_rs_path=generated_rs_path,
            timeout_seconds=timeout_seconds,
        )
        holds = check.get("holds")
        if holds is True:
            success = True
            outcome = "passed"
            passed += 1
        elif holds is False:
            success = False
            outcome = "failed"
            failed += 1
        else:
            success = None
            outcome = "unknown"
            unknown += 1
        details.append(
            {
                "id": f"{function_name or 'target'}_{clause_kind}_{index:03d}",
                "function": function_name,
                "clause": clause,
                "success": success,
                "outcome": outcome,
                "expected_implication": expected,
                "implication_check": check,
            }
        )

    total = len(source_clauses)
    determined = passed + failed
    status = "not_available" if total == 0 else "ok" if unknown == 0 else "partial"
    return {
        "status": status,
        "score": passed / determined if determined else None,
        "passed": passed,
        "failed": failed,
        "unknown": unknown,
        "total": total,
        "determined": determined,
        "coverage": determined / total if total else None,
        "function": function_name,
        "type_map": {},
        "source_side": source_side,
        "clause_kind": clause_kind,
        "implication_direction": implication_direction,
        "engine": "verus_lemma",
        "details": details,
        "implicit_true_contract_clauses": inject_implicit_true_contract_clauses,
    }


def clause_implication_metric(
    generated_contexts: Sequence[dict],
    reference_contexts: Sequence[dict],
    *,
    reference_rs_path: str,
    source_side: str,
    clause_kind: str,
    implication_direction: str,
    metric_kind: str,
    note: str,
    generated_rs_path: str | None = None,
    timeout_seconds: int = DEFAULT_LEMMA_TIMEOUT_SECONDS,
    inject_implicit_true_contract_clauses: bool = IMPLICIT_TRUE_CONTRACT_CLAUSES_DEFAULT,
) -> dict:
    candidate_map = {
        context.get("function"): context
        for context in generated_contexts
        if context.get("function")
    }
    references = [
        context
        for context in reference_contexts
        if context.get("function")
    ]
    function_results = [
        clause_implication_details_for_context(
            candidate=candidate_map.get(reference.get("function")),
            reference=reference,
            reference_rs_path=reference_rs_path,
            source_side=source_side,
            clause_kind=clause_kind,
            implication_direction=implication_direction,
            generated_rs_path=generated_rs_path,
            timeout_seconds=timeout_seconds,
            inject_implicit_true_contract_clauses=inject_implicit_true_contract_clauses,
        )
        for reference in references
    ]
    total = sum(int(result.get("total", 0) or 0) for result in function_results)
    passed = sum(int(result.get("passed", 0) or 0) for result in function_results)
    failed = sum(int(result.get("failed", 0) or 0) for result in function_results)
    unknown = sum(int(result.get("unknown", 0) or 0) for result in function_results)
    determined = passed + failed
    details: list[dict] = []
    for result in function_results:
        details.extend(result.get("details") or [])
    status = "not_available" if total == 0 else "ok" if unknown == 0 else "partial"
    return {
        "status": status,
        "score": passed / determined if determined else None,
        "passed": passed,
        "failed": failed,
        "unknown": unknown,
        "total": total,
        "determined": determined,
        "coverage": determined / total if total else None,
        "metric_kind": metric_kind,
        "method": "per_clause_verus_lemma",
        "engine": "verus_lemma",
        "source_side": source_side,
        "clause_kind": clause_kind,
        "implication_direction": implication_direction,
        "functions": function_results,
        "assertions": details[:80],
        "details": details[:80],
        "implicit_true_contract_clauses": inject_implicit_true_contract_clauses,
        "note": note,
    }


__all__ = [
    "UnsupportedExpression",
    "clause_implication_details_for_context",
    "clause_implication_metric",
    "lemma_implication_check",
    "precondition_satisfiability",
    "semantic_strength_comparison",
]
