from __future__ import annotations

from metrics_rebuild.share.functions import strength_contexts_for_path
from metrics_rebuild.share.lemma_implication import DEFAULT_LEMMA_TIMEOUT_SECONDS
from metrics_rebuild.share.smt import semantic_strength_comparison


def _strength_classification(
    generated_rs_path: str,
    ground_rs_path: str,
    *,
    inject_implicit_true_contract_clauses: bool = True,
    timeout_seconds: int = DEFAULT_LEMMA_TIMEOUT_SECONDS,
    rlimit: float | None = None,
) -> dict:
    result = semantic_strength_comparison(
        strength_contexts_for_path(generated_rs_path),
        strength_contexts_for_path(ground_rs_path),
        lemma_reference_path=ground_rs_path,
        generated_rs_path=generated_rs_path,
        timeout_seconds=timeout_seconds,
        rlimit=rlimit,
        inject_implicit_true_contract_clauses=inject_implicit_true_contract_clauses,
    )
    result["note"] = (
        "Strength classification is based on Verus lemma implication checks over "
        "executable-function contracts."
    )
    return result


def _core_gt_proportion_metric(
    generated_rs_path: str,
    ground_rs_path: str,
    *,
    direction: str,
    inject_implicit_true_contract_clauses: bool = True,
    timeout_seconds: int = DEFAULT_LEMMA_TIMEOUT_SECONDS,
    rlimit: float | None = None,
) -> dict:
    strength = _strength_classification(
        generated_rs_path,
        ground_rs_path,
        inject_implicit_true_contract_clauses=inject_implicit_true_contract_clauses,
        timeout_seconds=timeout_seconds,
        rlimit=rlimit,
    )
    functions = list(strength.get("functions") or [])
    total = len(functions)
    if direction == "at_least_gt":
        flag_key = "generated_refines_ground"
        name = "proportion_ge_gt"
        note = "Reliability core: P_GT -> P and Q -> Q_GT, i.e. generated contract is not weaker than GT."
    elif direction == "at_most_gt":
        flag_key = "ground_refines_generated"
        name = "proportion_le_gt"
        note = "Completeness core: P -> P_GT and Q_GT -> Q, i.e. generated contract is not stronger than GT."
    else:
        raise ValueError(f"unsupported GT proportion direction: {direction}")

    passed = sum(1 for item in functions if item.get(flag_key) is True)
    failed = sum(1 for item in functions if item.get(flag_key) is False)
    unknown = total - passed - failed
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
        "metric_kind": name,
        "source_metric": "strength_classification",
        "engine": strength.get("engine"),
        "method": strength.get("method"),
        "strength_status": strength.get("status"),
        "classification": strength.get("classification"),
        "functions": [
            {
                "function": item.get("function"),
                "status": item.get("status"),
                "success": item.get(flag_key),
                "contract_relation": item.get("contract_relation"),
                "precondition_relation": item.get("precondition_relation"),
                "postcondition_relation": item.get("postcondition_relation"),
                "requires": item.get("requires"),
                "ensures": item.get("ensures"),
            }
            for item in functions
        ],
        "strength_summary": {
            key: strength.get(key)
            for key in (
                "matched_functions",
                "functions_total",
                "missing_generated_functions",
                "extra_generated_functions",
                "relation_counts",
                "precondition_relation_summary",
                "postcondition_relation_summary",
            )
        },
        "implicit_true_contract_clauses": inject_implicit_true_contract_clauses,
        "note": note,
    }


def metric_proportion_at_least_gt(
    generated_rs_path: str,
    ground_rs_path: str,
    *,
    inject_implicit_true_contract_clauses: bool = True,
    timeout_seconds: int = DEFAULT_LEMMA_TIMEOUT_SECONDS,
    rlimit: float | None = None,
) -> dict:
    return _core_gt_proportion_metric(
        generated_rs_path,
        ground_rs_path,
        direction="at_least_gt",
        inject_implicit_true_contract_clauses=inject_implicit_true_contract_clauses,
        timeout_seconds=timeout_seconds,
        rlimit=rlimit,
    )


def metric_proportion_ge_gt(
    generated_rs_path: str,
    ground_rs_path: str,
    *,
    inject_implicit_true_contract_clauses: bool = True,
    timeout_seconds: int = DEFAULT_LEMMA_TIMEOUT_SECONDS,
    rlimit: float | None = None,
) -> dict:
    return metric_proportion_at_least_gt(
        generated_rs_path,
        ground_rs_path,
        inject_implicit_true_contract_clauses=inject_implicit_true_contract_clauses,
        timeout_seconds=timeout_seconds,
        rlimit=rlimit,
    )
