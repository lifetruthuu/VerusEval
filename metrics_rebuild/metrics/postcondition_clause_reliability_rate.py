from __future__ import annotations

from metrics_rebuild.metrics._clause_implication import compute_clause_implication_pair
from metrics_rebuild.share.lemma_implication import DEFAULT_LEMMA_TIMEOUT_SECONDS


def metric_postcondition_clause_reliability_rate(
    generated_rs_path: str,
    ground_rs_path: str,
    *,
    inject_implicit_true_contract_clauses: bool = True,
    timeout_seconds: int = DEFAULT_LEMMA_TIMEOUT_SECONDS,
) -> dict:
    result = compute_clause_implication_pair(
        generated_rs_path,
        ground_rs_path,
        source_side="reference",
        clause_kind="ensures",
        implication_direction="generated_implies_reference_clause",
        metric_kind="postcondition_clause_reliability_rate",
        note="Counts reference ensures clauses Qi_GT implied by generated postconditions Q.",
        inject_implicit_true_contract_clauses=inject_implicit_true_contract_clauses,
        timeout_seconds=timeout_seconds,
    )
    result["assertion_source"] = "reference_postconditions"
    result["metric_kind"] = "postcondition_clause_reliability_rate"
    result["renamed_from"] = "通过的断言数量"
    return result
