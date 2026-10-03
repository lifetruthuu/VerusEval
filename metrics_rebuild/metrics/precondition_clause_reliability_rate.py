from __future__ import annotations

from metrics_rebuild.metrics._clause_implication import compute_clause_implication_pair
from metrics_rebuild.share.lemma_implication import DEFAULT_LEMMA_TIMEOUT_SECONDS


def metric_precondition_clause_reliability_rate(
    generated_rs_path: str,
    ground_rs_path: str,
    *,
    inject_implicit_true_contract_clauses: bool = True,
    timeout_seconds: int = DEFAULT_LEMMA_TIMEOUT_SECONDS,
) -> dict:
    return compute_clause_implication_pair(
        generated_rs_path,
        ground_rs_path,
        source_side="generated",
        clause_kind="requires",
        implication_direction="reference_implies_generated_clause",
        metric_kind="precondition_clause_reliability_rate",
        note="Counts generated requires clauses Pi implied by reference preconditions P_GT.",
        inject_implicit_true_contract_clauses=inject_implicit_true_contract_clauses,
        timeout_seconds=timeout_seconds,
    )
