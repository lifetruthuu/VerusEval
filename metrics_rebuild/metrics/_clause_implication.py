from __future__ import annotations

from metrics_rebuild.share.functions import strength_contexts_for_path
from metrics_rebuild.share.lemma_implication import DEFAULT_LEMMA_TIMEOUT_SECONDS
from metrics_rebuild.share.smt import clause_implication_metric


def compute_clause_implication_pair(
    generated_rs_path: str,
    ground_rs_path: str,
    *,
    source_side: str,
    clause_kind: str,
    implication_direction: str,
    metric_kind: str,
    note: str,
    inject_implicit_true_contract_clauses: bool = True,
    timeout_seconds: int = DEFAULT_LEMMA_TIMEOUT_SECONDS,
) -> dict:
    generated_contexts = strength_contexts_for_path(generated_rs_path)
    ground_contexts = strength_contexts_for_path(ground_rs_path)
    generated = clause_implication_metric(
        generated_contexts,
        ground_contexts,
        reference_rs_path=ground_rs_path,
        source_side=source_side,
        clause_kind=clause_kind,
        implication_direction=implication_direction,
        metric_kind=metric_kind,
        note=note,
        generated_rs_path=generated_rs_path,
        timeout_seconds=timeout_seconds,
        inject_implicit_true_contract_clauses=inject_implicit_true_contract_clauses,
    )
    ground = clause_implication_metric(
        ground_contexts,
        ground_contexts,
        reference_rs_path=ground_rs_path,
        source_side=source_side,
        clause_kind=clause_kind,
        implication_direction=implication_direction,
        metric_kind=metric_kind,
        note=f"Self-check for reference {clause_kind} clauses.",
        timeout_seconds=timeout_seconds,
        inject_implicit_true_contract_clauses=inject_implicit_true_contract_clauses,
    )
    delta = None
    if generated.get("score") is not None and ground.get("score") is not None:
        delta = generated["score"] - ground["score"]
    return {"generated": generated, "ground": ground, "delta": delta}


__all__ = ["compute_clause_implication_pair"]
