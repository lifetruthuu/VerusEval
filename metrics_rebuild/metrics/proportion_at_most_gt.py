from __future__ import annotations

from metrics_rebuild.metrics.proportion_at_least_gt import _core_gt_proportion_metric
from metrics_rebuild.share.lemma_implication import DEFAULT_LEMMA_TIMEOUT_SECONDS


def metric_proportion_at_most_gt(
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
        direction="at_most_gt",
        inject_implicit_true_contract_clauses=inject_implicit_true_contract_clauses,
        timeout_seconds=timeout_seconds,
        rlimit=rlimit,
    )


def metric_proportion_le_gt(
    generated_rs_path: str,
    ground_rs_path: str,
    *,
    inject_implicit_true_contract_clauses: bool = True,
    timeout_seconds: int = DEFAULT_LEMMA_TIMEOUT_SECONDS,
    rlimit: float | None = None,
) -> dict:
    return metric_proportion_at_most_gt(
        generated_rs_path,
        ground_rs_path,
        inject_implicit_true_contract_clauses=inject_implicit_true_contract_clauses,
        timeout_seconds=timeout_seconds,
        rlimit=rlimit,
    )
