from __future__ import annotations

from typing import Optional

from .catalog import AVAILABLE_METRIC_IDS, DEFAULT_ENABLED_METRIC_IDS, EXTENSION_METRIC_IDS
from .metrics.correct_io_pass_rate import metric_correct_io_pass_rate
from .metrics.invalid_test_filtering_rate import metric_invalid_test_filtering_rate
from .metrics.llm_as_judge_intent_consistency import metric_llm_as_judge_intent_consistency
from .metrics.llm_as_judge_spec_code_intent_consistency import metric_llm_as_judge_spec_code_intent_consistency
from .metrics.mutation_kill_rate import metric_mutation_kill_rate
from .metrics.parse_rate import metric_parse_rate
from .metrics.postcondition_clause_completeness_rate import metric_postcondition_clause_completeness_rate
from .metrics.postcondition_clause_reliability_rate import metric_postcondition_clause_reliability_rate
from .metrics.precondition_clause_completeness_rate import metric_precondition_clause_completeness_rate
from .metrics.precondition_clause_reliability_rate import metric_precondition_clause_reliability_rate
from .metrics.proportion_at_least_gt import metric_proportion_at_least_gt
from .metrics.proportion_at_most_gt import metric_proportion_at_most_gt
from .metrics.spec_redundancy_rate import metric_spec_redundancy_rate
from .metrics.spec_size_complexity import (
    metric_spec_size_complexity,
    metric_spec_size_complexity_proof,
    metric_spec_size_complexity_spec_only,
)
from .metrics.trivial_spec_ratio import metric_trivial_spec_ratio
from .metrics.type_check_rate import metric_type_check_rate
from .metrics.verification_pass_rate import metric_verification_pass_rate
from .metrics.verification_time import metric_verification_time
from .metrics.verus_textual_similarity import metric_verus_textual_similarity, metric_verus_textual_similarity_spec_only, metric_verus_textual_similarity_proof
from .metrics.wrong_io_reject_rate import metric_wrong_io_reject_rate
from .share.scoring import exception_metric
from .share.text import get_text_metric_inputs, get_text_metric_scope, text_metric_scope

METRIC_FUNCTIONS = (
    metric_verus_textual_similarity,
    metric_verus_textual_similarity_spec_only,
    metric_verus_textual_similarity_proof,
    metric_trivial_spec_ratio,
    metric_llm_as_judge_intent_consistency,
    metric_parse_rate,
    metric_type_check_rate,
    metric_verification_pass_rate,
    metric_llm_as_judge_spec_code_intent_consistency,
    metric_proportion_at_least_gt,
    metric_postcondition_clause_reliability_rate,
    metric_precondition_clause_reliability_rate,
    metric_wrong_io_reject_rate,
    metric_proportion_at_most_gt,
    metric_postcondition_clause_completeness_rate,
    metric_precondition_clause_completeness_rate,
    metric_correct_io_pass_rate,
    metric_invalid_test_filtering_rate,
    metric_mutation_kill_rate,
    metric_spec_redundancy_rate,
    metric_spec_size_complexity,
    metric_spec_size_complexity_spec_only,
    metric_spec_size_complexity_proof,
    metric_verification_time,
)


def metric_verus_spec_score(generated_rs_path: str, ground_rs_path: str) -> dict:
    return metric_verus_textual_similarity(generated_rs_path, ground_rs_path)


def metric_llm_as_judge(generated_rs_path: str, ground_rs_path: str) -> dict:
    return metric_llm_as_judge_intent_consistency(generated_rs_path, ground_rs_path)


def metric_pass(generated_rs_path: str, ground_rs_path: str) -> dict:
    return metric_verification_pass_rate(generated_rs_path, ground_rs_path)


def metric_proportion_ge_gt(generated_rs_path: str, ground_rs_path: str) -> dict:
    return metric_proportion_at_least_gt(generated_rs_path, ground_rs_path)


def metric_proportion_le_gt(generated_rs_path: str, ground_rs_path: str) -> dict:
    return metric_proportion_at_most_gt(generated_rs_path, ground_rs_path)


def metric_redundancy_rate(generated_rs_path: str, ground_rs_path: str) -> dict:
    return metric_spec_redundancy_rate(generated_rs_path, ground_rs_path)


def compute_all_metrics(
    generated_rs_path: str,
    ground_rs_path: str,
    text_scope: Optional[str] = None,
    *,
    include_mutation: bool = True,
) -> dict:
    metric_functions = (
        METRIC_FUNCTIONS
        if include_mutation
        else tuple(fn for fn in METRIC_FUNCTIONS if fn is not metric_mutation_kill_rate)
    )
    with text_metric_scope(text_scope or get_text_metric_scope()):
        report = {
            "generated_rs_path": generated_rs_path,
            "ground_rs_path": ground_rs_path,
            "text_metric_scope": get_text_metric_scope(),
            "text_metric_inputs": get_text_metric_inputs(generated_rs_path, ground_rs_path),
            "metrics": {},
            "metric_catalog": {
                "enabled_metrics": [
                    fn.__name__.replace("metric_", "")
                    for fn in metric_functions
                ],
                "available_metrics": list(AVAILABLE_METRIC_IDS),
                "extension_metrics": list(EXTENSION_METRIC_IDS),
                "table_disabled_metrics": list(EXTENSION_METRIC_IDS),
            },
            "notes": [
                "metrics_rebuild is a side-by-side rebuild; default metrics use native metrics_rebuild implementations.",
                "Default metric set follows catalog.py and excludes bug_detection_rate.",
            ],
        }
        if not include_mutation:
            report["notes"].append("Mutation metric skipped by --no-mutation.")
        for metric_fn in metric_functions:
            name = metric_fn.__name__.replace("metric_", "")
            try:
                report["metrics"][name] = metric_fn(generated_rs_path, ground_rs_path)
            except Exception as exc:
                report["metrics"][name] = exception_metric(exc)
        return report
