"""Summarize the current target's metric results without running new checks."""
from __future__ import annotations


PRECONDITION_CHECKS = {
    "pre_ref_to_gen": "ground_implies_generated",
    "pre_gen_to_ref": "generated_implies_ground",
}
POSTCONDITION_METRICS = {
    "post_gen_to_ref": "postcondition_clause_reliability_rate",
    "post_ref_to_gen": "postcondition_clause_completeness_rate",
}


def resolve_summary_target(generated_path, reference_path):
    """Reuse the evaluator's target selection; retain resolution failures as data."""
    try:
        from metrics_rebuild.share.functions import resolve_pair_target

        resolved = resolve_pair_target(generated_path, reference_path)
        return {key: resolved[key] for key in
                ("status", "reason", "function", "selection", "diagnostics", "error")
                if key in resolved}
    except Exception as error:
        return {"status": "unavailable", "reason": "target_resolution_failed",
                "error": f"{type(error).__name__}: {error}"}


def generated_metric(metrics, name):
    node = metrics.get(name) or {}
    return node.get("generated", node)


def target_function(node, target):
    matches = [entry for entry in node.get("functions") or []
               if target and entry.get("function") == target]
    return matches[0] if len(matches) == 1 else {}


def implication_state(check):
    if not check:
        return "unavailable"
    if check.get("status") == "valid" and check.get("holds") is True:
        return "valid"
    if check.get("status") == "invalid" and check.get("holds") is False:
        return "invalid"
    return "unknown"


def clause_direction(node, target):
    function = target_function(node, target)
    if not function:
        return "unavailable"
    checks = [implication_state(detail.get("implication_check"))
              for detail in function.get("details") or []]
    # Incomplete or unresolved evidence cannot establish the whole direction.
    if not checks or len(checks) != function.get("total"):
        return "unknown"
    if all(state == "valid" for state in checks):
        return "valid"
    if all(state in {"valid", "invalid"} for state in checks):
        return "invalid"
    return "unknown"


def summarize_target(result, resolved, sample_id):
    """Build an additional summary, leaving all original metric nodes intact."""
    metrics = result.get("metrics") or {}
    target = resolved.get("function")
    eligible = True if resolved.get("status") == "ok" else (
        False if resolved.get("reason") in {
            "reference_target_not_found", "generated_target_not_found"} else None)
    verification = generated_metric(metrics, "verification_pass_rate")
    stage = {"verified": "accepted", "parse_error": "parse_failed",
             "frontend_failed": "frontend_failed", "verification_failed": "proof_failed"}.get(
                 verification.get("outcome_status"), "unknown")
    if verification.get("status") != "ok":
        stage = "unknown"

    whole = target_function(generated_metric(metrics, "proportion_at_least_gt"), target)
    requires = whole.get("requires") or {}
    directions = {key: implication_state(requires.get(check))
                  for key, check in PRECONDITION_CHECKS.items()}
    directions.update({key: clause_direction(generated_metric(metrics, metric), target)
                       for key, metric in POSTCONDITION_METRICS.items()})
    triviality = {}
    details = generated_metric(metrics, "trivial_spec_ratio").get("details") or {}
    for part, flag in (("precondition_falsity", "always_false"),
                       ("postcondition_truth", "always_true")):
        entry = target_function(details.get(part) or {}, target)
        state = "other"
        if entry.get(flag) is True:
            state = "unfavorable"
        elif entry.get("outcome") == "no_requires_default_true" or (
                entry.get("outcome") == "not_trivial" and entry.get("status") == "invalid"):
            state = "favorable"
        triviality[part] = state

    # A missing task function must not inherit results from helper functions.
    if eligible is not True:
        directions = {key: "unavailable" for key in directions}
        triviality = {key: "other" for key in triviality}
    return {
        "sample_id": sample_id,
        "target_function": target,
        "method": "summary_of_current_metrics",
        "target_resolution": resolved,
        "verification_stage": stage,
        "verification_source": "metrics.verification_pass_rate.generated",
        "analysis_eligible": eligible,
        "exclusion_reason": resolved.get("reason") if eligible is not True else None,
        "directions": directions,
        "triviality": triviality,
        "triviality_source": "metrics.trivial_spec_ratio.generated.details",
        "formal_evidence": {
            **{key: {"source_metric": "proportion_at_least_gt", "part": "requires", "check": check}
               for key, check in PRECONDITION_CHECKS.items()},
            **{key: {"source_metric": metric, "aggregation": "all_target_clauses"}
               for key, metric in POSTCONDITION_METRICS.items()},
        },
    }
