from __future__ import annotations

from metrics_rebuild.metrics.llm.common import llm_judge_prompt, source_hash
from metrics_rebuild.share.llm_client import call_llm_json
from metrics_rebuild.share.functions import resolve_pair_target, target_mismatch_metric
from metrics_rebuild.share.scoring import clamp_score, not_available_metric
from metrics_rebuild.share.verus_runner import verus_frontend_run_to_dict


def metric_llm_as_judge_intent_consistency(generated_rs_path: str, ground_rs_path: str) -> dict:
    generated_frontend = verus_frontend_run_to_dict(generated_rs_path)
    ground_frontend = verus_frontend_run_to_dict(ground_rs_path)
    generated_outcome = generated_frontend.get("outcome_status")
    ground_outcome = ground_frontend.get("outcome_status")
    if generated_outcome in {"parse_error", "tool_error", "timeout"} or generated_frontend.get("success") is not True:
        reason = f"generated_frontend_not_ok:{generated_outcome}"
        return {
            "status": "ok",
            "score": 0.0,
            "generated": {
                "status": "ok",
                "score": 0.0,
                "verdict": "invalid_frontend",
                "missing_constraints": [],
                "extra_or_overstrong_constraints": [],
                "vacuity_risks": [],
                "reasoning": "Generated file does not pass Verus frontend checks; semantic-intent judging is gated to avoid rewarding unparseable or ill-typed specifications.",
                "frontend": generated_frontend,
            },
            "ground": {
                "status": "ok" if ground_frontend.get("success") is True else "frontend_failed",
                "score": 1.0 if ground_frontend.get("success") is True else None,
                "verdict": "reference",
                "reasoning": "Reference specification is the comparison target.",
                "frontend": ground_frontend,
            },
            "delta": -1.0,
            "llm": {"status": "skipped", "reason": reason},
            "cached": False,
            "rubric": "LLM semantic judging is skipped when the generated file fails Verus frontend checks.",
            "raw_judge": {
                "score": 0.0,
                "verdict": "invalid_frontend",
                "reasoning": reason,
            },
        }
    if ground_outcome in {"parse_error", "tool_error", "timeout"} or ground_frontend.get("success") is not True:
        return not_available_metric(
            f"reference_frontend_not_ok:{ground_outcome}",
            source="llm_as_judge_frontend_gate",
        )
    alignment = resolve_pair_target(generated_rs_path, ground_rs_path)
    if alignment.get("status") != "ok":
        return target_mismatch_metric(alignment, metric_kind="llm_as_judge_intent_consistency")
    function_name = alignment["function"]
    if source_hash(generated_rs_path) == source_hash(ground_rs_path):
        identical = {
            "verdict": "equivalent",
            "score": 1.0,
            "missing_constraints": [],
            "extra_or_overstrong_constraints": [],
            "vacuity_risks": [],
            "reasoning": "Generated and reference files are identical after source hashing.",
        }
        return {
            "status": "ok",
            "score": 1.0,
            "generated": {
                "status": "ok",
                "score": 1.0,
                "verdict": "equivalent",
                "missing_constraints": [],
                "extra_or_overstrong_constraints": [],
                "vacuity_risks": [],
                "reasoning": identical["reasoning"],
            },
            "ground": {
                "status": "ok",
                "score": 1.0,
                "verdict": "reference",
                "reasoning": "Reference specification is the comparison target.",
            },
            "delta": 0.0,
            "llm": {"status": "skipped", "reason": "identical_files"},
            "cached": False,
            "rubric": "LLM compares behavioral intent of Verus spec/proof clauses; it is advisory and not used as an oracle for dynamic tests.",
            "raw_judge": identical,
            "target_alignment": alignment,
        }
    result = call_llm_json(
        task=(
            f"llm_as_judge:{function_name}:"
            f"{source_hash(generated_rs_path)}:{source_hash(ground_rs_path)}"
        ),
        messages=llm_judge_prompt(generated_rs_path, ground_rs_path, function_name),
        temperature=0.0,
        max_tokens=2048,
    )
    if result.get("status") != "ok":
        unavailable = not_available_metric(
            result.get("reason", "llm_judge_unavailable"),
            source="llm_as_judge",
        )
        unavailable["llm"] = result.get("llm")
        unavailable["attempts"] = result.get("attempts")
        if result.get("error"):
            unavailable["error"] = result.get("error")
        if result.get("http_body"):
            unavailable["http_body"] = result.get("http_body")
        if result.get("raw"):
            unavailable["raw"] = result.get("raw")
        return unavailable

    data = result.get("json")
    if not isinstance(data, dict):
        return not_available_metric("llm_judge_returned_non_object_json", source="llm_as_judge")
    score = clamp_score(data.get("score"))
    if score is None:
        return not_available_metric("llm_judge_missing_numeric_score", source="llm_as_judge")

    generated = {
        "status": "ok",
        "score": score,
        "verdict": data.get("verdict", "unknown"),
        "missing_constraints": data.get("missing_constraints", []),
        "extra_or_overstrong_constraints": data.get("extra_or_overstrong_constraints", []),
        "vacuity_risks": data.get("vacuity_risks", []),
        "reasoning": data.get("reasoning", ""),
    }
    ground = {
        "status": "ok",
        "score": 1.0,
        "verdict": "reference",
        "reasoning": "Reference specification is the comparison target.",
    }
    return {
        "status": "ok",
        "score": score,
        "generated": generated,
        "ground": ground,
        "delta": score - 1.0,
        "llm": result.get("llm"),
        "cached": result.get("cached"),
        "rubric": "LLM compares behavioral intent of Verus spec/proof clauses; it is advisory and not used as an oracle for dynamic tests.",
        "raw_judge": data,
        "target_alignment": alignment,
    }
