from __future__ import annotations

from metrics_rebuild.metrics.llm.common import (
    llm_spec_code_judge_prompt,
    source_hash,
    spec_code_alignment_contexts,
)
from metrics_rebuild.share.llm_client import call_llm_json
from metrics_rebuild.share.scoring import clamp_score, not_available_metric
from metrics_rebuild.share.verus_runner import verus_frontend_run_to_dict


def _llm_spec_code_alignment_judgment(
    rs_path: str,
    *,
    role: str,
) -> dict:
    contexts, alignment = spec_code_alignment_contexts(rs_path, rs_path)
    matched = int(alignment.get("matched_function_count", 0) or 0)
    if not contexts:
        # No executable/spec functions to judge: score 0, keep an explicit marker.
        return {
            "status": "ok",
            "score": 0.0,
            "verdict": "no_executable_spec_functions_found",
            "reason": "no_executable_spec_functions_found",
            "function_judgments": [],
            "missing_behavior": ["No executable specification functions were found in the evaluated file."],
            "overstrong_or_wrong_constraints": [],
            "vacuity_risks": [],
            "reasoning": "The file has no executable/spec functions that can be aligned for spec-code intent judging.",
            "alignment": alignment,
            "llm": {"status": "skipped", "reason": "no_executable_spec_functions_found"},
            "cached": False,
            "source": f"llm_spec_code_intent_{role}",
        }
    if matched == 0:
        return {
            "status": "ok",
            "score": 0.0,
            "verdict": "missing_code",
            "function_judgments": [
                {
                    "function": item.get("function"),
                    "score": 0.0,
                    "verdict": "missing_code",
                    "missing_behavior": ["No matching executable implementation function was found in the same file."],
                    "overstrong_or_wrong_constraints": [],
                    "vacuity_risks": [],
                    "reasoning": "Specification function could not be matched to an executable implementation function in the same file.",
                }
                for item in contexts
            ],
            "missing_behavior": ["No specification function matched an executable implementation function in the same file."],
            "overstrong_or_wrong_constraints": [],
            "vacuity_risks": [],
            "reasoning": "The metric requires matching specification and executable function names in the evaluated file.",
            "alignment": alignment,
            "llm": {"status": "skipped", "reason": "no_matched_functions"},
            "cached": False,
        }
    if not any(bool((item.get("spec") or {}).get("has_contract")) for item in contexts):
        return {
            "status": "ok",
            "score": 0.0,
            "verdict": "vacuous",
            "function_judgments": [{
                "function": item.get("function"),
                "score": 0.0,
                "verdict": "vacuous",
                "reasoning": "The file has no requires or ensures contract.",
            } for item in contexts],
            "missing_behavior": ["The file has no behavioral contract."],
            "overstrong_or_wrong_constraints": [],
            "vacuity_risks": ["No requires/ensures clauses in the evaluated file."],
            "reasoning": "A file without a behavioral contract is vacuous for spec-code intent evaluation.",
            "alignment": alignment,
            "llm": {"status": "skipped", "reason": "file_has_no_contract"},
            "cached": False,
        }

    result = call_llm_json(
        task=f"llm_spec_code_intent:{role}:{source_hash(rs_path)}",
        messages=llm_spec_code_judge_prompt(rs_path),
        temperature=0.0,
        max_tokens=2048,
    )
    if result.get("status") != "ok":
        unavailable = not_available_metric(
            result.get("reason", "llm_spec_code_judge_unavailable"),
            source=f"llm_spec_code_intent_{role}",
        )
        unavailable["alignment"] = alignment
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
        return not_available_metric(
            "llm_spec_code_judge_returned_non_object_json",
            source=f"llm_spec_code_intent_{role}",
        )
    score = clamp_score(data.get("score"))
    if score is None:
        return not_available_metric(
            "llm_spec_code_judge_missing_numeric_score",
            source=f"llm_spec_code_intent_{role}",
        )

    function_judgments = data.get("function_judgments", [])
    if not isinstance(function_judgments, list):
        function_judgments = []
    return {
        "status": "ok",
        "score": score,
        "verdict": data.get("verdict", "unknown"),
        "function_judgments": function_judgments,
        "missing_behavior": data.get("missing_behavior", []),
        "overstrong_or_wrong_constraints": data.get("overstrong_or_wrong_constraints", []),
        "vacuity_risks": data.get("vacuity_risks", []),
        "reasoning": data.get("reasoning", ""),
        "alignment": alignment,
        "llm": result.get("llm"),
        "cached": result.get("cached"),
        "raw_judge": data,
    }


def metric_llm_as_judge_spec_code_intent_consistency(generated_rs_path: str, ground_rs_path: str) -> dict:
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
                "function_judgments": [],
                "missing_behavior": [],
                "overstrong_or_wrong_constraints": [],
                "vacuity_risks": [],
                "reasoning": "Generated file does not pass Verus frontend checks; spec-code intent judging is gated to avoid rewarding unparseable or ill-typed specifications.",
                "frontend": generated_frontend,
            },
            "ground": {
                "status": "ok" if ground_frontend.get("success") is True else "frontend_failed",
                "score": 1.0 if ground_frontend.get("success") is True else None,
                "verdict": "reference_code",
                "reasoning": "Original/reference implementation is the code-intent target.",
                "frontend": ground_frontend,
            },
            "delta": -1.0,
            "llm": {"status": "skipped", "reason": reason},
            "cached": False,
            "rubric": "LLM spec-code judging is skipped when the candidate spec file fails Verus frontend checks.",
            "raw_judge": {
                "score": 0.0,
                "verdict": "invalid_frontend",
                "reasoning": reason,
            },
        }
    if ground_outcome in {"parse_error", "tool_error", "timeout"} or ground_frontend.get("success") is not True:
        return not_available_metric(
            f"code_frontend_not_ok:{ground_outcome}",
            source="llm_spec_code_frontend_gate",
        )
    generated = _llm_spec_code_alignment_judgment(
        generated_rs_path,
        role="generated",
    )
    if generated.get("status") != "ok":
        return generated
    ground = _llm_spec_code_alignment_judgment(
        ground_rs_path,
        role="ground",
    )
    if ground.get("status") != "ok":
        ground = {
            "status": ground.get("status"),
            "score": None,
            "verdict": "reference_unavailable",
            "reasoning": ground.get("reason", "Reference spec-code baseline unavailable."),
            "raw": ground,
        }
    generated_score = generated.get("score")
    ground_score = ground.get("score")
    delta = None
    if generated_score is not None and ground_score is not None:
        delta = generated_score - ground_score
    return {
        "status": "ok",
        "score": generated_score,
        "generated": generated,
        "ground": ground,
        "delta": delta,
        "llm": generated.get("llm"),
        "cached": generated.get("cached"),
        "rubric": (
            "LLM judges whether each complete Verus file's specification intent is consistent "
            "with its own executable code and proof context."
        ),
        "raw_judge": generated.get("raw_judge"),
    }
