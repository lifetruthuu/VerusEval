from __future__ import annotations

from metrics_rebuild.share.scoring import single_file_wrapper
from metrics_rebuild.share.verus_runner import verus_frontend_run_to_dict


def _type_check_rate_for_path(path: str) -> dict:
    frontend = verus_frontend_run_to_dict(path)
    outcome = frontend.get("outcome_status")
    if outcome == "tool_error":
        score = None
        status = "tool_error"
    elif outcome == "timeout":
        score = None
        status = "timeout"
    else:
        score = 1.0 if frontend.get("success") is True else 0.0
        status = "ok"
    return {
        "status": status,
        "score": score,
        "passed": score == 1.0 if score is not None else None,
        "method": "verus_no_verify",
        "frontend": frontend,
        "note": (
            "Runs Verus with --no-verify. Passing means the file completes all Verus/Rust frontend checks: "
            "syntax parsing, name resolution, type checking, and Verus mode checks. It does not run SMT "
            "verification and is not the same as Rust backend compilation."
        ),
    }


def metric_type_check_rate(generated_rs_path: str, ground_rs_path: str) -> dict:
    return single_file_wrapper(generated_rs_path, ground_rs_path, _type_check_rate_for_path)
