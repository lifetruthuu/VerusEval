from __future__ import annotations

from metrics_rebuild.share.scoring import single_file_wrapper
from metrics_rebuild.share.verus_runner import verus_frontend_run_to_dict


def _parse_rate_for_path(path: str) -> dict:
    frontend = verus_frontend_run_to_dict(path)
    outcome = frontend.get("outcome_status")
    if outcome == "tool_error":
        score = None
        status = "tool_error"
    elif outcome == "timeout":
        score = None
        status = "timeout"
    else:
        score = 0.0 if outcome == "parse_error" else 1.0
        status = "ok"
    return {
        "status": status,
        "score": score,
        "passed": score == 1.0 if score is not None else None,
        "method": "verus_no_verify_diagnostic_classification",
        "frontend": frontend,
        "note": (
            "Parse rate only checks whether the source is syntactically parseable by the Verus/Rust frontend. "
            "Verus does not expose a stable parse-only flag, so --no-verify runs all frontend checks and "
            "parse errors are identified by exclusion from the --error-format=json diagnostics: errors with "
            "Rust E-codes (type/name/resolution), VIR errors, or Verus-specific messages are NOT parse errors; "
            "only errors with no E-code and no Verus-specific indicators are classified as parse failures."
        ),
    }


def metric_parse_rate(generated_rs_path: str, ground_rs_path: str) -> dict:
    return single_file_wrapper(generated_rs_path, ground_rs_path, _parse_rate_for_path)
