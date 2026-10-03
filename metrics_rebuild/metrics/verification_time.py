from __future__ import annotations

from metrics_rebuild.share.scoring import single_file_wrapper
from metrics_rebuild.share.verus_runner import run_verus, verus_run_to_dict


def _verification_time_for_path(path: str) -> dict:
    result = verus_run_to_dict(run_verus(path, no_verify=False))
    elapsed = result.get("elapsed_seconds")
    return {
        "status": result.get("status"),
        "score": elapsed,
        "elapsed_seconds": elapsed,
        "verification_success": result.get("success"),
        "outcome_status": result.get("outcome_status"),
        "method": "verus_full_verification_elapsed_seconds",
        "verus": result,
        "note": "Score is elapsed seconds, so lower is better. This reruns full Verus verification for timing.",
    }


def metric_verification_time(generated_rs_path: str, ground_rs_path: str) -> dict:
    return single_file_wrapper(generated_rs_path, ground_rs_path, _verification_time_for_path)
