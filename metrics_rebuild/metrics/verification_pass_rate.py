from __future__ import annotations

from metrics_rebuild.share.scoring import single_file_wrapper
from metrics_rebuild.share.verus_runner import verus_staged_verification_to_dict, verus_verification_success


def _verification_pass_rate_for_path(path: str) -> dict:
    return verus_verification_success(verus_staged_verification_to_dict(path))


def metric_verification_pass_rate(generated_rs_path: str, ground_rs_path: str) -> dict:
    return single_file_wrapper(generated_rs_path, ground_rs_path, _verification_pass_rate_for_path)
