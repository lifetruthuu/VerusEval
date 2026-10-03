from __future__ import annotations

from metrics_rebuild.share.redundancy import spec_redundancy_for_path
from metrics_rebuild.share.scoring import single_file_wrapper


def metric_spec_redundancy_rate(generated_rs_path: str, ground_rs_path: str) -> dict:
    return single_file_wrapper(generated_rs_path, ground_rs_path, spec_redundancy_for_path)
