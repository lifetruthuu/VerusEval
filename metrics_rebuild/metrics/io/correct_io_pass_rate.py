from __future__ import annotations

from metrics_rebuild.share.io_cases import io_metric_pair


def metric_correct_io_pass_rate(generated_rs_path: str, ground_rs_path: str) -> dict:
    return io_metric_pair(generated_rs_path, ground_rs_path, "positive")
