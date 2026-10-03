from __future__ import annotations

from metrics_rebuild.share.mutation import generated_self_spec_robustness


def metric_mutation_kill_rate(generated_rs_path: str, ground_rs_path: str) -> dict:
    return generated_self_spec_robustness(generated_rs_path, ground_rs_path)
