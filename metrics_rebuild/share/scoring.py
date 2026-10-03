from __future__ import annotations

import math
from typing import Any, Callable, Optional


def exception_metric(exc: BaseException) -> dict[str, Any]:
    return {
        "status": "error",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def not_available_metric(reason: str, *, source: Optional[str] = None) -> dict:
    payload = {
        "score": None,
        "status": "not_available",
        "reason": reason,
    }
    if source:
        payload["source"] = source
    return {
        **payload,
        "generated": dict(payload),
        "ground": dict(payload),
        "delta": None,
    }


def clamp_score(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score) or math.isinf(score):
        return None
    return max(0.0, min(1.0, score))


def single_file_wrapper(
    generated_rs_path: str,
    ground_rs_path: str,
    analyzer: Callable[[str], dict],
) -> dict:
    generated = analyzer(generated_rs_path)
    ground = analyzer(ground_rs_path)
    score_key = "score" if isinstance(generated, dict) and "score" in generated else None
    delta: Optional[float] = None
    if score_key and generated.get(score_key) is not None and ground.get(score_key) is not None:
        delta = generated[score_key] - ground[score_key]
    return {"generated": generated, "ground": ground, "delta": delta}
