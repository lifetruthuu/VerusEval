#!/usr/bin/env python3
"""Run the six lemma-based metrics on a small, reproducible corpus sample.

The command reuses ``evaluate_analysis_dataset.discover_pairs`` but deliberately
does not call the full evaluator and does not write experiment artifacts.  Its
single stdout value is a compact JSON report suitable for CI logs or ``jq``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import io
import json
import random
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterator, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_ROOT = PROJECT_ROOT / "data/generated"
DEFAULT_REFERENCE_ROOT = PROJECT_ROOT / "data/references"
DEFAULT_IO_ROOT = PROJECT_ROOT / "data/io"


class SmokeTimeout(TimeoutError):
    """Raised when one public metric exceeds the smoke-test time budget."""


def _load_analysis_evaluator() -> ModuleType:
    """Load the existing analysis-dataset adapter without duplicating pairing."""
    module_path = PROJECT_ROOT / "scripts" / "evaluation" / "evaluate_analysis_dataset.py"
    spec = importlib.util.spec_from_file_location("veruseval_analysis_dataset", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _metric_deadline(timeout_seconds: int) -> Iterator[None]:
    """Apply a hard per-metric deadline while retaining the metric's own timeout."""
    if timeout_seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def handle_timeout(_signum: int, _frame: object) -> None:
        raise SmokeTimeout(f"metric exceeded {timeout_seconds}s smoke timeout")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _public_metrics() -> tuple[tuple[str, Callable[..., dict]], ...]:
    from metrics_rebuild.metrics.postcondition_clause_completeness_rate import (
        metric_postcondition_clause_completeness_rate,
    )
    from metrics_rebuild.metrics.postcondition_clause_reliability_rate import (
        metric_postcondition_clause_reliability_rate,
    )
    from metrics_rebuild.metrics.precondition_clause_completeness_rate import (
        metric_precondition_clause_completeness_rate,
    )
    from metrics_rebuild.metrics.precondition_clause_reliability_rate import (
        metric_precondition_clause_reliability_rate,
    )
    from metrics_rebuild.metrics.proportion_at_least_gt import metric_proportion_at_least_gt
    from metrics_rebuild.metrics.proportion_at_most_gt import metric_proportion_at_most_gt

    return (
        ("proportion_at_least_gt", metric_proportion_at_least_gt),
        ("postcondition_clause_reliability_rate", metric_postcondition_clause_reliability_rate),
        ("precondition_clause_reliability_rate", metric_precondition_clause_reliability_rate),
        ("proportion_at_most_gt", metric_proportion_at_most_gt),
        ("postcondition_clause_completeness_rate", metric_postcondition_clause_completeness_rate),
        ("precondition_clause_completeness_rate", metric_precondition_clause_completeness_rate),
    )


def _diagnostic_counts(value: object) -> dict[str, dict[str, int]]:
    counters = {key: Counter() for key in ("status", "phase", "reason")}

    def visit(current: object) -> None:
        if isinstance(current, dict):
            for key, child in current.items():
                if key in counters and child is not None and not isinstance(child, (dict, list)):
                    counters[key][str(child)[:240]] += 1
                visit(child)
        elif isinstance(current, list):
            for child in current:
                visit(child)

    visit(value)
    return {
        key: dict(sorted(counter.items()))
        for key, counter in counters.items()
        if counter
    }


def _concise_metric_result(raw: dict, elapsed_seconds: float) -> dict:
    metric_result = raw
    if "status" not in metric_result and isinstance(metric_result.get("generated"), dict):
        metric_result = metric_result["generated"]
    result = {
        "status": metric_result.get("status", "unknown"),
        "score": metric_result.get("score"),
        "passed": metric_result.get("passed"),
        "failed": metric_result.get("failed"),
        "unknown": metric_result.get("unknown"),
        "determined": metric_result.get("determined"),
        "total": metric_result.get("total"),
        "coverage": metric_result.get("coverage"),
        "phase": metric_result.get("phase"),
        "reason": metric_result.get("reason"),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    result["diagnostics"] = _diagnostic_counts(metric_result)
    return result


def _call_metric(
    metric_fn: Callable[..., dict],
    generated_path: str,
    reference_path: str,
    timeout_seconds: int,
) -> dict:
    kwargs = {}
    if "timeout_seconds" in inspect.signature(metric_fn).parameters:
        kwargs["timeout_seconds"] = timeout_seconds

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    started = time.monotonic()
    try:
        with _metric_deadline(timeout_seconds), contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            raw = metric_fn(generated_path, reference_path, **kwargs)
        if not isinstance(raw, dict):
            raise TypeError(f"metric returned {type(raw).__name__}, expected dict")
    except SmokeTimeout as exc:
        raw = {
            "status": "unknown",
            "phase": "smoke_timeout",
            "reason": "timeout",
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 - smoke report must retain all failures
        raw = {
            "status": "unknown",
            "phase": "metric_exception",
            "reason": type(exc).__name__,
            "error": str(exc),
        }
    return _concise_metric_result(raw, time.monotonic() - started)


def _aggregate(results: list[dict]) -> dict:
    status_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()

    for pair in results:
        for metric in pair["metrics"].values():
            status_counts[str(metric.get("status") or "unknown")] += 1
            for key, destination in (("passed", "valid"), ("failed", "invalid"), ("unknown", "unknown")):
                value = metric.get(key)
                if isinstance(value, int):
                    outcome_counts[destination] += value
            diagnostics = metric.get("diagnostics") or {}
            phase_counts.update(diagnostics.get("phase") or {})
            reason_counts.update(diagnostics.get("reason") or {})

    return {
        "status_counts": dict(sorted(status_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test the six Verus lemma implication metrics on real data/generated pairs.",
    )
    parser.add_argument("--methods", nargs="+", default=["alphaverus", "autoverus"])
    parser.add_argument("--analysis-root", default=str(DEFAULT_ANALYSIS_ROOT))
    parser.add_argument("--reference-root", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument("--io-root", default=str(DEFAULT_IO_ROOT), help="Used only by the shared pairing adapter")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of generated/reference pairs")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic sampling seed")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Per-metric timeout; <=0 disables the smoke-test deadline",
    )
    parser.add_argument("--verus-path", default=None, help="Override config.yaml/system PATH Verus")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")

    sys.path.insert(0, str(PROJECT_ROOT))
    pairing = _load_analysis_evaluator()
    analysis_root = Path(args.analysis_root)
    reference_root = Path(args.reference_root)
    io_root = Path(args.io_root)
    if not analysis_root.is_dir():
        parser.error(f"analysis root not found: {analysis_root}")
    if not reference_root.is_dir():
        parser.error(f"reference root not found: {reference_root}")

    candidates: list[dict] = []
    discovery_skips: Counter[str] = Counter()
    for method in args.methods:
        dataset_dir = analysis_root / method
        if not dataset_dir.is_dir():
            parser.error(f"dataset method directory not found: {dataset_dir}")
        pairs, _manifest, skipped = pairing.discover_pairs(
            method=method,
            dataset_dir=dataset_dir,
            reference_root=reference_root,
            io_root=io_root,
        )
        for filename, generated_path, reference_path in pairs:
            candidates.append(
                {
                    "method": method,
                    "filename": filename,
                    "generated_path": generated_path,
                    "reference_path": reference_path,
                }
            )
        discovery_skips.update(str(item.get("reason") or "unknown") for item in skipped)

    candidates.sort(key=lambda item: (item["method"], item["filename"]))
    selected = candidates
    if args.limit < len(candidates):
        selected = random.Random(args.seed).sample(candidates, args.limit)
        selected.sort(key=lambda item: (item["method"], item["filename"]))

    import metrics_rebuild

    verus_path: Optional[str] = args.verus_path or pairing._load_verus_path_from_config()
    if verus_path:
        metrics_rebuild.set_verus_binary(verus_path)

    metric_functions = _public_metrics()
    results = []
    started = time.monotonic()
    for pair in selected:
        metric_results = {
            metric_name: _call_metric(
                metric_fn,
                pair["generated_path"],
                pair["reference_path"],
                args.timeout_seconds,
            )
            for metric_name, metric_fn in metric_functions
        }
        results.append({**pair, "metrics": metric_results})

    report = {
        "mode": "lemma_implication_real_corpus_smoke",
        "methods": args.methods,
        "seed": args.seed,
        "limit": args.limit,
        "timeout_seconds": args.timeout_seconds,
        "verus_binary": verus_path or "system PATH",
        "candidate_pairs": len(candidates),
        "selected_pairs": len(selected),
        "discovery_skip_reasons": dict(sorted(discovery_skips.items())),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "summary": _aggregate(results),
        "results": results,
    }
    json.dump(report, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
