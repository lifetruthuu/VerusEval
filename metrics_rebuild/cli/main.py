from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .args import build_parser
from .output import render_json
from metrics_rebuild.registry import compute_all_metrics


def _validate_input_file(parser, path_value: str, label: str) -> None:
    path = Path(path_value)
    if not path.is_file():
        parser.error(f"{label} does not exist or is not a regular file: {path_value}")


def _has_metric_error(report: dict) -> bool:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return False
    return any(isinstance(result, dict) and result.get("status") == "error" for result in metrics.values())


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_input_file(parser, args.generated_rs_path, "generated_rs_path")
    _validate_input_file(parser, args.ground_rs_path, "ground_rs_path")
    report = compute_all_metrics(
        args.generated_rs_path,
        args.ground_rs_path,
        text_scope=args.text_scope,
        include_mutation=not args.no_mutation,
    )
    print(render_json(report, compact=args.compact))
    if args.fail_on_error and _has_metric_error(report):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
