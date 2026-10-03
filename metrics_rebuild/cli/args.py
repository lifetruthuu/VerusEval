from __future__ import annotations

import argparse

from metrics_rebuild.share.text import TEXT_METRIC_FULL, TEXT_METRIC_SCOPES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute VerusEval metrics through metrics_rebuild.")
    parser.add_argument("generated_rs_path", help="Path to generated Rust/Verus file")
    parser.add_argument("ground_rs_path", help="Path to ground-truth Rust/Verus file")
    parser.add_argument(
        "--no-mutation",
        action="store_true",
        help="Skip the expensive mutation_kill_rate metric in CLI output",
    )
    parser.add_argument(
        "--text-scope",
        choices=TEXT_METRIC_SCOPES,
        default=TEXT_METRIC_FULL,
        help="Token source for BLEU/ROUGE text metrics",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit the complete report as one-line JSON without omitting fields",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return exit code 1 when any metric reports status=error",
    )
    return parser
