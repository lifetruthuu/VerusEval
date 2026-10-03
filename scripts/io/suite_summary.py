from pathlib import Path
"""Refresh suite summaries after updating task records."""
from scripts.io import generate_io_tests_llm as _generate_io

def _rewrite_summary(v2_dir: Path, updated: dict[str, dict]) -> None:
    # Re-read every current meta/test pair so untouched tasks cannot retain
    # stale counts from an older summary.  Existing summary-only terminal
    # entries are preserved for legacy suites that do not yet have meta.json.
    _generate_io._write_summary(
        v2_dir,
        list(updated.values()),
        append_existing=True,
    )