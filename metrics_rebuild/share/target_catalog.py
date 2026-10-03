from __future__ import annotations

import csv
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_CATALOG = PROJECT_ROOT / "data/evaluation/target_functions.csv"


@lru_cache(maxsize=4)
def _catalog_rows(path: str) -> dict[tuple[str, str], dict[str, str]]:
    catalog_path = Path(path)
    if not catalog_path.is_file():
        return {}
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with catalog_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                str(row.get("benchmark") or ""),
                str(row.get("reference_file") or ""),
            )
            if not all(key) or not row.get("target_function"):
                continue
            if key in rows:
                raise ValueError(f"Duplicate target catalog key: {key}")
            rows[key] = {str(k): str(v or "") for k, v in row.items()}
    return rows


def target_catalog_entry_for_path(
    reference_path: str,
    *,
    catalog_path: Optional[str] = None,
) -> Optional[dict[str, str]]:
    """Return the canonical target row for a benchmark reference, if available.

    A matching catalog row is fail-loud when its source hash is stale. Temporary
    files and non-benchmark inputs simply fall back to source-based resolution.
    """

    path = Path(reference_path)
    resolved_catalog = str(Path(catalog_path) if catalog_path else DEFAULT_TARGET_CATALOG)
    row = _catalog_rows(resolved_catalog).get((path.parent.name, path.name))
    if row is None:
        return None
    expected_hash = row.get("reference_sha256")
    if expected_hash:
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"Target catalog source hash mismatch for {path}: "
                f"expected {expected_hash}, found {actual_hash}"
            )
    return dict(row)


__all__ = [
    "DEFAULT_TARGET_CATALOG",
    "target_catalog_entry_for_path",
]
