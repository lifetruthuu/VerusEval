#!/usr/bin/env python
"""Rebuild an IO suite summary from its authoritative per-task meta files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results: list[dict] = []
    for meta_path in sorted(args.suite_root.glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        old_function = str(meta.get("old_function") or "")
        function = str(meta.get("function") or "")
        strict = meta.get("strict_validation") or {}
        results.append(
            {
                "name": meta_path.parent.name,
                "benchmark": meta_path.parent.parent.name,
                "status": str(meta.get("status") or "error"),
                "positive": int(meta.get("positive_count", 0) or 0),
                "negative": int(meta.get("negative_count", 0) or 0),
                "invalid": int(meta.get("invalid_count", 0) or 0),
                "quarantined": int(strict.get("quarantined_count", 0) or 0),
                "wrong_target": bool(old_function and old_function != function),
                "resumed": True,
            }
        )

    statuses = Counter(str(item["status"]) for item in results)
    summary = {
        "schema_version": 3,
        "tasks": len(results),
        "stats": dict(sorted(statuses.items())),
        "wrong_target_tasks": sum(bool(item["wrong_target"]) for item in results),
        "quarantined_cases": sum(int(item["quarantined"]) for item in results),
        "results": results,
    }
    output = args.suite_root / "summary.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({key: summary[key] for key in summary if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
