from __future__ import annotations

import json
from typing import Any

def render_json(report: dict[str, Any], *, compact: bool = False) -> str:
    if compact:
        return json.dumps(report, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
