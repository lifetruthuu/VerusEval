from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

MetricFunction = Callable[[str, str], dict]


@dataclass(frozen=True)
class MetricCatalogEntry:
    metric_id: str
    display_name: str
    category: str
    direction: str
    implementation_status: str
