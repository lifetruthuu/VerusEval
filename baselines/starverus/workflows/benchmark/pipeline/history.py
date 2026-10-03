from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .schemas import RepairHistoryEntry


class RepairHistory:
    """History H(t), provided only to the selected Repairer."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.entries: List[RepairHistoryEntry] = []

    def append(self, entry: RepairHistoryEntry) -> None:
        self.entries.append(entry)
        self.output_path.write_text(
            json.dumps(
                [item.compact() for item in self.entries],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def for_repairer(self) -> str:
        if not self.entries:
            return "No previous repair rounds."

        lines = []
        for entry in self.entries[-5:]:
            lines.extend(
                [
                    f"Round {entry.epoch} (cycle {entry.repair_cycle})",
                    f"Repairer: {entry.repairer_name}",
                    f"Blueprint: {entry.blueprint}",
                    f"Outcome: {entry.result}",
                    f"Verifier feedback: {entry.verifier_feedback}",
                    "",
                ]
            )
        return "\n".join(lines).strip()
