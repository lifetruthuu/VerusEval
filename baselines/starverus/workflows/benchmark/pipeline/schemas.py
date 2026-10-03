from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class VerificationResult:
    path: Path
    verified_obligations: int
    errors: int
    stdout: str
    stderr: str
    errors_by_type: Dict[str, List[Dict[str, Any]]]

    @property
    def is_compilable(self) -> bool:
        return not (
            self.verified_obligations == -1 and self.errors == -1
        )

    @property
    def is_verified(self) -> bool:
        return self.verified_obligations > 0 and self.errors == 0

    @property
    def score_key(self) -> tuple[int, int]:
        return (-self.verified_obligations, self.errors)


@dataclass(frozen=True)
class PlannerDecision:
    root_cause_error_type: str
    selected_repairer: str
    fix_order: List[str]
    summary: str
    root_cause: str
    evidence: str
    repair_direction: str
    risk: str = ""
    used_fallback: bool = False


@dataclass(frozen=True)
class RepairBlueprint:
    repairer_name: str
    content: str


@dataclass
class RepairHistoryEntry:
    epoch: int
    repair_cycle: int
    repairer_name: str
    error_type: str
    blueprint: str
    verifier_feedback: str
    result: str
    verified_obligations: int
    errors: int
    accepted: bool
    planner_summary: str = ""

    def compact(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "repair_cycle": self.repair_cycle,
            "repairer": self.repairer_name,
            "error_type": self.error_type,
            "blueprint": self.blueprint,
            "verifier_feedback": self.verifier_feedback,
            "result": self.result,
            "verified_obligations": self.verified_obligations,
            "errors": self.errors,
            "accepted": self.accepted,
        }


@dataclass
class PipelineResult:
    verified: bool
    final_path: Path
    initial_score: tuple[int, int]
    final_score: tuple[int, int]
    alignment_iterations: int
    repair_epochs: int
    rewriter_invocations: int
    history: List[RepairHistoryEntry] = field(default_factory=list)
