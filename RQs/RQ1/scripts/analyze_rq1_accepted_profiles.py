"""Compute current RQ1 statistics from merged evaluation records."""

from __future__ import annotations

import csv
from pathlib import Path
from rq1_paths import RESULTS


ROOT = Path(__file__).resolve().parents[3]
PAPER_TAB = ROOT / "RQs/RQ1/tables"
OUTPUT = RESULTS
ARTIFACTS = ROOT / "data/evaluation/rq1_artifact_labels.csv"
OUTCOMES = ROOT / "data/evaluation/rq1_artifact_outcomes.csv"
FORMAL = {
    "soundness": ("pre_ref_to_gen", "post_gen_to_ref"),
    "completeness": ("pre_gen_to_ref", "post_ref_to_gen"),
    "equivalence": ("pre_ref_to_gen", "post_gen_to_ref", "pre_gen_to_ref", "post_ref_to_gen"),
}
IO = ("io_correct", "io_wrong", "io_invalid")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))




def formal_state(row: dict[str, str], keys: tuple[str, ...]) -> str:
    values = [row.get(key, "unavailable") for key in keys]
    if "invalid" in values:
        return "fail"
    if all(value == "valid" for value in values):
        return "pass"
    return "unresolved"


def io_score(row: dict[str, str], kind: str) -> float | None:
    """Fraction of checked cases that pass; an unresolved case counts as not passed."""
    passed = int(row[f"{kind}_passed"])
    checked = passed + int(row[f"{kind}_failed"]) + int(row[f"{kind}_unresolved"])
    return passed / checked if checked else None


def io_state(row: dict[str, str]) -> str:
    """Fail on a case that fails; pass when every checked case passes and none is unresolved."""
    passed, failed, unresolved = (sum(int(row[f"{kind}_{outcome}"]) for kind in IO)
                                  for outcome in ("passed", "failed", "unresolved"))
    if failed:
        return "fail"
    return "pass" if passed and not unresolved else "none"


def triviality_state(row: dict[str, str]) -> str:
    states = (row["triviality_pre_false_state"], row["triviality_post_true_state"])
    if "unfavorable" in states:
        return "fail"
    return "pass" if all(state == "favorable" for state in states) else "unresolved"
