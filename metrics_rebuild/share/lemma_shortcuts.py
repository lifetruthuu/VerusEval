from __future__ import annotations

from typing import Any, Mapping, Sequence


def _normalized_clause_texts(clauses: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(clause.get("normalized") or clause.get("text") or "").strip()
        for clause in clauses
        if str(clause.get("normalized") or clause.get("text") or "").strip()
    }


def safe_trivial_implication(
    antecedent: Sequence[Mapping[str, Any]],
    consequent: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Return only implication results that are independent of symbol binding."""
    consequent_texts = _normalized_clause_texts(consequent)
    if not consequent_texts:
        return {
            "holds": True,
            "status": "valid",
            "reason": "empty_consequent",
            "antecedent_total": len(antecedent),
            "consequent_total": 0,
            "engine": "syntactic",
        }
    if consequent_texts == {"true"}:
        return {
            "holds": True,
            "status": "valid",
            "reason": "true_consequent",
            "antecedent_total": len(antecedent),
            "consequent_total": len(consequent),
            "engine": "syntactic",
        }
    return None


__all__ = ["safe_trivial_implication"]
