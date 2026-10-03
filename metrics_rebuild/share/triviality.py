from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from metrics_rebuild.share.clauses import (
    extract_clauses,
    extract_clauses_from_text,
)
from metrics_rebuild.share.functions import (
    extract_functions,
    function_parameters_from_header,
    function_return_parameters_from_header,
)
from metrics_rebuild.share.logic_utils import (
    find_simple_contradictions,
    is_false_literal,
    is_simple_tautology,
    is_syntactic_tautology,
    is_true_literal,
    logic_tokens,
    strip_outer_parens,
)
from metrics_rebuild.share.text import strip_comments

TRIVIALITY_CLAUSE_KINDS = {
    "requires",
    "ensures",
    "default_ensures",
    "recommends",
    "invariant",
    "invariant_except_break",
    "assert",
    "decreases",
}

VACUOUS_POSTCONDITION_KINDS = {"ensures", "default_ensures", "recommends"}

_NONNEG_TYPES = {"nat", "u8", "u16", "u32", "u64", "u128", "usize"}


def _base_type(type_str: str) -> str:
    t = type_str.strip()
    while t.startswith("&") or t.startswith("*"):
        t = t.lstrip("&*").strip()
        if t.startswith("mut "):
            t = t[4:].strip()
    return t.split("<")[0].split("[")[0].strip()


def _nonneg_names_for_path(path: str) -> set[str]:
    names: set[str] = set()
    for fn in extract_functions(path):
        for p in function_parameters_from_header(fn.header):
            if _base_type(p["type"]) in _NONNEG_TYPES:
                names.add(p["name"])
        for p in function_return_parameters_from_header(fn.header):
            if _base_type(p["type"]) in _NONNEG_TYPES:
                names.add(p["name"])
    return names


def _is_unsigned_nonneg_clause(text: str, nonneg_names: set[str]) -> Optional[str]:
    toks = strip_outer_parens(logic_tokens(text))
    if len(toks) == 3 and toks[1] == ">=" and toks[2] == "0" and toks[0] in nonneg_names:
        return "unsigned_nonneg"
    if len(toks) == 3 and toks[0] == "0" and toks[1] == "<=" and toks[2] in nonneg_names:
        return "unsigned_nonneg"
    return None


def _clause_kind(value: Any) -> str:
    return str(value.get("kind", "") if isinstance(value, dict) else getattr(value, "kind", ""))


def _clause_text(value: Any) -> str:
    return str(value.get("text", "") if isinstance(value, dict) else getattr(value, "text", ""))


def _clause_normalized(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("normalized", value.get("text", "")))
    return str(getattr(value, "normalized", getattr(value, "text", "")))


def keyword_blacklist_flags(clean_text: str) -> dict:
    patterns = [
        ("assume_false", r"\bassume\s*\(\s*false\s*\)"),
        ("assume_any", r"\bassume\s*\((?!\s*false\s*\))[^)]*\)"),
        ("admit", r"\badmit\s*\(\s*\)"),
        ("assume_specification", r"\bassume_specification\b"),
        ("external_body", r"#\s*\[\s*verifier\s*::\s*external_body\s*\]|\bexternal_body\b|\bverifier\s*::\s*external\b"),
        ("no_verify", r"\bno_verify\b|--no-verify\b"),
        ("trusted", r"\btrusted\b|trusted\s+spec"),
        ("unimplemented", r"\b(?:unimplemented|todo)\s*!\s*\(\s*\)"),
        ("axiom", r"#\s*\[\s*verifier\s*::\s*axiom\s*\]|\baxiom\b"),
    ]
    flags: list[dict] = []
    for name, pattern in patterns:
        for match in re.finditer(pattern, clean_text, flags=re.DOTALL | re.IGNORECASE):
            snippet = " ".join(match.group(0).split())[:180]
            if name == "assume_any" and re.fullmatch(r"assume\s*\(\s*false\s*\)", snippet, flags=re.IGNORECASE):
                continue
            flags.append({"kind": name, "snippet": snippet})
    return {
        "score": 1.0 if flags else 0.0,
        "flag_count": len(flags),
        "flags": flags[:50],
    }


def keyword_blacklist_for_text(text: str) -> dict:
    return keyword_blacklist_flags(strip_comments(text))


def ensures_true_flags(clauses: Sequence[Any]) -> dict:
    flags: list[dict] = []
    for clause in clauses:
        if _clause_kind(clause) in {"ensures", "default_ensures"}:
            reason = is_syntactic_tautology(_clause_text(clause))
            if reason:
                flags.append({"clause": _clause_normalized(clause), "reason": reason})
    return {"score": 1.0 if flags else 0.0, "flag_count": len(flags), "flags": flags}


def vacuous_postcondition_flags_for_path(path: str) -> dict:
    function_of: dict[tuple[str, str], str] = {}
    for function in extract_functions(path):
        for clause in extract_clauses_from_text(function.header):
            if clause.kind in VACUOUS_POSTCONDITION_KINDS:
                function_of.setdefault((clause.kind, clause.normalized), function.name)

    flags: list[dict] = []
    total = 0
    for clause in extract_clauses(path):
        if clause.kind not in VACUOUS_POSTCONDITION_KINDS:
            continue
        total += 1
        reason = is_syntactic_tautology(clause.text)
        if reason:
            flags.append(
                {
                    "function": function_of.get((clause.kind, clause.normalized)),
                    "kind": clause.kind,
                    "clause": clause.normalized,
                    "reason": reason,
                    "tier": "syntactic",
                }
            )

    by_reason: dict[str, int] = {}
    for flag in flags:
        by_reason[flag["reason"]] = by_reason.get(flag["reason"], 0) + 1

    return {
        "score": 1.0 if flags else 0.0,
        "flag_count": len(flags),
        "flags": flags,
        "summary": {
            "total": total,
            "vacuous": len(flags),
            "by_reason": by_reason,
        },
    }


def trivial_clause_rate(
    clauses: Sequence[Any], nonneg_names: set[str] | None = None
) -> dict:
    spec_clauses = [clause for clause in clauses if _clause_kind(clause) in TRIVIALITY_CLAUSE_KINDS]
    trivial: list[dict] = []
    for clause in spec_clauses:
        reasons: list[str] = []
        text = _clause_text(clause)
        if is_true_literal(text):
            reasons.append("literal_true")
        if is_false_literal(text):
            reasons.append("literal_false")
        if is_simple_tautology(text):
            reasons.append("simple_tautology")
        if nonneg_names:
            nonneg_reason = _is_unsigned_nonneg_clause(text, nonneg_names)
            if nonneg_reason:
                reasons.append(nonneg_reason)
        contradictions = find_simple_contradictions(text)
        if contradictions:
            reasons.extend(contradictions)
        if reasons:
            trivial.append(
                {
                    "kind": _clause_kind(clause),
                    "clause": _clause_normalized(clause),
                    "reasons": sorted(set(reasons)),
                }
            )
    return {
        "score": len(trivial) / len(spec_clauses) if spec_clauses else 0.0,
        "clauses_total": len(spec_clauses),
        "trivial_clauses": len(trivial),
        "flags": trivial,
    }


def trivial_clause_rate_for_path(path: str) -> dict:
    clauses = extract_clauses(path)
    nonneg_names = _nonneg_names_for_path(path)
    return trivial_clause_rate(clauses, nonneg_names=nonneg_names)


__all__ = [
    "ensures_true_flags",
    "keyword_blacklist_flags",
    "keyword_blacklist_for_text",
    "trivial_clause_rate",
    "trivial_clause_rate_for_path",
    "vacuous_postcondition_flags_for_path",
]
