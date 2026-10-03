"""Target-contract parsing and edits used by the controlled variants."""
from __future__ import annotations
from dataclasses import dataclass, field
import re
from typing import Optional, Sequence
from metrics_rebuild.share.clauses import _token_depth_update, _top_level, find_matching_brace, split_top_level_comma_spans
from metrics_rebuild.share.functions import FN_RE, _find_top_level_signature_end, _leading_verifier_spec_attribute, function_mode, function_parameters_from_header
from metrics_rebuild.share.text import token_spans
HEADER_CLAUSE_KINDS = {"requires", "ensures", "default_ensures", "returns", "recommends", "opens_invariants", "decreases", "no_unwind", "invariant", "invariant_except_break"}

@dataclass
class ClauseGroup:
    kind: str
    keyword_start: int
    keyword_end: int
    group_end: int
    parts: list[tuple[str, int, int]] = field(default_factory=list)

@dataclass
class TargetLayout:
    name: str
    fn_start: int
    body_open: int
    body_close: int
    groups: list[ClauseGroup]

    def groups_of(self, kind: str) -> list[ClauseGroup]:
        return [group for group in self.groups if group.kind == kind]

    def parts_of(self, kind: str) -> list[tuple[ClauseGroup, int]]:
        return [(group, index) for group in self.groups_of(kind) for index in range(len(group.parts))]

def _header_groups(clean: str, header_start: int, header_end: int) -> list[ClauseGroup]:
    header = clean[header_start:header_end]
    spans = token_spans(header)
    depths = {"paren": 0, "bracket": 0, "brace": 0}
    keywords: list[tuple[str, int, int]] = []
    for index, span in enumerate(spans):
        if (
            _top_level(depths)
            and span.text in HEADER_CLAUSE_KINDS
            and not (index > 0 and spans[index - 1].text in {".", "::"})
        ):
            keywords.append((span.text, header_start + span.start, header_start + span.end))
        _token_depth_update(span.text, depths)
    groups: list[ClauseGroup] = []
    for position, (kind, keyword_start, keyword_end) in enumerate(keywords):
        group_end = keywords[position + 1][1] if position + 1 < len(keywords) else header_end
        parts = split_top_level_comma_spans(clean[keyword_end:group_end], keyword_end)
        groups.append(ClauseGroup(kind, keyword_start, keyword_end, group_end, parts))
    return groups

def locate_target(clean: str, target_name: str) -> Optional[TargetLayout]:
    for match in FN_RE.finditer(clean):
        if match.group("name") != target_name:
            continue
        prefix = match.group("prefix") or ""
        if function_mode(prefix) != "exec" or _leading_verifier_spec_attribute(clean, match.start()) is not None:
            continue
        signature_end = _find_top_level_signature_end(clean, match.end())
        if signature_end is None or not signature_end[1]:
            continue
        body_open = signature_end[0]
        body_close = find_matching_brace(clean, body_open)
        if body_close is None:
            continue
        return TargetLayout(
            name=target_name,
            fn_start=match.start(),
            body_open=body_open,
            body_close=body_close,
            groups=_header_groups(clean, match.start(), body_open),
        )
    return None

def removal_span_for_part(group: ClauseGroup, index: int, clean: str) -> tuple[int, int]:
    """Same comma discipline as the redundancy checker's removal candidates."""
    parts = group.parts
    if len(parts) == 1:
        return group.keyword_start, group.group_end
    _, part_start, part_end = parts[index]
    if index == 0:
        return part_start, parts[1][1]
    if index == len(parts) - 1:
        trailing = clean[part_end : group.group_end]
        start = part_start if "," in trailing else parts[index - 1][2]
        return start, group.group_end
    return part_start, parts[index + 1][1]

def splice(text: str, edits: Sequence[tuple[int, int, str]]) -> str:
    """Apply non-overlapping [start, end) -> replacement edits in one pass."""
    ordered = sorted(edits, key=lambda item: item[0])
    for left, right in zip(ordered, ordered[1:]):
        if left[1] > right[0]:
            raise ValueError("overlapping edits")
    out: list[str] = []
    cursor = 0
    for start, end, replacement in ordered:
        out.append(text[cursor:start])
        out.append(replacement)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)

def clause_insertion_point(layout: TargetLayout, kind: str) -> tuple[int, bool]:
    """Offset where a new ``kind`` clause goes and whether a keyword exists.

    Into an existing group: right after its keyword. New requires group: before
    the first header clause keyword (or the body brace). New ensures group:
    right before the body brace, after every other clause.
    """
    existing = layout.groups_of(kind)
    if existing:
        return existing[0].keyword_end, True
    if kind == "requires" and layout.groups:
        return layout.groups[0].keyword_start, False
    return layout.body_open, False

def insert_clause(clean: str, layout: TargetLayout, kind: str, clause: str) -> str:
    offset, has_keyword = clause_insertion_point(layout, kind)
    clause = clause.strip().rstrip(",;")
    if has_keyword:
        insertion = f"\n        {clause},"
    else:
        insertion = f"\n    {kind}\n        {clause},\n"
    return clean[:offset] + insertion + clean[offset:]

def _normalized_type(type_text: str) -> str:
    text = re.sub(r"\s+", "", type_text)
    text = re.sub(r"^&(mut)?", "", text)
    return text

def spurious_precondition_candidates(header: str) -> list[tuple[str, str]]:
    """(parameter_name, clause) candidates ordered by expected effectiveness."""
    candidates: list[tuple[str, str]] = []
    for param in function_parameters_from_header(header):
        name = str(param.get("name") or "")
        type_text = _normalized_type(str(param.get("type") or ""))
        if not name or name == "self":
            continue
        if re.match(r"^Vec<", type_text) or type_text in {"String"}:
            candidates += [(name, f"{name}.len() > 0"), (name, f"{name}.len() < 100"), (name, f"{name}@.len() > 0")]
        elif re.match(r"^\[", type_text) or re.match(r"^Seq<", type_text) or type_text in {"str"}:
            candidates += [(name, f"{name}@.len() > 0"), (name, f"{name}.len() > 0"), (name, f"{name}@.len() < 100")]
        elif type_text in {"i8", "i16", "i32", "i64", "i128", "isize", "int"}:
            candidates += [(name, f"{name} >= 0"), (name, f"{name} != 0"), (name, f"{name} <= 100")]
        elif type_text in {"u8", "u16", "u32", "u64", "u128", "usize", "nat"}:
            candidates += [(name, f"{name} > 0"), (name, f"{name} <= 100"), (name, f"{name} != 1")]
        elif type_text == "bool":
            candidates += [(name, f"{name} == true")]
        elif type_text == "char":
            candidates += [(name, f"{name} != 'x'")]
    # Interleave by parameter so the first attempts cover different parameters.
    by_param: dict[str, list[tuple[str, str]]] = {}
    for name, clause in candidates:
        by_param.setdefault(name, []).append((name, clause))
    ordered: list[tuple[str, str]] = []
    while any(by_param.values()):
        for name in list(by_param):
            if by_param[name]:
                ordered.append(by_param[name].pop(0))
    return ordered
