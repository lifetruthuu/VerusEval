from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from metrics_rebuild.share.text import (
    TokenSpan,
    _consume_raw_string,
    normalize_expr,
    normalized_tokens,
    read_text,
    strip_comments,
    token_spans,
    token_text,
)

CLAUSE_KINDS = (
    "requires",
    "ensures",
    "default_ensures",
    "returns",
    "recommends",
    "opens_invariants",
    "invariant",
    "invariant_except_break",
    "assert",
    "decreases",
    "no_unwind",
)
SPEC_CLAUSE_KINDS = (
    "requires",
    "ensures",
    "default_ensures",
    "returns",
    "recommends",
    "opens_invariants",
    "invariant",
    "invariant_except_break",
    "assert",
    "decreases",
    "no_unwind",
)
EXPRESSION_CLAUSE_KINDS = (
    "requires",
    "ensures",
    "default_ensures",
    "returns",
    "recommends",
    "opens_invariants",
    "invariant",
    "invariant_except_break",
    "decreases",
)
CLAUSE_BOUNDARY_KINDS = set(EXPRESSION_CLAUSE_KINDS) | {"assert", "no_unwind"}
FLAG_CLAUSE_KINDS = {"no_unwind"}

STOPWORDS = {
    "as",
    "bool",
    "else",
    "exists",
    "false",
    "fn",
    "forall",
    "if",
    "in",
    "int",
    "let",
    "mut",
    "nat",
    "open",
    "proof",
    "pub",
    "recommends",
    "requires",
    "ensures",
    "default_ensures",
    "returns",
    "invariant",
    "invariant_except_break",
    "opens_invariants",
    "decreases",
    "no_unwind",
    "assert",
    "return",
    "spec",
    "true",
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "usize",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "isize",
}

FN_RE = re.compile(
    r"""
    \b
    (?P<prefix>
        (?:
            (?:
                pub(?:\s*\([^)]*\))?|
                open|
                closed|
                uninterp|
                broadcast|
                spec(?:\s*\(\s*checked\s*\))?|
                proof|
                exec|
                unsafe
            )
            \s+
        )*
    )
    fn
    \s+
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.MULTILINE | re.VERBOSE,
)


@dataclass(frozen=True)
class Clause:
    kind: str
    text: str
    normalized: str


@dataclass(frozen=True)
class TextSimilarityItem:
    kind: str
    text: str
    normalized: str
    order: int = 0


_PIPE_BINDER_PREFIX_KEYWORDS = {"forall", "exists", "choose", "return"}


def _single_pipe(text: str, offset: int) -> bool:
    return (
        text[offset] == "|"
        and (offset == 0 or text[offset - 1] != "|")
        and (offset + 1 >= len(text) or text[offset + 1] != "|")
    )


def _pipe_opens_binder(text: str, offset: int) -> bool:
    """Return whether a single ``|`` starts a quantifier/closure binder.

    A plain toggle on every single pipe confuses Rust match alternatives and
    bitwise OR with ``forall|...|``/closure binders.  Binder openings occur at
    an expression boundary or immediately after a Verus quantifier keyword.
    """
    if not _single_pipe(text, offset):
        return False
    prefix = text[:offset].rstrip()
    if not prefix:
        return True
    word = re.search(r"([A-Za-z_][A-Za-z0-9_]*)$", prefix)
    if word and word.group(1) in _PIPE_BINDER_PREFIX_KEYWORDS:
        return True
    if prefix.endswith(("==>", "=>", "&&", "||")):
        return True
    return prefix[-1] in "([{,=:+-*/%!&^;"


def _scan_top_level_comma_offsets(text: str) -> tuple[list[int], bool]:
    offsets: list[int] = []
    paren = bracket = brace = 0
    in_pipe_binder = False
    state = "normal"
    i = 0

    while i < len(text):
        ch = text[i]
        if state == "normal":
            raw_end = _consume_raw_string(text, i)
            if raw_end is not None:
                i = raw_end - 1
            elif ch == '"':
                state = "string"
            elif ch == "'":
                consumed = _consume_lifetime_or_char(text, i)
                if consumed is not None:
                    i = consumed - 1
                else:
                    state = "char"
            elif _single_pipe(text, i):
                if in_pipe_binder:
                    in_pipe_binder = False
                elif _pipe_opens_binder(text, i):
                    in_pipe_binder = True
            elif ch == "(":
                paren += 1
            elif ch == ")" and paren > 0:
                paren -= 1
            elif ch == "[":
                bracket += 1
            elif ch == "]" and bracket > 0:
                bracket -= 1
            elif ch == "{":
                brace += 1
            elif ch == "}" and brace > 0:
                brace -= 1
            elif ch == "," and not in_pipe_binder and paren == bracket == brace == 0:
                offsets.append(i)
        elif state == "string":
            if ch == "\\":
                i += 1
            elif ch == '"':
                state = "normal"
        elif state == "char":
            if ch == "\\":
                i += 1
            elif ch == "'":
                state = "normal"
        i += 1
    return offsets, in_pipe_binder


def split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for offset in _scan_top_level_comma_offsets(text)[0]:
        part = text[start:offset].strip()
        if part:
            parts.append(part)
        start = offset + 1
    final = text[start:].strip()
    if final:
        parts.append(final)
    return parts


def split_top_level_type_commas(text: str) -> list[str]:
    """Split a Rust/Verus type list while preserving nested ``<...>``.

    Expression commas deliberately use :func:`split_top_level_commas`, because
    comparison operators also use ``<`` and ``>``.  Function signatures and
    generic declarations, however, need angle-bracket depth tracking.
    """
    parts: list[str] = []
    start = 0
    angle = paren = bracket = brace = 0
    state = "normal"
    i = 0
    while i < len(text):
        ch = text[i]
        if state == "normal":
            raw_end = _consume_raw_string(text, i)
            if raw_end is not None:
                i = raw_end - 1
            elif ch == '"':
                state = "string"
            elif ch == "'":
                consumed = _consume_lifetime_or_char(text, i)
                if consumed is not None:
                    i = consumed - 1
                else:
                    state = "char"
            elif ch == "<":
                angle += 1
            elif ch == ">" and angle > 0:
                angle -= 1
            elif ch == "(":
                paren += 1
            elif ch == ")" and paren > 0:
                paren -= 1
            elif ch == "[":
                bracket += 1
            elif ch == "]" and bracket > 0:
                bracket -= 1
            elif ch == "{":
                brace += 1
            elif ch == "}" and brace > 0:
                brace -= 1
            elif ch == "," and angle == paren == bracket == brace == 0:
                part = text[start:i].strip()
                if part:
                    parts.append(part)
                start = i + 1
        elif state == "string":
            if ch == "\\":
                i += 1
            elif ch == '"':
                state = "normal"
        elif state == "char":
            if ch == "\\":
                i += 1
            elif ch == "'":
                state = "normal"
        i += 1
    final = text[start:].strip()
    if final:
        parts.append(final)
    return parts


def _trimmed_absolute_span(text: str, start: int, end: int, base_offset: int = 0) -> Optional[Tuple[str, int, int]]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return text[start:end], base_offset + start, base_offset + end


def split_top_level_comma_spans(text: str, base_offset: int = 0) -> list[Tuple[str, int, int]]:
    parts: list[Tuple[str, int, int]] = []
    start = 0
    for offset in _scan_top_level_comma_offsets(text)[0]:
        part = _trimmed_absolute_span(text, start, offset, base_offset)
        if part is not None:
            parts.append(part)
        start = offset + 1

    final = _trimmed_absolute_span(text, start, len(text), base_offset)
    if final is not None:
        parts.append(final)
    return parts


def _token_depth_update(tok: str, depths: Dict[str, int]) -> None:
    if tok == "(":
        depths["paren"] += 1
    elif tok == ")" and depths["paren"] > 0:
        depths["paren"] -= 1
    elif tok == "[":
        depths["bracket"] += 1
    elif tok == "]" and depths["bracket"] > 0:
        depths["bracket"] -= 1
    elif tok == "{" and "brace" in depths:
        depths["brace"] += 1
    elif tok == "}" and "brace" in depths and depths["brace"] > 0:
        depths["brace"] -= 1


def _top_level(depths: Dict[str, int]) -> bool:
    return (
        depths.get("paren", 0) == 0
        and depths.get("bracket", 0) == 0
        and depths.get("brace", 0) == 0
    )


def _consume_lifetime_or_char(text: str, i: int) -> Optional[int]:
    """``text[i] == "'"``. Distinguish Rust lifetimes (``'ident``, e.g. ``'static``,
    ``'a``) from char literals (``'c'``, ``'\\n'``, ``' '``).

    Returns the index just past the consumed token (the next char to process) for
    lifetimes and single-identifier char literals such as ``'a'``; returns
    ``None`` for genuine char literals that still need the existing
    scan-to-close-quote handling (``'\\n'``, ``'%'``, ``' '``). Without this, a
    lifetime like ``'static`` is mistaken for an unterminated char literal and
    the scanner never recovers, dropping the enclosing function from parsing.
    """
    n = len(text)
    j = i + 1
    if j >= n:
        return None
    c = text[j]
    # Escaped char literal like '\n', '\'', '\\' -> let char-state scan to closing quote
    if c == "\\":
        return None
    # 'ident... : either a lifetime ('static / 'a) or a char literal 'a'.
    # Rust char literals hold exactly one char, so a multi-char identifier after
    # ' must be a lifetime.
    if c.isalpha() or c == "_":
        k = j
        while k < n and (text[k].isalnum() or text[k] == "_"):
            k += 1
        # 'a' (single char then closing quote) -> char literal, consume past close
        if k - j == 1 and k < n and text[k] == "'":
            return k + 1
        # 'static / 'a / '_ -> lifetime, consume to end of identifier
        return k
    # 'X' where X is a symbol/digit/space -> char literal, let char-state handle
    return None


def find_matching_delimiter(text: str, open_index: int, open_ch: str, close_ch: str) -> Optional[int]:
    depth = 0
    state = "normal"
    i = open_index
    while i < len(text):
        ch = text[i]
        if state == "normal":
            raw_end = _consume_raw_string(text, i)
            if raw_end is not None:
                i = raw_end - 1
            elif ch == '"':
                state = "string"
            elif ch == "'":
                consumed = _consume_lifetime_or_char(text, i)
                if consumed is not None:
                    i = consumed - 1
                else:
                    state = "char"
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return i
        elif state == "string":
            if ch == "\\":
                i += 1
            elif ch == '"':
                state = "normal"
        elif state == "char":
            if ch == "\\":
                i += 1
            elif ch == "'":
                state = "normal"
        i += 1
    return None


def find_matching_paren(text: str, open_index: int) -> Optional[int]:
    return find_matching_delimiter(text, open_index, "(", ")")


def find_matching_brace(text: str, open_index: int) -> Optional[int]:
    return find_matching_delimiter(text, open_index, "{", "}")


def _span_index_after_offset(spans: Sequence[TokenSpan], offset: int) -> int:
    for idx, span in enumerate(spans):
        if span.start > offset:
            return idx
    return len(spans)


def _skip_optional_semicolon(spans: Sequence[TokenSpan], idx: int) -> int:
    if idx < len(spans) and spans[idx].text == ";":
        return idx + 1
    return idx


def _skip_assert_tail(clean: str, spans: Sequence[TokenSpan], idx: int) -> int:
    if idx >= len(spans) or spans[idx].text != "by":
        return idx

    cursor = spans[idx].end
    while cursor < len(clean) and clean[cursor].isspace():
        cursor += 1

    if cursor >= len(clean):
        return idx

    if clean[cursor] == "{":
        block_close = find_matching_brace(clean, cursor)
        if block_close is None:
            return idx
        return _skip_optional_semicolon(spans, _span_index_after_offset(spans, block_close))

    if clean[cursor] == "(":
        tactic_close = find_matching_paren(clean, cursor)
        if tactic_close is None:
            return idx

        after_tactic = _span_index_after_offset(spans, tactic_close)
        if after_tactic < len(spans) and spans[after_tactic].text == "requires":
            semicolon = clean.find(";", spans[after_tactic].end)
            if semicolon == -1:
                return after_tactic
            return _span_index_after_offset(spans, semicolon)
        semicolon = clean.find(";", tactic_close)
        if semicolon == -1:
            return after_tactic
        return _span_index_after_offset(spans, semicolon)

    return idx


def _assert_by_block_span(clean: str, spans: Sequence[TokenSpan], by_idx: int) -> Optional[Tuple[int, int]]:
    if by_idx >= len(spans) or spans[by_idx].text != "by":
        return None
    cursor = spans[by_idx].end
    while cursor < len(clean) and clean[cursor].isspace():
        cursor += 1
    if cursor >= len(clean) or clean[cursor] != "{":
        return None
    block_close = find_matching_brace(clean, cursor)
    if block_close is None:
        return None
    return spans[by_idx].start, block_close + 1


def _find_assert_by_blocks(clean: str) -> list[TextSimilarityItem]:
    spans = token_spans(clean)
    items: list[TextSimilarityItem] = []
    i = 0
    while i < len(spans):
        if spans[i].text != "assert":
            i += 1
            continue
        if i + 1 < len(spans) and spans[i + 1].text == "!":
            i += 1
            continue

        cursor = spans[i].end
        while cursor < len(clean) and clean[cursor].isspace():
            cursor += 1

        by_idx: Optional[int] = None
        if cursor < len(clean) and clean[cursor] == "(":
            close = find_matching_paren(clean, cursor)
            if close is not None:
                candidate = _span_index_after_offset(spans, close)
                if candidate < len(spans) and spans[candidate].text == "by":
                    by_idx = candidate
        else:
            depths = {"paren": 0, "bracket": 0, "brace": 0}
            j = i + 1
            while j < len(spans):
                next_tok = spans[j].text
                if _top_level(depths):
                    if next_tok == "by":
                        by_idx = j
                        break
                    if next_tok == ";":
                        break
                    if next_tok == "{" and _is_probable_clause_body_open(
                        clean,
                        spans[i].end,
                        spans[j].start,
                    ):
                        break
                _token_depth_update(next_tok, depths)
                j += 1

        if by_idx is None:
            i += 1
            continue

        block_span = _assert_by_block_span(clean, spans, by_idx)
        if block_span is None:
            i = by_idx + 1
            continue
        start, end = block_span
        text = clean[start:end].strip()
        normalized = compact_text_similarity_text(text)
        if normalized:
            items.append(TextSimilarityItem("assert_by", text, normalized, start))
        i = _skip_optional_semicolon(spans, _span_index_after_offset(spans, end - 1))
    return items


_BLOCK_RHS_OPERATORS = {
    "==>",
    "<==>",
    "=>",
    "==",
    "!=",
    "<=",
    ">=",
    "<",
    ">",
    "+",
    "-",
    "*",
    "/",
    "%",
    "&&",
    "&&&",
    "||",
    "|||",
    "&",
    "|",
    "^",
    "=",
}


def _ends_with_turbofish(spans: Sequence[TokenSpan]) -> bool:
    """Whether the final ``>`` closes generic arguments such as ``None::<usize>``."""
    depth = 0
    for index in range(len(spans) - 1, -1, -1):
        tok = spans[index].text
        if tok == ">":
            depth += 1
        elif tok == "<":
            depth -= 1
            if depth == 0:
                return index > 0 and spans[index - 1].text == "::"
    return False


def _is_probable_clause_body_open(text: str, expr_start: int, brace_offset: int) -> bool:
    expression_prefix = text[expr_start:brace_offset].strip()
    if not expression_prefix:
        # A clause still needs an expression; ``requires { ... }`` is a block
        # expression, not the executable function body.
        return False
    if expression_prefix.rstrip().endswith(","):
        return True
    prefix_spans = token_spans(expression_prefix)
    if (
        expression_prefix.rstrip().endswith("else")
        or (
            prefix_spans
            and prefix_spans[-1].text in _BLOCK_RHS_OPERATORS
            and not (prefix_spans[-1].text == ">" and _ends_with_turbofish(prefix_spans))
        )
    ):
        return False

    line_start = text.rfind("\n", 0, brace_offset) + 1
    line_prefix = text[line_start:brace_offset].strip()
    if "==>" in line_prefix:
        return False
    if re.search(r"\b(if|else|match)\b", line_prefix):
        return False

    # Handle a construct whose opening brace is placed on the next line, while
    # still recognizing the function body after a completed match/if clause.
    segments = split_top_level_commas(expression_prefix)
    current = segments[-1] if segments else expression_prefix
    if "{" not in current and re.search(r"\b(if|else|match)\b", current):
        return False
    return True


def extract_clauses_from_text(text: str) -> list[Clause]:
    clean = strip_comments(text)
    spans = token_spans(clean)
    clauses: list[Clause] = []
    i = 0

    while i < len(spans):
        tok = spans[i].text
        if tok in EXPRESSION_CLAUSE_KINDS:
            start = spans[i].end
            depths = {"paren": 0, "bracket": 0, "brace": 0}
            j = i + 1
            while j < len(spans):
                next_tok = spans[j].text
                if _top_level(depths):
                    if (
                        next_tok in CLAUSE_BOUNDARY_KINDS
                        and not (j > 0 and spans[j - 1].text in {".", "::"})
                    ):
                        break
                    if next_tok == ";":
                        break
                    if next_tok == "{" and _is_probable_clause_body_open(
                        clean,
                        start,
                        spans[j].start,
                    ):
                        break
                _token_depth_update(next_tok, depths)
                j += 1
            end = spans[j].start if j < len(spans) else len(clean)
            for part in split_top_level_commas(clean[start:end]):
                normalized = normalize_expr(part)
                if normalized:
                    clauses.append(Clause(tok, part.strip(), normalized))
            i = j
            continue

        if tok == "no_unwind":
            start = spans[i].end
            depths = {"paren": 0, "bracket": 0, "brace": 0}
            j = i + 1
            while j < len(spans):
                next_tok = spans[j].text
                if _top_level(depths):
                    if next_tok in CLAUSE_BOUNDARY_KINDS:
                        break
                    if next_tok == ";":
                        break
                    if next_tok == "{" and _is_probable_clause_body_open(
                        clean,
                        start,
                        spans[j].start,
                    ):
                        break
                _token_depth_update(next_tok, depths)
                j += 1
            part = clean[start : spans[j].start if j < len(spans) else len(clean)].strip()
            if not part:
                part = "true"
            normalized = normalize_expr(part)
            if normalized:
                clauses.append(Clause(tok, part, normalized))
            i = j
            continue

        if tok == "assert":
            if i + 1 < len(spans) and spans[i + 1].text == "!":
                i += 1
                continue

            cursor = spans[i].end
            while cursor < len(clean) and clean[cursor].isspace():
                cursor += 1
            if cursor < len(clean) and clean[cursor] == "(":
                close = find_matching_paren(clean, cursor)
                if close is not None:
                    part = clean[cursor + 1 : close].strip()
                    normalized = normalize_expr(part)
                    if normalized:
                        clauses.append(Clause("assert", part, normalized))
                    i = _skip_assert_tail(clean, spans, _span_index_after_offset(spans, close))
                    continue
            start = spans[i].end
            depths = {"paren": 0, "bracket": 0, "brace": 0}
            j = i + 1
            while j < len(spans):
                next_tok = spans[j].text
                if _top_level(depths):
                    if next_tok in {"by", ";"}:
                        break
                    if next_tok == "{" and _is_probable_clause_body_open(
                        clean,
                        start,
                        spans[j].start,
                    ):
                        break
                _token_depth_update(next_tok, depths)
                j += 1
            part = clean[start : spans[j].start if j < len(spans) else len(clean)].strip()
            normalized = normalize_expr(part)
            if normalized:
                clauses.append(Clause("assert", part, normalized))
            if j < len(spans) and spans[j].text == "by":
                i = _skip_assert_tail(clean, spans, j)
            else:
                i = _skip_optional_semicolon(spans, j)
            continue

        i += 1

    return clauses


def extract_clauses(path: str) -> list[Clause]:
    return extract_clauses_from_text(read_text(path))


def clause_counts(clauses: Sequence[Clause]) -> Dict[str, int]:
    counts = {kind: 0 for kind in CLAUSE_KINDS}
    for clause in clauses:
        counts[clause.kind] = counts.get(clause.kind, 0) + 1
    counts["total"] = len(clauses)
    counts["spec_total"] = sum(counts[kind] for kind in SPEC_CLAUSE_KINDS)
    return counts


def clause_records(clauses: Sequence[Clause]) -> list[dict]:
    return [
        {"kind": clause.kind, "text": clause.text, "normalized": clause.normalized}
        for clause in clauses
    ]


def file_tokens(path: str) -> list[str]:
    return normalized_tokens(read_text(path))


def compact_text_similarity_text(text: str) -> str:
    return re.sub(r"\s+", " ", strip_comments(text)).strip()


def _find_top_level_signature_end(text: str, start: int) -> Optional[Tuple[int, bool]]:
    paren = bracket = brace = 0
    state = "normal"
    i = start
    while i < len(text):
        ch = text[i]
        if state == "normal":
            raw_end = _consume_raw_string(text, i)
            if raw_end is not None:
                i = raw_end - 1
            elif ch == '"':
                state = "string"
            elif ch == "'":
                consumed = _consume_lifetime_or_char(text, i)
                if consumed is not None:
                    i = consumed - 1
                else:
                    state = "char"
            elif ch == "(":
                paren += 1
            elif ch == ")" and paren > 0:
                paren -= 1
            elif ch == "[":
                bracket += 1
            elif ch == "]" and bracket > 0:
                bracket -= 1
            elif ch == "{" and paren == 0 and bracket == 0:
                if _is_probable_clause_body_open(text, start, i):
                    return i, True
                brace += 1
            elif ch == "}" and brace > 0:
                brace -= 1
            elif ch == ";" and paren == 0 and bracket == 0 and brace == 0:
                return i, False
        elif state == "string":
            if ch == "\\":
                i += 1
            elif ch == '"':
                state = "normal"
        elif state == "char":
            if ch == "\\":
                i += 1
            elif ch == "'":
                state = "normal"
        i += 1
    return None


def _function_mode(prefix: str) -> str:
    if re.search(r"\bspec\s*\(\s*checked\s*\)", prefix) or re.search(r"\bspec\b", prefix):
        return "spec"
    if re.search(r"\bproof\b", prefix):
        return "proof"
    if re.search(r"\bexec\b", prefix):
        return "exec"
    return "exec"


def text_similarity_function_items(path: str) -> list[TextSimilarityItem]:
    clean = strip_comments(read_text(path))
    items: list[TextSimilarityItem] = []
    for match in FN_RE.finditer(clean):
        prefix = match.group("prefix") or ""
        mode = _function_mode(prefix)
        if mode not in {"spec", "proof"}:
            continue
        signature_end = _find_top_level_signature_end(clean, match.end())
        if signature_end is None:
            continue
        end_offset, has_body = signature_end
        if mode == "spec":
            if has_body:
                body_close = find_matching_brace(clean, end_offset)
                item_end = body_close + 1 if body_close is not None else end_offset
                text = clean[match.start() : item_end]
            else:
                text = clean[match.start() : end_offset]
            kind = "spec_fn"
        else:
            if has_body:
                body_close = find_matching_brace(clean, end_offset)
                item_end = body_close + 1 if body_close is not None else end_offset
                text = clean[match.start() : item_end]
            else:
                text = clean[match.start() : end_offset]
            kind = "proof_fn"
        normalized = compact_text_similarity_text(text)
        if normalized:
            items.append(TextSimilarityItem(kind, text.strip(), normalized, match.start()))
    return items


def _mask_text_item_spans(text: str, items: Sequence[TextSimilarityItem]) -> str:
    chars = list(text)
    for item in items:
        if item.kind not in {"spec_fn", "proof_fn"}:
            continue
        start = max(0, item.order)
        end = min(len(chars), item.order + len(item.text))
        for idx in range(start, end):
            if chars[idx] != "\n":
                chars[idx] = " "
    return "".join(chars)


def text_similarity_items(path: str) -> list[TextSimilarityItem]:
    clean = strip_comments(read_text(path))
    function_items = text_similarity_function_items(path)
    masked = _mask_text_item_spans(clean, function_items)
    clauses = extract_clauses_from_text(masked)
    items: list[TextSimilarityItem] = list(function_items)
    search_start = 0
    for clause in clauses:
        pos = masked.find(clause.text, search_start)
        if pos < 0:
            pos = search_start
        else:
            search_start = pos + len(clause.text)
        normalized = compact_text_similarity_text(clause.text)
        items.append(TextSimilarityItem(clause.kind, clause.text, normalized, pos))
    items.extend(_find_assert_by_blocks(masked))
    return sorted(items, key=lambda item: item.order)


def spec_text_from_clauses(path: str) -> str:
    lines = []
    for clause in extract_clauses(path):
        lines.append(f"{clause.kind} {clause.normalized}")
    return "\n".join(lines)


def spec_text_for_text_similarity(path: str) -> str:
    lines = []
    for item in text_similarity_items(path):
        compact = compact_text_similarity_text(item.text)
        if compact:
            lines.append(f"{item.kind} {compact}")
        else:
            lines.append(item.kind)
    return "\n".join(lines)


def spec_tokens(path: str) -> list[str]:
    return normalized_tokens(spec_text_for_text_similarity(path))


def spec_text_metric_text(path: str) -> str:
    return spec_text_for_text_similarity(path)


def normalized_spec_sequence(path: str) -> list[str]:
    return normalized_tokens(spec_text_from_clauses(path))


def tokenized_spec_text(path: str) -> str:
    return token_text(spec_tokens(path))


def spec_item_kind_counts(path: str) -> dict:
    return dict(Counter(item.kind for item in text_similarity_items(path)))


__all__ = [
    "CLAUSE_KINDS",
    "SPEC_CLAUSE_KINDS",
    "STOPWORDS",
    "Clause",
    "TextSimilarityItem",
    "clause_counts",
    "clause_records",
    "compact_text_similarity_text",
    "extract_clauses",
    "extract_clauses_from_text",
    "file_tokens",
    "find_matching_brace",
    "find_matching_paren",
    "normalized_spec_sequence",
    "spec_item_kind_counts",
    "spec_text_for_text_similarity",
    "spec_text_from_clauses",
    "spec_text_metric_text",
    "spec_tokens",
    "split_top_level_comma_spans",
    "split_top_level_commas",
    "text_similarity_function_items",
    "text_similarity_items",
    "tokenized_spec_text",
]
