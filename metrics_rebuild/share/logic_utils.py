from __future__ import annotations

import re
from typing import Any, Optional

from metrics_rebuild.share.text import as_tokens

_LOGIC_TOKEN_RE = re.compile(
    r"<==>|==>|<=|>=|==|!=|::|&&|\|\||-?\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_]*|[@()!\[\]{}.,;:|+\-*/%<>]"
)


def logic_tokens(expr: Any) -> list[str]:
    if isinstance(expr, str):
        return _LOGIC_TOKEN_RE.findall(expr)
    tokens: list[str] = []
    for tok in as_tokens(expr):
        tokens.extend(_LOGIC_TOKEN_RE.findall(tok))
    return tokens


def strip_outer_parens(tokens: list[str]) -> list[str]:
    changed = True
    while changed and len(tokens) >= 2 and tokens[0] == "(" and tokens[-1] == ")":
        depth = 0
        changed = False
        for idx, tok in enumerate(tokens):
            if tok == "(":
                depth += 1
            elif tok == ")":
                depth -= 1
                if depth == 0 and idx != len(tokens) - 1:
                    return tokens
        if depth == 0:
            tokens = tokens[1:-1]
            changed = True
    return tokens


def is_true_literal(expr: str) -> bool:
    return strip_outer_parens(logic_tokens(expr)) == ["true"]


def is_false_literal(expr: str) -> bool:
    return strip_outer_parens(logic_tokens(expr)) == ["false"]


def is_low_information_true_expression_tokens(tokens: list[str]) -> bool:
    toks = strip_outer_parens(logic_tokens(tokens))
    if toks == ["true"]:
        return True
    if len(toks) == 3 and toks[0] == toks[2] and toks[1] in {"==", "<=", ">="}:
        return True
    return False


def _top_level_op_indices(toks: list[str], op: str) -> list[int]:
    depths = {"paren": 0, "bracket": 0, "brace": 0}
    indices: list[int] = []
    for idx, tok in enumerate(toks):
        if depths["paren"] == 0 and depths["bracket"] == 0 and depths["brace"] == 0:
            if tok == op:
                indices.append(idx)
        if tok == "(":
            depths["paren"] += 1
        elif tok == ")" and depths["paren"] > 0:
            depths["paren"] -= 1
        elif tok == "[":
            depths["bracket"] += 1
        elif tok == "]" and depths["bracket"] > 0:
            depths["bracket"] -= 1
        elif tok == "{":
            depths["brace"] += 1
        elif tok == "}" and depths["brace"] > 0:
            depths["brace"] -= 1
    return indices


def _split_top_level_op(toks: list[str], op: str) -> Optional[list[list[str]]]:
    indices = _top_level_op_indices(toks, op)
    if not indices:
        return None
    parts: list[list[str]] = []
    prev = 0
    for idx in indices:
        parts.append(toks[prev:idx])
        prev = idx + 1
    parts.append(toks[prev:])
    return parts


def _normalize_term(toks: list[str]) -> tuple[str, ...]:
    toks = strip_outer_parens(toks)
    changed = True
    while changed:
        changed = False
        if len(toks) == 3:
            a, op, b = toks
            if op == "+" and b == "0":
                toks = [a]
                changed = True
            elif op == "+" and a == "0":
                toks = [b]
                changed = True
            elif op == "-" and b == "0":
                toks = [a]
                changed = True
            elif op == "-" and a == b:
                toks = ["0"]
                changed = True
            elif op == "*" and b == "1":
                toks = [a]
                changed = True
            elif op == "*" and a == "1":
                toks = [b]
                changed = True
            elif op == "/" and b == "1":
                toks = [a]
                changed = True
            elif op == "*" and (a == "0" or b == "0"):
                toks = ["0"]
                changed = True
        new_toks = strip_outer_parens(toks)
        if new_toks != toks:
            toks = new_toks
            changed = True
    return tuple(toks)


def _is_equality_tautology(toks: list[str]) -> Optional[str]:
    for op in ("==", "<=", ">="):
        parts = _split_top_level_op(toks, op)
        if parts is not None and len(parts) == 2:
            left, right = parts
            if not left or not right:
                continue
            left_norm = _normalize_term(left)
            right_norm = _normalize_term(right)
            if left_norm == right_norm:
                if (
                    tuple(strip_outer_parens(left)) == left_norm
                    and tuple(strip_outer_parens(right)) == right_norm
                ):
                    return "self_comparison"
                return "arithmetic_identity"
    return None


def _is_implication_self(toks: list[str]) -> Optional[str]:
    parts = _split_top_level_op(toks, "==>")
    if parts is not None and len(parts) == 2:
        left, right = parts
        if left and right and strip_outer_parens(left) == strip_outer_parens(right):
            return "implication_self"
    return None


def _is_equivalence_self(toks: list[str]) -> Optional[str]:
    parts = _split_top_level_op(toks, "<==>")
    if parts is not None and len(parts) == 2:
        left, right = parts
        if left and right and strip_outer_parens(left) == strip_outer_parens(right):
            return "equivalence_self"
    return None


def _is_negation_of(a: list[str], b: list[str]) -> bool:
    if len(a) >= 2 and a[0] == "!":
        return strip_outer_parens(a[1:]) == strip_outer_parens(b)
    return False


def _is_excluded_middle(toks: list[str]) -> Optional[str]:
    parts = _split_top_level_op(toks, "||")
    if parts is None or len(parts) < 2:
        return None
    stripped = [strip_outer_parens(p) for p in parts]
    for i in range(len(stripped)):
        for j in range(i + 1, len(stripped)):
            if _is_negation_of(stripped[i], stripped[j]) or _is_negation_of(stripped[j], stripped[i]):
                return "excluded_middle"
    return None


def _ends_with_len_call(toks: list[str]) -> bool:
    return len(toks) >= 4 and toks[-4:] == [".", "len", "(", ")"]


_NONNEG_SKIP_OPS = frozenset({"==>", "<==>", "||", "&&", "|"})
_NONNEG_COMPARISON_OPS = frozenset({"<", ">", "<=", ">=", "==", "!="})


def _is_bare_len_side(toks: list[str]) -> bool:
    # The len-bearing side of `0 <= X` / `X >= 0` must be a bare `...len()` term,
    # not a chained comparison. Without this, `0 <= n < a.len()` is misread as the
    # tautology `0 <= a.len()` because the right side merely *ends* with `.len()`.
    stripped = strip_outer_parens(toks)
    if any(t in _NONNEG_COMPARISON_OPS for t in stripped):
        return False
    return _ends_with_len_call(stripped)


def _is_nonneg_tautology(toks: list[str]) -> Optional[str]:
    if any(t in _NONNEG_SKIP_OPS for t in toks):
        return None
    parts = _split_top_level_op(toks, ">=")
    if parts and len(parts) == 2:
        left, right = parts
        if _normalize_term(right) == ("0",) and _is_bare_len_side(left):
            return "nonneg_tautology"
    parts = _split_top_level_op(toks, "<=")
    if parts and len(parts) == 2:
        left, right = parts
        if _normalize_term(left) == ("0",) and _is_bare_len_side(right):
            return "nonneg_tautology"
    return None


def _is_low_information_implication(toks: list[str]) -> Optional[str]:
    parts = _split_top_level_op(toks, "==>")
    if parts is None or len(parts) != 2:
        return None
    antecedent, consequent = parts
    if strip_outer_parens(antecedent) == ["false"]:
        return "low_information_implication"
    cons_toks = strip_outer_parens(consequent)
    if cons_toks == ["true"]:
        return "low_information_implication"
    if is_low_information_true_expression_tokens(cons_toks):
        return "low_information_implication"
    if is_syntactic_tautology_tokens(cons_toks):
        return "low_information_implication"
    return None


def _is_quantifier_trivial_body(toks: list[str]) -> Optional[str]:
    if not toks or toks[0] not in ("forall", "exists"):
        return None
    if len(toks) < 2 or toks[1] != "|":
        return None
    pipe_count = 0
    body_start: Optional[int] = None
    for idx in range(1, len(toks)):
        if toks[idx] == "|":
            pipe_count += 1
            if pipe_count == 2:
                body_start = idx + 1
                break
    if body_start is None or body_start >= len(toks):
        return None
    body = strip_outer_parens(toks[body_start:])
    if is_syntactic_tautology_tokens(body):
        return "quantifier_trivial_body"
    return None


def _is_compound_tautology(toks: list[str]) -> Optional[str]:
    or_parts = _split_top_level_op(toks, "||")
    if or_parts is not None:
        for part in or_parts:
            if strip_outer_parens(part) == ["true"]:
                return "compound_tautology"
    and_parts = _split_top_level_op(toks, "&&")
    if and_parts is not None:
        all_taut = True
        for part in and_parts:
            if is_syntactic_tautology_tokens(strip_outer_parens(part)) is None:
                all_taut = False
                break
        if all_taut and and_parts:
            return "compound_tautology"
        non_true = [p for p in and_parts if strip_outer_parens(p) != ["true"]]
        if non_true and len(non_true) < len(and_parts):
            folded = list(non_true[0])
            for p in non_true[1:]:
                folded = folded + ["&&"] + list(p)
            if is_syntactic_tautology_tokens(strip_outer_parens(folded)) is not None:
                return "compound_tautology"
    if toks and toks[0] == "if":
        then_idx: Optional[int] = None
        else_idx: Optional[int] = None
        depths = {"paren": 0, "bracket": 0, "brace": 0}
        for idx, tok in enumerate(toks):
            if depths["paren"] == 0 and depths["bracket"] == 0 and depths["brace"] == 0:
                if tok == "then" and then_idx is None:
                    then_idx = idx
                elif tok == "else" and then_idx is not None and else_idx is None:
                    else_idx = idx
            if tok == "(":
                depths["paren"] += 1
            elif tok == ")" and depths["paren"] > 0:
                depths["paren"] -= 1
            elif tok == "[":
                depths["bracket"] += 1
            elif tok == "]" and depths["bracket"] > 0:
                depths["bracket"] -= 1
            elif tok == "{":
                depths["brace"] += 1
            elif tok == "}" and depths["brace"] > 0:
                depths["brace"] -= 1
        if then_idx is not None and else_idx is not None and then_idx < else_idx:
            then_part = strip_outer_parens(toks[then_idx + 1 : else_idx])
            else_part = strip_outer_parens(toks[else_idx + 1 :])
            if then_part and then_part == else_part:
                if is_syntactic_tautology_tokens(then_part) is not None:
                    return "compound_tautology"
    return None


def strip_logical_noise(tokens: list[str]) -> list[str]:
    return strip_outer_parens(tokens)


_TAUTOLOGY_CHECKS = (
    _is_equality_tautology,
    _is_implication_self,
    _is_equivalence_self,
    _is_excluded_middle,
    _is_nonneg_tautology,
    _is_low_information_implication,
    _is_quantifier_trivial_body,
    _is_compound_tautology,
)


def is_syntactic_tautology_tokens(toks: list[str]) -> Optional[str]:
    toks = strip_outer_parens(toks)
    if not toks:
        return None
    if toks == ["true"]:
        return "literal_true"
    for check in _TAUTOLOGY_CHECKS:
        reason = check(toks)
        if reason:
            return reason
    return None


def is_syntactic_tautology(expr: str) -> Optional[str]:
    return is_syntactic_tautology_tokens(logic_tokens(expr))


def is_simple_tautology(expr: str) -> bool:
    return is_syntactic_tautology(expr) is not None


def find_simple_contradictions(expr: str) -> list[str]:
    toks = strip_outer_parens(logic_tokens(expr))
    joined = " ".join(toks)
    findings: list[str] = []
    if toks == ["false"]:
        findings.append("literal_false")
    for idx in range(len(toks) - 2):
        left, op, right = toks[idx : idx + 3]
        if left == right and op in {"!=", "<", ">"}:
            findings.append(f"self_contradiction:{left} {op} {right}")

    simple_bounds: dict[str, list[tuple[str, float]]] = {}
    for match in re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(<=|<|>=|>|==|!=)\s*(-?\d+(?:\.\d+)?)\b",
        joined,
    ):
        var, op, value_text = match.groups()
        try:
            simple_bounds.setdefault(var, []).append((op, float(value_text)))
        except ValueError:
            continue

    for var, bounds in simple_bounds.items():
        equals = [value for op, value in bounds if op == "=="]
        not_equals = [value for op, value in bounds if op == "!="]
        lower_strict = [value for op, value in bounds if op == ">"]
        lower = [value for op, value in bounds if op == ">="]
        upper_strict = [value for op, value in bounds if op == "<"]
        upper = [value for op, value in bounds if op == "<="]
        for value in equals:
            if value in not_equals:
                findings.append(f"conflicting_equality:{var} == {value} && {var} != {value}")
            if any(value <= bound for bound in lower_strict) or any(value < bound for bound in lower):
                findings.append(f"equality_below_lower_bound:{var}")
            if any(value >= bound for bound in upper_strict) or any(value > bound for bound in upper):
                findings.append(f"equality_above_upper_bound:{var}")
        max_lower = max(lower + lower_strict, default=None)
        min_upper = min(upper + upper_strict, default=None)
        if max_lower is not None and min_upper is not None:
            if max_lower > min_upper:
                findings.append(f"inconsistent_bounds:{var}")
            if max_lower == min_upper and (
                max_lower in lower_strict or min_upper in upper_strict
            ):
                findings.append(f"strict_empty_interval:{var}")
    return sorted(set(findings))


__all__ = [
    "find_simple_contradictions",
    "is_false_literal",
    "is_low_information_true_expression_tokens",
    "is_simple_tautology",
    "is_syntactic_tautology",
    "is_syntactic_tautology_tokens",
    "is_true_literal",
    "logic_tokens",
    "strip_logical_noise",
    "strip_outer_parens",
]
