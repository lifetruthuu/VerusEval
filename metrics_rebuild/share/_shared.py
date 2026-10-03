from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Optional, Sequence


IMPLICIT_TRUE_CONTRACT_CLAUSES_DEFAULT = True


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def as_tokens(value: Any) -> list[str]:
    if isinstance(value, str):
        return [tok for tok in value.split() if tok]
    if isinstance(value, Sequence):
        return [str(tok) for tok in value if str(tok)]
    return [tok for tok in as_text(value).split() if tok]


def as_lines(value: Any) -> list[str]:
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, Sequence):
        return [str(line).strip() for line in value if str(line).strip()]
    text = as_text(value)
    return [line.strip() for line in text.splitlines() if line.strip()]


def get_field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def clause_kind(value: Any) -> str:
    return str(get_field(value, "kind", ""))


def clause_text(value: Any) -> str:
    return str(get_field(value, "text", ""))


def clause_normalized(value: Any) -> str:
    return str(get_field(value, "normalized", clause_text(value)))


def implicit_true_clause(kind: str) -> dict[str, str]:
    normalized_kind = "ensures" if kind == "default_ensures" else kind
    return {
        "kind": normalized_kind,
        "text": "true",
        "normalized": "true",
        "implicit": "true",
    }


def is_implicit_true_clause(clause: Any) -> bool:
    """Return True for placeholders produced by :func:`implicit_true_clause`."""
    if not isinstance(clause, Mapping):
        return False
    return str(clause.get("implicit", "")).lower() == "true"


def materialize_contract_clauses(
    value: Any,
    *,
    kind: str,
    inject_implicit_true: bool = IMPLICIT_TRUE_CONTRACT_CLAUSES_DEFAULT,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            text = clause_text(item).strip()
            normalized = clause_normalized(item).strip() or text
            if not text and not normalized:
                continue
            records.append(
                {
                    "kind": str(get_field(item, "kind", kind) or kind),
                    "text": text or normalized,
                    "normalized": normalized or text,
                }
            )
    if not records and inject_implicit_true and kind in {"requires", "ensures", "default_ensures"}:
        return [implicit_true_clause(kind)]
    return records


def nonempty_ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, 1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def counter_intersection_size(left: Counter, right: Counter) -> int:
    return sum((left & right).values())


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0
