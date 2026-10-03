from __future__ import annotations

import re
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

try:
    from sacrebleu.metrics import BLEU
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent.
    BLEU = None  # type: ignore[assignment]

TEXT_METRIC_FULL = "full"
TEXT_METRIC_SPEC_ONLY = "spec_only"
TEXT_METRIC_SCOPES = (TEXT_METRIC_FULL, TEXT_METRIC_SPEC_ONLY)

_TEXT_METRIC_SCOPE: ContextVar[str] = ContextVar(
    "text_metric_scope",
    default=TEXT_METRIC_FULL,
)

TOKEN_RE = re.compile(
    r"""
    <==>|=~=|&&&|\|\|\||==>|!=|==|<=|>=|&&|\|\||::|->|=>|\.\.|
    [A-Za-z_][A-Za-z0-9_]*|
    \d+\.\d+|\d+|
    .""",
    re.VERBOSE | re.DOTALL,
)


@dataclass(frozen=True)
class TokenSpan:
    text: str
    start: int
    end: int


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _is_identifier_continue(ch: str) -> bool:
    return ch == "_" or ch.isalnum()


def _consume_raw_string(text: str, i: int) -> Optional[int]:
    """Return the end offset of a Rust raw string literal starting at ``i``."""
    if i > 0 and _is_identifier_continue(text[i - 1]):
        return None

    n = len(text)
    for prefix in ("br", "cr", "r"):
        if not text.startswith(prefix, i):
            continue
        j = i + len(prefix)
        while j < n and text[j] == "#":
            j += 1
        if j >= n or text[j] != '"':
            continue
        hashes = j - (i + len(prefix))
        close = '"' + ("#" * hashes)
        end = text.find(close, j + 1)
        if end < 0:
            return n
        return end + len(close)
    return None


def _consume_quoted_literal(text: str, i: int, quote: str) -> int:
    """Return the end offset of a quoted string/char literal starting at ``i``."""
    j = i + 1
    while j < len(text):
        ch = text[j]
        if ch == "\\" and j + 1 < len(text):
            j += 2
            continue
        j += 1
        if ch == quote:
            return j
    return len(text)


def _consume_lifetime_or_char(text: str, i: int) -> Optional[int]:
    """Distinguish Rust lifetimes (``'a``) from char literals (``'\\n'``)."""
    n = len(text)
    j = i + 1
    if j >= n:
        return None
    c = text[j]
    if c == "\\":
        return None
    if c.isalpha() or c == "_":
        k = j
        while k < n and _is_identifier_continue(text[k]):
            k += 1
        if k - j == 1 and k < n and text[k] == "'":
            return k + 1
        return k
    return None


def normalize_text_metric_scope(scope: Optional[str]) -> str:
    if scope is None:
        return TEXT_METRIC_FULL
    normalized = str(scope).strip().lower().replace("-", "_")
    aliases = {
        "whole": TEXT_METRIC_FULL,
        "whole_file": TEXT_METRIC_FULL,
        "rust_and_spec": TEXT_METRIC_FULL,
        "all": TEXT_METRIC_FULL,
        "spec": TEXT_METRIC_SPEC_ONLY,
        "specs": TEXT_METRIC_SPEC_ONLY,
        "specification": TEXT_METRIC_SPEC_ONLY,
        "specification_only": TEXT_METRIC_SPEC_ONLY,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TEXT_METRIC_SCOPES:
        allowed = ", ".join(TEXT_METRIC_SCOPES)
        raise ValueError(f"Unsupported text metric scope {scope!r}; expected one of: {allowed}")
    return normalized


def get_text_metric_scope() -> str:
    return _TEXT_METRIC_SCOPE.get()


@contextmanager
def text_metric_scope(scope: Optional[str]) -> Iterator[None]:
    normalized = normalize_text_metric_scope(scope)
    token = _TEXT_METRIC_SCOPE.set(normalized)
    try:
        yield
    finally:
        _TEXT_METRIC_SCOPE.reset(token)


def strip_comments(text: str) -> str:
    """Remove Rust line/block comments while preserving strings and newlines."""
    out: list[str] = []
    i = 0
    state = "normal"
    block_depth = 0

    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if state == "normal":
            raw_end = _consume_raw_string(text, i)
            if raw_end is not None:
                out.append(text[i:raw_end])
                i = raw_end
                continue
            if ch == "/" and nxt == "/":
                state = "line_comment"
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block_comment"
                block_depth = 1
                i += 2
                continue
            if ch == '"':
                end = _consume_quoted_literal(text, i, '"')
                out.append(text[i:end])
                i = end
                continue
            elif ch == "'":
                end = _consume_lifetime_or_char(text, i)
                if end is None:
                    end = _consume_quoted_literal(text, i, "'")
                out.append(text[i:end])
                i = end
                continue
            out.append(ch)
            i += 1
            continue

        if state == "line_comment":
            if ch == "\n":
                out.append(ch)
                state = "normal"
            i += 1
            continue

        if state == "block_comment":
            if ch == "\n":
                out.append("\n")
                i += 1
                continue
            if ch == "/" and nxt == "*":
                block_depth += 1
                i += 2
                continue
            if ch == "*" and nxt == "/":
                block_depth -= 1
                i += 2
                if block_depth <= 0:
                    state = "normal"
                continue
            i += 1
            continue

    return "".join(out)


def token_spans(text: str) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        raw_end = _consume_raw_string(text, i)
        if raw_end is not None:
            spans.append(TokenSpan(text[i:raw_end], i, raw_end))
            i = raw_end
            continue
        if text[i] == '"':
            end = _consume_quoted_literal(text, i, '"')
            spans.append(TokenSpan(text[i:end], i, end))
            i = end
            continue
        if text[i] == "'":
            end = _consume_lifetime_or_char(text, i)
            if end is None:
                end = _consume_quoted_literal(text, i, "'")
            spans.append(TokenSpan(text[i:end], i, end))
            i = end
            continue

        match = TOKEN_RE.match(text, i)
        if match is None:  # Defensive fallback; TOKEN_RE's final "." should match.
            i += 1
            continue
        tok = match.group(0)
        spans.append(TokenSpan(tok, match.start(), match.end()))
        i = match.end()
    return spans


def tokens(text: str, *, strip_comments_from_text: bool = True) -> list[str]:
    source = strip_comments(text) if strip_comments_from_text else text
    return [span.text for span in token_spans(source) if span.text.strip()]


def normalized_tokens(text: str) -> list[str]:
    return [tok for tok in tokens(text) if tok.strip()]


def normalize_expr(text: str) -> str:
    return " ".join(normalized_tokens(text)).strip()


def token_text(tokens_: Sequence[str]) -> str:
    return " ".join(tokens_)


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


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for tok_a in a:
        current = [0]
        for j, tok_b in enumerate(b, 1):
            if tok_a == tok_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def counter_intersection_size(left: Counter, right: Counter) -> int:
    return sum((left & right).values())


def _text_metric_tokens(path: str) -> list[str]:
    from metrics_rebuild.share.clauses import file_tokens, spec_tokens

    if get_text_metric_scope() == TEXT_METRIC_SPEC_ONLY:
        return spec_tokens(path)
    return file_tokens(path)


def _text_metric_input_for_path(path: str) -> dict:
    from metrics_rebuild.share.clauses import spec_text_metric_text

    tokens_ = _text_metric_tokens(path)
    spec_only = get_text_metric_scope() == TEXT_METRIC_SPEC_ONLY
    return {
        "text_metric_scope": get_text_metric_scope(),
        "token_source": "spec_items" if spec_only else "whole_file",
        "text": spec_text_metric_text(path) if spec_only else token_text(tokens_),
        "tokens": len(tokens_),
    }


def get_text_metric_inputs(generated_rs_path: str, ground_rs_path: str) -> dict:
    """Return the preprocessed text inputs used by the text-style metrics."""
    generated = _text_metric_input_for_path(generated_rs_path)
    ground = _text_metric_input_for_path(ground_rs_path)
    return {
        "text_metric_scope": get_text_metric_scope(),
        "token_source": generated["token_source"],
        "generated": generated,
        "ground": ground,
    }


def bleu_score(candidate_text: Any, reference_text: Any, max_n: int = 4) -> dict:
    candidate = as_tokens(candidate_text)
    reference = as_tokens(reference_text)
    if not candidate and not reference:
        return {
            "score": 1.0,
            "candidate_tokens": 0,
            "reference_tokens": 0,
            "precisions": [1.0] * max_n,
            "brevity_penalty": 1.0,
            "sacrebleu_score": 100.0,
            "signature": None,
            "empty_match": True,
        }
    if not candidate:
        return {
            "score": 0.0,
            "candidate_tokens": 0,
            "reference_tokens": len(reference),
            "precisions": [0.0] * max_n,
            "brevity_penalty": 0.0,
            "sacrebleu_score": 0.0,
            "signature": None,
        }
    if not reference:
        return {
            "score": 0.0,
            "candidate_tokens": len(candidate),
            "reference_tokens": 0,
            "precisions": [0.0] * max_n,
            "brevity_penalty": 0.0,
            "sacrebleu_score": 0.0,
            "signature": None,
        }

    if BLEU is None:
        return {
            "score": 0.0,
            "candidate_tokens": len(candidate),
            "reference_tokens": len(reference),
            "precisions": [0.0] * max_n,
            "brevity_penalty": 0.0,
            "sacrebleu_score": 0.0,
            "signature": None,
            "status": "degraded",
            "note": "sacrebleu not installed; BLEU score unavailable.",
        }

    metric = BLEU(
        tokenize="none",
        smooth_method="exp",
        max_ngram_order=max_n,
        effective_order=True,
    )
    bleu = metric.corpus_score(
        [" ".join(candidate)],
        [[" ".join(reference)]],
    )
    precisions = [precision / 100 for precision in bleu.precisions[:max_n]]
    score = max(0.0, min(1.0, bleu.score / 100))
    return {
        "score": score,
        "candidate_tokens": len(candidate),
        "reference_tokens": len(reference),
        "precisions": precisions,
        "brevity_penalty": bleu.bp,
        "sacrebleu_score": bleu.score,
        "signature": str(metric.get_signature()),
    }


def rouge_l_score(candidate_text: Any, reference_text: Any) -> dict:
    candidate = as_tokens(candidate_text)
    reference = as_tokens(reference_text)
    if not candidate and not reference:
        return {
            "score": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "lcs_tokens": 0,
            "candidate_tokens": 0,
            "reference_tokens": 0,
            "empty_match": True,
        }
    lcs = lcs_length(candidate, reference)
    precision = nonempty_ratio(lcs, len(candidate))
    recall = nonempty_ratio(lcs, len(reference))
    score = f1(precision, recall)
    return {
        "score": score if score is not None else 0.0,
        "precision": precision,
        "recall": recall,
        "lcs_tokens": lcs,
        "candidate_tokens": len(candidate),
        "reference_tokens": len(reference),
    }
