from __future__ import annotations

import re
from collections import Counter

from metrics_rebuild.share.clauses import (
    SPEC_CLAUSE_KINDS,
    STOPWORDS,
    TextSimilarityItem,
    compact_text_similarity_text,
    spec_tokens,
    text_similarity_items,
)
from metrics_rebuild.share.text import (
    bleu_score,
    counter_intersection_size,
    normalized_tokens,
    rouge_l_score,
    token_text,
)

KEY_SPEC_TOKENS = {
    "forall",
    "exists",
    "old",
    "len",
    "contains",
    "index",
    "Some",
    "None",
    "Ok",
    "Err",
    "result",
    "&&",
    "||",
    "==>",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
}
KEY_SPEC_IGNORED_TOKENS = {
    ",",
    ";",
    ":",
    "::",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "|",
    ".",
}

# ── Clause-kind sets for each scope ──────────────────────────────────
# Classification principle: caller-visible obligations/promises = spec;
# annotations that only help the verifier get through = proof. Together the
# two sets cover every kind produced by text_similarity_items(). Known
# granularity limit: loop `ensures` shares the "ensures" kind with function
# signatures, so it lands in the spec scope.

SPEC_ONLY_KINDS = {
    "requires",
    "ensures",
    "returns",
    "default_ensures",
    "recommends",
    "opens_invariants",
    "no_unwind",
    "spec_fn",
}
PROOF_KINDS = {
    "assert",
    "assert_by",
    "proof_fn",
    "invariant",
    "invariant_except_break",
    "decreases",
}


# ── Text / token extraction per scope ────────────────────────────────

def _items_text(items: list[TextSimilarityItem]) -> str:
    """Render items to the same ``kind <normalized-text>`` format used by
    ``spec_text_for_text_similarity`` (see ``clauses.py``)."""
    lines: list[str] = []
    for item in items:
        compact = compact_text_similarity_text(item.text)
        if compact:
            lines.append(f"{item.kind} {compact}")
        else:
            lines.append(item.kind)
    return "\n".join(lines)


def _items_tokens(items: list[TextSimilarityItem]) -> list[str]:
    return normalized_tokens(_items_text(items))


# ── KeySpecMatch helpers (operate on item lists, not paths) ─────────

def _key_spec_feature_counter(items: list[TextSimilarityItem]) -> Counter:
    features: Counter = Counter()
    for item in items:
        if item.kind not in SPEC_CLAUSE_KINDS and item.kind not in {"spec_fn", "proof_fn", "assert_by"}:
            continue
        features[f"kind:{item.kind}"] += 1
        tokens = normalized_tokens(item.normalized)
        for tok in tokens:
            if tok in KEY_SPEC_IGNORED_TOKENS:
                continue
            if tok in KEY_SPEC_TOKENS:
                features[f"token:{tok}"] += 1
                continue
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok) and tok not in STOPWORDS:
                features[f"identifier:{tok}"] += 1
    return features


def _feature_rows(counter: Counter) -> list[dict]:
    return [
        {"feature": feature, "count": count}
        for feature, count in counter.most_common(30)
    ]


def _compute_key_spec_match(
    generated_items: list[TextSimilarityItem],
    reference_items: list[TextSimilarityItem],
) -> dict:
    generated = _key_spec_feature_counter(generated_items)
    reference = _key_spec_feature_counter(reference_items)
    reference_total = sum(reference.values())
    generated_total = sum(generated.values())
    matched = counter_intersection_size(generated, reference)
    if reference_total == 0:
        score = 1.0 if generated_total == 0 else 0.0
    else:
        score = matched / reference_total
    missing = reference - generated
    extra = generated - reference
    result = {
        "score": score,
        "matched_features": matched,
        "reference_features": reference_total,
        "generated_features": generated_total,
        "missing_features": _feature_rows(missing),
        "extra_features": _feature_rows(extra),
        "note": "KeySpecMatch compares clause kinds plus important Verus spec/proof tokens against the reference.",
    }
    result["status"] = "ok"
    if reference_total == 0 and generated_total == 0:
        result["empty_match"] = True
    return result


# ── Metric functions ─────────────────────────────────────────────────

def _build_metric(
    gen_tokens: list[str],
    ref_tokens: list[str],
    gen_items: list[TextSimilarityItem],
    ref_items: list[TextSimilarityItem],
    note: str,
) -> dict:
    bleu = bleu_score(token_text(gen_tokens), token_text(ref_tokens))
    rouge = rouge_l_score(token_text(gen_tokens), token_text(ref_tokens))
    key_match = _compute_key_spec_match(gen_items, ref_items)
    result = {
        "score": bleu["score"],
        "status": "ok" if bleu.get("score") is not None else "not_available",
        "components": {
            "bleu": bleu,
            "rouge_l": rouge,
            "key_spec_match": key_match,
        },
        "note": note,
    }
    if not gen_tokens and not ref_tokens:
        result["empty_match"] = True
    return result


def metric_verus_textual_similarity(generated_rs_path: str, ground_rs_path: str) -> dict:
    """Full scope: all Verus spec/proof clauses
    (requires, ensures, assert, assert_by, invariant, decreases, …).
    Report raw BLEU, ROUGE-L and KeySpecMatch without weighted combination."""
    return _build_metric(
        gen_tokens=spec_tokens(generated_rs_path),
        ref_tokens=spec_tokens(ground_rs_path),
        gen_items=text_similarity_items(generated_rs_path),
        ref_items=text_similarity_items(ground_rs_path),
        note="Full scope: all Verus spec/proof clauses (requires, ensures, assert, invariant, …). "
             "No weighted combination — each component score is reported independently.",
    )


def metric_verus_textual_similarity_spec_only(
    generated_rs_path: str, ground_rs_path: str,
) -> dict:
    """Spec scope: caller-visible contract clauses (requires, ensures, returns,
    default_ensures, recommends, opens_invariants, no_unwind) plus spec_fn
    definitions. Report raw BLEU, ROUGE-L and KeySpecMatch without weighted
    combination."""
    gen_items = [it for it in text_similarity_items(generated_rs_path) if it.kind in SPEC_ONLY_KINDS]
    ref_items = [it for it in text_similarity_items(ground_rs_path) if it.kind in SPEC_ONLY_KINDS]
    return _build_metric(
        gen_tokens=_items_tokens(gen_items),
        ref_tokens=_items_tokens(ref_items),
        gen_items=gen_items,
        ref_items=ref_items,
        note="Spec scope: caller-visible contract clauses (requires, ensures, returns, "
             "default_ensures, recommends, opens_invariants, no_unwind) plus spec_fn definitions. "
             "No weighted combination — each component score is reported independently.",
    )


def metric_verus_textual_similarity_proof(
    generated_rs_path: str, ground_rs_path: str,
) -> dict:
    """Proof scope: verifier-only scaffolding (assert, assert_by, proof_fn,
    invariant, invariant_except_break, decreases). Report raw BLEU, ROUGE-L
    and KeySpecMatch without weighted combination."""
    gen_items = [it for it in text_similarity_items(generated_rs_path) if it.kind in PROOF_KINDS]
    ref_items = [it for it in text_similarity_items(ground_rs_path) if it.kind in PROOF_KINDS]
    return _build_metric(
        gen_tokens=_items_tokens(gen_items),
        ref_tokens=_items_tokens(ref_items),
        gen_items=gen_items,
        ref_items=ref_items,
        note="Proof scope: verifier-only scaffolding (assert, assert_by, proof_fn, "
             "invariant, invariant_except_break, decreases). "
             "No weighted combination — each component score is reported independently.",
    )


# ── Backward-compatible alias ────────────────────────────────────────

metric_verus_spec_score = metric_verus_textual_similarity
