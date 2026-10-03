from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .text import token_spans


_DIRECTION_KEYS = frozenset(
    {
        "ground_implies_generated",
        "generated_implies_ground",
        "ground_pre_implies_gen_pre",
        "gen_pre_implies_ground_pre",
        "gen_post_implies_ground_post",
        "ground_post_implies_gen_post",
    }
)

_LEGACY_STRENGTH_DIRECTIONS = {
    "ground_pre_implies_gen_pre": "ground_implies_generated",
    "gen_pre_implies_ground_pre": "generated_implies_ground",
    "gen_post_implies_ground_post": "generated_implies_ground",
    "ground_post_implies_gen_post": "ground_implies_generated",
}

_STRENGTH_METRICS = frozenset(
    {
        "proportion_at_least_gt",
        "proportion_at_most_gt",
    }
)


def normalized_clause(text: Any) -> str:
    return " ".join(span.text for span in token_spans(str(text or ""))).strip()


@lru_cache(maxsize=4096)
def _relative_source_path_cached(path_text: str, project_root_text: str) -> str:
    value = Path(path_text)
    try:
        return value.resolve().relative_to(Path(project_root_text).resolve()).as_posix()
    except (OSError, ValueError):
        return value.as_posix()


def relative_source_path(path: Any, project_root: Path) -> str:
    """Project a source path onto the repository-relative audit identity.

    Results are cached because audit walks call this for every obligation with
    the same handful of paths; resolving foreign absolute paths (e.g. under
    the macOS ``/home`` automounter) can take tens of milliseconds per call.
    """
    return _relative_source_path_cached(str(path or ""), str(project_root))


def stable_obligation_id(
    *,
    generated_path: Any,
    reference_path: Any,
    metric: str,
    function: str,
    clause_kind: str,
    direction: str,
    antecedent: Sequence[Any],
    consequent: Sequence[Any],
    project_root: Path,
) -> str:
    payload = {
        "generated_path": relative_source_path(generated_path, project_root),
        "reference_path": relative_source_path(reference_path, project_root),
        "metric": metric,
        "function": function,
        "clause_kind": clause_kind,
        "direction": direction,
        "antecedent": [normalized_clause(item) for item in antecedent],
        "consequent": [normalized_clause(item) for item in consequent],
    }
    seed = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()


def legacy_obligation_id(
    record: Mapping[str, Any],
    *,
    project_root: Path,
) -> str:
    """Project a contextual obligation onto the pre-context audit identity."""
    metric = str(record.get("metric") or "")
    direction = str(record.get("direction") or "")
    clause_kind = str(record.get("clause_kind") or "")
    if metric in _STRENGTH_METRICS:
        try:
            legacy_direction = _LEGACY_STRENGTH_DIRECTIONS[direction]
        except KeyError as exc:
            raise ValueError(
                f"unsupported strength obligation direction for legacy projection: {direction}"
            ) from exc
        consequent: Sequence[Any] = []
    else:
        legacy_direction = "implication"
        consequent = list(record.get("consequent") or [])
    return stable_obligation_id(
        generated_path=record.get("generated_path"),
        reference_path=record.get("reference_path"),
        metric=metric,
        function=str(record.get("function") or ""),
        clause_kind=clause_kind,
        direction=legacy_direction,
        antecedent=[],
        consequent=consequent,
        project_root=project_root,
    )


def _status(check: Mapping[str, Any]) -> str:
    holds = check.get("holds")
    if holds is True:
        return "valid"
    if holds is False:
        return "invalid"
    return "unknown"


def _clause_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("normalized") or value.get("text") or "")
    return str(value or "")


def _clause_slot(value: Any) -> str:
    """Extract the stable per-clause suffix from a contextual check name."""
    text = str(value or "")
    match = re.search(r"_(?:requires|ensures)_(\d+)$", text)
    return match.group(1) if match else ""


def classify_harness_tags(record: Mapping[str, Any]) -> list[str]:
    tags: set[str] = set()
    if record.get("classification") != "unknown":
        return []
    check = record.get("check") if isinstance(record.get("check"), Mapping) else {}
    category = str(check.get("category") or "")
    reason = str(check.get("reason") or "")
    error = str(check.get("error") or "")
    clauses = " ".join(
        [
            *[str(value) for value in record.get("antecedent", [])],
            *[str(value) for value in record.get("consequent", [])],
        ]
    )
    if category in {"return_count", "return_name"}:
        tags.add("H1")
    if category == "return_type" and ",)" in re.sub(r"\s+", "", str(check)):
        tags.add("H2")
    if "dangling_>" in error and re.search(
        r"(?:::|[A-Za-z0-9_])\s*<[^<>]+>\s*$", clauses
    ):
        tags.add("H3")
    if check.get("phase") == "harness_frontend" and re.search(
        r"\b(?:if|match)\b[\s\S]*(?:&&|\|\||==>|<==>)", clauses
    ):
        tags.add("H4")
    if check.get("candidate_vstd_globs") or check.get("support_issue") in {
        "unsafe_generated_import",
        "ambiguous_generated_import",
    }:
        tags.add("H5")
    if "<==>" in clauses and reason in {
        "malformed_clause",
        "harness_frontend_failed",
        "unsupported_context",
    }:
        tags.add("H6")
    return sorted(tags)


def iter_metric_obligations(
    value: Any,
    *,
    metric: str,
    generated_path: Any,
    reference_path: Any,
    project_root: Path,
) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()

    def visit(
        node: Any,
        *,
        function: str = "",
        clause_kind: str = "",
        direction: str = "",
        clause: str = "",
        path: tuple[str, ...] = (),
    ) -> Iterator[dict[str, Any]]:
        if isinstance(node, list):
            for index, item in enumerate(node):
                yield from visit(
                    item,
                    function=function,
                    clause_kind=clause_kind,
                    direction=direction,
                    clause=clause,
                    path=(*path, str(index)),
                )
            return
        if not isinstance(node, Mapping):
            return

        function = str(node.get("function") or function)
        clause_kind = str(node.get("clause_kind") or clause_kind)
        if node.get("clause") is not None:
            clause = _clause_text(node.get("clause"))
        context = node.get("obligation_context")
        is_check = "holds" in node and "status" in node and (
            isinstance(context, Mapping)
            or
            "lemma_harness_version" in node
            or str(node.get("engine") or "").startswith("verus_lemma")
            or node.get("reason") in {
                "target_signature_mismatch",
                "generic_context_mismatch",
                "malformed_clause",
                "unsupported_context",
            }
        )
        if is_check:
            if isinstance(context, Mapping):
                function = str(context.get("function") or function)
                direction = str(context.get("check_name") or direction)
                antecedent = list(context.get("antecedent") or [])
                consequent = list(context.get("consequent") or [])
                kinds = [
                    *list(context.get("antecedent_kinds") or []),
                    *list(context.get("consequent_kinds") or []),
                ]
                clause_kind = clause_kind or next(
                    (str(kind) for kind in kinds if kind), ""
                )
            else:
                antecedent = []
                consequent = [clause] if clause else []
                direction = direction or next(
                    (part for part in reversed(path) if part in _DIRECTION_KEYS),
                    str(node.get("check_name") or "implication"),
                )
            obligation_id = stable_obligation_id(
                generated_path=generated_path,
                reference_path=reference_path,
                metric=metric,
                function=function,
                clause_kind=clause_kind,
                direction=direction,
                antecedent=antecedent,
                consequent=consequent,
                project_root=project_root,
            )
            if obligation_id not in seen:
                seen.add(obligation_id)
                record = {
                    "obligation_id": obligation_id,
                    "generated_path": relative_source_path(generated_path, project_root),
                    "reference_path": relative_source_path(reference_path, project_root),
                    "metric": metric,
                    "function": function,
                    "clause_kind": clause_kind,
                    "direction": direction,
                    "antecedent": [normalized_clause(item) for item in antecedent],
                    "consequent": [normalized_clause(item) for item in consequent],
                    "classification": _status(node),
                    "harness_sha256": node.get("harness_sha256"),
                    "lemma_harness_version": node.get("lemma_harness_version"),
                    "frontend_status": node.get("frontend_status"),
                    "selected_vstd_globs": node.get("selected_vstd_globs") or [],
                    "verus_runtime": node.get("verus_runtime"),
                    "reason": node.get("reason"),
                    "phase": node.get("phase"),
                    "audit_origin": (
                        "ground_self_check" if "ground" in path else "contextual_check"
                    ),
                    "clause_slot": _clause_slot(direction),
                    "check": dict(node),
                }
                record["harness_tags"] = classify_harness_tags(record)
                yield record

        for key, item in node.items():
            next_direction = key if key in _DIRECTION_KEYS else direction
            next_kind = key if key in {"requires", "ensures"} else clause_kind
            yield from visit(
                item,
                function=function,
                clause_kind=next_kind,
                direction=next_direction,
                clause=clause,
                path=(*path, str(key)),
            )

    yield from visit(value)


__all__ = [
    "classify_harness_tags",
    "iter_metric_obligations",
    "legacy_obligation_id",
    "normalized_clause",
    "relative_source_path",
    "stable_obligation_id",
]
