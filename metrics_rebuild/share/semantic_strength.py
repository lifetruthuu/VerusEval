from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Optional, Sequence

from ._shared import (
    IMPLICIT_TRUE_CONTRACT_CLAUSES_DEFAULT,
    get_field,
    materialize_contract_clauses,
)
from .lemma_implication import (
    DEFAULT_LEMMA_TIMEOUT_SECONDS,
    lemma_implication_check as _lemma_check,
)


def _records(
    value: Any,
    *,
    kind: str,
    source: str,
    inject_implicit_true: bool = IMPLICIT_TRUE_CONTRACT_CLAUSES_DEFAULT,
) -> list[dict[str, str]]:
    records = materialize_contract_clauses(
        value,
        kind=kind,
        inject_implicit_true=inject_implicit_true,
    )
    return [{**record, "_lemma_source": source} for record in records]


def _context_name(value: Any) -> str:
    return str(get_field(value, "function", "<unknown>"))


def _context_map(contexts: Sequence[Any]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for context in contexts:
        name = _context_name(context)
        counts[name] = counts.get(name, 0) + 1
        key = name if counts[name] == 1 else f"{name}#{counts[name]}"
        mapped[key] = context
    return mapped


def _implication_check(
    *,
    antecedent: Sequence[Mapping[str, str]],
    consequent: Sequence[Mapping[str, str]],
    reference_rs_path: Optional[str],
    function_name: str,
    check_name: str,
    reference_context: Mapping[str, Any],
    generated_context: Mapping[str, Any],
    timeout_seconds: int,
    rlimit: Optional[float] = None,
    generated_rs_path: Optional[str] = None,
) -> dict[str, Any]:
    if not reference_rs_path:
        return {
            "holds": None,
            "status": "unknown",
            "reason": "no_reference_path",
            "phase": "runtime",
            "engine": "verus_lemma",
            "antecedent_total": len(antecedent),
            "consequent_total": len(consequent),
        }
    return _lemma_check(
        reference_rs_path=reference_rs_path,
        reference_context=reference_context,
        generated_context=generated_context,
        function_name=function_name,
        check_name=check_name,
        antecedent=antecedent,
        consequent=consequent,
        generated_rs_path=generated_rs_path,
        timeout_seconds=timeout_seconds,
        rlimit=rlimit,
    )


def _binary_relation(*, left_implies_right: Any, right_implies_left: Any, left_name: str, right_name: str) -> str:
    if left_implies_right is None or right_implies_left is None:
        return "unknown"
    if left_implies_right is True and right_implies_left is True:
        return "equivalent"
    if left_implies_right is True:
        return f"{left_name}_stronger_or_equal"
    if right_implies_left is True:
        return f"{left_name}_weaker_or_equal"
    return "incomparable"


def _contract_relation(item: Mapping[str, Any]) -> str:
    if item.get("status") != "ok":
        if item.get("status") == "missing_generated":
            return "missing_generated"
        return "unknown"
    generated_refines_ground = item.get("generated_refines_ground")
    ground_refines_generated = item.get("ground_refines_generated")
    if generated_refines_ground is True and ground_refines_generated is True:
        return "equivalent"
    if generated_refines_ground is True:
        return "generated_refines_reference"
    if ground_refines_generated is True:
        return "reference_refines_generated"
    if generated_refines_ground is None or ground_refines_generated is None:
        return "unknown"
    return "incomparable"


def _combine_parts(parts: Sequence[Any]) -> Optional[bool]:
    if all(value is True for value in parts):
        return True
    if any(value is False for value in parts):
        return False
    return None


def _function_strength(
    generated: Mapping[str, Any],
    ground: Mapping[str, Any],
    timeout_seconds: int,
    *,
    rlimit: Optional[float] = None,
    lemma_reference_path: Optional[str] = None,
    generated_rs_path: Optional[str] = None,
    inject_implicit_true_contract_clauses: bool = IMPLICIT_TRUE_CONTRACT_CLAUSES_DEFAULT,
) -> dict[str, Any]:
    gen_requires = _records(
        get_field(generated, "requires", []),
        kind="requires",
        source="generated",
        inject_implicit_true=inject_implicit_true_contract_clauses,
    )
    ground_requires = _records(
        get_field(ground, "requires", []),
        kind="requires",
        source="reference",
        inject_implicit_true=inject_implicit_true_contract_clauses,
    )
    gen_ensures = _records(
        get_field(generated, "ensures", []),
        kind="ensures",
        source="generated",
        inject_implicit_true=inject_implicit_true_contract_clauses,
    )
    ground_ensures = _records(
        get_field(ground, "ensures", []),
        kind="ensures",
        source="reference",
        inject_implicit_true=inject_implicit_true_contract_clauses,
    )

    function_name = _context_name(generated)

    check_args = {
        "reference_rs_path": lemma_reference_path,
        "function_name": function_name,
        "reference_context": ground,
        "generated_context": generated,
        "generated_rs_path": generated_rs_path,
        "timeout_seconds": timeout_seconds,
        "rlimit": rlimit,
    }

    ground_pre_implies_gen_pre = _implication_check(
        antecedent=ground_requires,
        consequent=gen_requires,
        check_name="ground_pre_implies_gen_pre",
        **check_args,
    )
    gen_pre_implies_ground_pre = _implication_check(
        antecedent=gen_requires,
        consequent=ground_requires,
        check_name="gen_pre_implies_ground_pre",
        **check_args,
    )
    gen_post_implies_ground_post = _implication_check(
        antecedent=ground_requires + gen_requires + gen_ensures,
        consequent=ground_ensures,
        check_name="gen_post_implies_ground_post",
        **check_args,
    )
    ground_post_implies_gen_post = _implication_check(
        antecedent=ground_requires + gen_requires + ground_ensures,
        consequent=gen_ensures,
        check_name="ground_post_implies_gen_post",
        **check_args,
    )

    generated_parts = [
        ground_pre_implies_gen_pre.get("holds"),
        gen_post_implies_ground_post.get("holds"),
    ]
    ground_parts = [
        gen_pre_implies_ground_pre.get("holds"),
        ground_post_implies_gen_post.get("holds"),
    ]
    generated_refines_ground = _combine_parts(generated_parts)
    ground_refines_generated = _combine_parts(ground_parts)

    direction_values = [
        ground_pre_implies_gen_pre.get("holds"),
        gen_pre_implies_ground_pre.get("holds"),
        gen_post_implies_ground_post.get("holds"),
        ground_post_implies_gen_post.get("holds"),
    ]
    if any(value is None for value in direction_values):
        status = "partial"
    else:
        status = "ok"

    precondition_relation = _binary_relation(
        left_implies_right=gen_pre_implies_ground_pre.get("holds"),
        right_implies_left=ground_pre_implies_gen_pre.get("holds"),
        left_name="generated",
        right_name="reference",
    )
    postcondition_relation = _binary_relation(
        left_implies_right=gen_post_implies_ground_post.get("holds"),
        right_implies_left=ground_post_implies_gen_post.get("holds"),
        left_name="generated",
        right_name="reference",
    )

    result = {
        "function": function_name,
        "mode": str(get_field(generated, "mode", get_field(ground, "mode", "exec"))),
        "status": status,
        "parameter_types": {},
        "has_signature_spec": bool(
            get_field(generated, "has_signature_spec", False)
            or get_field(ground, "has_signature_spec", False)
        ),
        "has_contract": bool(
            get_field(generated, "has_contract", False)
            or get_field(ground, "has_contract", False)
        ),
        "requires": {
            "generated_total": len(gen_requires),
            "ground_total": len(ground_requires),
            "ground_implies_generated": ground_pre_implies_gen_pre,
            "generated_implies_ground": gen_pre_implies_ground_pre,
            "generated_is_weaker_or_equal": ground_pre_implies_gen_pre.get("holds"),
            "generated_is_stronger_or_equal": gen_pre_implies_ground_pre.get("holds"),
            "relation": precondition_relation,
        },
        "ensures": {
            "generated_total": len(gen_ensures),
            "ground_total": len(ground_ensures),
            "generated_implies_ground": gen_post_implies_ground_post,
            "ground_implies_generated": ground_post_implies_gen_post,
            "generated_is_stronger_or_equal": gen_post_implies_ground_post.get("holds"),
            "generated_is_weaker_or_equal": ground_post_implies_gen_post.get("holds"),
            "relation": postcondition_relation,
        },
        "precondition_relation": precondition_relation,
        "postcondition_relation": postcondition_relation,
        "generated_refines_ground": generated_refines_ground,
        "ground_refines_generated": ground_refines_generated,
        "implicit_true_contract_clauses": inject_implicit_true_contract_clauses,
    }
    result["contract_relation"] = _contract_relation(result)
    return result


def _classification(function_results: Sequence[Mapping[str, Any]]) -> str:
    if not function_results:
        return "no_matching_functions"

    gen_values = [item.get("generated_refines_ground") for item in function_results]
    ground_values = [item.get("ground_refines_generated") for item in function_results]
    all_gen = all(value is True for value in gen_values)
    all_ground = all(value is True for value in ground_values)

    if all_gen and all_ground:
        return "equivalent"
    if all_gen:
        return "generated_stronger_or_equal"
    if all_ground:
        return "generated_weaker_or_equal"
    if any(item.get("status") != "ok" for item in function_results):
        return "unknown"
    return "incomparable"


def semantic_strength_comparison(
    generated_contexts: Sequence[Any],
    ground_contexts: Sequence[Any],
    timeout_seconds: int = DEFAULT_LEMMA_TIMEOUT_SECONDS,
    *,
    rlimit: Optional[float] = None,
    lemma_reference_path: Optional[str] = None,
    generated_rs_path: Optional[str] = None,
    inject_implicit_true_contract_clauses: bool = IMPLICIT_TRUE_CONTRACT_CLAUSES_DEFAULT,
    **_kwargs: Any,
) -> dict[str, Any]:
    generated_map = _context_map(generated_contexts)
    ground_map = _context_map(ground_contexts)
    compared_mode_counts = Counter(str(get_field(context, "mode", "exec")) for context in ground_contexts)
    ground_keys = list(ground_map)
    matched_keys = [key for key in ground_keys if key in generated_map]
    missing_generated = [key for key in ground_keys if key not in generated_map]
    extra_generated = [key for key in generated_map if key not in ground_map]

    if not lemma_reference_path:
        return {
            "score": None,
            "status": "unavailable",
            "classification": "unknown",
            "engine": "verus_lemma",
            "method": "verus_lemma_implication",
            "reason": "no_reference_path",
            "matched_functions": 0,
            "compared_mode_counts": dict(compared_mode_counts),
            "missing_generated_functions": missing_generated,
            "extra_generated_functions": extra_generated,
            "functions": [],
        }

    function_results = [
        _function_strength(
            generated_map[key],
            ground_map[key],
            timeout_seconds,
            rlimit=rlimit,
            lemma_reference_path=lemma_reference_path,
            generated_rs_path=generated_rs_path,
            inject_implicit_true_contract_clauses=inject_implicit_true_contract_clauses,
        )
        for key in matched_keys
    ]
    for key in missing_generated:
        function_results.append(
            {
                "function": key,
                "status": "missing_generated",
                "contract_relation": "missing_generated",
                "generated_refines_ground": False,
                "ground_refines_generated": False,
                "requires": {},
                "ensures": {},
            }
        )

    classification = _classification(function_results)
    total = len(function_results)
    generated_refines_count = sum(1 for item in function_results if item.get("generated_refines_ground") is True)
    ground_refines_count = sum(1 for item in function_results if item.get("ground_refines_generated") is True)
    failed_count = sum(1 for item in function_results if item.get("generated_refines_ground") is False)
    unknown_count = total - generated_refines_count - failed_count
    determined = generated_refines_count + failed_count
    relation_counts = Counter(str(item.get("contract_relation") or _contract_relation(item)) for item in function_results)
    has_unresolved_relation = any(
        item.get("status") not in {"ok", "missing_generated"}
        for item in function_results
    )
    status = (
        "no_matching_functions"
        if not total
        else "partial"
        if unknown_count or has_unresolved_relation
        else "ok"
    )
    score = generated_refines_count / determined if determined else None
    gen_values = [item.get("generated_refines_ground") for item in function_results]
    ground_values = [item.get("ground_refines_generated") for item in function_results]
    generated_refines_ground: Optional[bool]
    ground_refines_generated: Optional[bool]
    if not total:
        generated_refines_ground = None
        ground_refines_generated = None
    elif all(value is True for value in gen_values):
        generated_refines_ground = True
    elif any(value is False for value in gen_values):
        generated_refines_ground = False
    else:
        generated_refines_ground = None
    if not total:
        ground_refines_generated = None
    elif all(value is True for value in ground_values):
        ground_refines_generated = True
    elif any(value is False for value in ground_values):
        ground_refines_generated = False
    else:
        ground_refines_generated = None

    return {
        "score": score,
        "status": status,
        "classification": classification,
        "engine": "verus_lemma",
        "method": "verus_lemma_implication",
        "generated_refines_ground": generated_refines_ground,
        "ground_refines_generated": ground_refines_generated,
        "matched_functions": len(matched_keys),
        "functions_total": total,
        "passed": generated_refines_count,
        "failed": failed_count,
        "unknown": unknown_count,
        "total": total,
        "determined": determined,
        "coverage": determined / total if total else None,
        "compared_mode_counts": dict(compared_mode_counts),
        "generated_refines_ground_functions": generated_refines_count,
        "ground_refines_generated_functions": ground_refines_count,
        "equivalent_functions": relation_counts.get("equivalent", 0),
        "strict_generated_refinement_functions": relation_counts.get("generated_refines_reference", 0),
        "strict_generated_weakening_functions": relation_counts.get("reference_refines_generated", 0),
        "incomparable_functions": relation_counts.get("incomparable", 0),
        "unknown_functions": unknown_count,
        "relation_counts": dict(relation_counts),
        "precondition_relation_summary": dict(
            Counter(str(item.get("precondition_relation", "unknown")) for item in function_results)
        ),
        "postcondition_relation_summary": dict(
            Counter(str(item.get("postcondition_relation", "unknown")) for item in function_results)
        ),
        "contract_relation": classification,
        "missing_generated_functions": missing_generated,
        "extra_generated_functions": extra_generated,
        "functions": function_results,
        "implicit_true_contract_clauses": inject_implicit_true_contract_clauses,
        "note": (
            "Verus-lemma-based refinement check: generated is at least as strong as reference when "
            "reference requires imply generated requires, and generated ensures imply reference "
            "ensures over the shared precondition domain."
        ),
    }
