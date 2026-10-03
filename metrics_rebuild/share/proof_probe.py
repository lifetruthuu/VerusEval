from __future__ import annotations

import hashlib
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .functions import (
    precondition_contexts_for_path,
    strength_contexts_for_path,
)
from .lemma_implication import DEFAULT_LEMMA_TIMEOUT_SECONDS, run_probe_lemma
from .text import read_text, token_spans
from .triviality_harness import erase_trigger_annotations, explicit_triggers, isolated_host

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
TRIVIALITY_RECOVERY_VERSION = "isolated-contract-v1"


def _normalize_probe_type(type_text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(type_text or "")).strip()
    # The probe must type-check in the same context as the source contract.
    # Stripping `&mut` makes `old(x)` ill-typed and stripping `&` can change
    # method resolution, producing synthetic E0308 errors.
    return cleaned or "int"


def _proof_params_for_context(ctx: Mapping[str, Any]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in [*ctx.get("parameters", []), *ctx.get("returns", [])]:
        name = str(item.get("name", "") or "")
        if not _IDENT_RE.fullmatch(name) or name in seen:
            continue
        seen.add(name)
        records.append({"name": name, "type": _normalize_probe_type(str(item.get("type", "") or ""))})
    return records


def _probe_name(kind: str, key: str, path: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]", "_", path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", key)
    return f"probe_{kind}_{stem}_{normalized}"


def _agg_status(results: Sequence[Mapping[str, Any]], checked: int) -> str:
    if checked == 0:
        return "not_available"
    statuses = [str(item.get("status", "") or "") for item in results]
    if statuses and all(status == "unavailable" for status in statuses):
        return "unavailable"
    if any(item.get("holds") is None or status in {"skipped", "timeout", "tool_error", "unavailable"} for item, status in zip(results, statuses)):
        return "partial"
    return "ok"


def _probe_isolated(
    path: str,
    ctx: Mapping[str, Any],
    *,
    part: str,
    timeout_seconds: int,
    rlimit: float | None,
) -> dict[str, Any]:
    """Retry one unresolved target contract in a minimal, trigger-repaired host.

    The target signature and logical clauses are copied from the source.  Other
    executable and proof functions are removed, while reachable spec functions
    remain definitions.  The retry never turns a proof body into an axiom.
    """
    target = str(ctx["function"])
    field = "requires" if part == "precondition_falsity" else "ensures"
    clauses = [str(item.get("text", "") or "").strip() for item in ctx.get(field, []) if str(item.get("text", "") or "").strip()]
    # Each lemma uses a frontend and a verification subprocess. Bound the
    # additional work across all retries, including individual conjuncts.
    budget = max(0, timeout_seconds * 3)
    deadline = time.monotonic() + budget
    attempts: list[dict[str, Any]] = []

    def finish(result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **result,
            "recovery_version": TRIVIALITY_RECOVERY_VERSION,
            "recovery_budget_seconds": budget,
            "fallback_attempts": attempts,
        }

    def logical_tokens(items: Sequence[str]) -> list[list[str]]:
        return [[span.text for span in token_spans(erase_trigger_annotations(item))]
                for item in items]

    def execute(host_text: str, method: str, audit: dict[str, Any],
                clause_index: int | None = None) -> dict[str, Any]:
        attempt_timeout = min(timeout_seconds, int((deadline - time.monotonic()) / 2))
        if attempt_timeout < 1:
            return {
                "holds": None, "status": "unknown",
                "reason": "recovery_budget_exhausted", "fallback_method": method,
            }
        with tempfile.TemporaryDirectory(prefix="veruseval_triviality_") as directory:
            host = Path(directory) / "contract.rs"
            host.write_text(host_text, encoding="utf-8")
            contexts = (precondition_contexts_for_path(str(host))
                        if part == "precondition_falsity"
                        else strength_contexts_for_path(str(host)))
            selected = [item for item in contexts if item.get("function") == target]
            expected = clauses if clause_index is None else [clauses[clause_index]]
            actual = ([str(item["text"]).strip() for item in selected[0].get(field, [])]
                      if len(selected) == 1 else [])
            if len(selected) != 1 or logical_tokens(actual) != logical_tokens(expected):
                retry = {
                    "holds": None, "status": "unknown",
                    "reason": "isolated_contract_mismatch",
                }
            else:
                # Use the reparsed clauses: these include trigger repairs and,
                # when splitting, only the selected original conjunct.
                retry = run_probe_lemma(
                    host_rs_path=str(host),
                    probe_name=f"triviality_{part}_{target}",
                    source_function_name=target,
                    parameters=_proof_params_for_context(selected[0]),
                    requires_clauses=[] if part == "postcondition_truth" else actual,
                    ensures_clauses=["false"] if part == "precondition_falsity" else actual,
                    timeout_seconds=attempt_timeout,
                    rlimit=rlimit,
                )
            retry = dict(retry)
            retry["fallback_method"] = method
            retry["fallback_audit"] = audit
            retry["isolated_host_sha256"] = hashlib.sha256(host_text.encode("utf-8")).hexdigest()
            if clause_index is not None:
                retry["fallback_clause_index"] = clause_index
            attempts.append(dict(retry))
            return retry

    source = read_text(path)
    try:
        host_text, audit = isolated_host(source, target, part)
    except (AssertionError, ValueError, IndexError) as exc:
        return finish({"holds": None, "status": "unknown", "reason": "isolated_host_failed",
                       "fallback_method": "isolated_contract", "error": f"{type(exc).__name__}: {exc}"})

    # Isolated hosts already erase trigger metadata. This first retry therefore
    # tests the same formulas with automatic trigger inference.
    retry = execute(host_text, "isolated_contract", audit)
    if retry.get("holds") is not None:
        return finish(retry)

    repaired, trigger_audit = explicit_triggers(host_text)
    if repaired != host_text:
        retry = execute(repaired, "explicit_triggers", {**audit, "trigger_repairs": trigger_audit})
        if retry.get("holds") is not None:
            return finish(retry)

    if part == "postcondition_truth" and len(clauses) > 1:
        # A rejected conjunct is sufficient for the existing non-detection
        # score, not a logical counterexample. Proved individual conjuncts do
        # not upgrade a failed whole-block attempt to a proved tautology.
        for index in range(len(clauses)):
            try:
                single, single_audit = isolated_host(source, target, part, index)
            except (AssertionError, ValueError, IndexError) as exc:
                attempts.append({"holds": None, "status": "unknown", "fallback_clause_index": index,
                                 "reason": "isolated_host_failed",
                                 "error": f"{type(exc).__name__}: {exc}"})
                continue
            single, repairs = explicit_triggers(single)
            result = execute(single, "conjunct", {**single_audit, "trigger_repairs": repairs}, index)
            if result.get("holds") is False:
                return finish(result)
            if result.get("reason") == "recovery_budget_exhausted":
                return finish(result)
    return finish(retry)


def _recover_unknown_probe(
    probe: dict[str, Any], path: str, ctx: Mapping[str, Any], *,
    part: str, timeout_seconds: int, rlimit: float | None, enabled: bool,
) -> dict[str, Any]:
    if not enabled or probe.get("holds") is not None:
        return probe
    if probe.get("status") == "unavailable" or probe.get("reason") in {
        "verus_binary_not_found", "reference_file_not_found", "reference_file_unreadable",
    }:
        return probe
    recovered = _probe_isolated(
        path, ctx, part=part, timeout_seconds=timeout_seconds, rlimit=rlimit,
    )
    recovered["initial_probe"] = dict(probe)
    recovered["source_rs_path"] = path
    recovered["source_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return recovered


def probe_precondition_falsity_for_path(
    path: str,
    *,
    timeout_seconds: int = DEFAULT_LEMMA_TIMEOUT_SECONDS,
    rlimit: float | None = None,
    function_name: str | None = None,
    recover_unknown: bool = True,
) -> dict[str, Any]:
    contexts = [ctx for ctx in precondition_contexts_for_path(path) if ctx.get("mode") == "exec"]
    if function_name is not None:
        contexts = [ctx for ctx in contexts if ctx.get("function") == function_name]
    fn_results: list[dict[str, Any]] = []
    checked = 0
    false_count = 0
    for ctx in contexts:
        requires = [str(clause.get("text", "") or "").strip() for clause in ctx.get("requires", []) if str(clause.get("text", "") or "").strip()]
        checked += 1
        if not requires:
            fn_results.append(
                {
                    "function": ctx["function"],
                    "requires": [],
                    "outcome": "no_requires_default_true",
                    "always_false": False,
                    "holds": False,
                    "status": "ok",
                    "counted": True,
                }
            )
            continue
        probe = run_probe_lemma(
            host_rs_path=path,
            probe_name=_probe_name("prefalse", str(ctx["function"]), path),
            source_function_name=str(ctx["function"]),
            parameters=_proof_params_for_context(ctx),
            requires_clauses=requires,
            ensures_clauses=["false"],
            timeout_seconds=timeout_seconds,
            rlimit=rlimit,
        )
        probe = _recover_unknown_probe(
            probe, path, ctx, part="precondition_falsity",
            timeout_seconds=timeout_seconds, rlimit=rlimit, enabled=recover_unknown,
        )
        always_false = probe.get("holds") is True
        if always_false:
            false_count += 1
        fn_results.append(
            {
                "function": ctx["function"],
                "requires": requires,
                "outcome": "always_false" if always_false else ("unknown" if probe.get("holds") is None else "not_trivial"),
                "always_false": always_false,
                "holds": probe.get("holds"),
                "status": probe.get("status"),
                "counted": True,
                "probe": probe,
            }
        )
    unresolved = sum(1 for result in fn_results if result.get("holds") is None)
    return {
        "status": _agg_status(fn_results, checked),
        "score": (false_count / checked) if checked else None,
        "checked_functions": checked,
        "always_false_functions": false_count,
        "unresolved": unresolved,
        "functions": fn_results,
        "engine": "verus_proof_probe",
    }


def probe_postcondition_truth_for_path(
    path: str,
    *,
    timeout_seconds: int = DEFAULT_LEMMA_TIMEOUT_SECONDS,
    rlimit: float | None = None,
    function_name: str | None = None,
    recover_unknown: bool = True,
) -> dict[str, Any]:
    contexts = strength_contexts_for_path(path)
    if function_name is not None:
        contexts = [ctx for ctx in contexts if ctx.get("function") == function_name]
    fn_results: list[dict[str, Any]] = []
    checked = 0
    true_count = 0
    for ctx in contexts:
        ensures = [str(clause.get("text", "") or "").strip() for clause in ctx.get("ensures", []) if str(clause.get("text", "") or "").strip()]
        checked += 1
        if not ensures:
            true_count += 1
            fn_results.append(
                {
                    "function": ctx["function"],
                    "ensures": [],
                    "outcome": "no_ensures_default_true",
                    "always_true": True,
                    "holds": True,
                    "status": "ok",
                    "counted": True,
                }
            )
            continue
        probe = run_probe_lemma(
            host_rs_path=path,
            probe_name=_probe_name("posttrue", str(ctx["function"]), path),
            source_function_name=str(ctx["function"]),
            parameters=_proof_params_for_context(ctx),
            requires_clauses=[],
            ensures_clauses=ensures,
            timeout_seconds=timeout_seconds,
            rlimit=rlimit,
        )
        probe = _recover_unknown_probe(
            probe, path, ctx, part="postcondition_truth",
            timeout_seconds=timeout_seconds, rlimit=rlimit, enabled=recover_unknown,
        )
        always_true = probe.get("holds") is True
        if always_true:
            true_count += 1
        fn_results.append(
            {
                "function": ctx["function"],
                "ensures": ensures,
                "outcome": "always_true" if always_true else ("unknown" if probe.get("holds") is None else "not_trivial"),
                "always_true": always_true,
                "holds": probe.get("holds"),
                "status": probe.get("status"),
                "counted": True,
                "unknown": probe.get("holds") is None,
                "probe": probe,
            }
        )
    unresolved = sum(1 for result in fn_results if result.get("unknown"))
    return {
        "status": _agg_status(fn_results, checked),
        "score": (true_count / checked) if checked else None,
        "checked_functions": checked,
        "always_true_functions": true_count,
        "skipped_old_mut": sum(1 for result in fn_results if result.get("outcome") == "skipped_old_mut"),
        "unresolved": unresolved,
        "functions": fn_results,
        "engine": "verus_proof_probe",
    }


__all__ = [
    "probe_postcondition_truth_for_path",
    "probe_precondition_falsity_for_path",
]
