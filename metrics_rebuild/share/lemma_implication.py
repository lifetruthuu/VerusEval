from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ._shared import clause_text, get_field
from .clauses import _scan_top_level_comma_offsets
from .functions import (
    function_declaration_context_for_path,
)
from .lemma_binding import (
    bind_function_contexts,
    build_clause_wrapper,
    immutable_snapshot_type,
    render_clause_wrapper,
)
from .lemma_harness import (
    GeneratedSupportContext,
    apply_generated_support_renames,
    append_lemma_to_reference as _append_lemma_to_reference,
    collect_generated_support,
    find_verus_block_close as _find_verus_block_close,
    make_lemma as _make_lemma,
    normalize_lemma_clause as _normalize_lemma_clause,
)
from .lemma_shortcuts import safe_trivial_implication
from .text import token_spans
from .verus_runner import verus_runtime_fingerprint


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_LEMMA_TIMEOUT_SECONDS = 20
LEMMA_HARNESS_VERSION = "bound-wrapper-hardening-v5"

_verus_binary: Optional[str] = None
_LEMMA_CACHE: dict[str, dict[str, Any]] = {}


def set_lemma_verus_binary(path: Optional[str]) -> None:
    global _verus_binary
    _verus_binary = path


def _load_verus_path_from_config() -> Optional[str]:
    try:
        lines = CONFIG_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        if key.strip().lower() == "verus_path":
            value = raw_value.split("#", 1)[0].strip().strip("\"'")
            return value or None
    return None


def _verus_bin() -> str:
    return _verus_binary or _load_verus_path_from_config() or "verus"


def _clause_texts_sources_and_kinds(
    clauses: Sequence[Any],
) -> tuple[list[str], list[str], list[str]]:
    texts: list[str] = []
    sources: list[str] = []
    kinds: list[str] = []
    for clause in clauses:
        text = clause_text(clause).strip()
        if text:
            texts.append(text)
            sources.append(str(get_field(clause, "_lemma_source", "") or ""))
            kinds.append(str(get_field(clause, "kind", "") or ""))
    return texts, sources, kinds


def _parameter_records(parameters: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in parameters:
        name = str(get_field(item, "name", "") or "")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        if name in seen:
            continue
        seen.add(name)
        records.append({"name": name, "type": str(get_field(item, "type", "int") or "int")})
    return records


def _mutable_snapshot_context(
    params: Sequence[Mapping[str, str]],
    antecedent: Sequence[str],
    consequent: Sequence[str],
    *,
    antecedent_kinds: Sequence[str] = (),
    consequent_kinds: Sequence[str] = (),
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    """Model each `&mut T` as independent immutable pre/post snapshots.

    Original requires clauses use `old(x)` for the pre-state while original
    ensures clauses use bare `x` for the post-state.  Once both formulas are
    moved into a new lemma contract, keeping an `&mut` parameter gives those
    names the *lemma's* pre/post meaning and makes post-state formulas illegal
    in `requires`.  Two immutable values preserve the intended implication.
    """
    rewritten_params: list[dict[str, str]] = []
    rewritten_antecedent = list(antecedent)
    rewritten_consequent = list(consequent)
    used_names = {str(item.get("name") or "") for item in params}

    for item in params:
        name = str(item.get("name") or "")
        type_text = str(item.get("type") or "int")
        immutable_type = immutable_snapshot_type(type_text)
        if immutable_type is None or not name:
            rewritten_params.append({"name": name, "type": type_text})
            continue

        old_name = f"__sqm_old_{name}"
        suffix = 2
        while old_name in used_names:
            old_name = f"__sqm_old_{name}_{suffix}"
            suffix += 1
        used_names.add(old_name)
        rewritten_params.append({"name": name, "type": immutable_type})
        rewritten_params.append({"name": old_name, "type": immutable_type})

        old_deref = re.compile(
            r"\bold\s*\(\s*\*\s*" + re.escape(name) + r"\s*\)"
        )
        old_value = re.compile(
            r"\bold\s*\(\s*" + re.escape(name) + r"\s*\)"
        )

        def rewrite(text: str, kind: str) -> str:
            text = old_deref.sub(f"*{old_name}", text)
            text = old_value.sub(old_name, text)
            # A mutable argument's bare name denotes its entry value in an
            # original precondition, but its post-state value in an original
            # postcondition.  Keep those states distinct after moving both
            # kinds of clauses into the synthetic lemma contract.
            if kind in {"requires", "recommends"}:
                text = re.sub(r"\b" + re.escape(name) + r"\b", old_name, text)
            return text

        rewritten_antecedent = [
            rewrite(text, antecedent_kinds[index] if index < len(antecedent_kinds) else "")
            for index, text in enumerate(rewritten_antecedent)
        ]
        rewritten_consequent = [
            rewrite(text, consequent_kinds[index] if index < len(consequent_kinds) else "")
            for index, text in enumerate(rewritten_consequent)
        ]

    return rewritten_params, rewritten_antecedent, rewritten_consequent


def _delimiter_issue(text: str) -> Optional[str]:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for span in token_spans(text):
        tok = span.text
        if tok in {"(", "[", "{"}:
            stack.append(tok)
        elif tok in pairs:
            if not stack or stack[-1] != pairs[tok]:
                return f"unexpected_{tok}"
            stack.pop()
    if stack:
        return f"unclosed_{stack[-1]}"
    return None


_DANGLING_CLAUSE_TOKENS = {
    "==>",
    "<==>",
    "=>",
    "&&",
    "&&&",
    "||",
    "|||",
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
    "&",
    "|",
    "^",
    "=",
    "::",
    ".",
    ",",
}


def _clause_syntax_issue(text: str) -> Optional[str]:
    if not text.strip():
        return "empty_expression"
    delimiter = _delimiter_issue(text)
    if delimiter is not None:
        return delimiter
    _, unclosed_pipe_binder = _scan_top_level_comma_offsets(text)
    if unclosed_pipe_binder:
        return "unclosed_pipe_binder"
    spans = token_spans(text)
    if not spans:
        return "empty_expression"
    if spans[-1].text == ">" and _terminal_generic_is_complete(spans):
        return None
    if spans[-1].text in _DANGLING_CLAUSE_TOKENS:
        return f"dangling_{spans[-1].text}"
    return None


def _terminal_generic_is_complete(spans: Sequence[Any]) -> bool:
    """Recognize a balanced terminal generic argument list such as ``::<T>``."""
    if not spans or spans[-1].text != ">":
        return False
    depth = 0
    open_index: Optional[int] = None
    for index in range(len(spans) - 1, -1, -1):
        token = spans[index].text
        if token == ">":
            depth += 1
        elif token == "<":
            depth -= 1
            if depth == 0:
                open_index = index
                break
        elif token in {"<==>", "==>", "<=", ">="}:
            return False
    if open_index is None or open_index == 0 or open_index + 1 == len(spans):
        return False
    previous = spans[open_index - 1]
    return previous.text == "::" or (
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", previous.text) is not None
        and previous.end == spans[open_index].start
    )


def _uses_associated_self(text: str) -> bool:
    return any(span.text in {"self", "Self"} for span in token_spans(text))


def _verus_diagnostic_text(run: subprocess.CompletedProcess[str]) -> tuple[str, list[dict[str, Any]]]:
    """Return searchable diagnostics, preferring Verus' JSON payloads.

    Verus emits one JSON object per line with ``--output-json``.  Keep the raw
    output as a fallback because older builds and early Rust frontend failures
    can still write plain text.
    """
    stderr = run.stderr or ""
    stdout = run.stdout or ""
    raw_text = stderr + "\n" + stdout
    diagnostics: list[dict[str, Any]] = []
    rendered: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        diagnostics.append(payload)
        rendered.extend(
            str(value)
            for value in (
                payload.get("message"),
                payload.get("rendered"),
                payload.get("note"),
            )
            if value
        )
    searchable = "\n".join([*rendered, raw_text])
    return searchable, diagnostics


def _classify_verus_run(run: subprocess.CompletedProcess[str], elapsed: float) -> dict[str, Any]:
    text, diagnostics = _verus_diagnostic_text(run)
    if run.returncode == 0:
        return {
            "holds": True,
            "status": "valid",
            "elapsed_seconds": elapsed,
            "returncode": run.returncode,
        }
    first_error = ""
    for line in text.splitlines():
        if line.startswith("error") or '"level":"error"' in line.replace(" ", ""):
            first_error = line[:300]
            break
    diagnostic_codes = set(re.findall(r"\berror\[([A-Z]\d{4})\]", text))
    diagnostic_codes.update(
        re.findall(
            r'"code"\s*:\s*\{\s*"code"\s*:\s*"([A-Z]\d{4})"',
            text,
        )
    )
    lowered = text.lower()
    verifier_issue = ""
    if "could not automatically infer triggers" in lowered:
        verifier_issue = "trigger_inference"
    elif "resource limit (rlimit) exceeded" in lowered:
        verifier_issue = "resource_limit"
    elif any(
        marker in lowered
        for marker in (
            "decreases not satisfied",
            "cannot prove termination",
            "might not terminate",
        )
    ):
        verifier_issue = "termination_check"
    if verifier_issue:
        return {
            "holds": None,
            "status": "unknown",
            "reason": "verification_unresolved",
            "phase": "verification",
            "verification_issue": verifier_issue,
            "elapsed_seconds": elapsed,
            "returncode": run.returncode,
            "error": first_error or text[:300],
            "diagnostic_codes": sorted(diagnostic_codes),
            "diagnostic_excerpt": text[:2000],
            "diagnostics": diagnostics[:20],
        }
    if "postcondition not satisfied" in lowered or "assertion failed" in lowered:
        return {
            "holds": False,
            "status": "invalid",
            "elapsed_seconds": elapsed,
            "returncode": run.returncode,
        }
    return {
        "holds": None,
        "status": "unknown",
        "reason": "unclassified_verus_failure",
        "phase": "verification",
        "elapsed_seconds": elapsed,
        "returncode": run.returncode,
        "error": first_error or text[:300],
        "diagnostic_codes": sorted(diagnostic_codes),
        "diagnostic_excerpt": text[:2000],
        "diagnostics": diagnostics[:20],
    }


def _cache_key(
    *,
    reference_path: Path,
    function_name: str,
    check_name: str,
    params: Sequence[Mapping[str, str]],
    generic_parameters: str,
    where_clause: str,
    antecedent: Sequence[str],
    consequent: Sequence[str],
    generated_path: Optional[Path] = None,
    support_context: Optional[GeneratedSupportContext] = None,
    runtime_fingerprint: Mapping[str, Any],
    timeout_seconds: int,
    rlimit: Optional[float] = None,
) -> str:
    reference_sha256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    payload: dict[str, Any] = {
        "reference": str(reference_path.resolve()),
        "reference_sha256": reference_sha256,
        "function": function_name,
        "check": check_name,
        "params": list(params),
        "generic_parameters": generic_parameters,
        "where_clause": where_clause,
        "antecedent": list(antecedent),
        "consequent": list(consequent),
        "lemma_harness_version": LEMMA_HARNESS_VERSION,
        "verus_runtime": dict(runtime_fingerprint),
        "timeout_seconds": timeout_seconds,
        "rlimit": rlimit,
    }
    if support_context is not None:
        payload["support_outer_uses"] = list(support_context.outer_uses)
        payload["support_inner_items"] = list(support_context.inner_items)
        payload["support_summary"] = dict(support_context.summary)
        payload["candidate_vstd_globs"] = list(support_context.candidate_vstd_globs)
    if generated_path is not None and generated_path.exists():
        payload["generated"] = str(generated_path.resolve())
        payload["generated_sha256"] = hashlib.sha256(generated_path.read_bytes()).hexdigest()
    seed = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()


def _lemma_result_metadata(
    support_context: Optional[GeneratedSupportContext] = None,
    *,
    harness_sha256: Optional[str] = None,
    frontend_status: str = "not_run",
    selected_vstd_globs: Sequence[str] = (),
) -> dict[str, Any]:
    summary = (
        dict(support_context.summary)
        if support_context is not None
        else dict(GeneratedSupportContext().summary)
    )
    return {
        "lemma_harness_version": LEMMA_HARNESS_VERSION,
        "harness_sha256": harness_sha256,
        "frontend_status": frontend_status,
        "selected_vstd_globs": sorted(set(selected_vstd_globs)),
        "candidate_vstd_globs": sorted(
            set(support_context.candidate_vstd_globs if support_context else ())
        ),
        "support_context_summary": summary,
    }


_UNRESOLVED_PATTERNS = (
    re.compile(r"cannot find (?:value|type|function|macro|trait|module|crate)[^`]*`([^`]+)`", re.I),
    re.compile(r"failed to resolve: use of undeclared (?:type|module) `([^`]+)`", re.I),
    re.compile(r"unresolved import `([^`]+)`", re.I),
)


def _unresolved_symbols(run: subprocess.CompletedProcess[str]) -> set[str]:
    text, _ = _verus_diagnostic_text(run)
    result: set[str] = set()
    for pattern in _UNRESOLVED_PATTERNS:
        result.update(match.group(1) for match in pattern.finditer(text))
    return result


def _non_unresolved_error_signatures(run: subprocess.CompletedProcess[str]) -> set[str]:
    text, diagnostics = _verus_diagnostic_text(run)
    signatures: set[str] = set()
    for payload in diagnostics:
        level = str(payload.get("level") or "").lower()
        message = str(payload.get("message") or "")
        if level == "error" and message and not any(
            pattern.search(message) for pattern in _UNRESOLVED_PATTERNS
        ):
            signatures.add(re.sub(r"\s+", " ", message).strip())
    if not diagnostics:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("error") and not any(
                pattern.search(stripped) for pattern in _UNRESOLVED_PATTERNS
            ):
                signatures.add(re.sub(r"\s+", " ", stripped)[:500])
    return signatures


def _verify_lemma(
    *,
    reference_path: Path,
    function_name: str,
    check_name: str,
    params: Sequence[Mapping[str, str]],
    generic_parameters: str = "",
    where_clause: str = "",
    antecedent_texts: Sequence[str],
    consequent_texts: Sequence[str],
    antecedent_kinds: Sequence[str] = (),
    consequent_kinds: Sequence[str] = (),
    timeout_seconds: int,
    rlimit: Optional[float] = None,
    support_context: Optional[GeneratedSupportContext] = None,
    generated_path: Optional[Path] = None,
    engine: str = "verus_lemma",
    lemma_prefix: str = "lemma_sqm",
) -> dict[str, Any]:
    support_context = support_context or GeneratedSupportContext()
    result_metadata = _lemma_result_metadata(support_context)
    antecedent_texts = [
        _normalize_lemma_clause(text) for text in antecedent_texts
    ]
    consequent_texts = [
        _normalize_lemma_clause(text) for text in consequent_texts
    ]
    if not consequent_texts:
        return {
            **result_metadata,
            "holds": True,
            "status": "valid",
            "reason": "empty_consequent",
            "engine": engine,
            "antecedent_total": len(antecedent_texts),
            "consequent_total": 0,
            "elapsed_seconds": 0.0,
        }

    if not reference_path.exists():
        return {
            **result_metadata,
            "holds": None,
            "status": "unknown",
            "reason": "reference_file_not_found",
            "phase": "harness_construction",
            "engine": engine,
            "reference_rs_path": str(reference_path),
            "antecedent_total": len(antecedent_texts),
            "consequent_total": len(consequent_texts),
        }

    params, antecedent_texts, consequent_texts = _mutable_snapshot_context(
        params,
        antecedent_texts,
        consequent_texts,
        antecedent_kinds=antecedent_kinds,
        consequent_kinds=consequent_kinds,
    )

    if any(
        _uses_associated_self(clause)
        for clause in (*antecedent_texts, *consequent_texts)
    ):
        return {
            **result_metadata,
            "holds": None,
            "status": "unknown",
            "reason": "unsupported_context",
            "support_issue": "associated_self",
            "phase": "support",
            "engine": "lemma_preflight",
            "antecedent_total": len(antecedent_texts),
            "consequent_total": len(consequent_texts),
        }

    for role, clauses in (
        ("antecedent", antecedent_texts),
        ("consequent", consequent_texts),
    ):
        for index, clause in enumerate(clauses, 1):
            issue = _clause_syntax_issue(clause)
            if issue is not None:
                return {
                    **result_metadata,
                    "holds": None,
                    "status": "unknown",
                    "reason": "malformed_clause",
                    "phase": "binding",
                    "error": f"{role}[{index}]: {issue}",
                    "engine": "lemma_preflight",
                    "antecedent_total": len(antecedent_texts),
                    "consequent_total": len(consequent_texts),
                }

    try:
        reference_text = reference_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {
            **result_metadata,
            "holds": None,
            "status": "unknown",
            "reason": "reference_file_unreadable",
            "phase": "harness_construction",
            "error": f"{type(exc).__name__}: {exc}",
            "engine": "lemma_preflight",
            "antecedent_total": len(antecedent_texts),
            "consequent_total": len(consequent_texts),
        }
    if _find_verus_block_close(reference_text) < 0:
        return {
            **result_metadata,
            "holds": None,
            "status": "unknown",
            "reason": "reference_verus_block_not_found",
            "phase": "harness_construction",
            "engine": "lemma_preflight",
            "antecedent_total": len(antecedent_texts),
            "consequent_total": len(consequent_texts),
        }

    verus_bin = _verus_bin()
    runtime_info = verus_runtime_fingerprint(verus_bin)
    key = _cache_key(
        reference_path=reference_path,
        function_name=function_name,
        check_name=check_name,
        params=params,
        generic_parameters=generic_parameters,
        where_clause=where_clause,
        antecedent=antecedent_texts,
        consequent=consequent_texts,
        generated_path=generated_path,
        support_context=support_context,
        runtime_fingerprint=runtime_info,
        timeout_seconds=timeout_seconds,
        rlimit=rlimit,
    )
    if key in _LEMMA_CACHE:
        cached = dict(_LEMMA_CACHE[key])
        cached["cached"] = True
        return cached

    command_prefix = [verus_bin, "--output-json", "--crate-type=lib", "--verify-root"]
    if rlimit is not None:
        command_prefix.extend(["--rlimit", str(rlimit)])
    if shutil.which(verus_bin) is None and not Path(verus_bin).is_file():
        result = {
            **result_metadata,
            "holds": None,
            "status": "unknown",
            "reason": "verus_binary_not_found",
            "phase": "runtime",
            "engine": engine,
            "verus_binary": verus_bin,
            "verus_runtime": runtime_info,
            "antecedent_total": len(antecedent_texts),
            "consequent_total": len(consequent_texts),
        }
        return result

    digest = key[:12]
    clean_function = re.sub(r"[^A-Za-z0-9_]", "_", function_name or "target")
    clean_check = re.sub(r"[^A-Za-z0-9_]", "_", check_name or "implication")
    lemma_name = f"{lemma_prefix}_{clean_function}_{clean_check}_{digest}"[:180]
    lemma = _make_lemma(
        lemma_name,
        params,
        antecedent_texts,
        consequent_texts,
        generic_parameters=generic_parameters,
        where_clause=where_clause,
    )

    try:
        with tempfile.TemporaryDirectory(prefix="sqm_lemma_") as tmp:
            tmpdir = Path(tmp)
            harness_path = tmpdir / f"{lemma_name}.rs"
            selected_globs: list[str] = []
            frontend_attempts: list[dict[str, Any]] = []

            def build_harness(globs: Sequence[str]) -> str:
                imports = [
                    *support_context.outer_uses,
                    *(f"use {path};" for path in sorted(set(globs))),
                ]
                return _append_lemma_to_reference(
                    reference_path,
                    lemma,
                    support_context.inner_items,
                    imports,
                )

            def run_frontend(harness_text: str) -> tuple[subprocess.CompletedProcess[str], float]:
                harness_path.write_text(harness_text, encoding="utf-8")
                frontend_command = command_prefix + ["--no-verify", str(harness_path)]
                frontend_started = time.monotonic()
                frontend_run = subprocess.run(
                    frontend_command,
                    cwd=str(PROJECT_ROOT),
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                return frontend_run, time.monotonic() - frontend_started

            harness = build_harness(selected_globs)
            try:
                frontend_run, frontend_elapsed = run_frontend(harness)
            except subprocess.TimeoutExpired as exc:
                return {
                    **_lemma_result_metadata(
                        support_context,
                        harness_sha256=hashlib.sha256(harness.encode("utf-8")).hexdigest(),
                        frontend_status="timeout",
                    ),
                    "holds": None,
                    "status": "unknown",
                    "reason": "harness_frontend_timeout",
                    "phase": "harness_frontend",
                    "frontend_error": str(exc),
                    "engine": engine,
                    "lemma": lemma_name,
                    "antecedent_total": len(antecedent_texts),
                    "consequent_total": len(consequent_texts),
                    "verus_runtime": runtime_info,
                }

            remaining = list(support_context.candidate_vstd_globs)
            while frontend_run.returncode != 0 and remaining:
                current_unresolved = _unresolved_symbols(frontend_run)
                if not current_unresolved:
                    break
                current_other_errors = _non_unresolved_error_signatures(frontend_run)
                viable: list[
                    tuple[str, set[str], str, subprocess.CompletedProcess[str], float]
                ] = []
                for candidate in remaining:
                    candidate_harness = build_harness([*selected_globs, candidate])
                    try:
                        candidate_run, candidate_elapsed = run_frontend(candidate_harness)
                    except subprocess.TimeoutExpired:
                        frontend_attempts.append(
                            {"import": candidate, "status": "timeout"}
                        )
                        continue
                    candidate_unresolved = _unresolved_symbols(candidate_run)
                    candidate_other_errors = _non_unresolved_error_signatures(candidate_run)
                    reduced = current_unresolved - candidate_unresolved
                    introduced = candidate_unresolved - current_unresolved
                    new_other_errors = candidate_other_errors - current_other_errors
                    frontend_attempts.append(
                        {
                            "import": candidate,
                            "returncode": candidate_run.returncode,
                            "reduced_unresolved": sorted(reduced),
                            "introduced_unresolved": sorted(introduced),
                            "introduced_errors": sorted(new_other_errors),
                        }
                    )
                    if reduced and not introduced and not new_other_errors:
                        viable.append(
                            (
                                candidate,
                                reduced,
                                candidate_harness,
                                candidate_run,
                                candidate_elapsed,
                            )
                        )

                ambiguous: set[str] = set()
                for index, left in enumerate(viable):
                    for right in viable[index + 1 :]:
                        if left[1] & right[1]:
                            ambiguous.update((left[0], right[0]))
                if ambiguous:
                    return {
                        **_lemma_result_metadata(
                            support_context,
                            harness_sha256=hashlib.sha256(harness.encode("utf-8")).hexdigest(),
                            frontend_status="ambiguous_import",
                            selected_vstd_globs=selected_globs,
                        ),
                        "holds": None,
                        "status": "unknown",
                        "reason": "unsupported_context",
                        "support_issue": "ambiguous_generated_import",
                        "detail": ",".join(sorted(ambiguous)),
                        "phase": "harness_frontend",
                        "frontend_attempts": frontend_attempts,
                        "engine": engine,
                        "lemma": lemma_name,
                        "antecedent_total": len(antecedent_texts),
                        "consequent_total": len(consequent_texts),
                        "verus_runtime": runtime_info,
                    }
                if not viable:
                    break
                chosen = min(viable, key=lambda item: item[0])
                selected_globs.append(chosen[0])
                remaining.remove(chosen[0])
                harness, frontend_run, frontend_elapsed = chosen[2], chosen[3], chosen[4]

            harness_path.write_text(harness, encoding="utf-8")
            harness_sha256 = hashlib.sha256(harness.encode("utf-8")).hexdigest()
            frontend_text, frontend_diagnostics = _verus_diagnostic_text(frontend_run)
            frontend_status = "passed" if frontend_run.returncode == 0 else "failed"
            result_metadata = _lemma_result_metadata(
                support_context,
                harness_sha256=harness_sha256,
                frontend_status=frontend_status,
                selected_vstd_globs=selected_globs,
            )
            result_metadata["support_context_summary"]["selected_glob_imports"] = len(
                selected_globs
            )
            if frontend_run.returncode != 0:
                return {
                    **result_metadata,
                    "holds": None,
                    "status": "unknown",
                    "reason": "harness_frontend_failed",
                    "phase": "harness_frontend",
                    "frontend_returncode": frontend_run.returncode,
                    "frontend_elapsed_seconds": frontend_elapsed,
                    "frontend_diagnostic_excerpt": frontend_text[:2000],
                    "frontend_diagnostics": frontend_diagnostics[:20],
                    "frontend_attempts": frontend_attempts,
                    "engine": engine,
                    "lemma": lemma_name,
                    "antecedent_total": len(antecedent_texts),
                    "consequent_total": len(consequent_texts),
                    "verus_runtime": runtime_info,
                }

            command = command_prefix + ["--verify-function", lemma_name, str(harness_path)]
            started = time.monotonic()
            try:
                run = subprocess.run(
                    command,
                    cwd=str(PROJECT_ROOT),
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                elapsed = time.monotonic() - started
            except subprocess.TimeoutExpired as exc:
                return {
                    **result_metadata,
                    "holds": None,
                    "status": "unknown",
                    "reason": "timeout",
                    "phase": "verification",
                    "error": str(exc),
                    "elapsed_seconds": time.monotonic() - started,
                    "engine": engine,
                    "lemma": lemma_name,
                    "antecedent_total": len(antecedent_texts),
                    "consequent_total": len(consequent_texts),
                    "verus_runtime": runtime_info,
                }
            result = _classify_verus_run(run, elapsed)
    except Exception as exc:
        result = {
            **result_metadata,
            "holds": None,
            "status": "unknown",
            "reason": "harness_construction_failed",
            "phase": "harness_construction",
            "error": f"{type(exc).__name__}: {exc}",
            "engine": engine,
            "antecedent_total": len(antecedent_texts),
            "consequent_total": len(consequent_texts),
            "verus_runtime": runtime_info,
        }
        return result

    result.update(
        {
            **result_metadata,
            "frontend_returncode": frontend_run.returncode,
            "frontend_elapsed_seconds": frontend_elapsed,
            "frontend_attempts": frontend_attempts,
            "engine": engine,
            "lemma": lemma_name,
            "antecedent_total": len(antecedent_texts),
            "consequent_total": len(consequent_texts),
            "verus_runtime": runtime_info,
        }
    )
    if result.get("holds") is True or result.get("holds") is False:
        _LEMMA_CACHE[key] = dict(result)
    return result


def lemma_implication_check(
    *,
    reference_rs_path: str,
    reference_context: Mapping[str, Any],
    generated_context: Mapping[str, Any],
    function_name: str,
    check_name: str,
    antecedent: Sequence[Any],
    consequent: Sequence[Any],
    generated_rs_path: Optional[str] = None,
    timeout_seconds: int = DEFAULT_LEMMA_TIMEOUT_SECONDS,
    rlimit: Optional[float] = None,
) -> dict[str, Any]:
    antecedent_texts, antecedent_sources, antecedent_kinds = _clause_texts_sources_and_kinds(
        antecedent
    )
    consequent_texts, consequent_sources, consequent_kinds = _clause_texts_sources_and_kinds(
        consequent
    )
    original_antecedent = list(antecedent_texts)
    original_consequent = list(consequent_texts)
    reference_path = Path(reference_rs_path)
    generated_path = Path(generated_rs_path) if generated_rs_path else None

    def audited(result: dict[str, Any]) -> dict[str, Any]:
        obligation_context = {
            "generated_path": str(generated_path or ""),
            "reference_path": str(reference_path),
            "function": function_name,
            "check_name": check_name,
            "antecedent": [
                _normalize_lemma_clause(text) for text in original_antecedent
            ],
            "consequent": [
                _normalize_lemma_clause(text) for text in original_consequent
            ],
            "antecedent_kinds": list(antecedent_kinds),
            "consequent_kinds": list(consequent_kinds),
        }
        seed = json.dumps(obligation_context, sort_keys=True, ensure_ascii=False)
        result["obligation_context"] = obligation_context
        result["obligation_sha256"] = hashlib.sha256(
            seed.encode("utf-8", errors="replace")
        ).hexdigest()
        result["run_config"] = {
            "timeout_seconds": timeout_seconds,
            "rlimit": rlimit,
            "lemma_harness_version": LEMMA_HARNESS_VERSION,
        }
        return result

    binding = bind_function_contexts(reference_context, generated_context)
    if binding.issue is not None:
        return audited({
            **binding.issue.to_dict(),
            "phase": "binding",
            "engine": "lemma_preflight",
            "antecedent_total": len(antecedent),
            "consequent_total": len(consequent),
        })
    plan = binding.plan
    assert plan is not None

    trivial = safe_trivial_implication(antecedent, consequent)
    if trivial is not None:
        return audited(trivial)
    support_context = GeneratedSupportContext()
    if (
        generated_path is not None
        and generated_path.exists()
        and reference_path.exists()
    ):
        support_context = collect_generated_support(
            generated_path,
            reference_path,
            generated_texts=[*antecedent_texts, *consequent_texts],
            generated_sources=[*antecedent_sources, *consequent_sources],
            parameter_types=[
                *(item.generated_type for item in plan.inputs),
                *(item.generated_type for item in plan.returns),
            ],
        )
        if support_context.issue is not None:
            return audited({
                **_lemma_result_metadata(support_context),
                **support_context.issue,
                "holds": None,
                "status": "unknown",
                "phase": "support",
                "engine": "lemma_preflight",
                "antecedent_total": len(antecedent_texts),
                "consequent_total": len(consequent_texts),
            })
        if support_context.call_renames or support_context.identifier_renames:
            antecedent_texts = apply_generated_support_renames(
                antecedent_texts,
                antecedent_sources,
                support_context,
            )
            consequent_texts = apply_generated_support_renames(
                consequent_texts,
                consequent_sources,
                support_context,
            )

    all_wrappers: list[str] = []
    antecedent_calls: list[str] = []
    consequent_calls: list[str] = []
    # A self-check can place the same source clause on both sides. Reusing one
    # wrapper keeps the implication syntactically identical without collapsing
    # same-text clauses from different source files.
    wrapper_calls: dict[tuple[Any, ...], str] = {}
    wrapper_seed = hashlib.sha256(
        json.dumps(
            {
                "reference": str(reference_path),
                "generated": str(generated_path or ""),
                "function": function_name,
                "check": check_name,
                "antecedent": antecedent_texts,
                "consequent": consequent_texts,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8", errors="replace")
    ).hexdigest()[:10]

    def add_wrappers(
        texts: Sequence[str],
        sources: Sequence[str],
        kinds: Sequence[str],
        role: str,
    ) -> tuple[Optional[dict[str, Any]], list[str]]:
        calls: list[str] = []
        for index, text in enumerate(texts, 1):
            syntax_issue = _clause_syntax_issue(text)
            if syntax_issue is not None:
                return {
                    "holds": None,
                    "status": "unknown",
                    "reason": "malformed_clause",
                    "phase": "binding",
                    "error": f"{role}[{index}]: {syntax_issue}",
                }, []
            if _uses_associated_self(text):
                return {
                    "holds": None,
                    "status": "unknown",
                    "reason": "unsupported_context",
                    "support_issue": "associated_self",
                    "phase": "support",
                }, []
            source = sources[index - 1] if index <= len(sources) else ""
            kind = kinds[index - 1] if index <= len(kinds) else ""
            wrapper_name = f"__sqm_{role}_{index}_{wrapper_seed}"
            wrapper_result = build_clause_wrapper(
                plan,
                {"text": text, "kind": kind, "source": source},
                wrapper_name=wrapper_name,
            )
            if wrapper_result.issue is not None:
                return {
                    **wrapper_result.issue.to_dict(),
                    "phase": "binding",
                }, []
            wrapper = wrapper_result.wrapper
            assert wrapper is not None
            # Ground/self checks omit a separate generated file, so the
            # generated/reference labels still refer to the same physical
            # source. Normalize that logical label before interning.
            source_identity = "reference" if generated_path is None else wrapper.source
            wrapper_key = (
                source_identity,
                wrapper.kind,
                wrapper.body_text,
                tuple(
                    (item.name, item.rust_type, item.argument, item.role)
                    for item in wrapper.parameters
                ),
                wrapper.generic_parameters,
                wrapper.where_clause,
            )
            call = wrapper_calls.get(wrapper_key)
            if call is None:
                call = wrapper.call
                wrapper_calls[wrapper_key] = call
                all_wrappers.append(render_clause_wrapper(wrapper))
            calls.append(call)
        return None, calls

    issue, antecedent_calls = add_wrappers(
        antecedent_texts,
        antecedent_sources,
        antecedent_kinds,
        "antecedent",
    )
    if issue is None:
        issue, consequent_calls = add_wrappers(
            consequent_texts,
            consequent_sources,
            consequent_kinds,
            "consequent",
        )
    if issue is not None:
        return audited({
            **_lemma_result_metadata(support_context),
            **issue,
            "engine": "lemma_preflight",
            "antecedent_total": len(antecedent_texts),
            "consequent_total": len(consequent_texts),
        })
    support_context.inner_items.extend(all_wrappers)

    result = _verify_lemma(
        reference_path=reference_path,
        function_name=function_name,
        check_name=check_name,
        params=plan.parameter_dicts(),
        generic_parameters=plan.generic_parameters,
        where_clause=plan.where_clause,
        antecedent_texts=antecedent_calls,
        consequent_texts=consequent_calls,
        antecedent_kinds=["requires"] * len(antecedent_calls),
        consequent_kinds=["ensures"] * len(consequent_calls),
        timeout_seconds=timeout_seconds,
        rlimit=rlimit,
        support_context=support_context,
        generated_path=generated_path,
        engine="verus_lemma",
        lemma_prefix="lemma_sqm",
    )
    return audited(result)


def run_probe_lemma(
    *,
    host_rs_path: str,
    probe_name: str,
    source_function_name: Optional[str] = None,
    parameters: Sequence[Mapping[str, str]],
    requires_clauses: Sequence[str],
    ensures_clauses: Sequence[str],
    timeout_seconds: int = DEFAULT_LEMMA_TIMEOUT_SECONDS,
    rlimit: Optional[float] = None,
) -> dict[str, Any]:
    host_path = Path(host_rs_path)
    antecedent_texts = [str(clause).strip() for clause in requires_clauses if str(clause).strip()]
    consequent_texts = [str(clause).strip() for clause in ensures_clauses if str(clause).strip()]
    declaration_context = function_declaration_context_for_path(
        str(host_path),
        source_function_name or probe_name,
    )
    result = _verify_lemma(
        reference_path=host_path,
        function_name=probe_name,
        check_name="proof_probe",
        params=_parameter_records(parameters),
        generic_parameters=declaration_context["generic_parameters"],
        where_clause=declaration_context["where_clause"],
        antecedent_texts=antecedent_texts,
        consequent_texts=consequent_texts,
        antecedent_kinds=["requires"] * len(antecedent_texts),
        consequent_kinds=["ensures"] * len(consequent_texts),
        timeout_seconds=timeout_seconds,
        rlimit=rlimit,
        engine="verus_proof_probe",
        lemma_prefix="probe_sqm",
    )
    result.setdefault("probe_name", probe_name)
    result.setdefault("host_rs_path", str(host_path))
    return result
