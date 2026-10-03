from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .clauses import _scan_top_level_comma_offsets, split_top_level_type_commas
from .functions import (
    constant_blocks_for_path,
    declared_type_names_for_path,
    declared_type_signatures_for_path,
    extract_functions,
    render_spec_fn_block,
    spec_fn_blocks_for_path,
    use_statements_for_path,
)
from .text import token_spans


@dataclass(frozen=True)
class _UseBinding:
    local_name: str
    source_path: str
    statement: str
    is_glob: bool = False


@dataclass
class GeneratedSupportContext:
    outer_uses: list[str] = field(default_factory=list)
    candidate_vstd_globs: list[str] = field(default_factory=list)
    inner_items: list[str] = field(default_factory=list)
    call_renames: dict[str, str] = field(default_factory=dict)
    identifier_renames: dict[str, str] = field(default_factory=dict)
    member_call_names: set[str] = field(default_factory=set)
    member_identifier_names: set[str] = field(default_factory=set)
    issue: Optional[dict[str, str]] = None
    summary: dict[str, int] = field(
        default_factory=lambda: {
            "copied_spec_functions": 0,
            "copied_constants": 0,
            "aliased_imports": 0,
            "copied_glob_imports": 0,
            "candidate_glob_imports": 0,
            "selected_glob_imports": 0,
            "renamed_symbols": 0,
            "reused_reference_items": 0,
        }
    )


def apply_generated_support_renames(
    texts: Sequence[str],
    sources: Sequence[str],
    support: GeneratedSupportContext,
) -> list[str]:
    result: list[str] = []
    for index, original in enumerate(texts):
        text = str(original)
        if index < len(sources) and sources[index] == "generated":
            text = _rewrite_support_symbols(
                text,
                support.call_renames,
                support.identifier_renames,
                member_call_names=support.member_call_names,
                member_identifier_names=support.member_identifier_names,
            )
        result.append(text)
    return result


def _rewrite_support_symbols(
    text: str,
    call_renames: Mapping[str, str],
    identifier_renames: Mapping[str, str],
    *,
    member_call_names: set[str] | frozenset[str] = frozenset(),
    member_identifier_names: set[str] | frozenset[str] = frozenset(),
) -> str:
    """Token-aware rewrite for generated-origin support code and clauses."""
    if not call_renames and not identifier_renames:
        return text
    pieces: list[str] = []
    cursor = 0
    spans = token_spans(text)
    for index, span in enumerate(spans):
        previous = spans[index - 1].text if index > 0 else ""
        member_access = previous in {".", "::"}
        declaration = previous in {"fn", "const"}
        replacement = identifier_renames.get(span.text)
        if replacement is not None and member_access and not (
            declaration or span.text in member_identifier_names
        ):
            replacement = None
        if replacement is None and span.text in call_renames:
            if _call_like_after_identifier(text, span.end) and (
                not member_access
                or declaration
                or span.text in member_call_names
            ):
                replacement = call_renames[span.text]
        if replacement is None:
            continue
        pieces.append(text[cursor:span.start])
        pieces.append(replacement)
        cursor = span.end
    if not pieces:
        return text
    pieces.append(text[cursor:])
    return "".join(pieces)


def _call_like_after_identifier(text: str, offset: int) -> bool:
    suffix = text[offset:]
    spans = token_spans(suffix)
    if not spans:
        return False
    if spans[0].text in {"(", "!"}:
        return True

    if len(spans) >= 2 and spans[0].text == "::" and spans[1].text == "<":
        open_index = 1
    elif spans[0].text == "<" and spans[0].start == 0:
        # Rust permits ``f<T>(x)`` only when the identifier and ``<`` are
        # adjacent.  This is also what separates a call from ``x > y``.
        open_index = 0
    else:
        return False

    depth = 0
    close_index: Optional[int] = None
    for index in range(open_index, len(spans)):
        token = spans[index].text
        if token == "<":
            depth += 1
        elif token == ">":
            depth -= 1
            if depth == 0:
                close_index = index
                break
        elif token in {"<==>", "==>", "<=", ">="}:
            return False
    return (
        close_index is not None
        and close_index + 1 < len(spans)
        and spans[close_index + 1].text == "("
    )


def _normalize_spec_body(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _spec_block_key(block: Mapping[str, Any]) -> tuple[str, str]:
    owner = _normalize_spec_body(str(block.get("owner") or ""))
    return owner, str(block.get("name") or "")


_SPEC_FN_NAME_RE = re.compile(r"\bfn\s+[A-Za-z_][A-Za-z0-9_]*\s*")


def _spec_fn_binder_names(text: str) -> Optional[tuple[list[str], list[str]]]:
    """Generic and value parameter names declared by one ``fn`` item."""
    match = _SPEC_FN_NAME_RE.search(text)
    if match is None:
        return None
    offset = match.end()
    generics: list[str] = []
    if offset < len(text) and text[offset] == "<":
        depth = 0
        close: Optional[int] = None
        for index in range(offset, len(text)):
            if text[index] == "<":
                depth += 1
            elif text[index] == ">":
                depth -= 1
                if depth == 0:
                    close = index
                    break
        if close is None:
            return None
        for part in split_top_level_type_commas(text[offset + 1 : close]):
            if part.strip().startswith("'"):
                # Lifetimes stay untouched; differing names simply fail the
                # comparison instead of risking an unsound token rewrite.
                continue
            name_match = re.match(r"\s*(?:const\s+)?([A-Za-z_][A-Za-z0-9_]*)", part)
            if name_match:
                generics.append(name_match.group(1))
        offset = close + 1
    paren = text.find("(", offset)
    if paren < 0:
        return None
    depth = 0
    close_paren: Optional[int] = None
    for index in range(paren, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                close_paren = index
                break
    if close_paren is None:
        return None
    params: list[str] = []
    for part in split_top_level_type_commas(text[paren + 1 : close_paren]):
        name_match = re.match(r"\s*(?:mut\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*:", part)
        if name_match:
            params.append(name_match.group(1))
    return generics, params


def _alpha_normalized_spec_fn(text: str, reserved: frozenset[str]) -> Optional[str]:
    """Normalize one spec fn with positional binder names, or None when unsafe.

    Two helper definitions that differ only in parameter/generic naming are the
    same function; copying such a helper under a fresh name would force the
    lemma to prove extensional equality of identical recursive functions,
    which Verus cannot do without induction.  Renames skip member-access
    positions and bail out when a binder name shadows a declared type or
    constant, where a plain token rewrite could change match-pattern meaning.
    """
    binder_names = _spec_fn_binder_names(text)
    if binder_names is None:
        return None
    generics, params = binder_names
    names = [name for name in (*generics, *params) if name != "_"]
    if not names or len(set(names)) != len(names):
        return None
    if any(name in reserved for name in names):
        return None
    mapping: dict[str, str] = {}
    for index, name in enumerate(generics):
        mapping[name] = f"__sqm_alpha_g{index}"
    for index, name in enumerate(params):
        mapping[name] = f"__sqm_alpha_p{index}"
    mapping.pop("_", None)
    spans = token_spans(text)
    pieces: list[str] = []
    cursor = 0
    for index, span in enumerate(spans):
        replacement = mapping.get(span.text)
        if replacement is None:
            continue
        if index > 0 and spans[index - 1].text in {".", "::"}:
            continue
        pieces.append(text[cursor : span.start])
        pieces.append(replacement)
        cursor = span.end
    pieces.append(text[cursor:])
    return _normalize_spec_body("".join(pieces))


def _private_support_item(text: str) -> str:
    """Limit copied support declarations to the synthetic lemma's module."""
    private = re.sub(
        r"\A((?:\s*#\s*\[[^\]]*\]\s*)*)pub(?:\s*\([^)]*\))?\s+",
        r"\1",
        str(text).strip(),
        count=1,
        flags=re.DOTALL,
    )
    # Verus requires every ``open`` function symbol itself to be externally
    # visible. Once copied, plain ``spec fn`` stays unfoldable by the lemma.
    return re.sub(
        r"\A((?:\s*#\s*\[[^\]]*\]\s*)*)open\s+(?=spec\b)",
        r"\1",
        private,
        count=1,
        flags=re.DOTALL,
    )


def _parse_use_statement(statement: str) -> tuple[list[_UseBinding], Optional[str]]:
    stripped = statement.strip()
    if not stripped.startswith("use ") or not stripped.endswith(";"):
        return [], "unsupported_import_form"
    body = stripped[4:-1].strip()

    def one_binding(item: str, prefix: str = "") -> Optional[_UseBinding]:
        item = item.strip()
        alias_match = re.fullmatch(
            r"(?P<path>.+?)\s+as\s+(?P<alias>[A-Za-z_][A-Za-z0-9_]*)",
            item,
        )
        source = alias_match.group("path").strip() if alias_match else item
        alias = alias_match.group("alias") if alias_match else ""
        if prefix:
            source = prefix if source == "self" else f"{prefix}::{source}"
        if source.endswith("::*"):
            return _UseBinding("*", source, stripped, True)
        if not re.fullmatch(
            r"(?:(?:crate|self|super|[A-Za-z_][A-Za-z0-9_]*)::)*"
            r"[A-Za-z_][A-Za-z0-9_]*",
            source,
        ):
            return None
        local = alias or source.rsplit("::", 1)[-1]
        return _UseBinding(local, source, stripped, False)

    if "{" in body or "}" in body:
        group = re.fullmatch(r"(?P<prefix>.+?)::\{(?P<items>.*)\}", body)
        if group is None or "{" in group.group("items") or "}" in group.group("items"):
            return [], "unsupported_import_group"
        bindings: list[_UseBinding] = []
        for part in split_top_level_type_commas(group.group("items")):
            binding = one_binding(part, group.group("prefix").strip())
            if binding is None:
                return [], "unsupported_import_group"
            bindings.append(binding)
        return bindings, None

    binding = one_binding(body)
    return ([binding], None) if binding is not None else ([], "unsupported_import_form")


def _use_bindings_for_path(path: Path) -> tuple[list[_UseBinding], Optional[str]]:
    bindings: list[_UseBinding] = []
    for statement in use_statements_for_path(str(path)):
        parsed, issue = _parse_use_statement(statement)
        if issue is not None:
            return [], issue
        bindings.extend(parsed)
    return bindings, None


def _text_has_identifier(text: str, name: str) -> bool:
    return any(span.text == name for span in token_spans(text))


def _text_has_unqualified_identifier(text: str, name: str) -> bool:
    spans = token_spans(text)
    return any(
        span.text == name
        and (index == 0 or spans[index - 1].text not in {".", "::"})
        for index, span in enumerate(spans)
    )


def _text_calls_name(text: str, name: str, *, allow_member: bool = True) -> bool:
    spans = token_spans(text)
    for index, span in enumerate(spans):
        if span.text != name or not _call_like_after_identifier(text, span.end):
            continue
        if allow_member or index == 0 or spans[index - 1].text not in {".", "::"}:
            return True
    return False


def _import_is_referenced(texts: Sequence[str], binding: _UseBinding) -> bool:
    if binding.is_glob:
        return bool(texts)
    if binding.local_name[:1].isupper():
        return any(
            _text_has_unqualified_identifier(text, binding.local_name)
            for text in texts
        )
    return any(
        _text_calls_name(text, binding.local_name, allow_member=False)
        for text in texts
    )


def _symbol_has_local_binding(text: str, name: str) -> bool:
    escaped = re.escape(name)
    common_binder = re.search(
        rf"(?:\b(?:let|ghost|tracked|mut)\s+|[,(|]\s*){escaped}\s*(?=:|=|,|\|)",
        text,
    )
    generic_binder = re.search(rf"<\s*(?:const\s+)?{escaped}\s*(?=[:,>])", text)
    destructuring_binder = re.search(
        rf"\b(?:let|ghost|tracked)\b[^;=]*\b{escaped}\b[^;=]*(?==)",
        text,
    )
    return any(
        match is not None
        for match in (common_binder, generic_binder, destructuring_binder)
    )


def _fresh_support_name(
    generated_path: Path,
    owner: str,
    name: str,
    occupied: set[str],
) -> str:
    seed = f"{generated_path.resolve()}\0{owner}\0{name}"
    digest = hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()
    clean_name = re.sub(r"[^A-Za-z0-9_]", "_", name) or "symbol"
    for width in (10, 16, 24, 32, 64):
        candidate = f"__sqm_gen_{clean_name}_{digest[:width]}"
        if candidate not in occupied:
            return candidate
    suffix = 2
    while f"__sqm_gen_{clean_name}_{digest}_{suffix}" in occupied:
        suffix += 1
    return f"__sqm_gen_{clean_name}_{digest}_{suffix}"


def _register_rename(
    mapping: dict[str, str],
    other_mapping: Mapping[str, str],
    old_name: str,
    new_name: str,
) -> Optional[dict[str, str]]:
    existing = mapping.get(old_name) or other_mapping.get(old_name)
    if existing is not None and existing != new_name:
        return {
            "reason": "unsupported_context",
            "support_issue": "ambiguous_symbol_rewrite",
            "detail": old_name,
        }
    mapping[old_name] = new_name
    return None


def _support_issue(code: str, detail: str = "") -> dict[str, str]:
    issue = {
        "reason": "unsupported_context",
        "support_issue": code,
    }
    if detail:
        issue["detail"] = detail
    return issue


def collect_generated_support(
    generated_rs_path: Path,
    reference_rs_path: Path,
    *,
    generated_texts: Sequence[str],
    generated_sources: Sequence[str],
    parameter_types: Sequence[str] = (),
) -> GeneratedSupportContext:
    support = GeneratedSupportContext()
    gen_blocks = spec_fn_blocks_for_path(str(generated_rs_path))
    ref_blocks = spec_fn_blocks_for_path(str(reference_rs_path))
    gen_constants = constant_blocks_for_path(str(generated_rs_path))
    ref_constants = constant_blocks_for_path(str(reference_rs_path))

    roots = [
        text
        for text, source in zip(generated_texts, generated_sources)
        if source == "generated"
    ]
    signature_type_texts = [
        str(type_text) for type_text in parameter_types if str(type_text).strip()
    ]

    gen_functions_by_name: dict[str, list[dict[str, Any]]] = {}
    for block in gen_blocks:
        gen_functions_by_name.setdefault(str(block["name"]), []).append(block)
    gen_constants_by_name: dict[str, list[dict[str, Any]]] = {}
    for block in gen_constants:
        gen_constants_by_name.setdefault(str(block["name"]), []).append(block)

    selected_functions: list[dict[str, Any]] = []
    selected_constants: list[dict[str, Any]] = []
    selected_function_ids: set[tuple[str, str, str]] = set()
    selected_constant_ids: set[tuple[str, str, str]] = set()
    dependency_texts = list(roots)
    changed = True
    while changed:
        changed = False
        for name, blocks in gen_functions_by_name.items():
            if not any(_text_calls_name(text, name) for text in dependency_texts):
                continue
            for block in blocks:
                block_id = (*_spec_block_key(block), str(block.get("text") or ""))
                if block_id in selected_function_ids:
                    continue
                selected_function_ids.add(block_id)
                selected_functions.append(block)
                dependency_texts.append(str(block.get("text") or ""))
                dependency_texts.append(str(block.get("owner") or ""))
                changed = True
        for name, blocks in gen_constants_by_name.items():
            if not any(_text_has_identifier(text, name) for text in dependency_texts):
                continue
            for block in blocks:
                block_id = (*_spec_block_key(block), str(block.get("text") or ""))
                if block_id in selected_constant_ids:
                    continue
                selected_constant_ids.add(block_id)
                selected_constants.append(block)
                dependency_texts.append(str(block.get("text") or ""))
                dependency_texts.append(str(block.get("owner") or ""))
                changed = True

    gen_declared_types = declared_type_names_for_path(str(generated_rs_path))
    ref_declared_types = declared_type_names_for_path(str(reference_rs_path))
    gen_type_signatures = declared_type_signatures_for_path(str(generated_rs_path))
    ref_type_signatures = declared_type_signatures_for_path(str(reference_rs_path))
    missing_types = gen_declared_types - ref_declared_types
    for block in [*selected_functions, *selected_constants]:
        owner = str(block.get("owner") or "")
        missing_owner = next(
            (name for name in missing_types if _text_has_identifier(owner, name)),
            None,
        )
        if missing_owner is not None:
            support.issue = _support_issue("missing_reference_owner", missing_owner)
            return support
    context_texts = [*roots, *dependency_texts]
    referenced_shared_types = {
        name
        for name in gen_declared_types & ref_declared_types
        if any(
            _text_has_identifier(text, name)
            for text in [*context_texts, *signature_type_texts]
        )
    }
    incompatible_type = next(
        (
            name
            for name in sorted(referenced_shared_types)
            if not gen_type_signatures.get(name)
            or not ref_type_signatures.get(name)
            or gen_type_signatures[name] != ref_type_signatures[name]
        ),
        None,
    )
    if incompatible_type is not None:
        support.issue = _support_issue("incompatible_type_declaration", incompatible_type)
        return support
    missing_type = next(
        (
            name
            for name in sorted(missing_types)
            if any(
                _text_has_identifier(text, name)
                for text in [*context_texts, *signature_type_texts]
            )
        ),
        None,
    )
    if missing_type is not None:
        support.issue = _support_issue("requires_type_declaration", missing_type)
        return support

    selected_spec_names = {str(block["name"]) for block in selected_functions}
    exec_names = {
        function.name
        for function in extract_functions(str(generated_rs_path))
        if function.mode == "exec" and function.name != "main"
    }
    unsupported_exec = next(
        (
            name
            for name in sorted(exec_names - selected_spec_names)
            if any(
                _text_calls_name(text, name, allow_member=False)
                for text in context_texts
            )
        ),
        None,
    )
    if unsupported_exec is not None:
        support.issue = _support_issue("requires_exec_declaration", unsupported_exec)
        return support

    ref_spec_by_key = {
        _spec_block_key(block): _normalize_spec_body(
            _private_support_item(str(block.get("text") or ""))
        )
        for block in ref_blocks
    }
    ref_spec_text_by_key: dict[tuple[str, str], str] = {}
    ref_spec_key_counts: dict[tuple[str, str], int] = {}
    for block in ref_blocks:
        key = _spec_block_key(block)
        ref_spec_key_counts[key] = ref_spec_key_counts.get(key, 0) + 1
        ref_spec_text_by_key[key] = _private_support_item(str(block.get("text") or ""))
    gen_spec_key_counts: dict[tuple[str, str], int] = {}
    for block in gen_blocks:
        key = _spec_block_key(block)
        gen_spec_key_counts[key] = gen_spec_key_counts.get(key, 0) + 1
    alpha_reserved = frozenset(
        (
            *gen_declared_types,
            *ref_declared_types,
            *(str(block["name"]) for block in gen_constants),
            *(str(block["name"]) for block in ref_constants),
        )
    )
    ref_const_by_key = {
        _spec_block_key(block): _normalize_spec_body(
            _private_support_item(str(block.get("text") or ""))
        )
        for block in ref_constants
    }
    copied_functions: list[dict[str, Any]] = []
    copied_constants: list[dict[str, Any]] = []
    reused_function_names: set[str] = set()
    reused_constant_names: set[str] = set()
    for block in selected_functions:
        key = _spec_block_key(block)
        private_text = _private_support_item(str(block.get("text") or ""))
        reused = ref_spec_by_key.get(key) == _normalize_spec_body(private_text)
        if (
            not reused
            and gen_spec_key_counts.get(key, 0) == 1
            and ref_spec_key_counts.get(key, 0) == 1
        ):
            generated_alpha = _alpha_normalized_spec_fn(private_text, alpha_reserved)
            reused = generated_alpha is not None and generated_alpha == (
                _alpha_normalized_spec_fn(ref_spec_text_by_key[key], alpha_reserved)
            )
        if reused:
            support.summary["reused_reference_items"] += 1
            reused_function_names.add(str(block["name"]))
        else:
            copied_functions.append(block)
    for block in selected_constants:
        normalized = _normalize_spec_body(
            _private_support_item(str(block.get("text") or ""))
        )
        if ref_const_by_key.get(_spec_block_key(block)) == normalized:
            support.summary["reused_reference_items"] += 1
            reused_constant_names.add(str(block["name"]))
        else:
            copied_constants.append(block)

    gen_uses, gen_use_issue = _use_bindings_for_path(generated_rs_path)
    ref_uses, ref_use_issue = _use_bindings_for_path(reference_rs_path)
    if gen_use_issue is not None:
        support.issue = _support_issue(gen_use_issue)
        return support
    if ref_use_issue is not None:
        # The reference already compiled in its original lexical context; an
        # unsupported form only prevents alias reuse for generated symbols.
        ref_uses = []

    ref_paths = {(binding.local_name, binding.source_path) for binding in ref_uses}
    for binding in gen_uses:
        if not _import_is_referenced(context_texts, binding):
            continue
        if binding.is_glob:
            if any(item.source_path == binding.source_path for item in ref_uses):
                continue
            if binding.source_path.startswith("vstd::"):
                support.candidate_vstd_globs.append(binding.source_path)
                continue
            support.issue = _support_issue("unsafe_generated_import", binding.source_path)
            return support
        if (binding.local_name, binding.source_path) in ref_paths:
            continue

    occupied = {function.name for function in extract_functions(str(reference_rs_path))}
    occupied.update(str(block["name"]) for block in ref_constants)
    occupied.update(ref_declared_types)
    occupied.update(
        binding.local_name for binding in ref_uses if not binding.is_glob
    )

    copied_functions_by_name: dict[str, list[dict[str, Any]]] = {}
    for block in copied_functions:
        copied_functions_by_name.setdefault(str(block["name"]), []).append(block)
    for name, blocks in copied_functions_by_name.items():
        if name not in occupied:
            occupied.add(name)
            continue
        if name in reused_function_names:
            support.issue = _support_issue("ambiguous_symbol_rewrite", name)
            return support
        owner_kinds = {bool(str(block.get("owner") or "").strip()) for block in blocks}
        if len(owner_kinds) > 1:
            support.issue = _support_issue("ambiguous_symbol_rewrite", name)
            return support
        new_name = _fresh_support_name(
            generated_rs_path,
            "|".join(sorted(str(block.get("owner") or "") for block in blocks)),
            name,
            occupied,
        )
        issue = _register_rename(
            support.call_renames,
            support.identifier_renames,
            name,
            new_name,
        )
        if issue is not None:
            support.issue = issue
            return support
        if owner_kinds == {True}:
            support.member_call_names.add(name)
        occupied.add(new_name)

    copied_constants_by_name: dict[str, list[dict[str, Any]]] = {}
    for block in copied_constants:
        copied_constants_by_name.setdefault(str(block["name"]), []).append(block)
    for name, blocks in copied_constants_by_name.items():
        if name not in occupied:
            occupied.add(name)
            continue
        if name in reused_constant_names:
            support.issue = _support_issue("ambiguous_symbol_rewrite", name)
            return support
        owner_kinds = {bool(str(block.get("owner") or "").strip()) for block in blocks}
        if len(owner_kinds) > 1:
            support.issue = _support_issue("ambiguous_symbol_rewrite", name)
            return support
        new_name = _fresh_support_name(
            generated_rs_path,
            "|".join(sorted(str(block.get("owner") or "") for block in blocks)),
            name,
            occupied,
        )
        issue = _register_rename(
            support.identifier_renames,
            support.call_renames,
            name,
            new_name,
        )
        if issue is not None:
            support.issue = issue
            return support
        if owner_kinds == {True}:
            support.member_identifier_names.add(name)
        occupied.add(new_name)

    for binding in gen_uses:
        if binding.is_glob or not _import_is_referenced(context_texts, binding):
            continue
        if (binding.local_name, binding.source_path) in ref_paths:
            continue
        if binding.source_path.startswith(("self::", "super::", "crate::")):
            support.issue = _support_issue("unsafe_generated_import", binding.source_path)
            return support
        if any(
            _symbol_has_local_binding(text, binding.local_name)
            for text in context_texts
        ):
            support.issue = _support_issue(
                "ambiguous_symbol_rewrite", binding.local_name
            )
            return support
        new_name = _fresh_support_name(
            generated_rs_path,
            binding.source_path,
            binding.local_name,
            occupied,
        )
        issue = _register_rename(
            support.identifier_renames,
            support.call_renames,
            binding.local_name,
            new_name,
        )
        if issue is not None:
            support.issue = issue
            return support
        occupied.add(new_name)
        support.outer_uses.append(f"use {binding.source_path} as {new_name};")

    all_renamed_names = set(support.call_renames) | set(support.identifier_renames)
    ambiguous_local = next(
        (
            name
            for name in sorted(all_renamed_names)
            if any(_symbol_has_local_binding(text, name) for text in context_texts)
        ),
        None,
    )
    if ambiguous_local is not None:
        support.issue = _support_issue("ambiguous_symbol_rewrite", ambiguous_local)
        return support
    for text, source in zip(generated_texts, generated_sources):
        if source:
            continue
        ambiguous = next(
            (name for name in all_renamed_names if _text_has_identifier(text, name)),
            None,
        )
        if ambiguous is not None:
            support.issue = _support_issue("ambiguous_clause_origin", ambiguous)
            return support

    for block in copied_functions:
        text = _private_support_item(str(block.get("text") or ""))
        text = _rewrite_support_symbols(
            text,
            support.call_renames,
            support.identifier_renames,
            member_call_names=support.member_call_names,
            member_identifier_names=support.member_identifier_names,
        )
        support.inner_items.append(render_spec_fn_block(block, text))
    for block in copied_constants:
        text = _private_support_item(str(block.get("text") or ""))
        text = _rewrite_support_symbols(
            text,
            support.call_renames,
            support.identifier_renames,
            member_call_names=support.member_call_names,
            member_identifier_names=support.member_identifier_names,
        )
        support.inner_items.append(render_spec_fn_block(block, text))

    support.summary.update(
        {
            "copied_spec_functions": len(copied_functions),
            "copied_constants": len(copied_constants),
            "aliased_imports": len(support.outer_uses),
            "copied_glob_imports": 0,
            "candidate_glob_imports": len(set(support.candidate_vstd_globs)),
            "renamed_symbols": len(all_renamed_names),
        }
    )
    support.outer_uses[:] = sorted(set(support.outer_uses))
    support.candidate_vstd_globs[:] = sorted(set(support.candidate_vstd_globs))
    return support


def normalize_lemma_clause(text: str) -> str:
    """Remove one top-level separator owned by the source contract list."""
    normalized = str(text).strip()
    if not normalized.endswith(","):
        return normalized
    comma_offsets, _ = _scan_top_level_comma_offsets(normalized)
    if comma_offsets and comma_offsets[-1] == len(normalized) - 1:
        return normalized[:-1].rstrip()
    return normalized


def find_verus_block_close(text: str) -> int:
    """Return the closing brace of the first top-level ``verus!`` block."""
    idx = text.find("verus!")
    if idx < 0:
        return -1
    brace = text.find("{", idx)
    if brace < 0:
        return -1
    depth = 0
    i = brace
    in_line_comment = False
    in_block_comment = False
    in_string = False
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
        elif in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 1
        elif in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == "/" and nxt == "/":
            in_line_comment = True
            i += 1
        elif ch == "/" and nxt == "*":
            in_block_comment = True
            i += 1
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def append_lemma_to_reference(
    reference_path: Path,
    lemma: str,
    inner_items: Sequence[str] = (),
    outer_uses: Sequence[str] = (),
) -> str:
    text = reference_path.read_text(encoding="utf-8", errors="replace")
    verus_offset = text.find("verus!")
    if verus_offset < 0:
        raise ValueError("could_not_find_verus_block")
    if outer_uses:
        injected_uses = "\n".join(
            sorted(set(str(item).strip() for item in outer_uses if str(item).strip()))
        )
        text = text[:verus_offset] + injected_uses + "\n\n" + text[verus_offset:]
    insert_at = find_verus_block_close(text)
    if insert_at < 0:
        raise ValueError("could_not_find_verus_block_end")
    injected = "\n\n".join([*inner_items, lemma])
    return text[:insert_at] + "\n\n" + injected + "\n" + text[insert_at:]


def make_lemma(
    name: str,
    params: Sequence[Mapping[str, str]],
    antecedent: Sequence[str],
    consequent: Sequence[str],
    *,
    generic_parameters: str = "",
    where_clause: str = "",
) -> str:
    signature = ", ".join(f"{item['name']}: {item['type']}" for item in params)
    lines = [f"proof fn {name}{generic_parameters.strip()}({signature})"]
    if where_clause.strip():
        lines.extend(f"    {line.strip()}" for line in where_clause.strip().splitlines())
    if antecedent:
        lines.append("    requires")
        lines.extend(f"        {normalize_lemma_clause(clause)}," for clause in antecedent)
    if consequent:
        lines.append("    ensures")
        lines.extend(f"        {normalize_lemma_clause(clause)}," for clause in consequent)
    lines.extend(["{", "}"])
    return "\n".join(lines)


__all__ = [
    "GeneratedSupportContext",
    "apply_generated_support_renames",
    "append_lemma_to_reference",
    "collect_generated_support",
    "find_verus_block_close",
    "make_lemma",
    "normalize_lemma_clause",
]
