from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

from metrics_rebuild.share.clauses import (
    CLAUSE_BOUNDARY_KINDS,
    EXPRESSION_CLAUSE_KINDS,
    FN_RE,
    _consume_lifetime_or_char,
    _is_probable_clause_body_open,
    _token_depth_update,
    _top_level,
    extract_clauses_from_text,
    find_matching_brace,
    find_matching_paren,
    split_top_level_commas,
    split_top_level_type_commas,
)
from metrics_rebuild.share.io_harness import extract_function, preferred_io_function_name
from metrics_rebuild.share.target_catalog import target_catalog_entry_for_path
from metrics_rebuild.share.text import read_text, strip_comments, token_spans
from metrics_rebuild.share.type_defs import (
    parse_type_definitions,
    render_type_definitions,
    resolve_alias_text,
)


@dataclass(frozen=True)
class FunctionInfo:
    name: str
    header: str
    return_vars: tuple[str, ...]
    has_contract: bool
    mode: str = "exec"
    modifiers: tuple[str, ...] = tuple()
    visibility: Optional[str] = None
    has_body: bool = True
    has_recommends: bool = False
    has_signature_spec: bool = False
    is_lemma_like: bool = False


def _normalized_signature_type(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def target_descriptor_for_path(path: str, function_name: str) -> Optional[dict]:
    """Return the exact executable target signature and strength context."""
    code = read_text(path)
    function = extract_function(code, function_name)
    if function is None or function.name != function_name:
        return None
    context = next(
        (
            item for item in strength_contexts_for_path(path)
            if item.get("function") == function_name
        ),
        None,
    )
    if context is None:
        return None
    returns = [
        {"name": str(item.get("name") or "ret"), "type": str(item.get("type") or "")}
        for item in context.get("returns") or []
        if isinstance(item, dict)
    ]
    return {
        "function": function.name,
        "parameters": [
            {"name": param.name, "type": param.rust_type}
            for param in function.params
        ],
        "returns": returns,
        "return_type": function.return_type or "()",
        "context": context,
    }


def target_signatures_match(reference: dict, generated: dict) -> bool:
    """Strict task signature match used by all target-aware metrics."""
    if reference.get("function") != generated.get("function"):
        return False
    for key in ("parameters", "returns"):
        ref_items = reference.get(key) or []
        gen_items = generated.get(key) or []
        if len(ref_items) != len(gen_items):
            return False
        for ref_item, gen_item in zip(ref_items, gen_items):
            if ref_item.get("name") != gen_item.get("name"):
                return False
            if _normalized_signature_type(ref_item.get("type")) != _normalized_signature_type(gen_item.get("type")):
                return False
    return _normalized_signature_type(reference.get("return_type")) == _normalized_signature_type(
        generated.get("return_type")
    )


def authoritative_target_for_path(
    path: str,
    *,
    use_target_catalog: bool = True,
) -> Optional[dict]:
    """Resolve the benchmark's ground-truth executable target.

    The canonical target catalog wins for known benchmark references. For
    uncatalogued inputs, an explicit ``<vc-spec>`` marker wins, followed by the
    last contracted executable function and then the last executable function.
    ``fn main`` is never a candidate, even when it is the last function in the
    file. Persisted IO metadata must never override this decision.
    """
    code = read_text(path)
    contexts = [
        item for item in strength_contexts_for_path(path)
        if item.get("function") and item.get("function") != "main"
    ]
    marked = preferred_io_function_name(code)
    if marked == "main":
        marked = None
    selected_name: Optional[str] = None
    selection: Optional[str] = None
    catalog_entry = (
        target_catalog_entry_for_path(path) if use_target_catalog else None
    )
    if catalog_entry is not None:
        catalog_name = str(catalog_entry.get("target_function") or "")
        if not any(item.get("function") == catalog_name for item in contexts):
            raise ValueError(
                f"Catalog target {catalog_name!r} is not executable in {path}"
            )
        selected_name = catalog_name
        selection = str(catalog_entry.get("selection") or "target_catalog")
    if marked and any(item.get("function") == marked for item in contexts):
        if selected_name is None:
            selected_name = marked
            selection = "vc_spec_marker"
    if selected_name is None:
        contracted = [item for item in contexts if item.get("has_contract")]
        selected = (contracted or contexts or [None])[-1]
        if selected is not None:
            selected_name = str(selected.get("function") or "")
            selection = "last_contracted_exec" if contracted else "last_exec"
    if not selected_name or selected_name == "main":
        return None
    descriptor = target_descriptor_for_path(path, selected_name)
    if descriptor is None:
        return None
    return {
        **descriptor,
        "selection": selection,
        "marked_function": marked,
        "target_catalog": (
            {
                "schema_version": catalog_entry.get("schema_version"),
                "reference_sha256": catalog_entry.get("reference_sha256"),
                "review_status": catalog_entry.get("review_status"),
            }
            if catalog_entry is not None
            else None
        ),
    }


def resolve_pair_target(
    generated_rs_path: str,
    reference_rs_path: str,
    *,
    preferred_name: Optional[str] = None,
) -> dict:
    """Resolve one authoritative task function shared by gen/ref metrics.

    Reference source selection is authoritative: the ``<vc-spec>`` marker wins,
    followed by the last contracted executable function and then the last
    executable function other than ``fn main``. ``preferred_name`` is retained only as stale-metadata diagnostics;
    it can never override the source. The generated file only needs an executable
    function with the selected name. Marker/signature differences are retained as
    diagnostics and left to each metric's actual comparison.
    """
    try:
        reference_code = read_text(reference_rs_path)
        generated_code = read_text(generated_rs_path)
    except OSError as exc:
        return {"status": "target_mismatch", "reason": "target_file_unavailable", "error": str(exc)}

    authoritative = authoritative_target_for_path(reference_rs_path)
    reference_marked = preferred_io_function_name(reference_code)
    if authoritative is None:
        return {"status": "target_mismatch", "reason": "reference_target_not_found"}
    selected_name = str(authoritative["function"])
    selection = str(authoritative.get("selection") or "")
    reference = {
        key: value for key, value in authoritative.items()
        if key not in {"selection", "marked_function"}
    }

    generated_marked = preferred_io_function_name(generated_code)
    generated = target_descriptor_for_path(generated_rs_path, selected_name)
    if generated is None:
        return {
            "status": "target_mismatch",
            "reason": "generated_target_not_found",
            "function": selected_name,
            "selection": selection,
            "reference": reference,
        }
    signature_match = target_signatures_match(reference, generated)
    diagnostics: list[str] = []
    if preferred_name and preferred_name != selected_name:
        diagnostics.append("preferred_metadata_points_to_different_function")
    if generated_marked and generated_marked != selected_name:
        diagnostics.append("generated_marker_points_to_different_function")
    if not signature_match:
        diagnostics.append("target_signature_differs")
    return {
        "status": "ok",
        "reason": None,
        "function": selected_name,
        "selection": selection,
        "reference_marked_function": reference_marked,
        "generated_marked_function": generated_marked,
        "signature_match": signature_match,
        "diagnostics": diagnostics,
        "reference": reference,
        "generated": generated,
    }


def target_mismatch_metric(alignment: dict, *, metric_kind: Optional[str] = None) -> dict:
    """Standard zero result for a gen/ref task-target mismatch."""
    payload = {
        "status": "target_mismatch",
        "score": 0.0,
        "reason": alignment.get("reason") or "target_mismatch",
        "target_alignment": alignment,
    }
    if metric_kind:
        payload["metric_kind"] = metric_kind
    return {
        **payload,
        "generated": dict(payload),
        "ground": {
            "status": "ok" if alignment.get("reference") else "target_mismatch",
            "score": 1.0 if alignment.get("reference") else None,
            "target": alignment.get("reference"),
        },
        "delta": -1.0 if alignment.get("reference") else None,
    }


def _latest_top_level_clause_expression_start(
    text: str,
    start: int,
    end: int,
) -> Optional[int]:
    depths = {"paren": 0, "bracket": 0, "brace": 0}
    latest: Optional[int] = None
    spans = token_spans(text[start:end])
    for index, span in enumerate(spans):
        if (
            _top_level(depths)
            and span.text in EXPRESSION_CLAUSE_KINDS
            and not (index > 0 and spans[index - 1].text in {".", "::"})
        ):
            latest = start + span.end
        _token_depth_update(span.text, depths)
    return latest


def _find_top_level_signature_end(text: str, start: int) -> Optional[tuple[int, bool]]:
    paren = bracket = brace = 0
    state = "normal"
    i = start
    while i < len(text):
        ch = text[i]
        if state == "normal":
            if ch == '"':
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
                # Braced expressions inside a contract (notably `match` arms)
                # are not the executable function body.  Only a brace seen at
                # the outermost brace depth can end the signature.
                if brace == 0:
                    clause_start = _latest_top_level_clause_expression_start(
                        text,
                        start,
                        i,
                    )
                    if clause_start is None or _is_probable_clause_body_open(
                        text,
                        clause_start,
                        i,
                    ):
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


def _normalize_prefix_modifiers(prefix: str) -> tuple[Optional[str], tuple[str, ...]]:
    modifiers: list[str] = []
    visibility: Optional[str] = None

    pub_match = re.search(r"\bpub(?:\s*\([^)]*\))?", prefix)
    if pub_match:
        visibility = pub_match.group(0).replace(" ", "")
        modifiers.append(visibility)

    for keyword in ("open", "closed", "uninterp", "broadcast", "unsafe"):
        if re.search(rf"\b{keyword}\b", prefix):
            modifiers.append(keyword)

    if re.search(r"\bspec\s*\(\s*checked\s*\)", prefix):
        modifiers.append("spec(checked)")
    elif re.search(r"\bspec\b", prefix):
        modifiers.append("spec")

    if re.search(r"\bproof\b", prefix):
        modifiers.append("proof")
    if re.search(r"\bexec\b", prefix):
        modifiers.append("exec")

    deduped: list[str] = []
    for item in modifiers:
        if item not in deduped:
            deduped.append(item)
    return visibility, tuple(deduped)


def function_mode(prefix: str) -> str:
    if re.search(r"\bspec\s*\(\s*checked\s*\)", prefix) or re.search(r"\bspec\b", prefix):
        return "spec"
    if re.search(r"\bproof\b", prefix):
        return "proof"
    if re.search(r"\bexec\b", prefix):
        return "exec"
    return "exec"


_LEADING_ATTRIBUTE_RE = re.compile(r"(?:#\s*\[[^\]]*\]\s*)+$", re.DOTALL)


def _leading_verifier_spec_attribute(clean: str, function_start: int) -> Optional[tuple[int, str]]:
    """Return a contiguous ``#[verifier::spec]`` attribute block before ``fn``."""
    match = _LEADING_ATTRIBUTE_RE.search(clean[:function_start])
    if match is None or not re.search(
        r"#\s*\[\s*verifier\s*::\s*spec\s*\]",
        match.group(0),
    ):
        return None
    return match.start(), match.group(0).strip()


def return_vars_from_header(header: str) -> tuple[str, ...]:
    match = re.search(r"->\s*\((.*?)\)", header, flags=re.DOTALL)
    if not match:
        return tuple()
    names: list[str] = []
    for name_match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:", match.group(1)):
        names.append(name_match.group(1))
    return tuple(names)


def _find_matching_angle(text: str, open_index: int) -> Optional[int]:
    """Find the close of a Rust generic parameter list.

    Token-based scanning avoids treating the `>` in `->` as a generic close.
    """
    depth = 0
    for span in token_spans(text[open_index:]):
        if span.text == "<":
            depth += 1
        elif span.text == ">":
            depth -= 1
            if depth == 0:
                return open_index + span.start
    return None


def _function_parameter_span(header: str) -> Optional[tuple[int, int]]:
    match = FN_RE.search(header)
    if not match:
        return None
    cursor = match.end()
    while cursor < len(header) and header[cursor].isspace():
        cursor += 1
    if cursor < len(header) and header[cursor] == "<":
        generic_close = _find_matching_angle(header, cursor)
        if generic_close is None:
            return None
        cursor = generic_close + 1
        while cursor < len(header) and header[cursor].isspace():
            cursor += 1
    if cursor >= len(header) or header[cursor] != "(":
        return None
    close_paren = find_matching_paren(header, cursor)
    if close_paren is None:
        return None
    return cursor, close_paren


def function_generic_parameters_from_header(header: str) -> str:
    """Return the exact `<...>` declaration between a function name and params."""
    match = FN_RE.search(header)
    parameter_span = _function_parameter_span(header)
    if not match or parameter_span is None:
        return ""
    open_paren, _ = parameter_span
    declaration = header[match.end() : open_paren].strip()
    return declaration if declaration.startswith("<") and declaration.endswith(">") else ""


def function_where_clause_from_header(header: str) -> str:
    """Return a function-level `where ...` clause without contract clauses."""
    parameter_span = _function_parameter_span(header)
    if parameter_span is None:
        return ""
    _, close_paren = parameter_span
    tail = header[close_paren + 1 :]
    spans = token_spans(tail)
    depths = {"paren": 0, "bracket": 0, "brace": 0}
    where_start: Optional[int] = None
    where_end = len(tail)
    for span in spans:
        tok = span.text
        if _top_level(depths):
            if where_start is None and tok == "where":
                where_start = span.start
            elif where_start is not None and tok in CLAUSE_BOUNDARY_KINDS:
                where_end = span.start
                break
        _token_depth_update(tok, depths)
    if where_start is None:
        return ""
    return tail[where_start:where_end].strip()


def function_declaration_prefix(header: str) -> str:
    """Return the declaration before the first top-level contract keyword."""
    spans = token_spans(header)
    depths = {"paren": 0, "bracket": 0, "brace": 0}
    for index, span in enumerate(spans):
        token = span.text
        if (
            _top_level(depths)
            and token in CLAUSE_BOUNDARY_KINDS
            and not (index > 0 and spans[index - 1].text in {".", "::"})
        ):
            return header[: span.start].strip().rstrip(",")
        _token_depth_update(token, depths)
    return header.strip().rstrip(",")


def function_parameters_from_header(header: str) -> list[dict]:
    parameter_span = _function_parameter_span(header)
    if parameter_span is None:
        return []
    open_paren, close_paren = parameter_span

    parameters: list[dict] = []
    for part in split_top_level_type_commas(header[open_paren + 1 : close_paren]):
        if ":" not in part:
            continue
        left, right = part.split(":", 1)
        # `mut x`, `ghost x`, and `tracked x` are binding modifiers.  Never
        # remove `mut` from the type side: doing so changes `&mut T` into `&T`
        # and makes otherwise valid `old(x)` clauses fail with E0308.
        left = re.sub(r"\b(?:tracked|ghost|mut)\b", " ", left).strip()
        name_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*$", left.strip())
        if not name_match:
            continue
        parameters.append({"name": name_match.group(1), "type": right.strip()})
    return parameters


def function_return_parameters_from_header(header: str) -> list[dict]:
    parameter_span = _function_parameter_span(header)
    if parameter_span is None:
        return []
    _, close_paren = parameter_span
    tail = header[close_paren + 1 :]
    arrow_in_tail: Optional[int] = None
    depths = {"paren": 0, "bracket": 0, "brace": 0}
    for span in token_spans(tail):
        tok = span.text
        if _top_level(depths):
            if tok == "->":
                arrow_in_tail = span.start
                break
            if tok == "where" or tok in CLAUSE_BOUNDARY_KINDS:
                break
        _token_depth_update(tok, depths)
    if arrow_in_tail is None:
        return []
    arrow = close_paren + 1 + arrow_in_tail

    start = arrow + 2
    spans = token_spans(header[start:])
    depths = {"paren": 0, "bracket": 0, "brace": 0}
    end = len(header)
    for span in spans:
        tok = span.text
        if _top_level(depths) and (tok == "where" or tok in CLAUSE_BOUNDARY_KINDS):
            end = start + span.start
            break
        _token_depth_update(tok, depths)

    return_text = header[start:end].strip()
    if not return_text or return_text == "()":
        return []

    # A Verus named-return list is distinct from an ordinary Rust tuple type.
    # Only treat the outer parentheses as a list when every semantic slot has
    # a valid ``name: type`` form.  All other return syntax denotes exactly one
    # (possibly tuple-typed) unnamed return slot.
    if not return_text.startswith("("):
        return [{"name": None, "type": return_text}]
    close_return = find_matching_paren(return_text, 0)
    if close_return is None or return_text[close_return + 1 :].strip():
        return [{"name": None, "type": return_text}]

    returns: list[dict] = []
    for part in split_top_level_type_commas(return_text[1:close_return]):
        if ":" not in part:
            return [{"name": None, "type": return_text}]
        left, right = part.split(":", 1)
        left = re.sub(r"\b(?:tracked|ghost|mut)\b", " ", left).strip()
        name_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*$", left.strip())
        if not name_match or not right.strip():
            return [{"name": None, "type": return_text}]
        returns.append({"name": name_match.group(1), "type": right.strip()})
    return returns or [{"name": None, "type": return_text}]


def extract_functions(path: str) -> list[FunctionInfo]:
    clean = strip_comments(_normalize_contract_markers(read_text(path)))
    functions: list[FunctionInfo] = []
    for match in FN_RE.finditer(clean):
        name = match.group("name")
        signature_end = _find_top_level_signature_end(clean, match.end())
        if signature_end is None:
            continue
        end_offset, has_body = signature_end
        header = clean[match.start() : end_offset]
        prefix = match.group("prefix") or ""
        mode = (
            "spec"
            if _leading_verifier_spec_attribute(clean, match.start()) is not None
            else function_mode(prefix)
        )
        visibility, modifiers = _normalize_prefix_modifiers(prefix)
        clauses = extract_clauses_from_text(header)
        clause_kinds = {clause.kind for clause in clauses}
        has_contract = bool(clause_kinds & {"requires", "ensures", "default_ensures"})
        has_recommends = "recommends" in clause_kinds
        has_signature_spec = bool(clause_kinds & {"requires", "ensures", "default_ensures", "invariant", "decreases"})
        functions.append(
            FunctionInfo(
                name=name,
                header=header,
                return_vars=return_vars_from_header(header),
                has_contract=has_contract,
                mode=mode,
                modifiers=modifiers,
                visibility=visibility,
                has_body=has_body,
                has_recommends=has_recommends,
                has_signature_spec=has_signature_spec,
                is_lemma_like=mode == "proof",
            )
        )
    return functions


def function_blocks_for_path(path: str) -> list[dict]:
    clean = strip_comments(read_text(path))
    blocks: list[dict] = []
    for match in FN_RE.finditer(clean):
        name = match.group("name")
        if name == "main":
            continue
        signature_end = _find_top_level_signature_end(clean, match.end())
        if signature_end is None:
            continue
        end_offset, has_body = signature_end
        if not has_body:
            continue
        body_open = end_offset
        body_close = find_matching_brace(clean, body_open)
        if body_close is None:
            continue
        header = clean[match.start() : body_open]
        prefix = match.group("prefix") or ""
        if (
            _leading_verifier_spec_attribute(clean, match.start()) is not None
            or function_mode(prefix) != "exec"
        ):
            continue
        start_line = clean.count("\n", 0, match.start()) + 1
        body_start_line = clean.count("\n", 0, body_open)
        blocks.append(
            {
                "function": name,
                "header": header,
                "body": clean[body_open + 1 : body_close],
                "start_line": start_line,
                "body_start_line": body_start_line,
                "parameters": function_parameters_from_header(header),
                "returns": function_return_parameters_from_header(header),
            }
        )
    return blocks


def _impl_blocks_from_text(clean: str) -> list[dict]:
    """Return lexical `impl ... { ... }` containers in stripped source text."""
    spans = token_spans(clean)
    blocks: list[dict] = []
    for index, span in enumerate(spans):
        if span.text != "impl":
            continue
        depths = {"paren": 0, "bracket": 0}
        body_open: Optional[int] = None
        for candidate in spans[index + 1 :]:
            tok = candidate.text
            if tok == "(":
                depths["paren"] += 1
            elif tok == ")" and depths["paren"] > 0:
                depths["paren"] -= 1
            elif tok == "[":
                depths["bracket"] += 1
            elif tok == "]" and depths["bracket"] > 0:
                depths["bracket"] -= 1
            elif tok == "{" and depths["paren"] == depths["bracket"] == 0:
                body_open = candidate.start
                break
            elif tok == ";" and depths["paren"] == depths["bracket"] == 0:
                break
        if body_open is None:
            continue
        body_close = find_matching_brace(clean, body_open)
        if body_close is None:
            continue
        blocks.append(
            {
                "header": clean[span.start:body_open].strip(),
                "body_open": body_open,
                "body_close": body_close,
            }
        )
    return blocks


def _enclosing_impl_header(clean: str, offset: int, impl_blocks: list[dict]) -> Optional[str]:
    owners = [
        block
        for block in impl_blocks
        if int(block["body_open"]) < offset < int(block["body_close"])
    ]
    if not owners:
        return None
    owner = max(owners, key=lambda block: int(block["body_open"]))
    return str(owner["header"])


def render_spec_fn_block(block: dict, text: Optional[str] = None) -> str:
    """Render a spec function in its original lexical owner.

    A method containing `self` must remain inside an `impl`; rendering only the
    function text as a top-level item produces an invalid Rust/Verus harness.
    """
    function_text = str(text if text is not None else block.get("text", "")).strip()
    owner = str(block.get("owner") or "").strip()
    if not owner:
        return function_text
    return f"{owner} {{\n{function_text}\n}}"


def spec_fn_blocks_for_path(path: str) -> list[dict]:
    clean = strip_comments(read_text(path))
    impl_blocks = _impl_blocks_from_text(clean)
    blocks: list[dict] = []
    for match in FN_RE.finditer(clean):
        name = match.group("name")
        prefix = match.group("prefix") or ""
        verifier_spec_attribute = _leading_verifier_spec_attribute(clean, match.start())
        if function_mode(prefix) != "spec" and verifier_spec_attribute is None:
            continue
        signature_end = _find_top_level_signature_end(clean, match.end())
        if signature_end is None:
            continue
        end_offset, has_body = signature_end
        if has_body:
            body_close = find_matching_brace(clean, end_offset)
            if body_close is None:
                continue
            item_start = verifier_spec_attribute[0] if verifier_spec_attribute else match.start()
            full_text = clean[item_start : body_close + 1]
            body_text = clean[end_offset + 1 : body_close].strip() or None
        else:
            item_start = verifier_spec_attribute[0] if verifier_spec_attribute else match.start()
            full_text = clean[item_start : end_offset + 1]
            body_text = None
        owner = _enclosing_impl_header(clean, match.start(), impl_blocks)
        block = {
            "name": name,
            "text": full_text.strip(),
            "body": body_text,
        }
        if verifier_spec_attribute is not None:
            block["attribute_spec"] = True
        # Keep the historical top-level block shape stable.  Only associated
        # spec functions need the additional lexical-owner metadata.
        if owner is not None:
            block["owner"] = owner
        blocks.append(block)
    return blocks


_CONST_RE = re.compile(
    r"""
    \b
    (?P<prefix>
        (?:(?:pub(?:\s*\([^)]*\))?|spec|exec|ghost|tracked)\s+)*
    )
    const\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:
    """,
    re.MULTILINE | re.VERBOSE,
)


def _function_body_ranges(clean: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for match in FN_RE.finditer(clean):
        signature_end = _find_top_level_signature_end(clean, match.end())
        if signature_end is None or not signature_end[1]:
            continue
        body_close = find_matching_brace(clean, signature_end[0])
        if body_close is not None:
            ranges.append((signature_end[0], body_close))
    return ranges


def constant_blocks_for_path(path: str) -> list[dict]:
    """Extract top-level/associated constants that can support a lemma."""
    clean = strip_comments(read_text(path))
    impl_blocks = _impl_blocks_from_text(clean)
    function_ranges = _function_body_ranges(clean)
    blocks: list[dict] = []
    for match in _CONST_RE.finditer(clean):
        if any(start < match.start() < end for start, end in function_ranges):
            continue
        depths = {"paren": 0, "bracket": 0, "brace": 0}
        end_offset: Optional[int] = None
        for span in token_spans(clean[match.end() :]):
            if _top_level(depths) and span.text == ";":
                end_offset = match.end() + span.end
                break
            _token_depth_update(span.text, depths)
        if end_offset is None:
            continue
        owner = _enclosing_impl_header(clean, match.start(), impl_blocks)
        block = {
            "name": match.group("name"),
            "text": clean[match.start() : end_offset].strip(),
            "mode": "spec" if re.search(r"\bspec\b", match.group("prefix") or "") else "const",
        }
        if owner is not None:
            block["owner"] = owner
        blocks.append(block)
    return blocks


_TYPE_DECL_RE = re.compile(
    r"\b(?:struct|enum|union|type)\s+([A-Za-z_][A-Za-z0-9_]*)"
)


def declared_type_names_for_path(path: str) -> set[str]:
    clean = strip_comments(read_text(path))
    return {match.group(1) for match in _TYPE_DECL_RE.finditer(clean)}


def declared_type_signatures_for_path(path: str) -> dict[str, str]:
    """Return conservative structural fingerprints for local type declarations.

    Visibility and attributes do not change the logical shape used by a spec
    helper, so they are ignored.  A declaration that cannot be delimited is
    omitted; callers must then treat the context as unsupported rather than
    assume two same-named types are interchangeable.
    """
    clean = strip_comments(read_text(path))
    signatures: dict[str, str] = {}
    for match in _TYPE_DECL_RE.finditer(clean):
        start = match.start()
        cursor = match.end()
        brace = clean.find("{", cursor)
        semicolon = clean.find(";", cursor)
        if brace >= 0 and (semicolon < 0 or brace < semicolon):
            close = find_matching_brace(clean, brace)
            if close is None:
                continue
            end = close + 1
        elif semicolon >= 0:
            end = semicolon + 1
        else:
            continue
        declaration = clean[start:end]
        declaration = re.sub(r"#\s*\[[^\]]*\]", "", declaration)
        declaration = re.sub(r"\bpub(?:\s*\([^)]*\))?\s+", "", declaration)
        signatures[match.group(1)] = re.sub(r"\s+", "", declaration)
    return signatures


_USE_STATEMENT_RE = re.compile(r"^\s*use\s+[^;\n]+;\s*$", re.MULTILINE)

_COMMENT_CONTRACT_MARKER_RE = re.compile(
    r"/\*\s*(requires|ensures|default_ensures|recommends|invariant|decreases|no_unwind)\s*\*/",
    re.IGNORECASE,
)
_AT_CONTRACT_MARKER_RE = re.compile(
    r"@\s*(requires|ensures|default_ensures|recommends|invariant|decreases|no_unwind)\b",
    re.IGNORECASE,
)


def _normalize_contract_markers(text: str) -> str:
    """Make common generated contract markers visible to the Rust scanner."""
    normalized = text.replace("『", " ").replace("』", " ")
    normalized = _COMMENT_CONTRACT_MARKER_RE.sub(
        lambda match: match.group(1).lower(), normalized
    )
    return _AT_CONTRACT_MARKER_RE.sub(
        lambda match: match.group(1).lower(), normalized
    )


def use_statements_for_path(path: str) -> list[str]:
    """提取源文件顶层 use 语句（verus! 块外），用于注入 contract check harness.

    返回去重后的 use 语句列表（不含 `use vstd::prelude::*;`，因为 harness 默认已加）。
    """
    clean = strip_comments(read_text(path))
    statements: list[str] = []
    seen: set[str] = set()
    for match in _USE_STATEMENT_RE.finditer(clean):
        stmt = match.group(0).strip()
        if not stmt or stmt in seen:
            continue
        # 跳过 verus! 块内的 use（通过位置近似判断：verus! 块从第一个 verus! 开始）
        # 简单起见，只取第一个 verus! 之前的 use 语句
        seen.add(stmt)
        statements.append(stmt)
    # 排除已默认存在的 prelude，并容忍生成器插入的空白差异。
    return [
        statement
        for statement in statements
        if re.sub(r"\s+", "", statement) != "usevstd::prelude::*;"
    ]


def precondition_contexts_for_path(path: str) -> list[dict]:
    use_statements = use_statements_for_path(path)
    contexts: list[dict] = []
    for function in extract_functions(path):
        if function.name == "main" or function.mode == "proof":
            continue
        precondition_kinds = {"requires"}
        if function.mode == "spec":
            precondition_kinds.add("recommends")
        requires = [
            {"kind": clause.kind, "text": clause.text, "normalized": clause.normalized}
            for clause in extract_clauses_from_text(function.header)
            if clause.kind in precondition_kinds
        ]
        contexts.append(
            {
                "function": function.name,
                "parameters": function_parameters_from_header(function.header),
                "requires": requires,
                "mode": function.mode,
                "precondition_kinds": sorted(precondition_kinds),
                "use_statements": use_statements,
            }
        )
    return contexts


def strength_contexts_for_path(path: str) -> list[dict]:
    """Return executable function contracts used by semantic strength metrics."""
    spec_blocks = spec_fn_blocks_for_path(path)
    # Associated spec functions must remain in their lexical ``impl`` owner.
    # Emitting only the inner ``spec fn`` makes ``self``/``Self`` illegal in the
    # synthetic contract harness even though the submitted source is valid.
    spec_preamble = (
        "\n\n".join(render_spec_fn_block(block) for block in spec_blocks)
        if spec_blocks
        else ""
    )
    use_statements = use_statements_for_path(path)
    # 文件内定义的 struct/enum:IO harness 依赖它构造/比较自定义类型的值。
    # 证明 harness 引用这些类型时需要其定义可见，故渲染进 spec_preamble。
    type_registry = parse_type_definitions(read_text(path))
    rendered_types = render_type_definitions(type_registry)
    if rendered_types:
        spec_preamble = f"{rendered_types}\n\n{spec_preamble}" if spec_preamble else rendered_types
    contexts: list[dict] = []
    for function in extract_functions(path):
        if function.name == "main" or function.mode != "exec":
            continue
        clauses = extract_clauses_from_text(function.header)
        requires = [
            {"kind": clause.kind, "text": clause.text, "normalized": clause.normalized}
            for clause in clauses
            if clause.kind == "requires"
        ]
        ensures = [
            {"kind": clause.kind, "text": clause.text, "normalized": clause.normalized}
            for clause in clauses
            if clause.kind in {"ensures", "default_ensures"}
        ]
        parameters = function_parameters_from_header(function.header)
        returns = function_return_parameters_from_header(function.header)
        if type_registry:
            # 类型别名（type Matrix = Vec<Vec<i8>>）在 context 层直接解析，
            # 让 coerce/证明 harness/候选生成统一看到底层类型。
            for item in [*parameters, *returns]:
                item["type"] = resolve_alias_text(type_registry, str(item.get("type") or ""))
        contexts.append(
            {
                "function": function.name,
                "parameters": parameters,
                "returns": returns,
                "requires": requires,
                "ensures": ensures,
                "mode": function.mode,
                "modifiers": list(function.modifiers),
                "has_signature_spec": function.has_signature_spec,
                "has_contract": function.has_contract,
                "spec_preamble": spec_preamble,
                "use_statements": use_statements,
                "generic_parameters": function_generic_parameters_from_header(function.header),
                "where_clause": function_where_clause_from_header(function.header),
                "type_registry": type_registry,
            }
        )
    return contexts


def precondition_satisfiability_for_path(path: str, analyzer) -> dict:
    functions = extract_functions(path)
    result = analyzer(precondition_contexts_for_path(path))
    mode_counts = Counter(function.mode for function in functions if function.name != "main")
    result["mode_counts"] = dict(mode_counts)
    result["considered_modes"] = ["exec", "spec"]
    result["skipped_proof_functions"] = mode_counts.get("proof", 0)
    return result


def function_declaration_context_for_path(path: str, function_name: str) -> dict[str, str]:
    """Return generic syntax needed to reconstruct a sibling proof function."""
    for function in extract_functions(path):
        if function.name == function_name:
            return {
                "generic_parameters": function_generic_parameters_from_header(function.header),
                "where_clause": function_where_clause_from_header(function.header),
            }
    return {"generic_parameters": "", "where_clause": ""}


__all__ = [
    "FunctionInfo",
    "constant_blocks_for_path",
    "declared_type_names_for_path",
    "declared_type_signatures_for_path",
    "extract_functions",
    "function_blocks_for_path",
    "function_declaration_context_for_path",
    "function_declaration_prefix",
    "function_mode",
    "function_generic_parameters_from_header",
    "function_parameters_from_header",
    "function_return_parameters_from_header",
    "function_where_clause_from_header",
    "authoritative_target_for_path",
    "precondition_contexts_for_path",
    "precondition_satisfiability_for_path",
    "resolve_pair_target",
    "return_vars_from_header",
    "render_spec_fn_block",
    "strength_contexts_for_path",
    "target_descriptor_for_path",
    "target_mismatch_metric",
    "target_signatures_match",
    "use_statements_for_path",
]
