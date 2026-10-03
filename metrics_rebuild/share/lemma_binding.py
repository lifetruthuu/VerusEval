from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from metrics_rebuild.share.text import token_spans


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MUTABLE_REFERENCE_RE = re.compile(
    r"^(?P<prefix>\s*&\s*(?:'[A-Za-z_][A-Za-z0-9_]*\s*)?)mut\b\s*"
)
_PRECONDITION_KINDS = frozenset({"requires", "recommends"})
_LIFETIME_RE = re.compile(r"'[A-Za-z_][A-Za-z0-9_]*\s*")


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalized_syntax(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _matching_paren(text: str, open_offset: int) -> Optional[int]:
    depth = 0
    for offset in range(open_offset, len(text)):
        if text[offset] == "(":
            depth += 1
        elif text[offset] == ")":
            depth -= 1
            if depth == 0:
                return offset
    return None


def _normalize_tuple_groups(text: str) -> str:
    pieces: list[str] = []
    cursor = 0
    while cursor < len(text):
        if text[cursor] != "(":
            pieces.append(text[cursor])
            cursor += 1
            continue
        close = _matching_paren(text, cursor)
        if close is None:
            pieces.append(text[cursor:])
            break
        inner = _normalize_tuple_groups(text[cursor + 1 : close])
        depths = {"paren": 0, "bracket": 0, "brace": 0, "angle": 0}
        commas: list[int] = []
        for index, char in enumerate(inner):
            if char == "(":
                depths["paren"] += 1
            elif char == ")" and depths["paren"]:
                depths["paren"] -= 1
            elif char == "[":
                depths["bracket"] += 1
            elif char == "]" and depths["bracket"]:
                depths["bracket"] -= 1
            elif char == "{":
                depths["brace"] += 1
            elif char == "}" and depths["brace"]:
                depths["brace"] -= 1
            elif char == "<":
                depths["angle"] += 1
            elif char == ">" and depths["angle"]:
                depths["angle"] -= 1
            elif char == "," and not any(depths.values()):
                commas.append(index)
        if len(commas) >= 2 and commas[-1] == len(inner) - 1:
            boundaries = [-1, *commas]
            elements = [
                inner[boundaries[index] + 1 : boundaries[index + 1]]
                for index in range(len(boundaries) - 1)
            ]
            if len(elements) >= 2 and all(elements):
                inner = inner[:-1]
        pieces.extend(("(", inner, ")"))
        cursor = close + 1
    return "".join(pieces)


def _normalized_type_syntax(value: Any) -> str:
    """Normalize layout, lifetimes and only the harmless multi-element tuple tail comma.

    A lifetime such as ``&'static str`` against ``&str`` does not change the
    value a contract describes.
    """
    return _normalize_tuple_groups(_normalized_syntax(_LIFETIME_RE.sub("", str(value or ""))))


def _wrapper_type(reference_type: str, generated_type: str) -> str:
    """The generated wrapper takes the lemma variable, so a differing lifetime follows the reference."""
    return reference_type if _LIFETIME_RE.search(generated_type) else generated_type


def _context_items(context: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in context.get(key) or []:
        raw_name = _field(item, "name", None)
        name = str(raw_name) if raw_name is not None else None
        rust_type = str(_field(item, "type", "") or "")
        items.append({"name": name, "type": rust_type})
    return items


def immutable_snapshot_type(rust_type: str) -> Optional[str]:
    match = _MUTABLE_REFERENCE_RE.match(rust_type)
    if match is None:
        return None
    return (match.group("prefix") + rust_type[match.end() :]).strip()


@dataclass(frozen=True)
class BindingIssue:
    reason: str
    category: Optional[str] = None
    position: Optional[int] = None
    reference: Any = None
    generated: Any = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "holds": None,
            "status": "unknown",
            "reason": self.reason,
        }
        if self.category is not None:
            result["category"] = self.category
        if self.position is not None:
            result["position"] = self.position
        if self.reference is not None:
            result["reference"] = self.reference
        if self.generated is not None:
            result["generated"] = self.generated
        return result


@dataclass(frozen=True)
class LemmaParameter:
    name: str
    rust_type: str
    role: str
    position: int

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "type": self.rust_type}


@dataclass(frozen=True)
class ValueBinding:
    category: str
    position: int
    reference_name: str
    generated_name: str
    reference_type: str
    generated_type: str
    canonical_name: str
    lemma_type: str
    old_canonical_name: Optional[str] = None

    @property
    def mutable(self) -> bool:
        return self.old_canonical_name is not None

    def source_name(self, source: str) -> str:
        return self.generated_name if source == "generated" else self.reference_name

    def source_type(self, source: str) -> str:
        return self.generated_type if source == "generated" else self.reference_type


@dataclass(frozen=True)
class BindingPlan:
    inputs: tuple[ValueBinding, ...]
    returns: tuple[ValueBinding, ...]
    lemma_parameters: tuple[LemmaParameter, ...]
    generic_parameters: str
    where_clause: str
    mode: str

    @property
    def has_source_renames(self) -> bool:
        return any(
            item.reference_name != item.generated_name
            for item in (*self.inputs, *self.returns)
        )

    def parameter_dicts(self) -> list[dict[str, str]]:
        return [item.to_dict() for item in self.lemma_parameters]


@dataclass(frozen=True)
class BindingResult:
    plan: Optional[BindingPlan] = None
    issue: Optional[BindingIssue] = None

    @property
    def ok(self) -> bool:
        return self.plan is not None and self.issue is None


def _signature_issue(
    category: str,
    reference_items: Sequence[Mapping[str, Any]],
    generated_items: Sequence[Mapping[str, Any]],
) -> Optional[BindingIssue]:
    if len(reference_items) != len(generated_items):
        return BindingIssue(
            reason="target_signature_mismatch",
            category=f"{category}_count",
            reference=len(reference_items),
            generated=len(generated_items),
        )
    for position, (reference, generated) in enumerate(
        zip(reference_items, generated_items)
    ):
        reference_name = reference.get("name")
        generated_name = generated.get("name")
        names_invalid = category == "parameter" and (
            not isinstance(reference_name, str)
            or not _IDENTIFIER_RE.fullmatch(reference_name)
            or not isinstance(generated_name, str)
            or not _IDENTIFIER_RE.fullmatch(generated_name)
        )
        if category == "return":
            names_invalid = any(
                name is not None
                and (not isinstance(name, str) or not _IDENTIFIER_RE.fullmatch(name))
                for name in (reference_name, generated_name)
            )
        if names_invalid:
            return BindingIssue(
                reason="target_signature_mismatch",
                category=f"{category}_name",
                position=position,
                reference=reference_name,
                generated=generated_name,
            )
        reference_type = str(reference.get("type") or "")
        generated_type = str(generated.get("type") or "")
        if not reference_type or not generated_type or _normalized_type_syntax(
            reference_type
        ) != _normalized_type_syntax(generated_type):
            return BindingIssue(
                reason="target_signature_mismatch",
                category=f"{category}_type",
                position=position,
                reference=reference_type,
                generated=generated_type,
            )
    return None


def bind_function_contexts(
    reference: Mapping[str, Any],
    generated: Mapping[str, Any],
) -> BindingResult:
    """Bind two function contexts by position, never by source variable name.

    The returned plan is deliberately renderer-neutral.  It provides canonical
    lemma variables and enough source-side information to build small wrapper
    predicates without rewriting ordinary identifiers in clause bodies.
    """
    reference_mode = str(reference.get("mode") or "exec")
    generated_mode = str(generated.get("mode") or "exec")
    if reference_mode != generated_mode:
        return BindingResult(
            issue=BindingIssue(
                reason="target_signature_mismatch",
                category="mode",
                reference=reference_mode,
                generated=generated_mode,
            )
        )

    reference_generics = str(reference.get("generic_parameters") or "").strip()
    generated_generics = str(generated.get("generic_parameters") or "").strip()
    reference_where = str(reference.get("where_clause") or "").strip()
    generated_where = str(generated.get("where_clause") or "").strip()
    if (
        _normalized_syntax(reference_generics)
        != _normalized_syntax(generated_generics)
        or _normalized_syntax(reference_where) != _normalized_syntax(generated_where)
    ):
        return BindingResult(
            issue=BindingIssue(
                reason="generic_context_mismatch",
                category="generic_context",
                reference={
                    "generic_parameters": reference_generics,
                    "where_clause": reference_where,
                },
                generated={
                    "generic_parameters": generated_generics,
                    "where_clause": generated_where,
                },
            )
        )

    reference_inputs = _context_items(reference, "parameters")
    generated_inputs = _context_items(generated, "parameters")
    reference_returns = _context_items(reference, "returns")
    generated_returns = _context_items(generated, "returns")
    for category, reference_items, generated_items in (
        ("parameter", reference_inputs, generated_inputs),
        ("return", reference_returns, generated_returns),
    ):
        issue = _signature_issue(category, reference_items, generated_items)
        if issue is not None:
            return BindingResult(issue=issue)

    inputs: list[ValueBinding] = []
    returns: list[ValueBinding] = []
    lemma_parameters: list[LemmaParameter] = []
    used_reference_names = {
        str(item["name"])
        for item in (*reference_inputs, *reference_returns)
        if item.get("name")
    }
    used_generated_names = {
        str(item["name"])
        for item in (*generated_inputs, *generated_returns)
        if item.get("name")
    }

    def return_name(item: Mapping[str, Any], position: int, used: set[str]) -> str:
        if item.get("name"):
            return str(item["name"])
        candidate = f"__lemma_ret_{position}"
        suffix = 2
        while candidate in used:
            candidate = f"__lemma_ret_{position}_{suffix}"
            suffix += 1
        used.add(candidate)
        return candidate

    for position, (reference_item, generated_item) in enumerate(
        zip(reference_inputs, generated_inputs)
    ):
        rust_type = reference_item["type"]
        snapshot_type = immutable_snapshot_type(rust_type)
        canonical_name = f"__arg{position}"
        old_name = f"__old_arg{position}" if snapshot_type is not None else None
        lemma_type = snapshot_type or rust_type
        inputs.append(
            ValueBinding(
                category="parameter",
                position=position,
                reference_name=str(reference_item["name"]),
                generated_name=str(generated_item["name"]),
                reference_type=rust_type,
                generated_type=_wrapper_type(rust_type, generated_item["type"]),
                canonical_name=canonical_name,
                lemma_type=lemma_type,
                old_canonical_name=old_name,
            )
        )
        lemma_parameters.append(
            LemmaParameter(canonical_name, lemma_type, "current", position)
        )
        if old_name is not None:
            lemma_parameters.append(
                LemmaParameter(old_name, lemma_type, "old", position)
            )

    for position, (reference_item, generated_item) in enumerate(
        zip(reference_returns, generated_returns)
    ):
        canonical_name = f"__ret{position}"
        rust_type = reference_item["type"]
        returns.append(
            ValueBinding(
                category="return",
                position=position,
                reference_name=return_name(reference_item, position, used_reference_names),
                generated_name=return_name(generated_item, position, used_generated_names),
                reference_type=rust_type,
                generated_type=_wrapper_type(rust_type, generated_item["type"]),
                canonical_name=canonical_name,
                lemma_type=rust_type,
            )
        )
        lemma_parameters.append(
            LemmaParameter(canonical_name, rust_type, "return", position)
        )

    return BindingResult(
        plan=BindingPlan(
            inputs=tuple(inputs),
            returns=tuple(returns),
            lemma_parameters=tuple(lemma_parameters),
            generic_parameters=reference_generics,
            where_clause=reference_where,
            mode=reference_mode,
        )
    )


@dataclass(frozen=True)
class WrapperParameter:
    name: str
    rust_type: str
    argument: str
    role: str


@dataclass(frozen=True)
class ClauseWrapper:
    name: str
    source: str
    kind: str
    original_text: str
    body_text: str
    parameters: tuple[WrapperParameter, ...]
    generic_parameters: str = ""
    where_clause: str = ""

    @property
    def call(self) -> str:
        arguments = ", ".join(item.argument for item in self.parameters)
        return f"{self.name}({arguments})"


@dataclass(frozen=True)
class WrapperResult:
    wrapper: Optional[ClauseWrapper] = None
    issue: Optional[BindingIssue] = None

    @property
    def ok(self) -> bool:
        return self.wrapper is not None and self.issue is None


def _unique_old_names(bindings: Sequence[ValueBinding], source: str) -> dict[str, str]:
    used = {
        item.source_name(source)
        for item in bindings
    }
    result: dict[str, str] = {}
    for item in bindings:
        if not item.mutable:
            continue
        source_name = item.source_name(source)
        candidate = f"__old_{source_name}"
        suffix = 2
        while candidate in used:
            candidate = f"__old_{source_name}_{suffix}"
            suffix += 1
        used.add(candidate)
        result[source_name] = candidate
    return result


def _rewrite_old_parameter_calls(
    text: str,
    replacements: Mapping[str, str],
) -> tuple[str, set[str]]:
    """Rewrite only exact `old(param)` and `old(*param)` token sequences."""
    spans = token_spans(text)
    edits: list[tuple[int, int, str]] = []
    used: set[str] = set()
    index = 0
    while index < len(spans):
        if spans[index].text != "old" or index + 3 >= len(spans):
            index += 1
            continue
        if index > 0 and spans[index - 1].text in {".", "::"}:
            index += 1
            continue
        if spans[index + 1].text != "(":
            index += 1
            continue
        dereference = spans[index + 2].text == "*"
        name_index = index + 3 if dereference else index + 2
        close_index = name_index + 1
        if close_index >= len(spans) or spans[close_index].text != ")":
            index += 1
            continue
        source_name = spans[name_index].text
        replacement = replacements.get(source_name)
        if replacement is None:
            index += 1
            continue
        edits.append(
            (
                spans[index].start,
                spans[close_index].end,
                f"(*{replacement})" if dereference else replacement,
            )
        )
        used.add(source_name)
        index = close_index + 1

    rewritten = text
    for start, end, replacement in reversed(edits):
        rewritten = rewritten[:start] + replacement + rewritten[end:]
    return rewritten, used


def build_clause_wrapper(
    plan: BindingPlan,
    clause: Any,
    *,
    wrapper_name: str,
) -> WrapperResult:
    text = str(_field(clause, "text", "") or "").strip()
    kind = str(_field(clause, "kind", "") or "")
    source_value = _field(clause, "source", None)
    if source_value is None:
        source_value = _field(clause, "_lemma_source", None)
    source = str(source_value or "").strip()
    if source not in {"reference", "generated"}:
        if plan.has_source_renames:
            return WrapperResult(
                issue=BindingIssue(
                    reason="ambiguous_clause_origin",
                    category="clause_source",
                    reference="reference",
                    generated="generated",
                )
            )
        source = "reference"
    if not _IDENTIFIER_RE.fullmatch(wrapper_name):
        return WrapperResult(
            issue=BindingIssue(
                reason="harness_wrapper_invalid",
                category="wrapper_name",
                reference=wrapper_name,
            )
        )

    all_bindings = (*plan.inputs, *plan.returns)
    old_names = _unique_old_names(all_bindings, source)
    old_replacements: dict[str, str] = {}
    for item in plan.inputs:
        source_name = item.source_name(source)
        if item.mutable:
            old_replacements[source_name] = old_names[source_name]
        else:
            old_replacements[source_name] = source_name
    body_text, used_old_names = _rewrite_old_parameter_calls(text, old_replacements)

    parameters: list[WrapperParameter] = []
    for item in all_bindings:
        source_name = item.source_name(source)
        rust_type = item.lemma_type if item.mutable else item.source_type(source)
        argument = item.canonical_name
        if item.mutable and kind in _PRECONDITION_KINDS:
            argument = str(item.old_canonical_name)
        parameters.append(
            WrapperParameter(source_name, rust_type, argument, item.category)
        )

    for item in plan.inputs:
        source_name = item.source_name(source)
        if not item.mutable or source_name not in used_old_names:
            continue
        parameters.append(
            WrapperParameter(
                old_names[source_name],
                item.lemma_type,
                str(item.old_canonical_name),
                "old",
            )
        )

    return WrapperResult(
        wrapper=ClauseWrapper(
            name=wrapper_name,
            source=source,
            kind=kind,
            original_text=text,
            body_text=body_text,
            parameters=tuple(parameters),
            generic_parameters=plan.generic_parameters,
            where_clause=plan.where_clause,
        )
    )


def render_clause_wrapper(wrapper: ClauseWrapper) -> str:
    parameters = ", ".join(
        f"{item.name}: {item.rust_type}" for item in wrapper.parameters
    )
    generics = wrapper.generic_parameters.strip()
    lines = [f"spec fn {wrapper.name}{generics}({parameters}) -> bool"]
    if wrapper.where_clause.strip():
        lines.extend(
            f"    {line.strip()}"
            for line in wrapper.where_clause.strip().splitlines()
        )
    lines.extend(["{", f"    {render_clause_expression(wrapper.body_text)}", "}"])
    return "\n".join(lines)


_LOGICAL_OPERATORS = frozenset({"&&", "&&&", "||", "|||", "==>", "<==>"})


def _matching_token(spans: Sequence[Any], start: int, opener: str, closer: str) -> Optional[int]:
    depth = 0
    for index in range(start, len(spans)):
        token = spans[index].text
        if token == opener:
            depth += 1
        elif token == closer:
            depth -= 1
            if depth == 0:
                return index
    return None


def _control_expression_end(spans: Sequence[Any], start: int) -> Optional[int]:
    """Return the final token index for one top-level ``if`` or ``match``."""
    if spans[start].text == "match":
        body = next(
            (index for index in range(start + 1, len(spans)) if spans[index].text == "{"),
            None,
        )
        return _matching_token(spans, body, "{", "}") if body is not None else None

    cursor = start
    while cursor < len(spans) and spans[cursor].text == "if":
        body = next(
            (index for index in range(cursor + 1, len(spans)) if spans[index].text == "{"),
            None,
        )
        if body is None:
            return None
        close = _matching_token(spans, body, "{", "}")
        if close is None:
            return None
        if close + 1 >= len(spans) or spans[close + 1].text != "else":
            return close
        if close + 2 < len(spans) and spans[close + 2].text == "if":
            cursor = close + 2
            continue
        if close + 2 >= len(spans) or spans[close + 2].text != "{":
            return None
        return _matching_token(spans, close + 2, "{", "}")
    return None


def render_clause_expression(text: str) -> str:
    """Parenthesize top-level control-flow operands in logical expressions.

    Verus accepts an ``if``/``match`` as a standalone expression, but Rust's
    parser needs parentheses when the control-flow expression is an operand of
    ``&&``, ``||`` or implication.  Token positions keep this transformation
    outside nested expression bodies and make it idempotent.
    """
    expression = str(text).strip()
    spans = token_spans(expression)
    if not spans:
        return expression

    depths: list[tuple[int, int, int]] = []
    paren = bracket = brace = 0
    for span in spans:
        depths.append((paren, bracket, brace))
        if span.text == "(":
            paren += 1
        elif span.text == ")" and paren:
            paren -= 1
        elif span.text == "[":
            bracket += 1
        elif span.text == "]" and bracket:
            bracket -= 1
        elif span.text == "{":
            brace += 1
        elif span.text == "}" and brace:
            brace -= 1

    top_level_operators = {
        index
        for index, span in enumerate(spans)
        if depths[index] == (0, 0, 0) and span.text in _LOGICAL_OPERATORS
    }
    if not top_level_operators:
        return expression

    edits: list[tuple[int, str]] = []
    consumed_until = -1
    for index, span in enumerate(spans):
        if index <= consumed_until or depths[index] != (0, 0, 0):
            continue
        if span.text not in {"if", "match"}:
            continue
        end = _control_expression_end(spans, index)
        if end is None:
            continue
        participates = index - 1 in top_level_operators or end + 1 in top_level_operators
        if participates:
            edits.append((spans[index].start, "("))
            edits.append((spans[end].end, ")"))
        consumed_until = end

    rendered = expression
    for offset, insertion in sorted(edits, reverse=True):
        rendered = rendered[:offset] + insertion + rendered[offset:]
    return rendered


__all__ = [
    "BindingIssue",
    "BindingPlan",
    "BindingResult",
    "ClauseWrapper",
    "LemmaParameter",
    "ValueBinding",
    "WrapperParameter",
    "WrapperResult",
    "bind_function_contexts",
    "build_clause_wrapper",
    "immutable_snapshot_type",
    "render_clause_expression",
    "render_clause_wrapper",
]
