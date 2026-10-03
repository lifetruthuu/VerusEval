from __future__ import annotations

import json
import math
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from metrics_rebuild.share.clauses import find_matching_paren, split_top_level_commas
from metrics_rebuild.share.text import token_spans
from metrics_rebuild.share.type_defs import registry_entry, registry_type_name
from metrics_rebuild.share.verus_runner import VerusRun, run_verus

_VEC_ELEMENT_CAP = 12
_FLAT_VEC_ELEMENT_CAP = 64
_NESTED_VEC_OUTER_CAP = 8
# Outputs come from executing the reference implementation, so their size is
# not under candidate-generation control (e.g. mt19937 returns 624 elements).
# Inputs keep the tight caps above; outputs get these generous ones.
_OUTPUT_FLAT_VEC_CAP = 2048
_OUTPUT_NESTED_OUTER_CAP = 256
_FINITE_FORALL_INDEX_CAP = 64

_NON_FINITE_FLOAT_STRINGS = {
    "inf": math.inf,
    "+inf": math.inf,
    "-inf": -math.inf,
    "infinity": math.inf,
    "+infinity": math.inf,
    "-infinity": -math.inf,
    "nan": math.nan,
    "-nan": math.nan,
}

_INT_BOUNDS = {
    "u8": (0, 255), "u16": (0, 65535), "u32": (0, 4294967295),
    "u64": (0, 2**64 - 1), "u128": (0, 2**128 - 1), "usize": (0, 2**64 - 1),
    "i8": (-128, 127), "i16": (-32768, 32767), "i32": (-2147483648, 2147483647),
    "i64": (-(2**63), 2**63 - 1), "i128": (-(2**127), 2**127 - 1), "isize": (-(2**63), 2**63 - 1),
}


def _coerce_vec_element(item: Any) -> tuple[bool, Any]:
    if isinstance(item, bool):
        return True, bool(item)
    if isinstance(item, int):
        return True, int(item)
    if isinstance(item, float):
        return True, float(item)
    if isinstance(item, str):
        return True, item
    return False, None


def base_type_name(type_text: str) -> str:
    """Return the scalar base type mentioned in ``type_text`` ("" when absent).

    Historically this defaulted to "int" for unknown types, which made custom
    types (e.g. ``Matrix``) coerce like integers. Returning "" keeps every
    membership check (`is_bool_type` 等) False for unrecognized types.
    """
    match = re.search(r"\b(bool|char|u8|u16|u32|u64|u128|usize|i8|i16|i32|i64|i128|isize|int|nat|f32|f64)\b", str(type_text))
    return match.group(1) if match else ""


def normalize_value_type(type_text: str, *, keep_reference: bool = False) -> str:
    """Normalize a possibly borrowed value type.

    Proof-side values never need the source lifetime or ``mut`` qualifier.  The
    caller may retain an immutable reference for a proof-function parameter;
    all other callers receive the underlying value type.
    """
    text = str(type_text).strip()
    reference = re.match(r"^&\s*(?:'[A-Za-z_][A-Za-z0-9_]*\s*)?(?:mut\s+)?", text)
    if reference:
        text = text[reference.end():]
    text = re.sub(r"^mut\s+", "", text).strip()
    return f"&{text}" if keep_reference and reference else text


def _is_string_like_type(type_text: str) -> bool:
    return normalize_value_type(type_text) in {"String", "str", "'static str"}


def _proof_scalar_binding_type(type_text: str) -> str:
    """Type used to let-bind a scalar parameter/return inside a proof fn.

    References are bound by value (proof-side values never need `&`/`&mut`),
    and string-like values are bound as `&str` literals — `String` has no
    literal form, while `&str` shares the same `Seq<char>` view, so both
    `name@` and bare `name == "lit"` clauses stay well-typed.
    """
    if _is_string_like_type(type_text):
        return "&str"
    return normalize_value_type(type_text)


def normalize_type_key(type_text: str) -> str:
    """Normalize a Rust/Verus type for equality grouping (e.g. SVR).

    Strips references and whitespace so `& Vec< bool >` and `Vec<bool>` match.
    Does **not** collapse nested scalars: `Vec<bool>` stays distinct from `bool`.
    Shared by mutation testing (SVR) and IO type helpers to avoid drifting rules.
    """
    text = normalize_value_type(type_text)
    return re.sub(r"\s+", "", text)


def _is_wrapper_or_composite_type(type_text: str) -> bool:
    """True for Vec/Seq/Option/tuple/String — top-level is not a scalar primitive."""
    text = normalize_value_type(type_text)
    if not text:
        return False
    if text.startswith("("):
        return True
    if re.match(r"Option\s*<", text):
        return True
    if is_vec_type(text):
        return True
    if text in {"String", "str", "'static str"} or text.startswith("str"):
        return True
    return False


def is_nested_vec_type(type_text: str) -> bool:
    return bool(re.search(r"\b(?:Vec|Seq)\s*<\s*(?:Vec|Seq)\s*<", str(type_text)))


def is_vec_type(type_text: str) -> bool:
    text = normalize_value_type(type_text)
    return bool(re.search(r"\b(?:Vec|Seq)\s*<", text)) or bool(re.search(r"\[\s*[^;\]]+", text))


def is_bool_type(type_text: str) -> bool:
    # Must not treat Vec<bool> / Option<bool> / (bool, _) as bool.
    if _is_wrapper_or_composite_type(type_text):
        return False
    return base_type_name(type_text) == "bool"


def is_char_type(type_text: str) -> bool:
    if _is_wrapper_or_composite_type(type_text):
        return False
    return base_type_name(type_text) == "char"


def is_float_type(type_text: str) -> bool:
    if _is_wrapper_or_composite_type(type_text):
        return False
    return base_type_name(type_text) in {"f32", "f64"}


def is_unsigned_type(type_text: str) -> bool:
    if _is_wrapper_or_composite_type(type_text):
        return False
    return base_type_name(type_text) in {"u8", "u16", "u32", "u64", "u128", "usize", "nat"}


def is_int_like_type(type_text: str) -> bool:
    if _is_wrapper_or_composite_type(type_text):
        return False
    if is_float_type(type_text):
        return False
    return base_type_name(type_text) in {
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "usize",
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "isize",
        "int",
        "nat",
    }


def coerce_value_for_type(
    value: Any,
    type_text: str,
    _nested_depth: int = 0,
    *,
    flat_cap: Optional[int] = None,
    nested_outer_cap: Optional[int] = None,
    registry: Optional[dict] = None,
) -> tuple[bool, Any]:
    stripped = normalize_value_type(type_text)
    if re.match(r"Option\s*<", stripped):
        if value is None:
            return True, None
        inner = re.sub(r"^Option\s*<(.*)>\s*$", r"\1", stripped)
        return coerce_value_for_type(
            value, inner, _nested_depth,
            flat_cap=flat_cap, nested_outer_cap=nested_outer_cap, registry=registry,
        )
    if stripped.startswith("("):
        # Checked before the vec/scalar branches: base_type_name / is_vec_type
        # would otherwise match a type name nested inside the tuple.
        if not isinstance(value, (list, tuple)):
            return False, None
        values = list(value)
        elem_types = _extract_tuple_element_types(stripped)
        if not elem_types:
            # Unparseable tuple type: keep the old permissive length check.
            return (True, values) if len(values) >= 2 else (False, None)
        if len(values) != len(elem_types):
            return False, None
        coerced = []
        for item, elem_type in zip(values, elem_types):
            ok, coerced_item = coerce_value_for_type(
                item, elem_type, _nested_depth,
                flat_cap=flat_cap, nested_outer_cap=nested_outer_cap, registry=registry,
            )
            if not ok:
                return False, None
            coerced.append(coerced_item)
        return True, coerced
    if is_vec_type(stripped):
        if not isinstance(value, list):
            return False, None
        # 定长数组 [T; N] 要求恰好 N 个元素（否则 harness 侧无法忠实构造）。
        array_match = re.match(r"^\[\s*.+?\s*;\s*(\d+)\s*\]$", stripped)
        if array_match and len(value) != int(array_match.group(1)):
            return False, None
        inner_type = _extract_vec_element_type(stripped)
        if _nested_depth > 0:
            cap = flat_cap if flat_cap is not None else _VEC_ELEMENT_CAP
        elif is_vec_type(inner_type):
            cap = nested_outer_cap if nested_outer_cap is not None else _NESTED_VEC_OUTER_CAP
        else:
            cap = flat_cap if flat_cap is not None else _FLAT_VEC_ELEMENT_CAP
        if len(value) > cap:
            return False, None
        coerced = []
        for item in value:
            if normalize_type_key(inner_type) == "char":
                # Vec<char> uses character strings throughout the IO harness;
                # scalar `char` payloads use code points. Preserve that public
                # representation while still validating each element shape.
                ok, coerced_item = _coerce_vec_element(item)
                ok = bool(ok and (not isinstance(item, str) or len(item) == 1))
            else:
                ok, coerced_item = coerce_value_for_type(
                    item, inner_type, _nested_depth + 1,
                    flat_cap=flat_cap, nested_outer_cap=nested_outer_cap, registry=registry,
                )
            if not ok:
                return False, None
            coerced.append(coerced_item)
        return True, coerced
    if stripped in ("String", "str", "&str", "'static str", "&'static str"):
        # 字符串类型：接受任意字符串（用于 String/&str 返回值的 IO 校验）
        if isinstance(value, str):
            return True, value
        return False, None
    if is_bool_type(stripped):
        if isinstance(value, bool):
            return True, value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return True, value.lower() == "true"
        return False, None
    if is_char_type(stripped) or base_type_name(stripped) == "char":
        if isinstance(value, bool):
            return False, None
        if isinstance(value, int):
            if 0 <= value <= 0x10FFFF and not (0xD800 <= value <= 0xDFFF):
                return True, int(value)
            return False, None
        if isinstance(value, str) and len(value) == 1:
            return True, ord(value)
        return False, None
    if is_float_type(stripped):
        if isinstance(value, bool):
            return False, None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
        elif isinstance(value, str) and re.fullmatch(r"-?\d+(\.\d+)?([eE][+-]?\d+)?", value.strip()):
            number = float(value.strip())
        elif isinstance(value, str) and value.strip().lower() in _NON_FINITE_FLOAT_STRINGS:
            number = _NON_FINITE_FLOAT_STRINGS[value.strip().lower()]
        else:
            return False, None
        # Non-finite values (reference implementations legitimately return
        # f32::INFINITY / NAN) are preserved; only finite out-of-range floats
        # are rejected.
        if math.isfinite(number):
            limit = 3.4e38 if base_type_name(stripped) == "f32" else 1.7e308
            if abs(number) > limit:
                return False, None
        return True, number
    if base_type_name(stripped) == "u8":
        if isinstance(value, bool):
            return False, None
        if isinstance(value, int) and 0 <= value <= 255:
            return True, int(value)
        if isinstance(value, str) and len(value) == 1 and ord(value) <= 255:
            return True, ord(value)
        return False, None
    if is_int_like_type(stripped):
        if isinstance(value, bool):
            return False, None
        if isinstance(value, int):
            number = int(value)
        elif isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            number = int(value.strip())
        else:
            return False, None
        base = base_type_name(stripped)
        bounds = _INT_BOUNDS.get(base)
        if bounds is not None and not (bounds[0] <= number <= bounds[1]):
            return False, None
        if is_unsigned_type(stripped) and number < 0:
            return False, None
        return True, number
    entry = registry_entry(registry, stripped)
    if entry is not None:
        if entry["kind"] == "enum":
            if isinstance(value, str) and value in entry["variants"]:
                return True, value
            return False, None
        if not isinstance(value, (list, tuple)):
            return False, None
        values = list(value)
        fields = entry["fields"]
        if len(values) != len(fields):
            return False, None
        coerced = []
        for item, (_fname, field_type) in zip(values, fields):
            ok, coerced_item = coerce_value_for_type(
                item, field_type, _nested_depth,
                flat_cap=flat_cap, nested_outer_cap=nested_outer_cap, registry=registry,
            )
            if not ok:
                return False, None
            coerced.append(coerced_item)
        return True, coerced
    if re.fullmatch(r"[A-Z]", stripped):
        # 单字母泛型参数：native harness 单态化为 i32，按 i32 值接受。
        if isinstance(value, bool):
            return False, None
        if isinstance(value, int):
            number = int(value)
        elif isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            number = int(value.strip())
        else:
            return False, None
        low, high = _INT_BOUNDS["i32"]
        if not (low <= number <= high):
            return False, None
        return True, number
    return False, None


def typed_input_payload(context: dict, raw_inputs: dict) -> Optional[dict]:
    registry = context.get("type_registry")
    typed: dict[str, Any] = {}
    for param in context.get("parameters") or []:
        name = param.get("name")
        if not name or name not in raw_inputs:
            return None
        ok, value = coerce_value_for_type(
            raw_inputs[name], param.get("type", ""), registry=registry,
        )
        if not ok:
            return None
        typed[name] = value
    return typed


def typed_output_payload(context: dict, output: Optional[dict]) -> Optional[dict]:
    if not isinstance(output, dict):
        return None
    registry = context.get("type_registry")
    typed: dict[str, Any] = {}
    for ret in context.get("returns") or []:
        name = ret.get("name") or "ret"
        if name not in output:
            return None
        ok, value = coerce_value_for_type(
            output[name], ret.get("type", ""),
            flat_cap=_OUTPUT_FLAT_VEC_CAP, nested_outer_cap=_OUTPUT_NESTED_OUTER_CAP,
            registry=registry,
        )
        if not ok:
            return None
        typed[name] = value
    return typed


def _tokenize_for_parens(text: str) -> list[str]:
    return re.findall(r"==>|<==>|&&&|\|\|\||[A-Za-z_][A-Za-z0-9_]*|-?\d+|[(){}\[\],;:|]|==|!=|<=|>=|&&|\|\||[^\s]", text)


def _strip_outer_parens(tokens: list[str]) -> list[str]:
    if len(tokens) < 2 or tokens[0] != "(" or tokens[-1] != ")":
        return tokens
    depth = 0
    for index, token in enumerate(tokens):
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth == 0 and index != len(tokens) - 1:
                return tokens
    return tokens[1:-1]


def top_level_split_once(expr: str, operator: str) -> Optional[tuple[str, str]]:
    paren = bracket = brace = 0
    in_pipe = False
    state = "normal"
    i = 0
    while i <= len(expr) - len(operator):
        ch = expr[i]
        nxt = expr[i + 1] if i + 1 < len(expr) else ""
        if state == "normal":
            if ch == '"':
                state = "string"
            elif ch == "'":
                state = "char"
            elif ch == "|" and nxt != "|":
                in_pipe = not in_pipe
            elif ch == "(":
                paren += 1
            elif ch == ")" and paren > 0:
                paren -= 1
            elif ch == "[":
                bracket += 1
            elif ch == "]" and bracket > 0:
                bracket -= 1
            elif ch == "{":
                brace += 1
            elif ch == "}" and brace > 0:
                brace -= 1
            elif not in_pipe and paren == bracket == brace == 0 and expr.startswith(operator, i):
                return expr[:i].strip(), expr[i + len(operator) :].strip()
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


def _finite_quantifier_domain(env: dict) -> list[int]:
    values: set[int] = set(range(-1, 11))
    for value in env.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            values.update({value - 1, value, value + 1})
        elif isinstance(value, list):
            values.update(range(-1, len(value) + 2))
            for item in value:
                if isinstance(item, int) and not isinstance(item, bool):
                    values.add(item)
    return sorted(value for value in values if -64 <= value <= 256)


_EVAL_INT_TYPE_BOUNDS: dict[str, tuple[Optional[int], Optional[int]]] = {
    "u8": (0, 2**8 - 1),
    "u16": (0, 2**16 - 1),
    "u32": (0, 2**32 - 1),
    "u64": (0, 2**64 - 1),
    "u128": (0, 2**128 - 1),
    "usize": (0, 2**64 - 1),
    "nat": (0, None),
    "i8": (-(2**7), 2**7 - 1),
    "i16": (-(2**15), 2**15 - 1),
    "i32": (-(2**31), 2**31 - 1),
    "i64": (-(2**63), 2**63 - 1),
    "i128": (-(2**127), 2**127 - 1),
    "isize": (-(2**63), 2**63 - 1),
}


def _quantifier_binders_with_types(binders_text: str) -> list[tuple[str, str]]:
    binders: list[tuple[str, str]] = []
    for part in split_top_level_commas(binders_text):
        if not part.strip():
            continue
        if ":" in part:
            name, type_text = part.split(":", 1)
        else:
            name, type_text = part, "int"
        name = name.strip()
        if name:
            binders.append((name, type_text.strip() or "int"))
    return binders


def _domain_for_quantifier_type(values: Sequence[int], type_text: str) -> list[int]:
    lower, upper = _EVAL_INT_TYPE_BOUNDS.get(base_type_name(type_text), (None, None))
    filtered = []
    for value in values:
        if lower is not None and value < lower:
            continue
        if upper is not None and value > upper:
            continue
        filtered.append(value)
    return filtered


def _normalize_eval_expr(expr: str) -> str:
    text = str(expr).strip()
    text = re.sub(r"#\s*\[\s*trigger\s*\]", " ", text)
    text = re.sub(r"#\s*\[[^\]]*\]", " ", text)
    text = text.replace("&&&", "&&").replace("|||", "||").replace("=~=", "==")
    text = re.sub(r"\.view\s*\(\s*\)", "", text)
    text = re.sub(
        r"\bold\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        lambda match: f"__ve_old_{match.group(1)}",
        text,
    )
    text = text.replace("@", "")
    # Strip unary Rust dereference (`*x`) without eating multiplication in
    # expressions such as `2 * N`. The prefix must be the start of the clause
    # or an operator/delimiter, not merely whitespace after an operand.
    text = re.sub(
        r"(^|[(,=!<>+\-/%&|])\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)",
        r"\1\2",
        text,
    )
    text = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\.(\d+)\b", r"\1[\2]", text)
    text = text.replace("seq![", "[")
    text = re.sub(
        r"\s+as\s+(?:int|nat|usize|u8|u16|u32|u64|u128|i8|i16|i32|i64|i128|isize)\b",
        "",
        text,
    )
    text = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*len\s*\(\s*\)", r"__ve_len(\1)", text)
    text = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*contains\s*\(", r"__ve_contains(\1, ", text)
    text = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*is_empty\s*\(\s*\)", r"(__ve_len(\1) == 0)", text)
    text = re.sub(r"(?<!\.)\babs\s*\(", "__ve_abs(", text)
    text = re.sub(r"(?<!\.)\bmin\s*\(", "__ve_min(", text)
    text = re.sub(r"(?<!\.)\bmax\s*\(", "__ve_max(", text)
    text = re.sub(r"(?<!\.)\bsum\s*\(", "__ve_sum(", text)
    text = re.sub(r"(?<!\.)\bis_sorted\s*\(", "__ve_is_sorted(", text)
    text = re.sub(r"\btrue\b", "True", text)
    text = re.sub(r"\bfalse\b", "False", text)
    text = text.replace("&&", " and ").replace("||", " or ")
    text = re.sub(r"!(?!=)", " not ", text)
    return text


def _resolve_nested_quantifiers(text: str, env: dict) -> str:
    result = text
    search_start = 0
    while True:
        match = re.search(r"\(\s*(forall|exists)\s*\|", result[search_start:])
        if not match:
            break
        abs_start = search_start + match.start()
        close = find_matching_paren(result, abs_start)
        if close is None:
            search_start = abs_start + 1
            continue
        sub_expr = result[abs_start + 1 : close].strip()
        sub_result = eval_contract_expr(sub_expr, env)
        if sub_result is not None:
            replacement = "True" if sub_result else "False"
            result = result[:abs_start] + replacement + result[close + 1 :]
            search_start = abs_start + len(replacement)
        else:
            search_start = close + 1
    return result


def eval_contract_expr(expr: str, env: dict) -> Optional[bool]:
    text = str(expr).strip().rstrip(",;")
    while text.startswith("(") and text.endswith(")"):
        tokens = _tokenize_for_parens(text)
        if _strip_outer_parens(tokens) == tokens:
            break
        text = text[1:-1].strip()
    if not text:
        return True

    quant_match = re.match(r"^(forall|exists)\s*\|(?P<binders>.*?)\|\s*(?P<body>.*)$", text, flags=re.DOTALL)
    if quant_match:
        kind = quant_match.group(1)
        binders = _quantifier_binders_with_types(quant_match.group("binders"))
        body = quant_match.group("body").strip()
        if not binders:
            return None
        domain = _finite_quantifier_domain(env)

        def check_binder(idx: int, local_env: dict) -> Optional[bool]:
            if idx == len(binders):
                return eval_contract_expr(body, local_env)
            saw_unknown = False
            binder_name, binder_type = binders[idx]
            typed_domain = _domain_for_quantifier_type(domain, binder_type)
            if kind == "forall":
                for value in typed_domain:
                    local_env[binder_name] = value
                    result = check_binder(idx + 1, local_env)
                    if result is False:
                        return False
                    if result is None:
                        saw_unknown = True
                return None if saw_unknown else True
            for value in typed_domain:
                local_env[binder_name] = value
                result = check_binder(idx + 1, local_env)
                if result is True:
                    return True
                if result is None:
                    saw_unknown = True
            return None if saw_unknown else False

        return check_binder(0, dict(env))

    implication = top_level_split_once(text, "==>")
    if implication is not None:
        left, right = implication
        left_value = eval_contract_expr(left, env)
        if left_value is False:
            return True
        if left_value is None:
            right_value = eval_contract_expr(right, env)
            return True if right_value is True else None
        return eval_contract_expr(right, env)

    py_expr = _normalize_eval_expr(_resolve_nested_quantifiers(text, env))
    safe_env = {
        "__ve_len": len,
        "__ve_contains": lambda seq, item: item in seq,
        "__ve_is_sorted": lambda seq: all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1)) if len(seq) > 1 else True,
        "__ve_sum": sum,
        "__ve_abs": abs,
        "__ve_min": min,
        "__ve_max": max,
    }
    safe_env.update(dict(env))
    try:
        return bool(eval(py_expr, {"__builtins__": {}}, safe_env))
    except Exception:
        return None


def _eval_clause_group(clauses: Sequence[dict], env: dict) -> tuple[Optional[bool], int, int]:
    unsupported = 0
    evaluated = 0
    for clause in clauses:
        result = eval_contract_expr(clause.get("normalized") or clause.get("text") or "", env)
        if result is None:
            unsupported += 1
            continue
        evaluated += 1
        if result is False:
            return False, unsupported, evaluated
    if unsupported:
        # Conservative: never report success while any clause went unevaluated —
        # a False could be hiding in a clause the Python evaluator did not
        # understand. (Previously non-strict mode accepted when
        # evaluated >= unsupported, which was an unsound softening.)
        return None, unsupported, evaluated
    return True, unsupported, evaluated


def contract_evaluation(context: dict, inputs: dict, output: Optional[dict] = None, *, strict: bool = False) -> dict:
    # `strict` now only affects reporting (the "strict" flag and the reported
    # unsupported count); the accept/None decision is always conservative, i.e.
    # any unevaluated clause yields accepted=None rather than a softened True.
    env = dict(inputs)
    env.update({f"__ve_old_{name}": value for name, value in inputs.items()})
    if output:
        env.update(output)
    requires_ok, req_unsupported, req_evaluated = _eval_clause_group(context.get("requires") or [], env)
    if requires_ok is False:
        result = {
            "accepted": False,
            "requires_ok": False,
            "ensures_ok": None,
            "unsupported_clauses": req_unsupported,
            "evaluated_clauses": req_evaluated,
            "reason": "requires_not_satisfied",
        }
        if strict:
            result["strict"] = True
        return result
    ensures_ok, ens_unsupported, ens_evaluated = _eval_clause_group(context.get("ensures") or [], env)
    if requires_ok is None or ensures_ok is None:
        result = {
            "accepted": None,
            "requires_ok": requires_ok,
            "ensures_ok": ensures_ok,
            "unsupported_clauses": req_unsupported + ens_unsupported,
            "evaluated_clauses": req_evaluated + ens_evaluated,
            "reason": "unsupported_expression",
        }
        if strict:
            result["strict"] = True
        return result
    result = {
        "accepted": bool(requires_ok and ensures_ok),
        "requires_ok": bool(requires_ok),
        "ensures_ok": bool(ensures_ok),
        "unsupported_clauses": 0 if not strict else req_unsupported + ens_unsupported,
        "evaluated_clauses": req_evaluated + ens_evaluated,
        "reason": "accepted" if requires_ok and ensures_ok else "ensures_not_satisfied",
    }
    if strict:
        result["strict"] = True
    return result


def _extract_tuple_element_types(type_text: str) -> list[str]:
    text = str(type_text).strip()
    if not text.startswith("("):
        return []
    inner = text[1:]
    if inner.endswith(")"):
        inner = inner[:-1]
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in inner:
        if ch in "<([":
            depth += 1
            current.append(ch)
        elif ch in ">)]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    trailing = "".join(current).strip()
    if trailing:
        parts.append(trailing)
    return parts


def _extract_vec_element_type(type_text: str) -> str:
    text = normalize_value_type(type_text)
    match = re.match(r"(?:Vec|Seq)\s*<\s*", text)
    if not match:
        return base_type_name(type_text)
    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "<":
            depth += 1
        elif text[i] == ">":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1].strip()
    return base_type_name(type_text)


def _spec_value_type(type_text: str) -> str:
    """Map runtime containers to their proof-side value types recursively."""
    stripped = normalize_value_type(type_text)
    if re.match(r"(?:Vec|Seq)\s*<", stripped):
        return f"Seq<{_spec_value_type(_extract_vec_element_type(stripped))}>"
    if re.match(r"Option\s*<", stripped):
        inner = re.sub(r"^Option\s*<(.*)>\s*$", r"\1", stripped)
        return f"Option<{_spec_value_type(inner)}>"
    if stripped.startswith("("):
        parts = _extract_tuple_element_types(stripped)
        suffix = "," if len(parts) == 1 else ""
        return "(" + ", ".join(_spec_value_type(part) for part in parts) + suffix + ")"
    return stripped


def _verus_int_literal(value: int, type_text: str) -> str:
    base = base_type_name(type_text)
    if base == "int":
        return f"{value}int"
    if base == "nat":
        return f"{value}int"
    if base in {"u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize"}:
        return f"{value}{base}"
    return str(value)


def _verus_char_literal(ordinal: int) -> str:
    if ordinal < 0 or ordinal > 0x10FFFF or 0xD800 <= ordinal <= 0xDFFF:
        # Not a Unicode scalar value. Emit the raw (invalid) escape so the
        # harness fails to compile and the case drops to None, rather than
        # silently masking it to a different, valid character.
        safe = ordinal if ordinal >= 0 else 0x110000
        return f"'\\u{{{safe:x}}}'"
    c = chr(ordinal)
    if c == "\\":
        return "'\\\\'"
    if c == "'":
        return "'\\''"
    if c == "\n":
        return "'\\n'"
    if c == "\t":
        return "'\\t'"
    if c == "\r":
        return "'\\r'"
    if c == "\0":
        return "'\\0'"
    if 0x20 <= ordinal <= 0x7E:
        return f"'{c}'"
    return f"'\\u{{{ordinal:04x}}}'"


def _is_bare_char_type(type_text: str) -> bool:
    return normalize_value_type(type_text) == "char"


def verus_literal(value: Any, type_text: str = "", registry: Optional[dict] = None) -> str:
    stripped = normalize_value_type(type_text)
    if re.match(r"Option\s*<", stripped):
        inner = re.sub(r"^Option\s*<(.*)>\s*$", r"\1", stripped)
        if value is None:
            return "None"
        return f"Some({verus_literal(value, inner, registry)})"
    entry = registry_entry(registry, stripped)
    if entry is not None:
        name = registry_type_name(stripped)
        if entry["kind"] == "enum":
            variants = entry["variants"]
            variant = value if isinstance(value, str) and value in variants else variants[0]
            return f"{name}::{variant}"
        fields = entry["fields"]
        values = list(value) if isinstance(value, (list, tuple)) else []
        while len(values) < len(fields):
            values.append(0)
        if entry.get("tuple"):
            parts = [
                verus_literal(item, field_type, registry)
                for item, (_fname, field_type) in zip(values, fields)
            ]
            return f"{name}({', '.join(parts)})"
        parts = [
            f"{fname}: {verus_literal(item, field_type, registry)}"
            for item, (fname, field_type) in zip(values, fields)
        ]
        return f"{name} {{ {', '.join(parts)} }}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        if _is_bare_char_type(type_text):
            return _verus_char_literal(value)
        return _verus_int_literal(value, type_text)
    if isinstance(value, float):
        base = base_type_name(stripped)
        suffix = base if base in {"f32", "f64"} else ""
        target = suffix or "f64"
        if math.isnan(value):
            return f"{target}::NAN"
        if math.isinf(value):
            return f"{target}::{'INFINITY' if value > 0 else 'NEG_INFINITY'}"
        literal = repr(value)
        return f"{literal}_{suffix}" if suffix else literal
    if isinstance(value, str) and len(value) == 1 and _is_bare_char_type(type_text):
        return _verus_char_literal(ord(value))
    if isinstance(value, (list, tuple)):
        if stripped.startswith("("):
            elem_types = _extract_tuple_element_types(stripped)
            parts = []
            for i, v in enumerate(value):
                t = elem_types[i] if i < len(elem_types) else ""
                parts.append(verus_literal(v, t, registry))
            suffix = "," if len(parts) == 1 else ""
            return "(" + ", ".join(parts) + suffix + ")"
        if isinstance(value, list):
            elem_type = _extract_vec_element_type(type_text)
            if not value:
                return f"Seq::<{elem_type}>::empty()"
            return "seq![" + ", ".join(verus_literal(item, elem_type, registry) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def _fixed_sequence_clauses(
    name: str,
    type_text: str,
    value: list,
    *,
    length_only: bool = False,
    registry: Optional[dict] = None,
) -> list[str]:
    def recurse(expr: str, current_type: str, current_value: Any, *, top: bool) -> list[str]:
        stripped = normalize_value_type(current_type)
        is_container = bool(re.match(r"(?:Vec|Seq)\s*<", stripped) or stripped.startswith("["))
        if not is_container or not isinstance(current_value, list):
            if _is_string_like_type(current_type):
                # String 元素与 &str 字面量直接 == 会类型不匹配；比较 Seq<char> 视图。
                return [f"{expr}@ == {verus_literal(current_value, current_type, registry)}@"]
            return [f"{expr} == {verus_literal(current_value, current_type, registry)}"]

        needs_view = bool(re.match(r"Vec\s*<", stripped) or stripped.startswith("["))
        view_expr = f"{expr}@" if needs_view else expr
        clauses = [f"{view_expr}.len() == {len(current_value)}"]
        if length_only and top:
            return clauses

        element_type = _extract_vec_element_type(stripped)
        element_is_container = bool(
            re.match(r"(?:Vec|Seq)\s*<", normalize_value_type(element_type))
        )
        for index, element in enumerate(current_value):
            element_expr = f"{view_expr}[{index}]"
            if element_is_container and isinstance(element, list):
                clauses.extend(recurse(element_expr, element_type, element, top=False))
            elif _is_string_like_type(element_type):
                clauses.append(f"{element_expr}@ == {verus_literal(element, element_type, registry)}@")
            elif not is_float_type(element_type):
                # Verus does not expose general proof-side equality for runtime
                # floats.  Shape facts are still exact and useful; value-based
                # float preconditions remain UNKNOWN instead of failing compile.
                clauses.append(f"{element_expr} == {verus_literal(element, element_type, registry)}")
        return clauses

    return recurse(name, type_text, value, top=True)


def _clause_expr_for_harness(clause: dict) -> str:
    text = str(clause.get("text") or clause.get("normalized") or "").strip()
    return text.rstrip(",;")


def _mutable_param_names(context: dict) -> set[str]:
    """Names of `&mut` parameters plus any explicitly tracked post-state names."""
    names = set(context.get("_mutable_post_state_names") or [])
    for param in context.get("parameters") or []:
        name = param.get("name")
        if name and re.match(r"^&\s*mut\b", str(param.get("type") or "").strip()):
            names.add(str(name))
    return names


def _strip_value_deref(expr: str, name: str) -> str:
    """Drop a unary `*` in front of ``name`` (value-bound, so deref is invalid).

    The prefix guard mirrors ``_normalize_eval_expr``: the `*` must follow an
    operator/delimiter or the expression start, so multiplication like
    ``a * name`` is preserved.
    """
    return re.sub(
        rf"(^|[(,=!<>+\-/%&|])\s*\*\s*{re.escape(name)}\b",
        rf"\1{name}",
        expr,
    )


def _substitute_pre_state(expr: str, name: str, replacement: str, *, strip_deref: bool) -> str:
    """Rewrite pre-state references of a `&mut` parameter in a clause.

    ``old(name)`` and bare ``name`` both denote the pre-state in requires
    clauses, so both collapse onto ``replacement`` (the concrete binding).
    """
    expr = re.sub(rf"\bold\s*\(\s*{re.escape(name)}\s*\)", name, expr)
    expr = re.sub(rf"\b{re.escape(name)}\b", replacement, expr)
    if strip_deref:
        expr = _strip_value_deref(expr, replacement)
    return expr


_SPEC_FN_DECREASES_RE = re.compile(r"\bspec\s+fn\s+([A-Za-z_]\w*)\b[^{};]*\bdecreases\b")
_SPEC_FN_HEADER_RE = re.compile(r"\bspec\s+fn\s+([A-Za-z_]\w*)")


def _spec_fn_bodies(preamble: str) -> dict[str, str]:
    """Map each `spec fn` declared in the preamble to its body text."""
    bodies: dict[str, str] = {}
    for match in _SPEC_FN_HEADER_RE.finditer(preamble):
        start = preamble.find("{", match.end())
        if start < 0:
            continue
        depth = 0
        for index in range(start, len(preamble)):
            if preamble[index] == "{":
                depth += 1
            elif preamble[index] == "}":
                depth -= 1
                if depth == 0:
                    bodies[match.group(1)] = preamble[start + 1:index]
                    break
    return bodies


def _spec_fn_calls(bodies: dict[str, str]) -> dict[str, set[str]]:
    return {
        name: {
            other for other in bodies
            if re.search(rf"\b{re.escape(other)}\s*\(", body)
        }
        for name, body in bodies.items()
    }


def _really_recursive_spec_fns(bodies: dict[str, str]) -> set[str]:
    """Spec fns that call themselves, directly or through another spec fn.

    A `decreases` clause is not proof of recursion, and Verus rejects
    `reveal_with_fuel` above 1 for a non-recursive function — which fails the
    whole harness, not just the one clause that wanted the fuel.
    """
    calls = _spec_fn_calls(bodies)
    recursive: set[str] = set()
    for name in bodies:
        seen: set[str] = set()
        frontier = list(calls[name])
        while frontier:
            current = frontier.pop()
            if current == name:
                recursive.add(name)
                break
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(calls.get(current, ()))
    return recursive


def _fuel_reveal_lines(context: dict, clause_texts: Sequence[str]) -> list[str]:
    """`reveal_with_fuel` lines for recursive spec fns reachable from the clauses.

    Concrete-input proofs over recursive spec fns (e.g. ``triangle(3)``) need
    more unfolding fuel than the default of 1; revealing is additive and never
    invalidates an otherwise-passing proof.  Reachability is transitive: a
    clause often names only a non-recursive wrapper (``fibo_fits_i32(n)``) whose
    body is what actually needs the fuel (``fibo(n)``).
    """
    preamble = str(context.get("spec_preamble") or "")
    if not preamble:
        return []
    declared = {match.group(1) for match in _SPEC_FN_DECREASES_RE.finditer(preamble)}
    if not declared:
        return []
    bodies = _spec_fn_bodies(preamble)
    recursive = declared & _really_recursive_spec_fns(bodies)
    if not recursive:
        return []
    reached: set[str] = set()
    frontier = [" ".join(clause_texts)]
    while frontier:
        text = frontier.pop()
        for name, body in bodies.items():
            if name in reached or not re.search(rf"\b{re.escape(name)}\s*\(", text):
                continue
            reached.add(name)
            frontier.append(body)
    return [
        f"    reveal_with_fuel({name}, 12);"
        for name in bodies
        if name in reached and name in recursive
    ]


_STRING_FACT_CHAR_CAP = 40


def _string_view_fact_lines(
    bindings: Sequence[tuple[str, Any]],
) -> tuple[list[str], list[str]]:
    """`reveal_strlit` calls plus per-character view facts for `&str` bindings.

    Verus keeps a string literal's `Seq<char>` view opaque until the literal is
    revealed, so a concrete-pair proof over `text@[i]` cannot decide either
    direction and stays unresolved.  Revealing the literal and pinning its
    length and characters lets the solver instantiate quantified clauses.  The
    facts follow from the literal itself, so they only ever add information.
    """
    reveals: list[str] = []
    facts: list[str] = []
    revealed: set[str] = set()
    for name, value in bindings:
        if not isinstance(value, str) or len(value) > _STRING_FACT_CHAR_CAP:
            continue
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            continue
        literal = verus_literal(value, "str")
        if literal not in revealed:
            revealed.add(literal)
            reveals.append(f"    reveal_strlit({literal});")
        facts.append(f"    assert({name}@.len() == {len(value)});")
        facts.extend(
            f"    assert({name}@[{index}] == {_verus_char_literal(ord(char))});"
            for index, char in enumerate(value)
        )
    return reveals, facts


def _case_has_string_literal(context: dict, case: dict) -> bool:
    """Whether this concrete pair binds any `&str` literal inside the harness."""
    for group, values in (
        ("parameters", case.get("inputs") or {}),
        ("returns", case.get("output") or {}),
    ):
        for item in context.get(group) or []:
            if not _is_string_like_type(str(item.get("type") or "")):
                continue
            name = str(item.get("name") or ("ret" if group == "returns" else ""))
            if isinstance(values.get(name), str):
                return True
    return False


def _proof_fn_header_lines(context: dict, harness_name: str, sig_params: Sequence[str]) -> list[str]:
    """Build a sibling proof signature with the target's generic declaration."""
    generic_parameters = str(context.get("generic_parameters") or "").strip()
    lines = [f"proof fn {harness_name}{generic_parameters}({', '.join(sig_params)})"]
    where_clause = str(context.get("where_clause") or "").strip()
    if where_clause:
        lines.extend(f"    {line.strip()}" for line in where_clause.splitlines())
    return lines


def _delimiter_issue(text: str) -> Optional[str]:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for span in token_spans(text):
        token = span.text
        if token in {"(", "[", "{"}:
            stack.append(token)
        elif token in pairs:
            if not stack or stack[-1] != pairs[token]:
                return f"unexpected_{token}"
            stack.pop()
    return f"unclosed_{stack[-1]}" if stack else None


def _generic_type_binders(context: dict) -> set[str]:
    declaration = str(context.get("generic_parameters") or "").strip()
    if not (declaration.startswith("<") and declaration.endswith(">")):
        return set()
    binders: set[str] = set()
    for part in split_top_level_commas(declaration[1:-1]):
        match = re.match(r"\s*(?:const\s+)?([A-Za-z_][A-Za-z0-9_]*)", part)
        if match:
            binders.add(match.group(1))
    return binders


def _type_uses_generic_binder(type_text: str, binders: set[str]) -> bool:
    return any(span.text in binders for span in token_spans(str(type_text)))


def _parameter_is_length_only(context: dict, name: str, *, include_ensures: bool) -> bool:
    clauses = list(context.get("requires") or [])
    if include_ensures:
        clauses.extend(context.get("ensures") or [])
    text = " ".join(_clause_expr_for_harness(clause) for clause in clauses)
    without_len = re.sub(
        rf"\b{re.escape(name)}\s*@?\s*\.\s*len\s*\(\s*\)",
        "",
        text,
    )
    return not re.search(rf"\b{re.escape(name)}\b", without_len)


def _generic_value_needs_literal(
    value: Any,
    type_text: str,
    binders: set[str],
    *,
    length_only: bool = False,
) -> bool:
    """Whether encoding this value would force a concrete literal to have type T."""
    if not _type_uses_generic_binder(type_text, binders):
        return False
    stripped = normalize_value_type(type_text)
    if re.match(r"Option\s*<", stripped):
        if value is None:
            return False
        inner = re.sub(r"^Option\s*<(.*)>\s*$", r"\1", stripped)
        return _generic_value_needs_literal(value, inner, binders)
    if is_vec_type(stripped):
        if length_only or value == []:
            return False
        inner = _extract_vec_element_type(stripped)
        return any(
            _generic_value_needs_literal(item, inner, binders)
            for item in (value if isinstance(value, list) else [value])
        )
    if stripped.startswith("(") and isinstance(value, (list, tuple)):
        element_types = _extract_tuple_element_types(stripped)
        return any(
            _generic_value_needs_literal(item, element_types[index], binders)
            for index, item in enumerate(value)
            if index < len(element_types)
        )
    return True


def _type_supported_by_harness(type_text: str, binders: set[str], registry: Optional[dict] = None) -> bool:
    """Reject custom runtime types whose declarations are absent from the mini harness.

    References are supported: containers go through their proof-side views,
    scalars behind ``&``/``&mut`` are let-bound by value, and string-like
    parameters are bound as ``&str`` literals (see ``_proof_scalar_binding_type``).
    ``registry`` 中的文件内 struct/enum 用其定义展开后判定。
    """
    stripped = re.sub(
        r"^&\s*(?:'[A-Za-z_][A-Za-z0-9_]*\s*)?(?:mut\s+)?", "", str(type_text).strip(),
    )
    if stripped in binders:
        return True
    if stripped in {
        "bool", "char", "u8", "u16", "u32", "u64", "u128", "usize",
        "i8", "i16", "i32", "i64", "i128", "isize", "int", "nat",
        "f32", "f64", "()",
        "str", "'static str", "String",
    }:
        return True
    if re.match(r"(?:Vec|Seq|Option)\s*<", stripped):
        inner = _extract_vec_element_type(stripped) if not stripped.startswith("Option") else re.sub(
            r"^Option\s*<(.*)>\s*$", r"\1", stripped,
        )
        return _type_supported_by_harness(inner, binders, registry)
    if stripped.startswith("["):
        inner = _extract_vec_element_type(stripped)
        return _type_supported_by_harness(inner, binders, registry)
    if stripped.startswith("("):
        parts = _extract_tuple_element_types(stripped)
        return bool(parts) and all(_type_supported_by_harness(part, binders, registry) for part in parts)
    entry = registry_entry(registry, stripped)
    if entry is not None:
        if entry["kind"] == "enum":
            return True
        return all(
            _type_supported_by_harness(field_type, binders, registry)
            for _fname, field_type in entry["fields"]
        )
    return False


def _contract_case_support_issue(
    context: dict,
    inputs: dict,
    output: Optional[dict] = None,
    *,
    include_ensures: bool,
) -> Optional[dict[str, str]]:
    clause_groups = ["requires", "ensures"] if include_ensures else ["requires"]
    for group in clause_groups:
        for clause in context.get(group) or []:
            expression = _clause_expr_for_harness(clause)
            issue = _delimiter_issue(expression)
            if issue:
                return {"reason": "malformed_clause", "detail": issue}

    binders = _generic_type_binders(context)
    registry = context.get("type_registry")
    for param in context.get("parameters") or []:
        name = str(param.get("name") or "")
        type_text = str(param.get("type") or "")
        if not name or name not in inputs:
            return {"reason": "input_type_mismatch", "detail": name or "missing_parameter_name"}
        if not _type_supported_by_harness(type_text, binders, registry):
            return {"reason": "unsupported_type", "detail": type_text}
        length_only = bool(
            is_vec_type(type_text)
            and _parameter_is_length_only(context, name, include_ensures=include_ensures)
        )
        if _generic_value_needs_literal(
            inputs[name], type_text, binders, length_only=length_only,
        ):
            return {"reason": "unresolved_generic_instantiation", "detail": name}
        coerced, _ = coerce_value_for_type(inputs[name], type_text, registry=registry)
        if not coerced:
            return {"reason": "input_type_mismatch", "detail": name}

    if include_ensures:
        output = output if isinstance(output, dict) else {}
        for ret in context.get("returns") or []:
            name = str(ret.get("name") or "ret")
            type_text = str(ret.get("type") or "")
            if name not in output:
                return {"reason": "output_type_mismatch", "detail": name}
            if not _type_supported_by_harness(type_text, binders, registry):
                return {"reason": "unsupported_type", "detail": type_text}
            if _generic_value_needs_literal(output[name], type_text, binders):
                return {"reason": "unresolved_generic_instantiation", "detail": name}
            coerced, _ = coerce_value_for_type(
                output[name], type_text,
                flat_cap=_OUTPUT_FLAT_VEC_CAP, nested_outer_cap=_OUTPUT_NESTED_OUTER_CAP,
                registry=registry,
            )
            if not coerced:
                return {"reason": "output_type_mismatch", "detail": name}
    return None


def _finite_index_forall_assertion_lines(
    expr: str,
    sequence_lengths: dict[str, int],
) -> Optional[list[str]]:
    """Add a finite-domain hint for a concrete sequence-indexed ``forall``.

    Verus does not always split a bounded integer quantifier into its concrete
    indices, especially when the body contains sequence indexing and modulo.
    The sequence values are already pinned by harness preconditions, so an
    explicit exhaustive disjunction is a proof hint, not a sampled fallback.
    """
    text = str(expr).strip()
    while text.startswith("(") and find_matching_paren(text, 0) == len(text) - 1:
        text = text[1:-1].strip()
    match = re.match(
        r"^forall\s*\|(?P<binders>.*?)\|\s*(?P<body>.*)$",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return None
    binders = _quantifier_binders_with_types(match.group("binders"))
    if len(binders) != 1:
        return None
    binder, binder_type = binders[0]
    if base_type_name(binder_type) not in {"int", "nat"}:
        return None

    implication = top_level_split_once(match.group("body"), "==>")
    if implication is None:
        implication = top_level_split_once(match.group("body"), "implies")
    if implication is None:
        return None
    antecedent, consequent = implication

    escaped_binder = re.escape(binder)
    nonnegative = base_type_name(binder_type) == "nat" or bool(
        re.search(rf"(?:\b0\s*<=\s*\b{escaped_binder}\b|\b{escaped_binder}\b\s*>=\s*0\b)", antecedent)
    )
    if not nonnegative:
        return None

    bound: Optional[int] = None
    for name, length in sequence_lengths.items():
        escaped_name = re.escape(name)
        length_expr = rf"\b{escaped_name}\b\s*@?\s*\.\s*len\s*\(\s*\)"
        if re.search(rf"\b{escaped_binder}\b\s*<\s*{length_expr}", antecedent):
            bound = length
            break
    if bound is None or bound > _FINITE_FORALL_INDEX_CAP:
        return None

    domain = " || ".join(f"{binder} == {index}" for index in range(bound)) or "false"
    # A trigger attribute must directly follow the binders, outside the parentheses.
    prefix = ""
    stripped = antecedent.lstrip()
    if stripped.startswith("#!["):
        depth = 0
        for end, char in enumerate(stripped):
            depth += {"[": 1, "]": -1}.get(char, 0)
            if char == "]" and depth == 0:
                prefix, antecedent = stripped[:end + 1] + " ", stripped[end + 1:].strip()
                break
    return [
        f"    assert forall|{match.group('binders')}| {prefix}({antecedent}) implies ({consequent}) by {{",
        f"        assert({domain});",
        "    }",
    ]


_BITWISE_OPS = {"^": lambda x, y: x ^ y, "|": lambda x, y: x | y, "&": lambda x, y: x & y}
_BITWISE_FACT_CAP = 128


def _bitwise_ops_in(clause_texts: Sequence[str]) -> list[str]:
    """Binary ^, | and & in the clauses, ignoring binders, &&, || and references."""
    text = re.sub(r"\b(?:forall|exists|choose)\s*\|[^|]*\|", " ", " ".join(clause_texts))
    return [
        op for op in _BITWISE_OPS
        if re.search(rf"[\w\])]\s*{re.escape(op)}(?![&|=])\s*[\w(#-]", text)
    ]


def _bitwise_fact_lines(
    context: dict, inputs: dict, clause_texts: Sequence[str], registry: Optional[dict] = None,
) -> list[str]:
    """Bit-vector facts on the concrete element pairs of equally long integer inputs.

    Outside bit_vector mode Verus treats ^, | and & as uninterpreted, so even
    ``a[0] ^ b[0]`` with pinned elements stays undecided. Each fact is proved
    by Verus in bit_vector mode and then usable in both proof directions.
    Both operand orders are emitted because the operators are uninterpreted.
    """
    ops = _bitwise_ops_in(clause_texts)
    if not ops:
        return []
    sequences = []
    for param in context.get("parameters") or []:
        values = inputs.get(param.get("name"))
        type_text = str(param.get("type") or "")
        if not is_vec_type(type_text) or not isinstance(values, list):
            continue
        element = normalize_value_type(_extract_vec_element_type(type_text))
        if element in _INT_BOUNDS and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            sequences.append((element, values))
    cast_types = set(re.findall(r"\bas\s+([ui](?:8|16|32|64|128|size))\b", " ".join(clause_texts)))
    facts: list[str] = []
    for index, (left_type, left) in enumerate(sequences):
        for other, (right_type, right) in enumerate(sequences):
            if index == other or left_type != right_type or len(left) != len(right):
                continue
            for type_name in sorted({left_type} | (cast_types & set(_INT_BOUNDS))):
                low, high = _INT_BOUNDS[type_name]
                for x, y in zip(left, right):
                    if not (low <= x <= high and low <= y <= high):
                        continue
                    for op in ops:
                        value = _BITWISE_OPS[op](x, y)
                        fact = (
                            f"    assert({verus_literal(x, type_name, registry)} {op} "
                            f"{verus_literal(y, type_name, registry)} == "
                            f"{verus_literal(value, type_name, registry)}) by (bit_vector);"
                        )
                        if fact not in facts:
                            facts.append(fact)
    return facts[:_BITWISE_FACT_CAP]


def _return_param_type(type_text: str) -> Optional[str]:
    """Owned parameter type for a Vec, array or String return, else None."""
    stripped = normalize_value_type(type_text)
    if re.match(r"Vec\s*<", stripped) or stripped.startswith("[") or stripped == "String":
        return stripped
    return None


def _contract_check_proof_fn_lines(
    context: dict,
    inputs: dict,
    output: dict,
    harness_name: str,
    *,
    negate_contract: bool = False,
    string_facts: bool = False,
    return_params: bool = False,
) -> list[str]:
    params = list(context.get("parameters") or [])
    returns = list(context.get("returns") or [])
    requires = context.get("requires") or []
    ensures = context.get("ensures") or []
    # Optionally bind Vec, array and String returns like vector inputs: as
    # parameters pinned by requires, so clauses expecting the runtime type
    # still typecheck where a Seq or &str literal would not.
    return_param_types = {
        str(ret.get("name") or "ret"): _return_param_type(ret.get("type", ""))
        for ret in returns
        if return_params and _return_param_type(ret.get("type", ""))
        and isinstance(output.get(ret.get("name") or "ret"), (list, str))
    }

    registry = context.get("type_registry")
    mutable_names = _mutable_param_names(context)
    vector_params = [p for p in params if is_vec_type(p.get("type", "")) and isinstance(inputs.get(p.get("name")), list)]
    scalar_params = [p for p in params if p not in vector_params]
    # `&mut` vector params are bound as immutable `&Vec<..>` pre-state views;
    # proof fns cannot take `&mut` parameters.
    sig_params = [
        f"{'__old_' if p.get('name') in mutable_names else ''}{p.get('name')}: "
        f"{normalize_value_type(p.get('type', ''), keep_reference=True)}"
        for p in vector_params if p.get("name") and p.get("type")
    ]
    # Scalar params (including references) are let-bound by value, so any
    # unary deref of them in the clauses must be stripped.
    deref_strip_names = {
        (f"__old_{p['name']}" if p.get("name") in mutable_names else str(p.get("name")))
        for p in scalar_params
        if p.get("name") and str(p.get("type") or "").strip().startswith("&")
    }

    sig_params.extend(f"{name}: {type_text}" for name, type_text in return_param_types.items())
    lines = _proof_fn_header_lines(context, harness_name, sig_params)
    contract_text = " ".join(
        _clause_expr_for_harness(clause)
        for clause in [*requires, *ensures]
    )
    fixed_clauses: list[str] = []
    for p in vector_params:
        fixed_name = f"__old_{p.get('name')}" if p.get("name") in mutable_names else p.get("name", "")
        without_len = re.sub(
            rf"\b{re.escape(str(p.get('name') or ''))}\s*@?\s*\.\s*len\s*\(\s*\)",
            "",
            contract_text,
        )
        length_only = not re.search(
            rf"\b{re.escape(str(p.get('name') or ''))}\b", without_len,
        )
        fixed_clauses.extend(
            _fixed_sequence_clauses(
                fixed_name,
                p.get("type", ""),
                inputs.get(p.get("name"), []),
                length_only=length_only,
                registry=registry,
            )
        )
    for name, type_text in return_param_types.items():
        fixed_clauses.extend(_fixed_sequence_clauses(name, type_text, output[name], registry=registry))
    if fixed_clauses:
        lines.append("    requires")
        for index, clause in enumerate(fixed_clauses):
            comma = "," if index < len(fixed_clauses) - 1 else ""
            lines.append(f"        {clause}{comma}")
    lines.append("{")
    lines.extend(_fuel_reveal_lines(
        context,
        [_clause_expr_for_harness(clause) for clause in [*requires, *ensures]],
    ))
    reveal_slot = len(lines)
    string_bindings: list[tuple[str, Any]] = []
    for param in scalar_params:
        name = param.get("name")
        if name and name in inputs:
            bound_name = f"__old_{name}" if name in mutable_names else name
            binding_type = _proof_scalar_binding_type(param.get("type", ""))
            lines.append(f"    let {bound_name}: {binding_type} = {verus_literal(inputs[name], param.get('type', ''), registry)};")
            if binding_type == "&str":
                string_bindings.append((bound_name, inputs[name]))
    return_names: set[str] = set()
    string_return_names: set[str] = set()
    return_sequence_facts: list[str] = []
    for ret in returns:
        name = ret.get("name") or "ret"
        if name not in output or name in return_param_types:
            continue
        return_names.add(name)
        type_text = ret.get("type", "")
        value = output[name]
        stripped_ret = normalize_value_type(type_text)
        is_option = bool(re.match(r"Option\s*<", stripped_ret))
        is_top_level_vec = bool(re.match(r"(?:Vec|Seq)\s*<", stripped_ret))
        if isinstance(value, list) and is_top_level_vec and not is_option:
            elem = _extract_vec_element_type(type_text)
            lines.append(f"    let {name}: Seq<{_spec_value_type(elem)}> = {verus_literal(value, type_text, registry)};")
            # Besides defining the literal, surface its concrete shape and
            # elements as terms.  Quantified postconditions often trigger on
            # ``result[i]``; without these facts a false bounded ``forall`` can
            # remain undecided in the rejection direction even for a fully
            # concrete output.
            return_sequence_facts.extend(
                _fixed_sequence_clauses(
                    name,
                    f"Seq<{_spec_value_type(elem)}>",
                    value,
                    registry=registry,
                )
            )
        elif _is_string_like_type(type_text) and isinstance(value, str):
            # `String` has no literal; a `&str` binding shares the Seq<char>
            # view, so `name@` and bare `name == "lit"` clauses both typecheck.
            string_return_names.add(name)
            lines.append(f"    let {name}: &str = {verus_literal(value, type_text)};")
            string_bindings.append((name, value))
        else:
            lines.append(f"    let {name}: {_spec_value_type(type_text)} = {verus_literal(value, type_text, registry)};")
    lines.extend(f"    assert({fact});" for fact in return_sequence_facts)
    lines.extend(_bitwise_fact_lines(
        context,
        inputs,
        [_clause_expr_for_harness(clause) for clause in [*requires, *ensures]],
        registry,
    ))
    if string_facts and string_bindings:
        reveals, facts = _string_view_fact_lines(string_bindings)
        lines[reveal_slot:reveal_slot] = reveals
        lines.extend(facts)
    scalar_mutable_names = {
        str(p.get("name")) for p in scalar_params if p.get("name") in mutable_names
    }
    require_exprs: list[str] = []
    for clause in requires:
        expr = _clause_expr_for_harness(clause)
        for name in mutable_names:
            expr = _substitute_pre_state(
                expr, name, f"__old_{name}", strip_deref=name in scalar_mutable_names,
            )
        for name in deref_strip_names:
            expr = _strip_value_deref(expr, name)
        if expr:
            require_exprs.append(expr)
    if not negate_contract:
        for expr in require_exprs:
            lines.append(f"    assert({expr});")
    ensure_exprs: list[str] = []
    sequence_lengths = {
        (
            f"__old_{p.get('name')}"
            if p.get("name") in mutable_names
            else str(p.get("name"))
        ): len(inputs[p.get("name")])
        for p in vector_params
        if p.get("name") in inputs and isinstance(inputs[p.get("name")], list)
    }
    sequence_lengths.update(
        {
            str(ret.get("name") or "ret"): len(output[ret.get("name") or "ret"])
            for ret in returns
            if isinstance(output.get(ret.get("name") or "ret"), list)
            and re.match(r"(?:Vec|Seq)\s*<", normalize_value_type(ret.get("type", "")))
        }
    )
    vector_mutable_names = {
        str(p.get("name"))
        for p in vector_params
        if p.get("name") in mutable_names
    }
    for clause in ensures:
        expr = _clause_expr_for_harness(clause)
        if not expr:
            continue
        # 返回变量声明为 Seq<T>（用 seq! 字面量），而源码 ensures 中写 result@ (View)。
        # Seq 没有 view 方法，但语义上 result 已是 Seq，故把 <retname>@ 替换为 <retname>。
        # 字符串返回绑定为 &str，自身有 view，保留 <retname>@。
        for rname in return_names - string_return_names:
            expr = re.sub(rf"\b{re.escape(rname)}@", rname, expr)
            expr = re.sub(
                rf"\b{re.escape(rname)}((?:\.\d+)+)@",
                lambda match: f"{rname}{match.group(1)}",
                expr,
            )
        for name in mutable_names:
            if name in vector_mutable_names:
                # The source may spell a Vec snapshot as either ``old(x)``
                # (coerced by Verus in the function contract) or explicitly
                # as ``old(x)@``.  The harness binds ``__old_x`` as ``&Vec``;
                # consume an existing view suffix before adding the one view
                # required by the proof-side ``Seq`` expression.
                expr = re.sub(
                    rf"\bold\s*\(\s*{re.escape(name)}\s*\)\s*@?",
                    f"__old_{name}@",
                    expr,
                )
            else:
                expr = re.sub(
                    rf"\bold\s*\(\s*{re.escape(name)}\s*\)",
                    f"__old_{name}",
                    expr,
                )
            if name in scalar_mutable_names:
                expr = _strip_value_deref(expr, f"__old_{name}")
        # 后置状态的标量返回值（如 &mut u32 的 *sum）也按值绑定，剥去解引用。
        for rname in return_names:
            expr = _strip_value_deref(expr, rname)
        for name in deref_strip_names:
            expr = _strip_value_deref(expr, name)
        ensure_exprs.append(expr)
    if negate_contract:
        if ensure_exprs:
            # Rejection is the negation of accepting the concrete IO pair.
            # This includes either a rejected input or a violated postcondition.
            conjunction = " && ".join(
                f"({expr})" for expr in [*require_exprs, *ensure_exprs]
            )
            lines.append(f"    assert(!({conjunction}));")
        else:
            # A missing/transformation-failed postcondition is not evidence of
            # rejection.  Force this proof direction to remain unresolved.
            lines.append("    assert(false);")
    else:
        for expr in ensure_exprs:
            finite_forall = _finite_index_forall_assertion_lines(expr, sequence_lengths)
            if finite_forall is not None:
                lines.extend(finite_forall)
            else:
                lines.append(f"    assert({expr});")
    lines.append("}")
    return lines


def _requires_check_proof_fn_lines(
    context: dict,
    inputs: dict,
    harness_name: str,
    *,
    negate_requires: bool = False,
    string_facts: bool = False,
) -> list[str]:
    params = list(context.get("parameters") or [])
    requires = context.get("requires") or []

    registry = context.get("type_registry")
    mutable_names = _mutable_param_names(context)
    vector_params = [p for p in params if is_vec_type(p.get("type", "")) and isinstance(inputs.get(p.get("name")), list)]
    scalar_params = [p for p in params if p not in vector_params]
    # Requires clauses only see the pre-state, so `&mut` params bind under
    # their own name as immutable `&Vec<..>` (proof fns cannot take `&mut`).
    sig_params = [
        f"{p.get('name')}: {normalize_value_type(p.get('type', ''), keep_reference=True)}"
        for p in vector_params if p.get("name") and p.get("type")
    ]
    deref_strip_names = {
        str(p.get("name"))
        for p in scalar_params
        if p.get("name") and str(p.get("type") or "").strip().startswith("&")
    }

    lines = _proof_fn_header_lines(context, harness_name, sig_params)
    fixed_clauses: list[str] = []
    requires_text = " ".join(_clause_expr_for_harness(clause) for clause in requires)
    for p in vector_params:
        name = str(p.get("name") or "")
        # If a parameter is used only through len(), do not pin its elements.
        # This avoids irrelevant f32/custom-element proof failures for length
        # preconditions such as a.len() == b.len().
        without_len = re.sub(rf"\b{re.escape(name)}\s*@?\s*\.\s*len\s*\(\s*\)", "", requires_text)
        length_only = not re.search(rf"\b{re.escape(name)}\b", without_len)
        fixed_clauses.extend(
            _fixed_sequence_clauses(
                name,
                p.get("type", ""),
                inputs.get(p.get("name"), []),
                length_only=length_only,
                registry=registry,
            )
        )
    if fixed_clauses:
        lines.append("    requires")
        for index, clause in enumerate(fixed_clauses):
            comma = "," if index < len(fixed_clauses) - 1 else ""
            lines.append(f"        {clause}{comma}")
    lines.append("{")
    lines.extend(_fuel_reveal_lines(
        context, [_clause_expr_for_harness(clause) for clause in requires],
    ))
    reveal_slot = len(lines)
    string_bindings: list[tuple[str, Any]] = []
    for param in scalar_params:
        name = param.get("name")
        if name and name in inputs:
            binding_type = _proof_scalar_binding_type(param.get("type", ""))
            lines.append(f"    let {name}: {binding_type} = {verus_literal(inputs[name], param.get('type', ''), registry)};")
            if binding_type == "&str":
                string_bindings.append((name, inputs[name]))
    if string_facts and string_bindings:
        reveals, facts = _string_view_fact_lines(string_bindings)
        lines[reveal_slot:reveal_slot] = reveals
        lines.extend(facts)
    require_exprs: list[str] = []
    for clause in requires:
        expr = _clause_expr_for_harness(clause)
        if not expr:
            continue
        for name in mutable_names:
            expr = _substitute_pre_state(
                expr, name, name, strip_deref=name in deref_strip_names,
            )
        for name in deref_strip_names:
            expr = _strip_value_deref(expr, name)
        require_exprs.append(expr)
    if negate_requires:
        if require_exprs:
            conjunction = " && ".join(f"({expr})" for expr in require_exprs)
            lines.append(f"    assert(!({conjunction}));")
        else:
            lines.append("    assert(false);")
    else:
        for expr in require_exprs:
            lines.append(f"    assert({expr});")
    lines.append("}")
    return lines


def _run_verus_on_text(text: str, filename: str, timeout_seconds: int = 10):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / filename
        path.write_text(text, encoding="utf-8")
        return run_verus(str(path), timeout_seconds=timeout_seconds)


def _unresolved_run_reason(run: VerusRun) -> str:
    if run.status == "timeout":
        return "timeout"
    if run.status == "unavailable":
        return "unavailable"
    stderr = (run.stderr or "").lower()
    if any(
        token in stderr
        for token in (
            "not supported",
            "unsupported",
            "no method named",
            "trait bound",
            "cannot find type",
        )
    ):
        return "unsupported_type"
    # Verus rejects the same quantifiers and spec functions in the target's own
    # contract, so these errors mark an ill-formed contract, not a harness bug.
    if any(
        token in stderr
        for token in (
            "could not automatically infer triggers",
            "could not prove termination",
            "recursive function must have a decreases clause",
            "trigger must be a function call",
            "triggers cannot contain",
            "in trigger cannot appear in both arithmetic and non-arithmetic",
            "trigger does not cover variable",
        )
    ):
        return "contract_ill_formed"
    return "compile_error"


def _contract_expression_undefined(run: VerusRun) -> bool:
    """Whether Verus found an unmet recommendation in a concrete contract.

    Sequence indexing and narrowing/widening casts in specifications carry
    ``recommends`` obligations.  With concrete I/O bindings, an unmet
    recommendation means that the contract expression is not defined for that
    pair; it is more informative than an ordinary undecided proof.
    """
    return "recommendation not met" in str(run.stderr or "").lower()


def _hidden_spec_index_is_out_of_bounds(context: dict, case: dict) -> bool:
    """Detect a concrete OOB index hidden inside a directly called spec fn.

    Verus reports an unmet recommendation when an indexed expression occurs
    directly in the harness.  An open spec-function body can instead leave both
    proof directions undecided without surfacing that diagnostic.  This narrow
    check handles direct ``helper(sequence, index)`` calls whose helper body
    indexes the sequence by that index without guarding it by the sequence
    length.
    """
    preamble = str(context.get("spec_preamble") or "")
    bodies = _spec_fn_bodies(preamble)
    if not bodies:
        return False
    env = dict(case.get("inputs") or {})
    env.update(case.get("output") or {})
    for header in _SPEC_FN_HEADER_RE.finditer(preamble):
        name = header.group(1)
        body = bodies.get(name)
        if not body:
            continue
        params_open = preamble.find("(", header.end())
        if params_open < 0:
            continue
        params_close = find_matching_paren(preamble, params_open)
        if params_close is None:
            continue
        formal_names = [
            part.split(":", 1)[0].strip()
            for part in split_top_level_commas(preamble[params_open + 1 : params_close])
            if ":" in part
        ]
        for clause in context.get("ensures") or []:
            expression = _clause_expr_for_harness(clause)
            for call in re.finditer(rf"\b{re.escape(name)}\s*\(", expression):
                call_open = expression.find("(", call.start())
                call_close = find_matching_paren(expression, call_open)
                if call_close is None:
                    continue
                actuals = split_top_level_commas(expression[call_open + 1 : call_close])
                if len(actuals) != len(formal_names):
                    continue
                actual_by_formal = dict(zip(formal_names, (item.strip() for item in actuals)))
                for sequence_formal, sequence_actual in actual_by_formal.items():
                    sequence_value = env.get(sequence_actual)
                    if not isinstance(sequence_value, list):
                        continue
                    for index_formal, index_actual in actual_by_formal.items():
                        index_value = env.get(index_actual)
                        if isinstance(index_value, bool) or not isinstance(index_value, int):
                            continue
                        indexed = re.search(
                            rf"\b{re.escape(sequence_formal)}\s*@?\s*\[\s*[^\]]*\b{re.escape(index_formal)}\b[^\]]*\]",
                            body,
                        )
                        if indexed is None:
                            continue
                        length_expr = rf"\b{re.escape(sequence_formal)}\b\s*@?\s*\.\s*len\s*\(\s*\)"
                        guarded = bool(
                            re.search(rf"\b{re.escape(index_formal)}\b\s*<\s*{length_expr}", body)
                        )
                        if not guarded and not (0 <= index_value < len(sequence_value)):
                            return True
    return False


def _quantifier_trigger_witness_assertions(context: dict, case: dict) -> list[str]:
    """Candidate concrete trigger facts for a false quantified postcondition."""
    values = [
        value
        for value in [*(case.get("inputs") or {}).values(), *(case.get("output") or {}).values()]
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    assertions: list[str] = []
    seen: set[str] = set()
    for clause in context.get("ensures") or []:
        expression = _clause_expr_for_harness(clause).strip()
        match = re.match(r"^forall\s*\|(?P<binders>.*?)\|\s*(?P<body>.*)$", expression, re.DOTALL)
        if not match:
            continue
        binders = _quantifier_binders_with_types(match.group("binders"))
        if len(binders) != 1:
            continue
        binder, binder_type = binders[0]
        for trigger in re.finditer(
            r"#\s*\[\s*trigger\s*\]\s*(?P<call>[A-Za-z_]\w*\s*\([^()]*(?:\([^()]*\)[^()]*)*\))",
            match.group("body"),
        ):
            call = trigger.group("call")
            if not re.search(rf"\b{re.escape(binder)}\b", call):
                continue
            for value in values:
                literal = verus_literal(value, binder_type)
                instantiated = re.sub(rf"\b{re.escape(binder)}\b", literal, call)
                assertion = f"    assert({instantiated});"
                if assertion not in seen:
                    seen.add(assertion)
                    assertions.append(assertion)
                if len(assertions) >= 16:
                    return assertions
    return assertions


def _diagnose_contract_case(context: dict, case: dict, index: int) -> dict[str, Any]:
    """Diagnose one unresolved pair without weakening the dual-proof rule."""
    expression_undefined = False
    for negate_contract, direction in ((False, "accept"), (True, "reject")):
        item = _BatchItem(
            key=str(case["key"]),
            fn_name=f"__sqm_detail_{direction}_{index:04d}",
        )
        item.lines = _contract_check_proof_fn_lines(
            context,
            case.get("inputs") or {},
            case.get("output") or {},
            item.fn_name,
            negate_contract=negate_contract,
        )
        harness = _build_batch_harness(
            [item],
            spec_preamble=context.get("spec_preamble", ""),
            use_statements=context.get("use_statements") or (),
        )
        try:
            run = _run_verus_on_text(harness, f"contract_{direction}_detail.rs", timeout_seconds=15)
        except Exception:
            return {"accepted": None, "reason": "compile_error"}
        expression_undefined = expression_undefined or _contract_expression_undefined(run)
        parsed = _parse_batch_result(run, [item]).get(item.key)
        if parsed is True:
            return {
                "accepted": False if negate_contract else True,
                "reason": "contract_rejected" if negate_contract else "accepted",
            }
        if parsed is None:
            if expression_undefined:
                return {"accepted": None, "reason": "contract_expression_undefined"}
            return {"accepted": None, "reason": _unresolved_run_reason(run)}
        # A verification failure in one direction is not a decision. Try the
        # opposite direction before returning verification_unresolved.
    if expression_undefined or _hidden_spec_index_is_out_of_bounds(context, case):
        return {"accepted": None, "reason": "contract_expression_undefined"}

    # A quantified rejection can need one explicit ground trigger term.  Try
    # candidates independently so an inapplicable witness never poisons the
    # ordinary batch result.
    for assertion in _quantifier_trigger_witness_assertions(context, case):
        item = _BatchItem(
            key=str(case["key"]),
            fn_name=f"__sqm_detail_reject_witness_{index:04d}",
        )
        item.lines = _contract_check_proof_fn_lines(
            context,
            case.get("inputs") or {},
            case.get("output") or {},
            item.fn_name,
            negate_contract=True,
        )
        insertion = next(
            (position for position, line in enumerate(item.lines) if line.strip().startswith("assert(!")),
            len(item.lines) - 1,
        )
        item.lines.insert(insertion, assertion)
        harness = _build_batch_harness(
            [item],
            spec_preamble=context.get("spec_preamble", ""),
            use_statements=context.get("use_statements") or (),
        )
        try:
            run = _run_verus_on_text(harness, "contract_reject_witness.rs", timeout_seconds=15)
        except Exception:
            continue
        if _parse_batch_result(run, [item]).get(item.key) is True:
            return {"accepted": False, "reason": "contract_rejected"}
    return {"accepted": None, "reason": "verification_unresolved"}


# ---------------------------------------------------------------------------
# Batch verification infrastructure
# ---------------------------------------------------------------------------

# Maximum bisection depth for batch-poisoning recovery. A fully-poisoned batch
# (every case failing to compile) would otherwise fan out into a binary tree of
# up to ~2*N Verus runs, which can dominate the tail of a batch-mode run. Capping
# the depth bounds the worst case to ~2^(depth+1)-1 runs; once the cap is hit we
# stop subdividing and leave the still-unresolved cases as None. With depth 4 the
# common batch sizes (<=16) are still fully isolated, so healthy cases are only
# ever surrendered in the rare large-and-poisoned case.
_BATCH_BISECT_MAX_DEPTH = 4


@dataclass
class _BatchItem:
    key: str
    fn_name: str
    lines: list[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0


def _extend_physical_lines(file_lines: list[str], lines: Sequence[str]) -> None:
    """Append ``lines`` while splitting embedded newlines into physical lines.

    Clause texts copied verbatim from specifications frequently span several
    physical lines inside a single list element. Item line ranges are matched
    against Verus diagnostic line numbers, so they must count rendered lines,
    not list elements; otherwise every later item's range drifts and errors
    get attributed to neighbouring proof functions.
    """
    for text in lines:
        pieces = str(text).splitlines()
        file_lines.extend(pieces if pieces else [""])


def _build_batch_harness(
    items: list[_BatchItem],
    spec_preamble: str = "",
    use_statements: Sequence[str] = (),
) -> str:
    file_lines = ["use vstd::prelude::*;"]
    for stmt in use_statements:
        if stmt and stmt not in file_lines:
            file_lines.append(stmt)
    file_lines.append("")
    file_lines.append("verus! {")
    if spec_preamble:
        file_lines.append("")
        file_lines.extend(spec_preamble.splitlines())
    for item in items:
        file_lines.append("")
        item.start_line = len(file_lines) + 1
        _extend_physical_lines(file_lines, item.lines)
        item.end_line = len(file_lines)
    file_lines.extend(["", "} // verus!", ""])
    return "\n".join(file_lines)


def _parse_batch_result(
    run: VerusRun, items: list[_BatchItem],
) -> dict[str, Optional[bool]]:
    if run.success is True:
        return {item.key: True for item in items}
    if run.status in ("unavailable", "timeout"):
        return {item.key: None for item in items}

    stderr = run.stderr or ""
    verification_error_lines: set[int] = set()
    other_error_lines: set[int] = set()
    has_verification_error = False
    verification_diagnostic_count = 0

    for text_line in stderr.splitlines():
        text_line = text_line.strip()
        if not text_line.startswith("{"):
            continue
        try:
            diag = json.loads(text_line)
        except json.JSONDecodeError:
            continue
        if diag.get("$message_type") != "diagnostic" or diag.get("level") != "error":
            continue
        msg = diag.get("message", "")
        if "aborting due to" in msg:
            continue
        is_verification = (
            "postcondition not satisfied" in msg
            or "assertion failed" in msg
            or "arithmetic overflow" in msg
            or "arithmetic underflow" in msg
            or "underflow/overflow" in msg
        )
        if is_verification:
            has_verification_error = True
            verification_diagnostic_count += 1
        for span in diag.get("spans") or []:
            line_start = span.get("line_start")
            if isinstance(line_start, int):
                if is_verification:
                    verification_error_lines.add(line_start)
                else:
                    other_error_lines.add(line_start)

    if not has_verification_error:
        return {item.key: None for item in items}

    # Diagnostic-loss guard: Verus reports its verification error count in the
    # stdout summary, which run_verus parses before any output truncation. If
    # stderr carries fewer verification diagnostics than that count, part of
    # the diagnostics stream was lost (e.g. output capping), and an item
    # without visible errors can no longer be presumed proved. Returning
    # all-None lets the caller bisect instead of misreading missing
    # diagnostics as success.
    if run.errors is not None and verification_diagnostic_count < run.errors:
        return {item.key: None for item in items}

    all_error_lines = verification_error_lines | other_error_lines
    unmapped = all_error_lines.copy()
    result: dict[str, Optional[bool]] = {}
    for item in items:
        item_verif = {ln for ln in verification_error_lines if item.start_line <= ln <= item.end_line}
        item_other = {ln for ln in other_error_lines if item.start_line <= ln <= item.end_line}
        if item_verif:
            result[item.key] = False
        elif item_other:
            result[item.key] = None
        else:
            result[item.key] = True
        unmapped -= (item_verif | item_other)

    if unmapped:
        return {item.key: None for item in items}

    return result


def _run_and_parse_batch(
    items: list[_BatchItem],
    spec_preamble: str,
    filename: str,
    *,
    isolate: bool = True,
    depth: int = 0,
    use_statements: Sequence[str] = (),
) -> dict[str, Optional[bool]]:
    harness_text = _build_batch_harness(
        items, spec_preamble=spec_preamble, use_statements=use_statements,
    )
    timeout = min(120, 10 + len(items) // 2)
    try:
        run = _run_verus_on_text(harness_text, filename, timeout_seconds=timeout)
    except Exception:
        return {item.key: None for item in items}
    parsed = _parse_batch_result(run, items)

    # Batch poisoning recovery: a single case that fails to compile (or an error
    # line we cannot attribute to any case) makes _parse_batch_result return
    # all-None, discarding verdicts for every healthy sibling. Detect that
    # "poisoned" state and recover the good cases by bisecting and re-running.
    # We only do this on a hard failure (not on unavailable/timeout, where a
    # re-run cannot help, nor on full success) and only when at least one case
    # could still carry a real verdict.
    if not isolate or len(items) <= 1:
        return parsed
    if run.success is True or run.status in ("unavailable", "timeout"):
        return parsed
    if any(value is not None for value in parsed.values()):
        return parsed

    # Depth-bounded bisection: stop subdividing once the recursion cap is hit and
    # leave the remaining cases unresolved (parsed is already all-None here). This
    # clamps the worst-case tail of a pathologically poisoned batch instead of
    # descending all the way to singletons. See _BATCH_BISECT_MAX_DEPTH.
    if depth >= _BATCH_BISECT_MAX_DEPTH:
        return parsed

    mid = len(items) // 2
    left = _run_and_parse_batch(
        items[:mid], spec_preamble, filename, depth=depth + 1, use_statements=use_statements,
    )
    right = _run_and_parse_batch(
        items[mid:], spec_preamble, filename, depth=depth + 1, use_statements=use_statements,
    )
    return {**left, **right}


def batch_verus_contract_check(
    context: dict, cases: list[dict], *, string_facts: bool = False, return_params: bool = False,
) -> dict[str, Optional[bool]]:
    if not cases:
        return {}
    if not (context.get("ensures") or []):
        return {case["key"]: None for case in cases}

    items: list[_BatchItem] = []
    for i, case in enumerate(cases):
        fn_name = f"__sqm_batch_check_{i:04d}"
        fn_lines = _contract_check_proof_fn_lines(
            context, case["inputs"], case.get("output") or {}, fn_name,
            string_facts=string_facts, return_params=return_params,
        )
        items.append(_BatchItem(key=case["key"], fn_name=fn_name, lines=fn_lines))

    return _run_and_parse_batch(
        items, context.get("spec_preamble", ""), "batch_check.rs",
        use_statements=context.get("use_statements") or (),
    )


def batch_verus_contract_check_detailed(
    context: dict, cases: list[dict],
) -> dict[str, dict]:
    """Compatibility-preserving detailed view of contract verdicts.

    Unresolved batch members are re-run individually so timeout, unavailable,
    unsupported-type and ordinary compile failures are not collapsed into a
    single opaque ``None`` count.
    """
    raw = batch_verus_contract_check(context, cases)
    details: dict[str, dict] = {}
    unresolved: list[tuple[dict, _BatchItem]] = []
    for index, case in enumerate(cases):
        key = case["key"]
        accepted = raw.get(key)
        reason = (
            "accepted" if accepted is True
            else "verification_failed" if accepted is False
            else "verification_unresolved"
        )
        details[key] = {"accepted": accepted, "reason": reason, "engine": "verus_batch"}
        if accepted is None:
            fn_name = f"__sqm_detail_contract_{index:04d}"
            unresolved.append((case, _BatchItem(
                key=key,
                fn_name=fn_name,
                lines=_contract_check_proof_fn_lines(context, case["inputs"], case.get("output") or {}, fn_name),
            )))
    for detail_index, (case, item) in enumerate(unresolved):
        if detail_index >= 8:
            # Bound diagnostic overhead for large mutation batches. Remaining
            # cases retain the explicit verification_unresolved reason.
            continue
        harness = _build_batch_harness(
            [item],
            spec_preamble=context.get("spec_preamble", ""),
            use_statements=context.get("use_statements") or (),
        )
        try:
            run = _run_verus_on_text(harness, "contract_detail.rs", timeout_seconds=15)
        except Exception:
            details[case["key"]]["reason"] = "compile_error"
            continue
        parsed = _parse_batch_result(run, [item]).get(case["key"])
        if parsed is not None:
            details[case["key"]].update({
                "accepted": parsed,
                "reason": "accepted" if parsed else "verification_failed",
            })
            continue
        stderr = (run.stderr or "").lower()
        if run.status == "timeout":
            reason = "timeout"
        elif run.status == "unavailable":
            reason = "unavailable"
        elif any(token in stderr for token in ("not supported", "unsupported", "no method named", "trait bound")):
            reason = "unsupported_type"
        else:
            reason = "compile_error"
        details[case["key"]]["reason"] = reason
    return details


def batch_verus_contract_decide(
    context: dict, cases: list[dict],
) -> dict[str, Optional[bool]]:
    """Decide concrete contract acceptance with two independent proofs.

    ``False`` is returned only when Verus proves the negation of concrete-pair
    acceptance (preconditions and postconditions taken together). A failed
    acceptance proof alone is never treated as rejection.
    """
    if not cases:
        return {}
    if not (context.get("ensures") or []):
        # Without ensures clauses the postcondition is true, so accepting a
        # concrete pair reduces to accepting its input. Clauses that exist but
        # fail to transform keep their own unresolved path below.
        return batch_verus_requires_decide(context, cases)

    result: dict[str, Optional[bool]] = {case["key"]: None for case in cases}
    supported = [
        case
        for case in cases
        if _contract_case_support_issue(
            context,
            case.get("inputs") or {},
            case.get("output") or {},
            include_ensures=True,
        )
        is None
    ]
    if not supported:
        return result

    passes = [(False, False), (True, False)]
    if any(_return_param_type(ret.get("type", "")) for ret in context.get("returns") or []):
        # Seq/&str return literals can clash with clauses typed on Vec or
        # String; retry the remaining cases with parameter-bound returns.
        passes.append((False, True))
    for string_facts, return_params in passes:
        unresolved = [case for case in supported if result.get(case["key"]) is None]
        if not unresolved:
            break
        if string_facts and not any(
            _case_has_string_literal(context, case) for case in unresolved
        ):
            continue
        accepted_raw = batch_verus_contract_check(
            context, unresolved, string_facts=string_facts, return_params=return_params,
        )
        for case in unresolved:
            if accepted_raw.get(case["key"]) is True:
                result[case["key"]] = True
        still_unresolved = [case for case in unresolved if result.get(case["key"]) is None]
        if not still_unresolved:
            break
        reject_items: list[_BatchItem] = []
        for index, case in enumerate(still_unresolved):
            fn_name = f"__sqm_batch_reject_{index:04d}"
            reject_items.append(_BatchItem(
                key=case["key"],
                fn_name=fn_name,
                lines=_contract_check_proof_fn_lines(
                    context,
                    case["inputs"],
                    case.get("output") or {},
                    fn_name,
                    negate_contract=True,
                    string_facts=string_facts,
                    return_params=return_params,
                ),
            ))
        rejected_raw = _run_and_parse_batch(
            reject_items,
            context.get("spec_preamble", ""),
            "batch_reject.rs",
            use_statements=context.get("use_statements") or (),
        )
        for case in still_unresolved:
            if rejected_raw.get(case["key"]) is True:
                result[case["key"]] = False
    return result


def batch_verus_contract_decide_detailed(
    context: dict, cases: list[dict],
) -> dict[str, dict]:
    raw = batch_verus_contract_decide(context, cases)
    details: dict[str, dict] = {}
    unresolved: list[tuple[int, dict]] = []
    for index, case in enumerate(cases):
        key = case["key"]
        accepted = raw.get(key)
        has_postcondition = bool(context.get("ensures") or [])
        issue = _contract_case_support_issue(
            context,
            case.get("inputs") or {},
            case.get("output") or {},
            include_ensures=has_postcondition,
        )
        # A decision is authoritative. Support issues only explain undecided
        # cases, e.g. an unchecked type when a contract has no clauses at all.
        reason = (
            "accepted"
            if accepted is True
            else "contract_rejected"
            if accepted is False
            else str(issue["reason"])
            if issue
            else "verification_unresolved"
        )
        details[key] = {
            "accepted": accepted,
            "reason": reason,
            "engine": "verus_dual_proof",
        }
        if not has_postcondition:
            details[key]["empty_postcondition"] = True
        if accepted is None and issue and issue.get("detail"):
            details[key]["reason_detail"] = issue["detail"]
        if accepted is None and issue is None and has_postcondition:
            unresolved.append((index, case))

    # Diagnose one representative unresolved case, never every case. Batch
    # decisions remain the source of truth and this keeps the common all-unknown
    # path at O(1) extra Verus runs rather than degrading to per-case execution.
    if unresolved:
        index, case = unresolved[0]
        diagnosis = _diagnose_contract_case(context, case, index)
        details[case["key"]].update(diagnosis)
        if diagnosis.get("accepted") is None and diagnosis.get("reason") in {
            "compile_error", "contract_ill_formed", "timeout", "unavailable", "unsupported_type",
        }:
            for _, unresolved_case in unresolved[1:]:
                details[unresolved_case["key"]].update({
                    "accepted": None,
                    "reason": diagnosis["reason"],
                    "diagnostic_scope": "representative_unresolved_case",
                })
    return details


def batch_verus_requires_check(
    context: dict, input_cases: list[dict], *, string_facts: bool = False,
) -> dict[str, Optional[bool]]:
    if not input_cases:
        return {}
    if not (context.get("requires") or []):
        return {case["key"]: True for case in input_cases}

    items: list[_BatchItem] = []
    for i, case in enumerate(input_cases):
        fn_name = f"__sqm_batch_requires_{i:04d}"
        fn_lines = _requires_check_proof_fn_lines(
            context, case["inputs"], fn_name, string_facts=string_facts,
        )
        items.append(_BatchItem(key=case["key"], fn_name=fn_name, lines=fn_lines))

    return _run_and_parse_batch(
        items, context.get("spec_preamble", ""), "batch_requires.rs",
        use_statements=context.get("use_statements") or (),
    )


def batch_verus_requires_decide(
    context: dict, input_cases: list[dict],
) -> dict[str, Optional[bool]]:
    """Decide preconditions; proof failure is UNKNOWN, never implicit false."""
    if not input_cases:
        return {}
    if not (context.get("requires") or []):
        return {case["key"]: True for case in input_cases}

    result: dict[str, Optional[bool]] = {case["key"]: None for case in input_cases}
    supported = [
        case
        for case in input_cases
        if _contract_case_support_issue(
            context,
            case.get("inputs") or {},
            include_ensures=False,
        )
        is None
    ]
    if not supported:
        return result

    for string_facts in (False, True):
        unresolved = [case for case in supported if result.get(case["key"]) is None]
        if not unresolved:
            break
        if string_facts and not any(
            _case_has_string_literal(context, case) for case in unresolved
        ):
            break
        accepted_raw = batch_verus_requires_check(
            context, unresolved, string_facts=string_facts,
        )
        for case in unresolved:
            if accepted_raw.get(case["key"]) is True:
                result[case["key"]] = True
        still_unresolved = [case for case in unresolved if result.get(case["key"]) is None]
        if not still_unresolved:
            break
        reject_items: list[_BatchItem] = []
        for index, case in enumerate(still_unresolved):
            fn_name = f"__sqm_batch_requires_reject_{index:04d}"
            reject_items.append(_BatchItem(
                key=case["key"],
                fn_name=fn_name,
                lines=_requires_check_proof_fn_lines(
                    context,
                    case["inputs"],
                    fn_name,
                    negate_requires=True,
                    string_facts=string_facts,
                ),
            ))
        rejected_raw = _run_and_parse_batch(
            reject_items,
            context.get("spec_preamble", ""),
            "batch_requires_reject.rs",
            use_statements=context.get("use_statements") or (),
        )
        for case in still_unresolved:
            if rejected_raw.get(case["key"]) is True:
                result[case["key"]] = False
    return result


def batch_verus_requires_check_detailed(
    context: dict, input_cases: list[dict],
) -> dict[str, dict]:
    raw = batch_verus_requires_check(context, input_cases)
    return {
        case["key"]: {
            "accepted": raw.get(case["key"]),
            "reason": (
                "accepted" if raw.get(case["key"]) is True
                else "verification_failed" if raw.get(case["key"]) is False
                else "verification_unresolved"
            ),
            "engine": "verus_batch",
        }
        for case in input_cases
    }


def batch_strict_requires_check(
    context: dict, input_list: list[dict],
) -> list[dict]:
    if not input_list:
        return []
    cases = [{"inputs": item["inputs"], "key": item["key"]} for item in input_list]
    raw = batch_verus_requires_check(context, cases)
    results = []
    for item in input_list:
        accepted = raw.get(item["key"])
        results.append({
            "accepted": accepted,
            "engine": "verus_batch",
            "reason": "ok" if accepted is not None else "unsupported_expression",
        })
    return results


def batch_contract_check(
    context: dict, cases: list[dict],
) -> list[dict]:
    if not cases:
        return []
    raw = batch_verus_contract_check(context, cases)
    results = []
    for case in cases:
        accepted = raw.get(case["key"])
        results.append({
            "accepted": accepted,
            "requires_ok": None,
            "ensures_ok": None,
            "engine": "verus_batch",
            "reason": "accepted" if accepted is True else "not_accepted" if accepted is False else "unsupported_expression",
        })
    return results


def io_harness_for_case(context: dict, case: dict, kind: str) -> str:
    params = list(context.get("parameters") or [])
    returns = list(context.get("returns") or [])
    output = case.get("mutated_output") if kind == "negative" else case.get("output")
    output = output or {}
    vector_params = [
        param for param in params
        if is_vec_type(param.get("type", "")) and isinstance(case.get("inputs", {}).get(param.get("name")), list)
    ]
    scalar_params = [param for param in params if param not in vector_params]
    signature_params = [
        f"{param.get('name')}: {param.get('type', '')}"
        for param in vector_params
        if param.get("name") and param.get("type")
    ]
    raw_name = f"__sqm_{kind}_{case.get('id', 'case')}_{context.get('function', 'target')}"
    harness_name = re.sub(r"[^A-Za-z0-9_]", "_", raw_name)
    lines = ["// Temporary Verus harness sketch generated from this concrete test case."]
    if kind == "positive":
        lines.append("// Expected result: all assertions should verify.")
    elif kind == "negative":
        lines.append("// Expected result: at least one postcondition assertion should fail.")
    elif kind == "invalid":
        lines.append("// Expected result: at least one precondition assertion should fail.")
    else:
        lines.append("// Expected result: this harness explains the behavior tags for the case.")
    lines.append(f"proof fn {harness_name}({', '.join(signature_params)})")
    fixed_clauses: list[str] = []
    for param in vector_params:
        input_val = case.get("inputs", {}).get(param.get("name"))
        if input_val is None:
            input_val = []
        fixed_clauses.extend(
            _fixed_sequence_clauses(param.get("name", "input"), param.get("type", ""), input_val)
        )
    if fixed_clauses:
        lines.append("    requires")
        for index, clause in enumerate(fixed_clauses):
            comma = "," if index < len(fixed_clauses) - 1 else ""
            lines.append(f"        {clause}{comma}")
    lines.append("{")
    for param in scalar_params:
        name = param.get("name")
        if not name or name not in case.get("inputs", {}):
            continue
        type_text = param.get("type", "")
        value = case["inputs"][name]
        lines.append(f"    let {name}: {type_text} = {verus_literal(value, type_text)};")
    for ret in returns:
        name = ret.get("name") or "ret"
        if name not in output:
            continue
        type_text = ret.get("type", "")
        value = output[name]
        if isinstance(value, list) and is_vec_type(type_text):
            seq_type = _extract_vec_element_type(type_text)
            lines.append(f"    let {name}: Seq<{seq_type}> = {verus_literal(value, type_text)};")
        else:
            lines.append(f"    let {name}: {type_text} = {verus_literal(value, type_text)};")
    if context.get("requires"):
        lines.append("")
        lines.append("    // Preconditions checked against the concrete input.")
        for clause in context.get("requires") or []:
            expr = _clause_expr_for_harness(clause)
            if expr:
                lines.append(f"    assert({expr});")
    if kind != "invalid" and context.get("ensures"):
        lines.append("")
        lines.append("    // Postconditions checked against the concrete output state.")
        for clause in context.get("ensures") or []:
            expr = _clause_expr_for_harness(clause)
            if expr:
                lines.append(f"    assert({expr});")
    lines.append("}")
    return "\n".join(lines)


__all__ = [
    "base_type_name",
    "batch_contract_check",
    "batch_strict_requires_check",
    "batch_verus_contract_check",
    "batch_verus_contract_decide",
    "batch_verus_contract_decide_detailed",
    "batch_verus_contract_check_detailed",
    "batch_verus_requires_check",
    "batch_verus_requires_decide",
    "batch_verus_requires_check_detailed",
    "coerce_value_for_type",
    "contract_evaluation",
    "eval_contract_expr",
    "io_harness_for_case",
    "is_bool_type",
    "is_char_type",
    "is_float_type",
    "is_int_like_type",
    "is_nested_vec_type",
    "is_unsigned_type",
    "is_vec_type",
    "normalize_type_key",
    "normalize_value_type",
    "typed_input_payload",
    "typed_output_payload",
    "verus_literal",
]
