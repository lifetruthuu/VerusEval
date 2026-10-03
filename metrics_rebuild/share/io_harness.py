#!/usr/bin/env python3
"""IO 测试 harness 的共享工具：函数签名解析 + 值格式化 + harness 输出解析.

这些符号被以下模块复用：
- scripts/io/generate_io_tests_llm.py（LLM 驱动 + verus --no-verify --compile 后端）
- test/test_test_io.py

历史上它们曾定义在已删除的 scripts/generate_io_tests.py（strip Verus + rustc 后端）里，
随该脚本被 verus --compile 路径取代而抽取到此处，作为唯一的公共实现。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from metrics_rebuild.share.text import strip_comments

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ParamInfo:
    name: str
    rust_type: str
    is_ref: bool = False
    is_mut_ref: bool = False
    is_slice: bool = False
    inner_type: str = ""  # e.g. "i32" for Vec<i32>


@dataclass
class FuncInfo:
    name: str
    params: List[ParamInfo]
    return_type: str
    return_name: str = "result"
    has_ghost_types: bool = False
    helper_fns: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 1: Parse function signatures
# ---------------------------------------------------------------------------

_RUST_PRIMITIVE_RUNTIME_TYPES = {
    "bool", "char", "String", "str", "&str", "'static str", "&'static str",
    "i8", "i16", "i32", "i64", "i128", "isize",
    "u8", "u16", "u32", "u64", "u128", "usize",
    "f32", "f64", "int", "nat",
}


_ARRAY_TYPE_RE = re.compile(r"^\[\s*(.+?)\s*;\s*(\d+)\s*\]$")


def array_type_parts(type_text: str) -> Optional[Tuple[str, int]]:
    """`[T; N]` -> (T, N)；非定长数组返回 None。引用前缀先剥掉。"""
    t = str(type_text).strip()
    if t.startswith("&"):
        t = t[1:].strip()
        if t.startswith("mut "):
            t = t[4:].strip()
    match = _ARRAY_TYPE_RE.match(t)
    if not match:
        return None
    return match.group(1).strip(), int(match.group(2))


def parse_type(raw: str) -> Tuple[str, bool, bool, bool, str]:
    """Parse a Rust type string. Returns (type, is_ref, is_mut_ref, is_slice, inner_type)."""
    raw = raw.strip()
    is_ref = False
    is_mut_ref = False
    is_slice = False
    inner = ""

    if raw.startswith("&mut "):
        is_ref = True
        is_mut_ref = True
        raw = raw[5:].strip()
    elif raw.startswith("&"):
        is_ref = True
        raw = raw[1:].strip()

    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        prefix = "&mut " if is_mut_ref else "&" if is_ref else ""
        array = array_type_parts(raw)
        if array is not None:
            # 定长数组不是 slice：调用侧传 &var 而非 &var[..]。
            return (f"{prefix}[{inner}]", is_ref, is_mut_ref, False, array[0])
        return (f"{prefix}[{inner}]", is_ref, is_mut_ref, True, inner)

    m = re.match(r'Vec\s*<\s*(.+)\s*>', raw)
    if m:
        inner = m.group(1).strip()
        base = f"Vec<{inner}>"
        prefix = "&mut " if is_mut_ref else "&" if is_ref else ""
        return (f"{prefix}{base}", is_ref, is_mut_ref, False, inner)

    prefix = "&mut " if is_mut_ref else "&" if is_ref else ""
    return (f"{prefix}{raw}", is_ref, is_mut_ref, False, inner)


def _extract_verus_inner(code: str) -> str:
    vm = re.search(r'verus!\s*\{(.*)\}\s*//\s*verus!', code, re.DOTALL)
    if not vm:
        vm = re.search(r'verus!\s*\{(.*)\}', code, re.DOTALL)
    return vm.group(1) if vm else code


def _find_matching(text: str, start: int, open_ch: str, close_ch: str) -> int:
    if start >= len(text) or text[start] != open_ch:
        return -1
    depth = 1
    i = start + 1
    while i < len(text):
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _read_type_until_boundary(text: str, start: int) -> tuple[str, int]:
    depth_angle = 0
    depth_paren = 0
    depth_bracket = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == '<':
            depth_angle += 1
        elif ch == '>':
            if depth_angle == 0:
                break
            depth_angle -= 1
        elif ch == '(':
            depth_paren += 1
        elif ch == ')':
            if depth_paren == 0:
                break
            depth_paren -= 1
        elif ch == '[':
            depth_bracket += 1
        elif ch == ']':
            if depth_bracket == 0:
                break
            depth_bracket -= 1
        elif depth_angle == 0 and depth_paren == 0 and depth_bracket == 0:
            if ch == '{':
                break
            if ch.isspace():
                lookahead = text[i:].lstrip()
                if re.match(r'^(?:requires|ensures|decreases|recommends|opens_invariants|no_unwind)\b', lookahead):
                    break
        i += 1
    return text[start:i].strip().rstrip(",").strip(), i


def _split_top_level_colon(text: str) -> Optional[int]:
    depth_angle = 0
    depth_paren = 0
    depth_bracket = 0
    for i, ch in enumerate(text):
        if ch == '<':
            depth_angle += 1
        elif ch == '>':
            depth_angle -= 1
        elif ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren -= 1
        elif ch == '[':
            depth_bracket += 1
        elif ch == ']':
            depth_bracket -= 1
        elif ch == ':' and depth_angle == 0 and depth_paren == 0 and depth_bracket == 0:
            return i
    return None


def _parse_return_signature(text: str, arrow_pos: int) -> tuple[str, str]:
    i = arrow_pos + 2
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text):
        return "result", "()"

    if text[i] == '(':
        close = _find_matching(text, i, "(", ")")
        if close < 0:
            return "result", "()"
        inner = text[i + 1:close].strip()
        colon = _split_top_level_colon(inner)
        if colon is not None:
            name = inner[:colon].strip()
            type_text = inner[colon + 1:].strip().rstrip(",").strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) and type_text:
                return name, type_text
        return "result", text[i:close + 1].strip()

    type_text, _end = _read_type_until_boundary(text, i)
    return "result", type_text or "()"


def parse_params(params_str: str) -> List[ParamInfo]:
    """Parse function parameter list."""
    params = []
    if not params_str.strip():
        return params

    depth = 0
    current = ""
    for ch in params_str:
        if ch in '<(':
            depth += 1
        elif ch in '>)':
            depth -= 1
        if ch == ',' and depth == 0:
            if current.strip():
                params.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        params.append(current.strip())

    result = []
    for p in params:
        p = p.strip()
        if ':' not in p:
            continue
        name, type_str = p.split(':', 1)
        name = name.strip()
        type_str = type_str.strip()
        ty, is_ref, is_mut_ref, is_slice, inner = parse_type(type_str)
        result.append(ParamInfo(name=name, rust_type=ty, is_ref=is_ref, is_mut_ref=is_mut_ref, is_slice=is_slice, inner_type=inner))
    return result


def _skip_optional_generics(text: str, start: int) -> int:
    """Skip an optional `<...>` generic parameter list after a function name.

    Needed for signatures like ``fn copy<T: Copy>(...)`` where generics sit
    between the name and the parameter list.  Returns the index of the first
    non-space character after the generics (or ``start`` when absent).
    """
    cursor = start
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] != "<":
        return cursor
    close = _find_matching(text, cursor, "<", ">")
    if close < 0:
        return start
    cursor = close + 1
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor


def _iter_exec_fn_matches(text: str):
    """Yield ``(match, name, params_open_index)`` for executable ``fn`` headers.

    Accepts optional generic parameters between the name and ``(``.
    """
    fn_re = re.compile(r"(?:pub\s+)?fn\s+(\w+)\b", re.DOTALL)
    for m in fn_re.finditer(text):
        params_start = _skip_optional_generics(text, m.end())
        if params_start >= len(text) or text[params_start] != "(":
            continue
        yield m, m.group(1), params_start


def _iter_function_headers(inner: str):
    for m, fn_name, params_start in _iter_exec_fn_matches(inner):
        params_end = _find_matching(inner, params_start, "(", ")")
        if params_end < 0:
            continue
        cursor = params_end + 1
        while cursor < len(inner) and inner[cursor].isspace():
            cursor += 1
        ret_name = "result"
        ret_type = "()"
        if inner.startswith("->", cursor):
            ret_name, ret_type = _parse_return_signature(inner, cursor)
        yield m, fn_name, inner[params_start + 1:params_end], ret_name, ret_type


def preferred_io_function_name(code: str) -> Optional[str]:
    """Return the explicitly marked benchmark target, when present.

    VeriCoding files may place executable helpers before the function under
    ``// <vc-spec>``.  Selecting the first executable function therefore
    evaluates the helper instead of the benchmark target.
    """
    marker = re.search(
        r"//\s*<vc-spec>\s*(.*?)//\s*</vc-spec>",
        code,
        re.DOTALL,
    )
    if marker is None:
        return None
    segment = strip_comments(marker.group(1))
    for match, fn_name, _params_start in _iter_exec_fn_matches(segment):
        prefix = segment[max(0, match.start() - 30):match.start()].split()[-3:]
        if not any(word in {"spec", "proof"} for word in prefix):
            return fn_name
    return None


def extract_function(code: str, preferred_name: Optional[str] = None) -> Optional[FuncInfo]:
    """Extract the primary executable function from a .rs file.

    ``preferred_name`` is optional for backwards compatibility.  When it
    names an executable function, that function wins over the historical
    "first suitable function" heuristic.
    """
    inner = strip_comments(_extract_verus_inner(code))

    best = None
    for m, fn_name, params_str, ret_name, ret_type in _iter_function_headers(inner):
        if fn_name == "main":
            continue

        # Check if this fn is preceded by spec/proof keywords
        # Look at the ~30 chars immediately before this match
        prefix_start = max(0, m.start() - 30)
        prefix_text = inner[prefix_start:m.start()].strip()
        # Get the last few words before "fn"
        prefix_words = prefix_text.split()[-3:] if prefix_text else []
        is_spec_or_proof = any(w in ("spec", "proof") for w in prefix_words)
        if is_spec_or_proof:
            continue

        params = parse_params(params_str)

        has_ghost = any(
            p.rust_type in ("int", "nat") or "Vec<int>" in p.rust_type or "Vec<nat>" in p.rust_type
            for p in params
        ) or ret_type in ("int", "nat") or "Vec<int>" in ret_type or "Vec<nat>" in ret_type

        helper_fns = []
        for hm, hname, _ in _iter_exec_fn_matches(inner):
            if hname != fn_name and hname != "main":
                hp = inner[:hm.start()].rstrip()
                hl = hp[hp.rfind('\n')+1:].strip()
                if "spec fn" not in hl and "proof fn" not in hl:
                    helper_fns.append(hname)

        info = FuncInfo(
            name=fn_name, params=params, return_type=ret_type,
            return_name=ret_name, has_ghost_types=has_ghost,
            helper_fns=helper_fns
        )

        if preferred_name and fn_name == preferred_name:
            return info
        if params and ret_type != "()":
            if best is None:
                best = info
            if not preferred_name:
                break
        if best is None:
            best = info

    return best


def _strip_runtime_ref(type_text: str) -> str:
    t = str(type_text).strip()
    if t.startswith("&"):
        rest = t[1:].strip()
        if rest.startswith("mut "):
            rest = rest[4:].strip()
        if rest in ("str", "'static str"):
            return "&" + rest
        return rest
    return t


def _extract_generic_inner_runtime(t: str, prefix: str) -> str:
    m = re.match(re.escape(prefix) + r'\s*<\s*', t)
    if not m:
        return ""
    start = m.end()
    depth = 1
    i = start
    while i < len(t) and depth > 0:
        if t[i] == '<':
            depth += 1
        elif t[i] == '>':
            depth -= 1
        i += 1
    if depth != 0:
        return ""
    return t[start:i - 1].strip()


def _split_tuple_runtime_types(t: str) -> list[str]:
    inner = t[1:-1] if t.startswith("(") and t.endswith(")") else t
    parts: list[str] = []
    depth_angle = 0
    depth_paren = 0
    depth_bracket = 0
    cur = ""
    for ch in inner:
        if ch == '<':
            depth_angle += 1
            cur += ch
        elif ch == '>':
            depth_angle -= 1
            cur += ch
        elif ch == '(':
            depth_paren += 1
            cur += ch
        elif ch == ')':
            depth_paren -= 1
            cur += ch
        elif ch == '[':
            depth_bracket += 1
            cur += ch
        elif ch == ']':
            depth_bracket -= 1
            cur += ch
        elif ch == "," and depth_angle == 0 and depth_paren == 0 and depth_bracket == 0:
            if cur.strip():
                parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def runtime_type_support_issue(type_text: str, registry: Optional[dict] = None) -> Optional[str]:
    """Return why a native runtime IO type is unsupported, or None when supported.

    ``registry`` 为文件内 struct/enum/alias 定义（见 type_defs.parse_type_definitions）；
    注册过的类型按其字段/variant/别名目标递归判定。
    """
    if registry:
        from metrics_rebuild.share.type_defs import resolve_alias_text
        type_text = resolve_alias_text(registry, str(type_text))
    t = _strip_runtime_ref(type_text)
    if not t or t == "()":
        return None
    array = array_type_parts(t)
    if array is not None:
        return runtime_type_support_issue(array[0], registry)
    if t.startswith("[") and t.endswith("]"):
        return runtime_type_support_issue(t[1:-1].strip(), registry)
    if t in _RUST_PRIMITIVE_RUNTIME_TYPES:
        return None
    if t.startswith("Vec<"):
        inner = _extract_generic_inner_runtime(t, "Vec")
        return runtime_type_support_issue(inner, registry) if inner else "unsupported_generic_type"
    if t.startswith("Seq<"):
        # Seq 是 Verus 数学类型：定义能过编译，但 exec harness 无法构造其值。
        return "verus_math_runtime_type"
    if t.startswith("Option<"):
        inner = _extract_generic_inner_runtime(t, "Option")
        return runtime_type_support_issue(inner, registry) if inner else "unsupported_generic_type"
    if t.startswith("(") and t.endswith(")"):
        for elem_type in _split_tuple_runtime_types(t):
            issue = runtime_type_support_issue(elem_type, registry)
            if issue:
                return issue
        return None
    if re.match(r"^[A-Za-z_][A-Za-z0-9_:]*\s*<", t):
        return "unsupported_generic_type"
    if re.fullmatch(r"[A-Z]", t):
        # 单字母泛型参数：harness 会单态化成 i32 运行（见 build_verus_harness）。
        return None
    if registry:
        entry = registry.get(t)
        if entry is not None:
            if entry.get("kind") == "enum":
                return None
            for _field_name, field_type in entry.get("fields") or []:
                issue = runtime_type_support_issue(field_type, registry)
                if issue:
                    return issue
            return None
    return "custom_runtime_type"


# ---------------------------------------------------------------------------
# Value formatting: Python value -> Rust literal
# ---------------------------------------------------------------------------


def _rust_string_literal(val: Any) -> str:
    """把任意值转成合法的 Rust 字符串字面量（含转义），如 "a\\"b"."""
    s = str(val)
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return f'"{s}"'


def _default_value_for_type(ty: str) -> Any:
    ty = ty.replace("&mut ", "").replace("&", "").strip()
    if ty == "bool":
        return False
    if ty == "char":
        return "x"
    if ty in ("String", "str", "'static str"):
        return ""
    if ty.startswith("Vec<") or ty.startswith("["):
        return []
    if ty.startswith("Option<"):
        return None
    if ty.startswith("(") and ty.endswith(")"):
        return [_default_value_for_type(t) for t in _split_tuple_runtime_types(ty)]
    if ty in ("f32", "f64"):
        return 0.0
    return 0


def format_value_rust(val: Any, ty: str, registry: Optional[dict] = None) -> str:
    """Format a Python value as a Rust literal.

    ``registry`` 支持文件内 struct/enum/alias：enum 值为 variant 名字符串，
    struct 值为按字段顺序的列表，alias 解析为底层类型格式化。
    """
    if registry:
        from metrics_rebuild.share.type_defs import resolve_alias_text
        ty = resolve_alias_text(registry, str(ty))
    ty = ty.replace("&", "").strip()

    if ty.startswith("Option<"):
        inner = _extract_generic_inner_runtime(ty, "Option")
        if val is None:
            return "None"
        return f"Some({format_value_rust(val, inner, registry)})"

    if ty.startswith("(") and ty.endswith(")"):
        elem_types = _split_tuple_runtime_types(ty)
        values = list(val) if isinstance(val, (list, tuple)) else []
        while len(values) < len(elem_types):
            values.append(_default_value_for_type(elem_types[len(values)]))
        parts = [
            format_value_rust(value, elem_type, registry)
            for value, elem_type in zip(values[:len(elem_types)], elem_types)
        ]
        if len(parts) == 1:
            return f"({parts[0]},)"
        return f"({', '.join(parts)})"

    if ty == "bool" or isinstance(val, bool):
        return "true" if val else "false"

    if ty == "int":
        return f"{int(val)}int"

    if ty in ("i32", "i64", "i8", "i16", "i128", "isize"):
        real_ty = ty
        v = int(val)
        # Clamp to type range to avoid literal overflow
        type_ranges = {
            "i8": (-128, 127),
            "i16": (-32768, 32767),
            "i32": (-2147483648, 2147483647),
            "i64": (-2**63, 2**63-1),
            "i128": (-2**127, 2**127-1),
            "isize": (-2**63, 2**63-1),
        }
        lo, hi = type_ranges[real_ty]
        v = max(lo, min(hi, v))
        if v == lo:
            # Rust can't parse i32::MIN directly as literal, use wrapping
            return f"{real_ty}::MIN"
        return f"{v}_{real_ty}"

    if ty == "nat":
        return f"{max(0, int(val))}nat"

    if ty in ("u32", "u64", "u8", "u16", "usize", "u128"):
        real_ty = ty
        v = max(0, int(val))
        type_ranges = {"u8": 255, "u16": 65535, "u32": 2**32-1, "u64": 2**64-1, "usize": 2**64-1, "u128": 2**128-1}
        v = min(v, type_ranges.get(real_ty, 2**64-1))
        return f"{v}_{real_ty}"

    if ty in ("f32", "f64"):
        try:
            v = float(val)  # 也接受 "inf"/"-inf"/"NaN" 等字符串形式
        except (TypeError, ValueError):
            v = 0.0
        # 非有限值用常量表达，不再静默钳成 0.0（参考实现可合法返回无穷/NaN）。
        if math.isnan(v):
            return f"{ty}::NAN"
        if math.isinf(v):
            return f"{ty}::INFINITY" if v > 0 else f"{ty}::NEG_INFINITY"
        limit = 3.4e38 if ty == "f32" else 1.7e308
        v = max(-limit, min(limit, v))
        return f"{v}_{ty}"

    if ty == "char":
        if isinstance(val, bool):
            c = 'x'
        elif isinstance(val, int):
            try:
                c = chr(val)
            except (ValueError, OverflowError):
                c = 'x'
        else:
            c = str(val)
        if len(c) != 1:
            c = 'x'  # fallback for invalid chars
        # 转义会破坏字符字面量的特殊字符：反斜杠、单引号、换行等控制符。
        # 否则 verus --compile 会报 E0762 / "character constant must be escaped"，
        # 导致整题 harness 编译失败、no_valid_cases。
        code = ord(c)
        if c == '\\':
            return r"'\\'"
        if c == "'":
            return r"'\''"
        if c == '\n':
            return r"'\n'"
        if c == '\r':
            return r"'\r'"
        if c == '\t':
            return r"'\t'"
        if c == '\0':
            return r"'\0'"
        if code < 0x20 or code == 0x7f:
            return f"'\\u{{{code:x}}}'"
        return f"'{c}'"

    if ty == "String":
        return f'String::from({_rust_string_literal(val)})'

    # str / &str / 'static str（函数开头已剥离 &）
    if ty in ("str", "'static str"):
        return _rust_string_literal(val)

    if ty.startswith("Vec<Vec<"):
        inner = _extract_generic_inner_runtime(ty, "Vec")
        it = _extract_generic_inner_runtime(inner, "Vec") if inner.startswith("Vec<") else "i32"
        if not isinstance(val, list):
            val = []
        parts = []
        for sublist in val:
            if not isinstance(sublist, list):
                sublist = [sublist]
            elems = ", ".join(format_value_rust(e, it) for e in sublist)
            parts.append(f"vec![{elems}]")
        return f"vec![{', '.join(parts)}]"

    array = array_type_parts(ty)
    if array is not None:
        elem_type, length = array
        values = list(val) if isinstance(val, list) else [val]
        # 定长数组必须恰好 N 个元素；截断/补默认值保证可编译。
        values = values[:length]
        while len(values) < length:
            values.append(_default_value_for_type(elem_type))
        elems = ", ".join(format_value_rust(e, elem_type, registry) for e in values)
        return f"[{elems}]"

    if ty.startswith("Vec<") or ty.startswith("["):
        it = _extract_generic_inner_runtime(ty, "Vec") if ty.startswith("Vec<") else ty[1:-1].strip()
        if not isinstance(val, list):
            val = [val]
        elems = ", ".join(format_value_rust(e, it, registry) for e in val)
        return f"vec![{elems}]"

    if registry:
        entry = registry.get(ty)
        if entry is not None:
            if entry.get("kind") == "enum":
                variants = entry.get("variants") or []
                variant = val if isinstance(val, str) and val in variants else (variants[0] if variants else "")
                return f"{ty}::{variant}"
            fields = entry.get("fields") or []
            values = list(val) if isinstance(val, (list, tuple)) else []
            while len(values) < len(fields):
                values.append(_default_value_for_type(fields[len(values)][1]))
            if entry.get("tuple"):
                parts = [
                    format_value_rust(value, field_type, registry)
                    for value, (_fname, field_type) in zip(values, fields)
                ]
                return f"{ty}({', '.join(parts)})"
            parts = [
                f"{fname}: {format_value_rust(value, field_type, registry)}"
                for value, (fname, field_type) in zip(values, fields)
            ]
            return f"{ty} {{ {', '.join(parts)} }}"

    return str(val)


# ---------------------------------------------------------------------------
# Value formatting: Python value -> JSON (VeriScale-compatible)
# ---------------------------------------------------------------------------


def format_value_json(val: Any, ty: str) -> Any:
    """Format a Python value for JSON (VeriScale-compatible)."""
    ty = ty.replace("&", "").strip()

    if ty == "bool" or isinstance(val, bool):
        return val

    if ty in ("i32", "i64", "i8", "i16", "u32", "u64", "u8", "u16", "usize", "int", "nat"):
        return val if isinstance(val, int) else int(val) if str(val).lstrip('-').isdigit() else val

    if ty == "char":
        if isinstance(val, int) and not isinstance(val, bool):
            try:
                return chr(val)
            except (ValueError, OverflowError):
                return str(val)
        return str(val)

    if ty == "String":
        return str(val)

    if isinstance(val, list):
        return str(val)

    return val


def format_input_json(inp: Dict[str, Any], func: FuncInfo) -> Dict[str, Any]:
    """Format function inputs for JSON output."""
    result = {}
    for p in func.params:
        v = inp.get(p.name)
        result[p.name] = format_value_json(v, p.rust_type)
    return result


# ---------------------------------------------------------------------------
# Harness output parsing
# ---------------------------------------------------------------------------


def parse_results(stdout: str) -> Dict[int, Tuple[str, str]]:
    """Parse harness output lines. Returns {index: (status, value)}."""
    results = {}
    for line in stdout.strip().split('\n'):
        line = line.strip()
        if not line.startswith("RESULT:"):
            continue
        parts = line.split(':', 3)
        if len(parts) < 3:
            continue
        idx = int(parts[1])
        status = parts[2]
        value = parts[3] if len(parts) > 3 else ""
        results[idx] = (status, value)
    return results
