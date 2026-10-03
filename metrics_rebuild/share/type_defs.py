"""解析参考文件内定义的 struct/enum,为 IO harness 提供构造与格式化依据。

只接受 harness 能忠实构造/比较的形态:
- enum:所有 variant 均为 unit variant(无字段);
- struct:命名字段或 tuple struct,字段类型受 harness 支持(含嵌套注册类型)。

值的 JSON 表示约定:
- enum 值 = variant 名字符串,如 ``"Int8"``;
- struct 值 = 按字段声明顺序的列表,如 ``[3, [1, 2]]``。
"""

from __future__ import annotations

import re
from typing import Any, Optional

from metrics_rebuild.share.text import strip_comments

_ENUM_RE = re.compile(r"\benum\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{")
_STRUCT_RE = re.compile(r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\{|\()")
_ALIAS_RE = re.compile(r"\btype\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);")


def _find_matching(text: str, start: int, open_ch: str, close_ch: str) -> int:
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


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in text:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == sep and depth == 0:
            if cur.strip():
                parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _strip_attributes(text: str) -> str:
    return re.sub(r"#\s*\[[^\]]*\]", " ", text)


def parse_type_definitions(code: str) -> dict[str, dict]:
    """从源码提取文件内 struct/enum 定义。

    返回 {name: {"kind": "enum", "variants": [...]}
          | {"kind": "struct", "fields": [(name, type), ...], "tuple": bool}}。
    不合形态(带字段的 enum variant、泛型类型定义等)不会进入注册表。
    """
    clean = strip_comments(code)
    registry: dict[str, dict] = {}

    for match in _ALIAS_RE.finditer(clean):
        name, target = match.group(1), match.group(2).strip()
        if target and "<" not in name:
            registry[name] = {"kind": "alias", "target": target}

    for match in _ENUM_RE.finditer(clean):
        name = match.group(1)
        body_open = match.end() - 1
        body_close = _find_matching(clean, body_open, "{", "}")
        if body_close < 0:
            continue
        body = _strip_attributes(clean[body_open + 1:body_close])
        variants: list[str] = []
        ok = True
        for part in _split_top_level(body):
            variant = part.strip()
            if not variant:
                continue
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variant):
                ok = False  # 带字段/判别值的 variant,不支持
                break
            variants.append(variant)
        if ok and variants:
            registry[name] = {"kind": "enum", "variants": variants}

    for match in _STRUCT_RE.finditer(clean):
        name = match.group(1)
        opener = match.group(2)
        body_open = match.end() - 1
        closer = "}" if opener == "{" else ")"
        body_close = _find_matching(clean, body_open, opener, closer)
        if body_close < 0:
            continue
        body = _strip_attributes(clean[body_open + 1:body_close])
        fields: list[tuple[str, str]] = []
        ok = True
        if opener == "{":
            for part in _split_top_level(body):
                part = re.sub(r"\bpub(?:\s*\([^)]*\))?\s+", "", part).strip()
                if not part:
                    continue
                if ":" not in part:
                    ok = False
                    break
                field_name, field_type = part.split(":", 1)
                field_name = field_name.strip()
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field_name):
                    ok = False
                    break
                fields.append((field_name, field_type.strip()))
        else:
            for index, part in enumerate(_split_top_level(body)):
                part = re.sub(r"\bpub(?:\s*\([^)]*\))?\s+", "", part).strip()
                if not part:
                    continue
                fields.append((str(index), part))
        if ok and fields:
            registry[name] = {
                "kind": "struct",
                "fields": fields,
                "tuple": opener == "(",
            }

    return registry


def registry_entry(registry: Optional[dict], type_text: str) -> Optional[dict]:
    """按类型名查注册表;剥引用,拒绝带泛型参数的用法。alias 会解析到底层定义。"""
    if not registry:
        return None
    t = str(type_text).strip()
    t = re.sub(r"^&\s*(?:'[A-Za-z_][A-Za-z0-9_]*\s*)?(?:mut\s+)?", "", t)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t):
        return None
    entry = registry.get(t)
    for _ in range(4):
        if entry is None or entry.get("kind") != "alias":
            break
        target = str(entry.get("target") or "").strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", target):
            return None  # alias 指向复合类型:调用方应先做 resolve_alias_text
        entry = registry.get(target)
    return entry


def resolve_alias_text(registry: Optional[dict], type_text: str) -> str:
    """把类型文本中出现的 alias 名替换为其目标类型(含嵌套,如 Vec<Matrix>)。"""
    if not registry:
        return type_text
    text = str(type_text)
    for _ in range(4):
        changed = False
        for name, entry in registry.items():
            if entry.get("kind") != "alias":
                continue
            new_text = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
                str(entry.get("target") or "").strip(),
                text,
            )
            if new_text != text:
                text = new_text
                changed = True
        if not changed:
            break
    return text


def registry_type_name(type_text: str) -> str:
    t = str(type_text).strip()
    return re.sub(r"^&\s*(?:'[A-Za-z_][A-Za-z0-9_]*\s*)?(?:mut\s+)?", "", t)


def default_value_for_entry(entry: dict, registry: dict, scalar_default: Any = 0) -> Any:
    """注册类型的默认 JSON 值(enum→首个 variant;struct→字段默认列表)。"""
    if entry["kind"] == "enum":
        return entry["variants"][0]
    values: list[Any] = []
    for _name, field_type in entry["fields"]:
        nested = registry_entry(registry, field_type)
        if nested is not None:
            values.append(default_value_for_entry(nested, registry, scalar_default))
        elif re.search(r"\bbool\b", field_type):
            values.append(False)
        elif re.search(r"\b(?:f32|f64)\b", field_type):
            values.append(0.0)
        elif re.search(r"\b(?:String|str)\b", field_type):
            values.append("")
        elif re.search(r"Vec\s*<|\[", field_type):
            values.append([])
        elif re.search(r"Option\s*<", field_type):
            values.append(None)
        else:
            values.append(scalar_default)
    return values


def render_type_definitions(registry: Optional[dict]) -> str:
    """把注册表渲染回 Verus 可编译的类型定义(用于证明 harness 前导)。"""
    if not registry:
        return ""
    blocks: list[str] = []
    for name, entry in registry.items():
        if entry.get("kind") == "alias":
            blocks.append(f"pub type {name} = {entry.get('target')};")
        elif entry.get("kind") == "enum":
            variants = ", ".join(entry.get("variants") or [])
            blocks.append(f"pub enum {name} {{ {variants} }}")
        elif entry.get("tuple"):
            fields = ", ".join(f"pub {ftype}" for _fname, ftype in entry.get("fields") or [])
            blocks.append(f"pub struct {name}({fields});")
        else:
            fields = ", ".join(
                f"pub {fname}: {ftype}" for fname, ftype in entry.get("fields") or []
            )
            blocks.append(f"pub struct {name} {{ {fields} }}")
    return "\n\n".join(blocks)


__all__ = [
    "default_value_for_entry",
    "parse_type_definitions",
    "registry_entry",
    "registry_type_name",
    "render_type_definitions",
]
