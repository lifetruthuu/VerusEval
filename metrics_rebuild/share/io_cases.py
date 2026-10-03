from __future__ import annotations

import ast
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

from metrics_rebuild.share.contract_eval import (
    _extract_vec_element_type,
    _is_string_like_type,
    batch_verus_contract_decide_detailed,
    batch_verus_requires_decide,
    coerce_value_for_type,
    io_harness_for_case,
    is_bool_type,
    is_char_type,
    is_float_type,
    is_int_like_type,
    is_nested_vec_type,
    is_unsigned_type,
    is_vec_type,
    normalize_value_type,
    typed_input_payload,
    typed_output_payload,
)
from metrics_rebuild.share.functions import (
    authoritative_target_for_path,
    resolve_pair_target,
    strength_contexts_for_path,
    target_signatures_match,
)
from metrics_rebuild.share.io_harness import array_type_parts, extract_function
from metrics_rebuild.share.type_defs import default_value_for_entry, registry_entry
from metrics_rebuild.share.llm_client import call_llm_json
from metrics_rebuild.share.paths import CONFIG_PATH
from metrics_rebuild.share.scoring import not_available_metric
from metrics_rebuild.share.text import read_text
from metrics_rebuild.share.verus_runner import get_verus_binary

DEFAULT_IO_POSITIVE_CANDIDATES = 10
DEFAULT_IO_INVALID_CANDIDATES = 12
_IO_SUITE_DIR_OVERRIDE: Optional[str] = None


def set_io_suite_dir(path: Optional[str]) -> None:
    global _IO_SUITE_DIR_OVERRIDE
    _IO_SUITE_DIR_OVERRIDE = str(path) if path else None


def source_hash(path: str) -> str:
    return hashlib.sha256(read_text(path).encode("utf-8", errors="replace")).hexdigest()[:16]


def _audit_case_key(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def copy_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _load_io_suite_dir_from_config() -> Optional[str]:
    """从 config.yaml 读取 io_suite_dir（离线 IO 套件根目录）。未配置返回 None."""
    if _IO_SUITE_DIR_OVERRIDE:
        return _IO_SUITE_DIR_OVERRIDE
    if not CONFIG_PATH.exists():
        return None
    try:
        for line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            if key.strip().lower() == "io_suite_dir":
                path = value.strip().strip("\"' ")
                return path if path else None
    except OSError:
        pass
    return None


def choose_io_target_context(path: str) -> Optional[dict]:
    target = authoritative_target_for_path(path)
    return target.get("context") if target is not None else None


def context_for_suite_function(
    path: str,
    function_name: Optional[str],
    *,
    fallback: bool = True,
) -> Optional[dict]:
    contexts = strength_contexts_for_path(path)
    if function_name:
        for context in contexts:
            if context.get("function") == function_name:
                return context
    return choose_io_target_context(path) if fallback else None


def _target_descriptor_from_meta(meta: dict, context: dict) -> dict:
    parameters = meta.get("parameters") if isinstance(meta.get("parameters"), list) else context.get("parameters") or []
    returns = meta.get("returns") if isinstance(meta.get("returns"), list) else context.get("returns") or []
    return_type = meta.get("return_type")
    if not return_type:
        if len(returns) == 1:
            return_type = returns[0].get("type") or "()"
        elif len(returns) > 1:
            return_type = "(" + ", ".join(str(item.get("type") or "") for item in returns) + ")"
        else:
            return_type = "()"
    return {
        "function": str(meta.get("function") or context.get("function") or ""),
        "parameters": [
            {"name": str(item.get("name") or ""), "type": str(item.get("type") or "")}
            for item in parameters if isinstance(item, dict)
        ],
        "returns": [
            {"name": str(item.get("name") or "ret"), "type": str(item.get("type") or "")}
            for item in returns if isinstance(item, dict)
        ],
        "return_type": str(return_type),
    }


def _target_descriptor_for_generated(path: str, function_name: str, context: dict) -> Optional[dict]:
    code = read_text(path)
    func = extract_function(code, function_name)
    if func is None or func.name != function_name:
        return None
    returns = context.get("returns") or []
    return {
        "function": func.name,
        "parameters": [{"name": param.name, "type": param.rust_type} for param in func.params],
        "returns": [
            {"name": str(item.get("name") or "ret"), "type": str(item.get("type") or "")}
            for item in returns if isinstance(item, dict)
        ],
        "return_type": func.return_type or "()",
    }


def _observable_io_context(context: dict, descriptor: dict) -> dict:
    if context.get("returns"):
        return context
    mutable = [
        item for item in descriptor.get("parameters") or []
        if re.match(r"^&\s*mut\b", str(item.get("type") or "").strip())
    ]
    if not mutable:
        return context
    patched = dict(context)
    patched["returns"] = [
        {
            "name": item.get("name"),
            "type": re.sub(r"^&\s*mut\s+", "", str(item.get("type") or "")).strip(),
        }
        for item in mutable
    ]
    patched["_mutable_post_state_names"] = [str(item.get("name")) for item in mutable]
    return patched


def _observable_output_fields(descriptor: dict) -> list[dict]:
    returns = list(descriptor.get("returns") or [])
    if returns:
        return returns
    return [
        {
            "name": item.get("name"),
            "type": re.sub(r"^&\s*mut\s+", "", str(item.get("type") or "")).strip(),
        }
        for item in descriptor.get("parameters") or []
        if re.match(r"^&\s*mut\b", str(item.get("type") or "").strip())
    ]


def _remap_payload_by_position(
    payload: dict,
    reference_fields: list[dict],
    generated_fields: list[dict],
) -> Optional[dict]:
    if len(reference_fields) != len(generated_fields):
        return None
    mapped: dict[str, Any] = {}
    for reference, generated in zip(reference_fields, generated_fields):
        reference_name = str(reference.get("name") or "")
        generated_name = str(generated.get("name") or "")
        if not reference_name or not generated_name or reference_name not in payload:
            return None
        mapped[generated_name] = payload[reference_name]
    return mapped


def _normalized_io_type(type_text: Any) -> str:
    """Drop whitespace, lifetimes and trailing commas that do not change the type."""
    text = re.sub(r"\s+", "", re.sub(r"'[A-Za-z_]\w*\s*", "", str(type_text or "")))
    chars: list[str] = []
    commas = [0]
    for char in text:
        if char in "(<[":
            commas.append(0)
        elif char in ")>]":
            # `(T,)` is a one-element tuple, so only drop a comma after two or more elements.
            if chars and chars[-1] == "," and commas[-1] > 1:
                chars.pop()
            if len(commas) > 1:
                commas.pop()
        elif char == ",":
            commas[-1] += 1
        chars.append(char)
    text = "".join(chars)
    # A shared `&Vec<T>` and `&[T]` expose the same Seq<T> view and accept the same inputs.
    match = re.fullmatch(r"&Vec<(.+)>", text)
    return f"&[{match.group(1)}]" if match else text


def _positional_signature_issue(
    reference_fields: Sequence[dict],
    generated_fields: Sequence[dict],
    *,
    category: str,
) -> Optional[dict[str, Any]]:
    """Return an IO incompatibility while allowing harmless parameter renames."""
    if len(reference_fields) != len(generated_fields):
        return {
            "reason": "signature_type_mismatch",
            "detail": f"{category}_arity:{len(reference_fields)}!={len(generated_fields)}",
        }
    for index, (reference, generated) in enumerate(zip(reference_fields, generated_fields)):
        reference_type = _normalized_io_type(reference.get("type"))
        generated_type = _normalized_io_type(generated.get("type"))
        if reference_type != generated_type:
            return {
                "reason": "signature_type_mismatch",
                "detail": f"{category}[{index}]:{reference_type}!={generated_type}",
                "reference_type": reference.get("type"),
                "generated_type": generated.get("type"),
            }
    return None


def _target_mismatch_score(suite: dict, kind: str, reason: str, **details: Any) -> dict:
    cases = [
        case for case in suite.get("cases", [])
        if case.get("kind") == kind and case.get("status") == "validated"
    ]
    return {
        "status": "target_mismatch",
        "score": 0.0,
        "coverage": 0.0,
        "decidable_score": None,
        "reason": reason,
        "passed": 0,
        "failed": 0,
        "unknown": len(cases),
        "unknown_reason_summary": {reason: len(cases)} if cases else {},
        "total": len(cases),
        "evaluated": 0,
        "scoring_policy": "all_validated_cases_unknown_zero",
        "target": suite.get("target"),
        **details,
    }


def _run_generated_invalid_inputs(
    path: str,
    function_name: str,
    cases: Sequence[dict],
) -> dict[str, dict]:
    """Run invalid inputs and report only explicit runtime outcomes.

    This is intentionally a lazy import: the native compilation harness lives
    in the offline generator, while ordinary IO scoring should not pay that
    import cost unless an invalid case was not rejected by ``requires``.
    """
    if not cases:
        return {}
    from scripts.io import generate_io_tests_llm as generator

    target = generator.resolve_target(path, function_name)
    if target is None or target.func.name != function_name:
        return {
            str(case.get("key")): {"status": "UNKNOWN", "reason": "generated_target_not_found"}
            for case in cases
        }
    inputs: list[dict] = []
    keys: list[str] = []
    rejected: dict[str, dict] = {}
    for case in cases:
        key = str(case.get("key") or "")
        typed = typed_input_payload(target.context, case.get("inputs") or {})
        if not key or typed is None:
            if key:
                rejected[key] = {"status": "UNKNOWN", "reason": "input_type_mismatch"}
            continue
        keys.append(key)
        inputs.append(typed)
    if not inputs:
        return rejected
    with tempfile.TemporaryDirectory() as work_dir:
        harness = generator.build_verus_harness(target, inputs)
        results, reason = generator.verus_compile_and_run(
            harness, work_dir, get_verus_binary(), len(inputs),
        )
    outcomes = dict(rejected)
    for index, key in enumerate(keys):
        status, _value = results.get(index, ("UNKNOWN", ""))
        outcomes[key] = {
            "status": status if status in {"OK", "PANIC", "TIMEOUT", "UNKNOWN"} else "UNKNOWN",
            "reason": reason,
        }
    return outcomes


def _default_value_for_param(param: dict, registry: Optional[dict] = None) -> Any:
    type_text = param.get("type", "")
    array = array_type_parts(type_text)
    if array is not None:
        return [_vec_elem_filler({"type": f"Vec<{array[0]}>"})] * array[1]
    if is_nested_vec_type(type_text):
        return []
    if is_vec_type(type_text):
        return []
    if is_bool_type(type_text):
        return False
    if is_char_type(type_text):
        return ord("a")
    if is_float_type(type_text):
        return 0.0
    if _is_string_like_type(type_text):
        return ""
    entry = registry_entry(registry, type_text)
    if entry is not None:
        return default_value_for_entry(entry, registry or {})
    return 0


def _registry_value_variants(entry: dict, registry: Optional[dict]) -> list[Any]:
    """注册类型的候选值:enum 枚举所有 variant,struct 给默认值加一组扰动。"""
    if entry["kind"] == "enum":
        return list(entry["variants"][:8])
    base = default_value_for_entry(entry, registry or {})
    variants: list[Any] = [copy_jsonable(base)]
    alt = []
    for value, (_fname, field_type) in zip(base, entry["fields"]):
        nested = registry_entry(registry, field_type)
        if nested is not None and nested["kind"] == "enum" and len(nested["variants"]) > 1:
            alt.append(nested["variants"][1])
        elif isinstance(value, bool):
            alt.append(True)
        elif isinstance(value, (int, float)):
            alt.append(type(value)(1))
        elif isinstance(value, str):
            alt.append("a")
        elif isinstance(value, list):
            alt.append(value)
        else:
            alt.append(value)
    if alt != base:
        variants.append(alt)
    return variants


def _unescape_literal(raw: str) -> str:
    if "\\" not in raw:
        return raw
    try:
        return raw.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return raw


def _string_literals_from_context(context: dict) -> tuple[list[str], list[str]]:
    """Extract string and char literals mentioned by the task's contracts.

    Deterministic string candidates need domain-relevant values (e.g. "SUN"
    for a weekday task, '<'/'>' for a bracketing task); numeric-style pools
    cannot satisfy such preconditions.
    """
    blobs = [
        str(clause.get("normalized") or clause.get("text") or "")
        for clause in [*(context.get("requires") or []), *(context.get("ensures") or [])]
    ]
    blobs.append(str(context.get("spec_preamble") or ""))
    text = "\n".join(blobs)
    strings: list[str] = []
    for match in re.finditer(r'"((?:\\.|[^"\\]){0,24})"', text):
        value = _unescape_literal(match.group(1))
        if value not in strings:
            strings.append(value)
    chars: list[str] = []
    for match in re.finditer(r"'((?:\\.|[^'\\]))'", text):
        value = _unescape_literal(match.group(1))
        if len(value) == 1 and value not in chars:
            chars.append(value)
    return strings, chars


def _string_candidate_pool(context: dict, *, limit: int = 16) -> list[str]:
    pool: list[str] = []

    def add(value: str) -> None:
        if isinstance(value, str) and len(value) <= 24 and value not in pool:
            pool.append(value)

    strings, chars = _string_literals_from_context(context)
    for value in strings[:8]:
        add(value)
    for c in chars[:4]:
        add(c)
    # 字符组合覆盖配对/嵌套/逆序模式（如括号匹配任务的 "<>"、"<<>>"、"><"）。
    for c1 in chars[:3]:
        for c2 in chars[:3]:
            if c1 != c2:
                add(c1 + c2)
                add(c1 + c1 + c2 + c2)
    for value in ("", "a", "ab", "abc", "aA", "xyz"):
        add(value)
    return pool[:limit]


def _vec_string_values(pool: Sequence[str]) -> list[list[str]]:
    head = [value for value in pool if value][:2] or ["a"]
    values: list[list[str]] = [[], [""], [head[0]]]
    if len(head) > 1:
        values.append([head[0], head[1]])
    values.extend([["a", "b"], ["a", "b", "c"]])
    deduped: list[list[str]] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _vec_elem_filler(param: dict) -> Any:
    """Fill element used when synthesizing Vec values of a given length."""
    type_text = str(param.get("type", ""))
    inner = re.sub(r"^(?:Vec|Seq)\s*<\s*(.*)\s*>\s*$", r"\1", normalize_value_type(type_text))
    if is_bool_type(inner):
        return False
    if is_char_type(inner):
        return ord("a")
    if is_float_type(inner):
        return 0.0
    if _is_string_like_type(inner):
        return ""
    return 0


def _list_value_for_length(param: dict, length: int) -> list[Any]:
    # Only construct small upper-bound witnesses. Larger domains are marked
    # unconstructable by the category-target estimator.
    type_text = param.get("type", "")
    array = array_type_parts(type_text)
    if array is not None:
        # 定长数组的长度由类型决定，忽略请求长度。
        return [_vec_elem_filler({"type": f"Vec<{array[0]}>"})] * array[1]
    length = max(0, min(length, 12))
    if is_nested_vec_type(type_text):
        return [[_vec_elem_filler(param)] for _ in range(length)]
    filler = _vec_elem_filler(param)
    return [filler for _ in range(length)]


def tag_input_values(inputs: dict) -> list[str]:
    tags: set[str] = set()
    for value in inputs.values():
        if isinstance(value, list):
            if len(value) == 0:
                tags.add("empty")
            if len(value) == 1:
                tags.add("single")
            if len(value) > 1:
                tags.add("multi")
            try:
                if len(set(value)) < len(value):
                    tags.add("duplicate")
            except TypeError:
                pass
            if any(item in {0, 1, -1} for item in value if isinstance(item, int)):
                tags.add("boundary")
        elif isinstance(value, bool):
            tags.add("bool")
        elif isinstance(value, int):
            if value in {-1, 0, 1}:
                tags.add("boundary")
    return sorted(tags)


def _requires_search_hit(context: dict) -> bool:
    requires_text = " ".join(
        str(clause.get("normalized") or clause.get("text") or "")
        for clause in context.get("requires") or []
    ).lower()
    return "exists" in requires_text and ("==" in requires_text or "contains" in requires_text)


def expected_path_tags(context: dict) -> set[str]:
    tags = {"normal"}
    params = context.get("parameters") or []
    if any(is_vec_type(param.get("type", "")) for param in params):
        tags.update({"empty", "single", "multi", "duplicate", "boundary"})
    if context.get("requires"):
        tags.add("requires_boundary")
    name = str(context.get("function", "")).lower()
    if "search" in name or "find" in name:
        tags.update({"found", "first_match"})
        if not _requires_search_hit(context):
            tags.add("not_found")
    return tags


def _validate_llm_input_case(context: dict, raw: Any) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    payload = raw.get("inputs") if isinstance(raw.get("inputs"), dict) else raw
    if not isinstance(payload, dict):
        return None
    item: dict[str, Any] = {}
    for param in context.get("parameters") or []:
        name = param.get("name")
        if not name or name not in payload:
            return None
        ok, value = coerce_value_for_type(payload[name], param.get("type", ""))
        if not ok:
            return None
        item[name] = value
    return item


def _validate_llm_output_case(context: dict, raw: Any) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    item: dict[str, Any] = {}
    returns = list(context.get("returns") or [])
    if not returns:
        return None
    for ret in returns:
        name = ret.get("name") or "ret"
        if name not in raw:
            return None
        ok, value = coerce_value_for_type(raw[name], ret.get("type", ""))
        if not ok:
            return None
        item[name] = value
    return item


def _dedupe_input_cases(candidates: Sequence[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for item in candidates:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(copy_jsonable(item))
    return deduped


def llm_candidate_inputs(context: dict, *, budget: int, attempt: int = 0) -> tuple[list[dict], dict]:
    """生成候选输入。

    当 context 含有 requires 子句时使用「requires 优先」模式：将 spec_preamble（spec fn 定义）
    和 requires 子句明确注入 prompt，要求 LLM 生成满足所有前置条件的输入；启发式生成无法做到这一
    点（它不理解 spec fn 语义），所以这里 LLM 是主力。
    attempt > 0 表示重试轮次，使用更高 temperature 提升多样性，同时使用不同缓存键。
    """
    params = [
        {"name": param.get("name"), "type": param.get("type")}
        for param in context.get("parameters") or []
        if param.get("name")
    ]
    if not params:
        return [], {"status": "skipped", "reason": "no_parameters"}

    requires_clauses = [
        clause.get("normalized") or clause.get("text")
        for clause in context.get("requires") or []
    ]
    ensures_clauses = [
        clause.get("normalized") or clause.get("text")
        for clause in context.get("ensures") or []
    ]
    spec_preamble = (context.get("spec_preamble") or "").strip()
    has_requires = bool(requires_clauses)

    if has_requires:
        # --- requires 优先模式 ---
        # LLM 是主策略：需要理解 spec fn 语义才能生成满足 requires 的输入
        system_content = (
            "You are generating concrete input test cases for a formally verified Rust/Verus function. "
            "The function has `requires` preconditions that ALL inputs MUST satisfy — inputs violating "
            "any precondition are discarded immediately by the verifier. "
            "If `spec_definitions` are provided, read them carefully to understand what each `requires` "
            "clause means before generating inputs. "
            "Return strict JSON only."
        )
        prompt: dict = {
            "function": context.get("function"),
            "parameters": params,
            "requires": requires_clauses,
            "ensures": ensures_clauses,
            "budget": budget,
        }
        if spec_preamble:
            prompt["spec_definitions"] = spec_preamble
        prompt["task"] = (
            f"Generate {budget} diverse concrete inputs for `{context.get('function')}` that ALL satisfy "
            "every `requires` clause listed above. "
            "Steps: (1) analyse each `requires` clause (and `spec_definitions` if provided) to determine "
            "the valid input domain; "
            "(2) generate the minimal boundary inputs (e.g. if requires len >= 2, include exactly len=2); "
            "(3) generate diverse interior values; "
            "(4) generate near-boundary values. "
            "Do NOT generate inputs that violate any requires clause. "
            "The expected_output field is a best-effort guess; the actual output will be verified by "
            "running the compiled function."
        )
        prompt["output_schema"] = {
            "inputs": [
                {
                    "inputs": {"param_name": "concrete JSON value satisfying requires"},
                    "expected_output": {"return_name": "best-effort guess"},
                    "tags": ["boundary", "normal"],
                    "rationale": "why this input satisfies requires and is diverse",
                }
            ]
        }
        # 重试时用更高温度提升多样性
        temperature = 0.7 if attempt > 0 else 0.3
        task_key = f"io_candidate_inputs_req:{context.get('function')}:{budget}"
    else:
        # --- 通用模式（无 requires，保持原有行为）---
        system_content = (
            "You generate input-output test cases for formal-spec testing. "
            "Return strict JSON only. Your expected outputs are candidates; "
            "the verifier/reference contract determines final correctness."
        )
        prompt = {
            "function": context.get("function"),
            "parameters": params,
            "returns": context.get("returns") or [],
            "requires": requires_clauses,
            "ensures": ensures_clauses,
            "budget": budget,
            "task": (
                "Generate diverse concrete input-output test cases for Verus specification testing. "
                "For each case, provide the inputs AND the expected output that satisfies the ensures clauses. "
                "Cover normal paths, boundary values, empty/single/multi Vec or Seq values, duplicates, "
                "requires boundary cases, and different return branches. "
                "The expected_output field should map return parameter names to their concrete values."
            ),
            "output_schema": {
                "inputs": [
                    {
                        "inputs": {"param_name": "concrete JSON value"},
                        "expected_output": {"return_name": "concrete JSON value"},
                        "tags": ["boundary", "normal"],
                        "rationale": "why this input is useful",
                    }
                ]
            },
        }
        temperature = 0.2
        task_key = f"io_candidate_io_pairs:{context.get('function')}:{budget}"

    if attempt > 0:
        task_key = f"{task_key}:attempt{attempt}"

    result = call_llm_json(
        task=task_key,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, indent=2)},
        ],
        temperature=temperature,
    )
    if result.get("status") != "ok":
        return [], result
    data = result.get("json")
    raw_inputs = data.get("inputs") if isinstance(data, dict) else data
    if not isinstance(raw_inputs, list):
        return [], {
            "status": "not_available",
            "reason": "llm_inputs_not_a_list",
            "llm": result.get("llm"),
            "raw": result.get("raw"),
            "cached": result.get("cached"),
        }
    candidates = []
    llm_outputs: list[Optional[dict]] = []
    for raw_item in raw_inputs:
        validated = _validate_llm_input_case(context, raw_item)
        if validated is not None:
            candidates.append(validated)
            raw_output = raw_item.get("expected_output") if isinstance(raw_item, dict) else None
            llm_outputs.append(_validate_llm_output_case(context, raw_output))
    deduped = _dedupe_input_cases(candidates)[:budget]
    return deduped, {
        "status": "ok",
        "accepted_candidates": len(deduped),
        "raw_candidates": len(raw_inputs),
        "llm": result.get("llm"),
        "cached": result.get("cached"),
        "llm_outputs": llm_outputs[: len(deduped)],
        "requires_mode": has_requires,
    }


def first_vec_scalar_pair(context: dict, inputs: dict) -> tuple[Optional[str], Optional[str]]:
    vec_name = None
    scalar_name = None
    for param in context.get("parameters") or []:
        name = param.get("name")
        if not name:
            continue
        if vec_name is None and is_vec_type(param.get("type", "")) and isinstance(inputs.get(name), list):
            vec_name = name
        elif scalar_name is None and is_int_like_type(param.get("type", "")) and isinstance(inputs.get(name), int):
            scalar_name = name
    return vec_name, scalar_name


def generate_candidate_inputs(context: dict, *, budget: int = DEFAULT_IO_POSITIVE_CANDIDATES) -> list[dict]:
    params = list(context.get("parameters") or [])
    if not params:
        return [{}]
    registry = context.get("type_registry")

    vec_params = [param for param in params if is_vec_type(param.get("type", "")) and not is_nested_vec_type(param.get("type", ""))]
    nested_vec_params = [param for param in params if is_nested_vec_type(param.get("type", ""))]
    # 快速路径的标量必须能吃整数字面量；字符串/char 参数交给下方类型感知笛卡尔组合。
    numeric_scalar_params = [
        param for param in params
        if not is_vec_type(param.get("type", ""))
        and (is_int_like_type(param.get("type", "")) or is_float_type(param.get("type", "")))
    ]
    bool_params = [param for param in params if is_bool_type(param.get("type", ""))]
    cases: list[dict] = []

    if len(vec_params) == 1 and len(numeric_scalar_params) >= 1 and not nested_vec_params:
        vec_name = vec_params[0]["name"]
        scalar_name = numeric_scalar_params[0]["name"]
        patterns = [
            ([], 0),
            ([5], 5),
            ([5], 4),
            ([2, 4, 4], 4),
            ([2, 4, 4], 2),
            ([1, 3, 5], 4),
            ([0, 0], 0),
            ([3, 1, 4, 1, 5], 1),
        ]
        for array_value, scalar_value in patterns:
            item = {param["name"]: _default_value_for_param(param, registry) for param in params}
            item[vec_name] = list(array_value)
            item[scalar_name] = scalar_value
            for bool_param in bool_params:
                item[bool_param["name"]] = len(cases) % 2 == 0
            cases.append(item)
            if len(cases) >= budget:
                return cases

    string_pool: Optional[list[str]] = None

    def pool() -> list[str]:
        nonlocal string_pool
        if string_pool is None:
            string_pool = _string_candidate_pool(context)
        return string_pool

    value_options: list[tuple[str, list[Any]]] = []
    for param in params:
        name = param["name"]
        type_text = param.get("type", "")
        entry = registry_entry(registry, type_text)
        array = array_type_parts(type_text)
        if entry is not None:
            values = _registry_value_variants(entry, registry)
        elif array is not None:
            values = _array_value_variants(*array)
        elif is_nested_vec_type(type_text):
            values = [[], [[1]], [[1, 2], [3, 4]], [[1, 2, 3]], [[1], [2], [3]], [[1, 2], [3, 4, 5]]]
        elif is_vec_type(type_text):
            inner = _extract_vec_element_type(type_text)
            if _is_string_like_type(inner):
                values = _vec_string_values(pool())
            elif is_char_type(inner):
                values = [[], [ord("a")], [ord("a"), ord("b"), ord("c")], [ord("0"), ord("1")]]
            elif is_bool_type(inner):
                values = [[], [True], [True, False], [False, False, True]]
            else:
                values = [[], [0], [1], [1, 2, 3], [2, 4, 4], [3, 1, 4, 1, 5]]
        elif is_bool_type(type_text):
            values = [False, True]
        elif _is_string_like_type(type_text):
            values = list(pool()) or [""]
        elif is_char_type(type_text):
            values = [ord("a"), ord("b"), ord("z"), ord("0"), ord("A")]
        elif is_unsigned_type(type_text):
            values = [0, 1, 2, 3, 5, 10]
        else:
            values = [0, 1, -1, 2, 3, 5]
        value_options.append((name, values))

    def build(idx: int, current: dict) -> None:
        if len(cases) >= budget:
            return
        if idx == len(value_options):
            cases.append(dict(current))
            return
        name, values = value_options[idx]
        for value in values:
            current[name] = list(value) if isinstance(value, list) else value
            build(idx + 1, current)
            if len(cases) >= budget:
                return

    build(0, {})
    return cases


def _requires_clause_texts(context: dict) -> list[str]:
    texts: list[str] = []
    for clause in context.get("requires") or []:
        if isinstance(clause, dict):
            text = str(clause.get("normalized") or clause.get("text") or "")
        else:
            text = str(clause)
        if text.strip():
            texts.append(_normalize_requires_pattern_text(text.strip()))
    return texts


def _normalize_requires_pattern_text(text: str) -> str:
    """Normalize Verus pretty-printed clauses so regex patterns can match.

    Strength contexts often emit `x1 . len ( ) == x2 . len ( )`; collapse that to
    `x1.len() == x2.len()` while keeping ordinary spacing around operators.
    """
    text = re.sub(r"\bold\s*\(\s*([A-Za-z_]\w*)\s*\)", r"\1", text)
    text = re.sub(r"\s*@\s*", "", text)
    text = re.sub(r"\s*\.\s*", ".", text)
    text = re.sub(r"\s*\(\s*\)", "()", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _int_values_for_relation(op: str, value: int, *, unsigned: bool) -> list[int]:
    if op == ">":
        values = [value + 1, value + 2, max(value + 10, 10)]
    elif op == ">=":
        values = [value, value + 1, value + 2]
    elif op == "<":
        values = [value - 1, value - 2, min(value - 10, -1)]
    elif op == "<=":
        values = [value, value - 1, value - 2]
    elif op == "==":
        values = [value]
    elif op == "!=":
        values = [value + 1, value - 1, 0 if value != 0 else 1]
    else:
        values = [value]
    if unsigned:
        values = [v for v in values if v >= 0]
    return values or [0]


def _len_values_for_relation(op: str, value: int) -> list[int]:
    if op == ">":
        return [value + 1, value + 2, max(value + 3, 3)]
    if op == ">=":
        return [value, value + 1, value + 2]
    if op == "==":
        return [value]
    if op == "<":
        return [max(0, value - 1), 0]
    if op == "<=":
        return [max(0, value), max(0, value - 1)]
    if op == "!=":
        return [value + 1, max(0, value - 1)]
    return [value]


def _mod_values_for_relation(modulus: int, op: str, value: int, *, unsigned: bool) -> list[int]:
    modulus = abs(modulus)
    if modulus == 0:
        return []
    base = value % modulus
    if op == "==":
        values = [base, base + modulus, base + 2 * modulus]
    else:
        values = [base + 1, base + modulus + 1, base + 2 * modulus + 1]
    if unsigned:
        values = [v for v in values if v >= 0]
    return values or [0]


def generate_requires_satisfying_candidate_inputs(context: dict, *, budget: int) -> list[dict]:
    """Generate simple positive candidates from common requires patterns.

    These are only seed candidates. The native pipeline still validates them with
    Verus requires checks before accepting positive IO.
    """
    params = [param for param in context.get("parameters") or [] if param.get("name")]
    if not params:
        return [{}]
    registry = context.get("type_registry")
    param_by_name = {str(param.get("name")): param for param in params}
    texts = _requires_clause_texts(context)
    if not texts:
        return []

    value_options: dict[str, list[Any]] = {
        str(param.get("name")): [_default_value_for_param(param, registry)]
        for param in params
    }

    def add_value(name: str, value: Any) -> None:
        if name not in value_options:
            return
        current = value_options[name]
        if value not in current:
            current.append(copy_jsonable(value))

    for text in texts:
        for match in re.finditer(
            r"\b(?P<name>[A-Za-z_]\w*)\s*(?P<op>>=|<=|==|!=|>|<)\s*(?P<value>-?\d+)\b",
            text,
        ):
            name = match.group("name")
            param = param_by_name.get(name)
            if not param or not is_int_like_type(param.get("type", "")):
                continue
            unsigned = is_unsigned_type(param.get("type", ""))
            for value in _int_values_for_relation(match.group("op"), int(match.group("value")), unsigned=unsigned):
                add_value(name, value)

        for match in re.finditer(
            r"\b(?P<name>[A-Za-z_]\w*)\.len\(\)\s*(?P<op>>=|<=|==|!=|>|<)\s*(?P<value>\d+)\b",
            text,
        ):
            name = match.group("name")
            param = param_by_name.get(name)
            if not param:
                continue
            if is_vec_type(param.get("type", "")):
                for length in _len_values_for_relation(match.group("op"), int(match.group("value"))):
                    add_value(name, _list_value_for_length(param, length))
            elif _is_string_like_type(param.get("type", "")):
                # String/&str 长度约束（如 sep@.len() > 0）：构造对应长度的串。
                for length in _len_values_for_relation(match.group("op"), int(match.group("value"))):
                    add_value(name, "a" * max(0, min(length, 12)))

        for match in re.finditer(
            r"\b(?P<idx>[A-Za-z_]\w*)\s*(?P<op><=|<)\s*(?P<vec>[A-Za-z_]\w*)\.len\(\)",
            text,
        ):
            idx_name = match.group("idx")
            vec_name = match.group("vec")
            idx_param = param_by_name.get(idx_name)
            vec_param = param_by_name.get(vec_name)
            if not idx_param or not vec_param:
                continue
            if not is_int_like_type(idx_param.get("type", "")) or not is_vec_type(vec_param.get("type", "")):
                continue
            length = 3
            add_value(vec_name, _list_value_for_length(vec_param, length))
            add_value(idx_name, 0)
            add_value(idx_name, 1)
            if match.group("op") == "<=":
                add_value(idx_name, length)

        for match in re.finditer(
            r"\b(?P<name>[A-Za-z_]\w*)\s*%\s*(?P<mod>-?\d+)\s*(?P<op>==|!=)\s*(?P<value>-?\d+)\b",
            text,
        ):
            name = match.group("name")
            param = param_by_name.get(name)
            if not param or not is_int_like_type(param.get("type", "")):
                continue
            unsigned = is_unsigned_type(param.get("type", ""))
            for value in _mod_values_for_relation(
                int(match.group("mod")),
                match.group("op"),
                int(match.group("value")),
                unsigned=unsigned,
            ):
                add_value(name, value)

        # a.len() == b.len()  (and a.len() == b.len() == ... loosely via pairwise)
        for match in re.finditer(
            r"\b(?P<a>[A-Za-z_]\w*)\.len\(\)\s*==\s*(?P<b>[A-Za-z_]\w*)\.len\(\)",
            text,
        ):
            a_name, b_name = match.group("a"), match.group("b")
            a_param, b_param = param_by_name.get(a_name), param_by_name.get(b_name)
            if not a_param or not b_param:
                continue
            if not is_vec_type(a_param.get("type", "")) or not is_vec_type(b_param.get("type", "")):
                continue
            for length in (0, 1, 2, 3, 5):
                add_value(a_name, _list_value_for_length(a_param, length))
                add_value(b_name, _list_value_for_length(b_param, length))

        # 'a' <= c <= 'y'  char ranges
        for match in re.finditer(
            r"'(?P<lo>(?:\\.|[^'\\]))'\s*<=\s*(?P<name>[A-Za-z_]\w*)\s*<=\s*'(?P<hi>(?:\\.|[^'\\]))'",
            text,
        ):
            name = match.group("name")
            param = param_by_name.get(name)
            if not param or not is_char_type(param.get("type", "")):
                continue
            lo = match.group("lo")
            hi = match.group("hi")
            if len(lo) == 1 and len(hi) == 1 and ord(lo) <= ord(hi):
                mid = chr((ord(lo) + ord(hi)) // 2)
                for ch in (lo, mid, hi):
                    add_value(name, ord(ch))

    # 字符串参数补充契约相关的字面量候选（如 valid_day 的 "SUN"、括号任务的 "<>"）。
    string_pool: Optional[list[str]] = None
    for param in params:
        name = str(param.get("name"))
        type_text = param.get("type", "")
        if _is_string_like_type(type_text):
            if string_pool is None:
                string_pool = _string_candidate_pool(context)
            for value in string_pool[:6]:
                add_value(name, value)
        elif is_vec_type(type_text) and not is_nested_vec_type(type_text) and _is_string_like_type(_extract_vec_element_type(type_text)):
            if string_pool is None:
                string_pool = _string_candidate_pool(context)
            for value in _vec_string_values(string_pool)[:4]:
                add_value(name, value)
        else:
            entry = registry_entry(registry, type_text)
            if entry is not None:
                for value in _registry_value_variants(entry, registry)[:4]:
                    add_value(name, value)

    cases: list[dict] = []
    max_options = max((len(values) for values in value_options.values()), default=1)
    for offset in range(max_options):
        item: dict[str, Any] = {}
        for param in params:
            name = str(param.get("name"))
            values = value_options.get(name) or [_default_value_for_param(param, registry)]
            value = values[offset % len(values)]
            item[name] = copy_jsonable(value)
        # Re-sync equal-length pairs after independent offset picking.
        for text in texts:
            for match in re.finditer(
                r"\b(?P<a>[A-Za-z_]\w*)\.len\(\)\s*==\s*(?P<b>[A-Za-z_]\w*)\.len\(\)",
                text,
            ):
                a_name, b_name = match.group("a"), match.group("b")
                a_val, b_val = item.get(a_name), item.get(b_name)
                if isinstance(a_val, list) and isinstance(b_val, list) and len(a_val) != len(b_val):
                    target_len = len(a_val)
                    b_param = param_by_name.get(b_name)
                    if b_param is not None:
                        item[b_name] = _list_value_for_length(b_param, target_len)
        cases.append(item)
        if len(cases) >= budget:
            break
    return _dedupe_input_cases(cases)[:budget]


def _array_value_variants(elem_type: str, length: int) -> list[list[Any]]:
    filler = _vec_elem_filler({"type": f"Vec<{elem_type}>"})
    variants: list[list[Any]] = [[filler] * length]
    if isinstance(filler, (int, float)) and not isinstance(filler, bool):
        one = 1.0 if isinstance(filler, float) else 1
        variants.append([one] * length)
        variants.append([type(one)(i % 7) for i in range(length)])
        variants.append([one if i % 2 else type(one)(0) for i in range(length)])
    elif isinstance(filler, bool):
        variants.append([True] * length)
        variants.append([bool(i % 2) for i in range(length)])
    deduped: list[list[Any]] = []
    for value in variants:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _param_value_variants(param: dict, registry: Optional[dict] = None) -> list[Any]:
    type_text = param.get("type", "")
    entry = registry_entry(registry, type_text)
    if entry is not None:
        return _registry_value_variants(entry, registry)
    array = array_type_parts(type_text)
    if array is not None:
        return _array_value_variants(*array)
    if is_nested_vec_type(type_text):
        return [
            [], [[]], [[0]], [[1, 2], [3]], [[1], [2], [3]], [[0, 0], [1, 1]],
            # 含空子 Vec 的混合结构：违反 forall|i| s[i].len() > 0 类元素约束
            [[1], []], [[], [2]], [[1], [], [3]], [[], [], [1]], [[1, 2], [], [3]],
            [[], [0], []],
        ]
    if is_vec_type(type_text):
        inner = _extract_vec_element_type(type_text)
        if _is_string_like_type(inner):
            return [[], [""], ["a"], ["a", "b"], ["", "a"]]
        return [[], [0], [1], [-1], [0, 0], [1, 2, 3], [2, 4, 4], [3, 1, 4, 1, 5], list(range(12))]
    if is_bool_type(type_text):
        return [False, True]
    if _is_string_like_type(type_text):
        return ["", "a", "ab", "abc"]
    if is_unsigned_type(type_text):
        return [0, 1, 2, 3, 5, 10]
    if is_int_like_type(type_text):
        return [0, 1, -1, 2, 3, 5, 10, -10]
    return [_default_value_for_param(param, registry)]


def _extract_numeric_constants_from_clauses(clauses: Sequence[dict]) -> list[int]:
    values: list[int] = []
    for clause in clauses:
        text = str(clause.get("normalized") or clause.get("text") or "")
        for match in re.finditer(r"(?<![A-Za-z0-9_])-?\d+", text):
            try:
                value = int(match.group(0))
            except ValueError:
                continue
            for candidate in (value - 1, value, value + 1):
                if -256 <= candidate <= 256 and candidate not in values:
                    values.append(candidate)
    return values


def _build_invalid_seed_inputs(context: dict, budget: int = DEFAULT_IO_INVALID_CANDIDATES) -> list[dict]:
    params = list(context.get("parameters") or [])
    if not params:
        return [{}]
    registry = context.get("type_registry")
    constants = _extract_numeric_constants_from_clauses(context.get("requires") or [])
    seed = {param["name"]: _default_value_for_param(param, registry) for param in params if param.get("name")}
    cases: list[dict] = []
    param_by_name = {str(param.get("name")): param for param in params if param.get("name")}
    texts = _requires_clause_texts(context)

    for param in params:
        name = param.get("name")
        if not name:
            continue
        variants = _param_value_variants(param, registry)
        if is_vec_type(param.get("type", "")):
            for value in constants[:6]:
                variants.extend([[], [value], [value + 1], [value, value]])
        elif is_int_like_type(param.get("type", "")):
            variants.extend(constants[:8])
            if is_unsigned_type(param.get("type", "")):
                variants = [value for value in variants if not isinstance(value, int) or value >= 0]
        elif is_char_type(param.get("type", "")):
            variants.extend([ord("a"), ord("z"), ord("A"), ord("0"), 0, ord("{")])
        elif is_float_type(param.get("type", "")):
            variants.extend([0.0, 1.0, -1.0, 0.5, 1e6])
        for value in variants:
            item = dict(seed)
            item[name] = list(value) if isinstance(value, list) else value
            cases.append(item)
            if len(cases) >= budget * 3:
                break
        if len(cases) >= budget * 3:
            break

    if len(params) >= 2:
        names = [param.get("name") for param in params if param.get("name")]
        for left, right in zip(names, names[1:]):
            item = dict(seed)
            left_value = item.get(left)
            right_value = item.get(right)
            if isinstance(left_value, list) and isinstance(right_value, list):
                item[left] = [0, 1]
                item[right] = []
            elif isinstance(left_value, int) and isinstance(right_value, int):
                item[left] = 0
                item[right] = 1
            cases.append(item)

    # Pattern-directed requires violations (equal length, char range, index bounds).
    for text in texts:
        for match in re.finditer(
            r"\b(?P<name>[A-Za-z_]\w*)\.len\(\)\s*(?P<op>>=|<=|==|>|<)\s*(?P<value>\d+)\b",
            text,
        ):
            name = match.group("name")
            param = param_by_name.get(name)
            if not param or not is_vec_type(param.get("type", "")):
                continue
            bound = int(match.group("value"))
            op = match.group("op")
            if op in (">", ">="):
                bad_len = bound if op == ">" else max(0, bound - 1)
            elif op in ("<", "<="):
                bad_len = bound if op == "<" else bound + 1
            else:
                bad_len = bound + 1
            if bad_len <= 12:
                item = dict(seed)
                item[name] = _list_value_for_length(param, bad_len)
                cases.append(item)

        for match in re.finditer(
            r"\b(?P<a>[A-Za-z_]\w*)\.len\(\)\s*==\s*(?P<b>[A-Za-z_]\w*)\.len\(\)",
            text,
        ):
            a_name, b_name = match.group("a"), match.group("b")
            a_param, b_param = param_by_name.get(a_name), param_by_name.get(b_name)
            if not a_param or not b_param:
                continue
            for a_len, b_len in ((0, 1), (1, 0), (2, 1), (3, 0), (1, 3)):
                item = dict(seed)
                item[a_name] = _list_value_for_length(a_param, a_len)
                item[b_name] = _list_value_for_length(b_param, b_len)
                cases.append(item)
        for match in re.finditer(
            r"'(?P<lo>(?:\\.|[^'\\]))'\s*<=\s*(?P<name>[A-Za-z_]\w*)\s*<=\s*'(?P<hi>(?:\\.|[^'\\]))'",
            text,
        ):
            name = match.group("name")
            param = param_by_name.get(name)
            if not param or not is_char_type(param.get("type", "")):
                continue
            lo, hi = match.group("lo"), match.group("hi")
            if len(lo) == 1 and len(hi) == 1:
                item = dict(seed)
                item[name] = ord(lo) - 1 if ord(lo) > 0 else ord(hi) + 1
                cases.append(item)
                item2 = dict(seed)
                item2[name] = ord(hi) + 1
                cases.append(item2)
        for match in re.finditer(
            r"\b(?P<idx>[A-Za-z_]\w*)\s*<\s*(?P<vec>[A-Za-z_]\w*)\.len\(\)",
            text,
        ):
            idx_name, vec_name = match.group("idx"), match.group("vec")
            idx_param, vec_param = param_by_name.get(idx_name), param_by_name.get(vec_name)
            if not idx_param or not vec_param:
                continue
            if not is_int_like_type(idx_param.get("type", "")) or not is_vec_type(vec_param.get("type", "")):
                continue
            for length in (0, 1, 2, 3):
                item = dict(seed)
                item[vec_name] = _list_value_for_length(vec_param, length)
                item[idx_name] = length  # violates idx < len
                cases.append(item)
                item2 = dict(seed)
                item2[vec_name] = _list_value_for_length(vec_param, length)
                item2[idx_name] = length + 1
                cases.append(item2)

    for item in generate_candidate_inputs(context, budget=budget * 2):
        cases.append(item)
    return _dedupe_input_cases(cases)[: budget * 4]


def generate_requires_violating_from_positives(
    context: dict,
    positive_inputs: Sequence[dict],
    *,
    budget: int = DEFAULT_IO_INVALID_CANDIDATES,
) -> list[dict]:
    """Derive invalid candidates by minimally mutating known-valid positive inputs.

    This is often more reliable than cold heuristic seeds because the base inputs
    already satisfy most of the requires conjunction.
    """
    params = [param for param in context.get("parameters") or [] if param.get("name")]
    if not params or not positive_inputs:
        return []
    param_by_name = {str(param.get("name")): param for param in params}
    texts = _requires_clause_texts(context)
    cases: list[dict] = []

    def push(item: dict) -> None:
        cases.append(copy_jsonable(item))

    for base in positive_inputs:
        if not isinstance(base, dict):
            continue
        seed = {param["name"]: copy_jsonable(base.get(param["name"], _default_value_for_param(param, context.get("type_registry")))) for param in params}

        for text in texts:
            for match in re.finditer(
                r"(?P<lo>-?\d+)\s*<=\s*(?P<name>[A-Za-z_]\w*)\s*<=\s*(?P<hi>-?\d+)",
                text,
            ):
                name = match.group("name")
                param = param_by_name.get(name)
                if not param or not is_int_like_type(param.get("type", "")):
                    continue
                lo, hi = int(match.group("lo")), int(match.group("hi"))
                for value in (lo - 1, hi + 1):
                    if is_unsigned_type(param.get("type", "")) and value < 0:
                        continue
                    item = dict(seed)
                    item[name] = value
                    push(item)

            for match in re.finditer(
                r"\b(?P<a>[A-Za-z_]\w*)\.len\(\)\s*==\s*(?P<b>[A-Za-z_]\w*)\.len\(\)",
                text,
            ):
                a_name, b_name = match.group("a"), match.group("b")
                a_val = seed.get(a_name)
                b_param = param_by_name.get(b_name)
                if not isinstance(a_val, list) or b_param is None:
                    continue
                item = dict(seed)
                item[b_name] = _list_value_for_length(b_param, len(a_val) + 1)
                push(item)
                item2 = dict(seed)
                if len(a_val) == 0:
                    item2[b_name] = [_vec_elem_filler(b_param)]
                else:
                    item2[b_name] = _list_value_for_length(b_param, max(0, len(a_val) - 1))
                push(item2)

            for match in re.finditer(
                r"'(?P<lo>(?:\\.|[^'\\]))'\s*<=\s*(?P<name>[A-Za-z_]\w*)\s*<=\s*'(?P<hi>(?:\\.|[^'\\]))'",
                text,
            ):
                name = match.group("name")
                if name not in seed or not is_char_type((param_by_name.get(name) or {}).get("type", "")):
                    continue
                lo, hi = match.group("lo"), match.group("hi")
                if len(lo) == 1:
                    item = dict(seed)
                    item[name] = max(0, ord(lo) - 1)
                    push(item)
                if len(hi) == 1:
                    item = dict(seed)
                    item[name] = ord(hi) + 1
                    push(item)

            for match in re.finditer(
                r"\b(?P<name>[A-Za-z_]\w*)\s*(?P<op>>=|<=|==|>|<)\s*(?P<value>-?\d+)\b",
                text,
            ):
                name = match.group("name")
                param = param_by_name.get(name)
                if not param or not is_int_like_type(param.get("type", "")):
                    continue
                value = int(match.group("value"))
                op = match.group("op")
                item = dict(seed)
                if op in (">", ">="):
                    item[name] = value - 1 if op == ">=" else value
                elif op in ("<", "<="):
                    item[name] = value + 1 if op == "<=" else value
                elif op == "==":
                    item[name] = value + 1
                else:
                    item[name] = value
                if is_unsigned_type(param.get("type", "")) and isinstance(item[name], int) and item[name] < 0:
                    continue
                push(item)

            for match in re.finditer(
                r"\b(?P<name>[A-Za-z_]\w*)\.len\(\)\s*(?P<op>>=|<=|==|>|<)\s*(?P<value>\d+)\b",
                text,
            ):
                name = match.group("name")
                param = param_by_name.get(name)
                if not param or not is_vec_type(param.get("type", "")):
                    continue
                bound = int(match.group("value"))
                op = match.group("op")
                item = dict(seed)
                if op in (">", ">="):
                    bad_len = max(0, bound - 1) if op == ">=" else bound
                elif op in ("<", "<="):
                    bad_len = bound + 1 if op == "<=" else bound
                elif op == "==":
                    bad_len = bound + 1
                else:
                    bad_len = bound
                item[name] = _list_value_for_length(param, bad_len)
                push(item)

            for match in re.finditer(
                r"\b(?P<idx>[A-Za-z_]\w*)\s*<\s*(?P<vec>[A-Za-z_]\w*)\.len\(\)",
                text,
            ):
                idx_name, vec_name = match.group("idx"), match.group("vec")
                vec_val = seed.get(vec_name)
                if not isinstance(vec_val, list) or idx_name not in seed:
                    continue
                item = dict(seed)
                item[idx_name] = len(vec_val)
                push(item)
                item2 = dict(seed)
                item2[idx_name] = len(vec_val) + 1
                push(item2)

        # Generic single-parameter perturbations as fallback.
        for param in params:
            name = param["name"]
            type_text = param.get("type", "")
            item = dict(seed)
            cur = item.get(name)
            if is_vec_type(type_text) and isinstance(cur, list):
                item[name] = list(cur) + [_vec_elem_filler(param)]
                push(item)
                item2 = dict(seed)
                item2[name] = list(cur[:-1]) if cur else [_vec_elem_filler(param)]
                push(item2)
            elif is_int_like_type(type_text) and isinstance(cur, int):
                item[name] = cur + 1
                push(item)
                item2 = dict(seed)
                item2[name] = cur - 1 if not is_unsigned_type(type_text) or cur > 0 else cur + 2
                push(item2)
            elif is_char_type(type_text) and isinstance(cur, int):
                item[name] = cur + 1
                push(item)
            elif is_bool_type(type_text) and isinstance(cur, bool):
                item[name] = not cur
                push(item)
            if len(cases) >= budget * 4:
                break
        if len(cases) >= budget * 4:
            break

    return _dedupe_input_cases(cases)[: budget * 4]


def generate_boundary_candidate_inputs(context: dict, *, budget: int = 20) -> list[dict]:
    """生成边界/极端候选输入（不依赖显式 requires），用于运行时 panic 检测的兜底补充.

    复用 _build_invalid_seed_inputs 的边界生成策略：空向量、单元素、极值、
    多参数组合等。即使没有 requires 子句也能产出边界候选。
    """
    return _build_invalid_seed_inputs(context, budget=budget)


def llm_invalid_candidate_inputs(
    context: dict,
    *,
    budget: int = DEFAULT_IO_INVALID_CANDIDATES,
    attempt: int = 0,
    feedback: Optional[dict] = None,
) -> tuple[list[dict], dict]:
    params = [
        {"name": param.get("name"), "type": param.get("type")}
        for param in context.get("parameters") or []
        if param.get("name")
    ]
    if not params:
        return [], {"status": "skipped", "reason": "no_parameters"}
    requires = [clause.get("normalized") or clause.get("text") for clause in context.get("requires") or []]
    if not requires:
        return [], {"status": "skipped", "reason": "no_reference_requires"}
    spec_preamble = (context.get("spec_preamble") or "").strip()

    prompt = {
        "function": context.get("function"),
        "parameters": params,
        "requires": requires,
        "budget": budget,
        "task": (
            "Generate concrete INVALID input cases for Verus specification testing. "
            "Each case must violate at least one reference requires clause. "
            "Prefer MINIMAL single-clause violations derived from a nearly-valid input: "
            "unequal vector lengths when requires equal lengths, empty vectors for non-empty requirements, "
            "out-of-range indices (i >= a.len()), chars just outside quoted ranges, "
            "absent search elements for existential membership requirements, zero divisors, "
            "negative values for positive constraints, and values just outside numeric bounds. "
            "For nested collections, violate element-wise constraints (e.g. one empty inner vec). "
            "Return only inputs; do not include outputs."
        ),
        "output_schema": {
            "invalid_inputs": [
                {
                    "inputs": {"param_name": "concrete JSON value"},
                    "violated_requires": ["copied or paraphrased violated requires clause"],
                    "rationale": "why this input violates the reference precondition",
                }
            ]
        },
    }
    if spec_preamble:
        prompt["spec_definitions"] = spec_preamble
    if feedback:
        prompt["previous_attempt_feedback"] = feedback
    temperature = 0.7 if attempt > 0 else 0.15
    task_key = f"io_invalid_inputs:{context.get('function')}:{budget}"
    if attempt > 0:
        task_key = f"{task_key}:attempt{attempt}"
    result = call_llm_json(
        task=task_key,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate invalid concrete inputs for formal-spec testing. "
                    "Return strict JSON only. Every suggested case should violate the reference requires clauses. "
                    "If `spec_definitions` are provided, use them to understand helper predicates before violating them. "
                    "For nested collection types (e.g. Vec<Vec<T>>, Vec<String>), produce cases where an inner "
                    "element violates the element-wise requires (e.g. an empty inner vector when the requires "
                    "demands every inner element be non-empty). Vary structures across cases."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, indent=2)},
        ],
        temperature=temperature,
    )
    if result.get("status") != "ok":
        return [], result
    data = result.get("json")
    raw_inputs = data.get("invalid_inputs") if isinstance(data, dict) else data
    if not isinstance(raw_inputs, list):
        return [], {
            "status": "not_available",
            "reason": "llm_invalid_inputs_not_a_list",
            "llm": result.get("llm"),
            "raw": result.get("raw"),
            "cached": result.get("cached"),
        }
    candidates = []
    for raw_item in raw_inputs:
        validated = _validate_llm_input_case(context, raw_item)
        if validated is not None:
            candidates.append(validated)
    return _dedupe_input_cases(candidates)[:budget], {
        "status": "ok",
        "accepted_candidates": len(candidates[:budget]),
        "raw_candidates": len(raw_inputs),
        "llm": result.get("llm"),
        "cached": result.get("cached"),
    }


def llm_panic_candidate_inputs(
    context: dict,
    *,
    source_code: str = "",
    budget: int = DEFAULT_IO_INVALID_CANDIDATES,
    attempt: int = 0,
) -> tuple[list[dict], dict]:
    """用 LLM 分析函数体，生成可能导致运行时 panic 的输入候选.

    不依赖显式 requires：通过阅读函数体源码推断隐式前置条件（数组越界、
    除零、整数溢出、unwrap on None、长度不匹配等），生成违反这些隐式
    条件的输入。需配合 verus --compile 运行验证确认是否真的 panic。

    source_code 为函数所在源文件文本；若为空则仅基于签名/requires/ensures 推断。
    """
    params = [
        {"name": param.get("name"), "type": param.get("type")}
        for param in context.get("parameters") or []
        if param.get("name")
    ]
    if not params:
        return [], {"status": "skipped", "reason": "no_parameters"}

    requires_clauses = [
        clause.get("normalized") or clause.get("text")
        for clause in context.get("requires") or []
    ]
    prompt: dict = {
        "function": context.get("function"),
        "parameters": params,
        "requires": requires_clauses,
        "budget": budget,
        "task": (
            "Generate concrete INVALID inputs that VIOLATE the function's preconditions — "
            "both explicit `requires` clauses AND implicit runtime assumptions in the body. "
            "These inputs are used for spec-testing the function's robustness against illegal inputs. "
            "The `requires` clauses tell you what the function ASSUMES — your job is to BREAK those "
            "assumptions and find inputs that cause a runtime panic or undefined behaviour: "
            "empty vec where len>=1 is required, index out of bounds, division by zero, integer "
            "overflow/underflow (e.g. usize 0 - 1), unwrap on None, slice length mismatch, etc. "
            "Prefer DIVERSE boundary violations: empty vectors, single-element where >=2 needed, "
            "zero divisors, negative values for unsigned types, out-of-range indices, unequal-length "
            "pairs. Do NOT repeat the same violation (e.g. only one empty-vec case is needed). "
            "If the function truly handles ALL inputs gracefully (no possible panic even when "
            "preconditions are violated), return an empty list. "
            "Return only inputs; do not include outputs."
        ),
        "output_schema": {
            "panic_inputs": [
                {
                    "inputs": {"param_name": "concrete JSON value"},
                    "panic_reason": "which precondition is violated and the resulting runtime failure",
                }
            ]
        },
    }
    if source_code:
        prompt["function_body"] = source_code

    temperature = 0.6 if attempt > 0 else 0.35
    task_key = f"io_panic_invalid_inputs:{context.get('function')}:{budget}"
    if attempt > 0:
        task_key = f"{task_key}:attempt{attempt}"

    result = call_llm_json(
        task=task_key,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate INVALID inputs that violate a Rust/Verus function's preconditions "
                    "to trigger runtime panics. The `requires` clauses are the function's assumptions — "
                    "your job is to BREAK them and find inputs that cause panic (index out of bounds, "
                    "division by zero, overflow, empty-vec underflow, unwrap on None, etc.). "
                    "Return strict JSON only. If the function cannot panic even with violated "
                    "preconditions, return {\"panic_inputs\": []}."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, indent=2)},
        ],
        temperature=temperature,
    )
    if result.get("status") != "ok":
        return [], result
    data = result.get("json")
    raw_inputs = data.get("panic_inputs") if isinstance(data, dict) else data
    if not isinstance(raw_inputs, list):
        return [], {
            "status": "not_available",
            "reason": "llm_panic_inputs_not_a_list",
            "llm": result.get("llm"),
            "raw": result.get("raw"),
            "cached": result.get("cached"),
        }
    candidates = []
    for raw_item in raw_inputs:
        validated = _validate_llm_input_case(context, raw_item)
        if validated is not None:
            candidates.append(validated)
    deduped = _dedupe_input_cases(candidates)[:budget]
    return deduped, {
        "status": "ok",
        "accepted_candidates": len(deduped),
        "raw_candidates": len(raw_inputs),
        "llm": result.get("llm"),
        "cached": result.get("cached"),
    }


def generate_invalid_cases_from_reference_requires(
    context: dict,
    *,
    existing_cases: Sequence[dict] = (),
    budget: int = DEFAULT_IO_INVALID_CANDIDATES,
    positive_inputs: Sequence[dict] = (),
) -> tuple[list[dict], dict]:
    if not context.get("requires"):
        return [], {"status": "skipped", "reason": "no_reference_requires"}
    existing_keys = {
        json.dumps(case.get("inputs", {}), ensure_ascii=False, sort_keys=True)
        for case in existing_cases
        if isinstance(case, dict)
    }
    heuristic_candidates = _build_invalid_seed_inputs(context, budget=budget)
    positive_derived = generate_requires_violating_from_positives(
        context, positive_inputs, budget=budget,
    )
    invalid_cases: list[dict] = []
    uncertain = 0
    tested_keys = set(existing_keys)

    def validate(candidates: Sequence[dict], source: str) -> tuple[int, int, int]:
        nonlocal uncertain
        filtered: list[dict] = []
        for inputs in _dedupe_input_cases(candidates):
            key = json.dumps(inputs, ensure_ascii=False, sort_keys=True)
            if key in tested_keys:
                continue
            tested_keys.add(key)
            filtered.append(inputs)
        batch_cases = [{"inputs": inp, "key": f"inv_{source}_{i}"} for i, inp in enumerate(filtered)]
        verdicts = batch_verus_requires_decide(context, batch_cases)
        before = len(invalid_cases)
        local_uncertain = 0
        for i, inputs in enumerate(filtered):
            accepted = verdicts.get(f"inv_{source}_{i}")
            if accepted is False:
                invalid_cases.append({
                    "id": f"invalid_generated_{len(invalid_cases) + 1:03d}",
                    "kind": "invalid",
                    "inputs": inputs,
                    "tags": sorted(set(tag_input_values(inputs)) | {"requires_boundary", "generated_invalid"}),
                    "oracle": "reference_requires_negation",
                    "source": source,
                    "status": "validated",
                    "reference_requires_evaluation": {"accepted": False, "engine": "verus_batch"},
                })
                if len(invalid_cases) >= budget:
                    break
            elif accepted is None:
                uncertain += 1
                local_uncertain += 1
        return len(invalid_cases) - before, len(filtered), local_uncertain

    deterministic = _dedupe_input_cases([*positive_derived, *heuristic_candidates])
    det_added, det_tested, det_unknown = validate(deterministic, "heuristic")

    llm_candidates: list[dict] = []
    llm_generations: list[dict] = []
    no_progress = False
    if len(invalid_cases) < budget:
        feedback: Optional[dict] = None
        for attempt in (0, 1):
            generated, generation = llm_invalid_candidate_inputs(
                context,
                budget=budget,
                attempt=attempt,
                feedback=feedback,
            )
            llm_generations.append(generation)
            llm_candidates = _dedupe_input_cases([*llm_candidates, *generated])
            added, tested, unknown = validate(generated, f"llm{attempt}")
            if len(invalid_cases) >= budget:
                break
            feedback = {
                "accepted_invalid": added,
                "tested_candidates": tested,
                "verification_unknown": unknown,
                "duplicate_or_rejected": max(0, len(generated) - tested),
                "instruction": "Generate different, type-correct single-clause violations that address these failures.",
            }
            if added == 0 and (attempt == 1 or not generated):
                no_progress = True
                if not generated:
                    break

    llm_generation = llm_generations[-1] if llm_generations else {"status": "skipped", "reason": "deterministic_target_met"}

    metadata = {
        "status": "ok" if invalid_cases else "not_available",
        "reason": None if invalid_cases else "no_generated_invalid_cases",
        "llm_generation": llm_generation,
        "llm_candidate_inputs": len(llm_candidates),
        "heuristic_candidate_inputs": len(heuristic_candidates),
        "positive_derived_candidates": len(positive_derived),
        "deterministic_tested": det_tested,
        "deterministic_validated": det_added,
        "deterministic_unknown": det_unknown,
        "validated_invalid": len(invalid_cases),
        "uncertain_invalid_candidates": uncertain,
        "llm_attempts": len(llm_generations),
        "stopped_no_progress": no_progress,
        "method": "reference_requires_negation_llm_plus_heuristic",
    }
    return invalid_cases[:budget], metadata


def _search_output_mutation_label(context: dict, inputs: dict, output: dict, name: str, candidate: Any, label: str) -> str:
    function_name = str(context.get("function", "")).lower()
    if "search" not in function_name and "find" not in function_name:
        return label
    if not isinstance(candidate, int):
        return label
    vec_name, scalar_name = first_vec_scalar_pair(context, inputs)
    seq = inputs.get(vec_name) if vec_name else None
    needle = inputs.get(scalar_name) if scalar_name else None
    correct = output.get(name)
    if (
        isinstance(seq, list)
        and isinstance(needle, int)
        and isinstance(correct, int)
        and 0 <= candidate < len(seq)
        and needle in seq
        and correct == seq.index(needle)
        and candidate != correct
        and seq[candidate] == needle
    ):
        return "non_first_match"
    return label


def _split_tuple_type_parts(type_text: str) -> list[str]:
    text = str(type_text).strip()
    if not (text.startswith("(") and text.endswith(")")):
        return []
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text[1:-1]:
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        if char == "," and depth == 0:
            if "".join(current).strip():
                parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if "".join(current).strip():
        parts.append("".join(current).strip())
    return parts


def mutate_output_values(context: dict, inputs: dict, output: dict) -> list[tuple[dict, str]]:
    """Mutate expected outputs to produce negative IO cases.

    Semantic cousin of mutation.py BOUNDARY/ROR-style perturbations, but operates on
    *runtime JSON values* rather than source tokens. Type dispatch must use top-level
    predicates (is_vec_type before is_bool_type) so Vec<bool> is never treated as bool —
    the same nesting pitfall mutation testing avoids by only rewriting true/false literals.
    """
    returns = list(context.get("returns") or [])
    registry = context.get("type_registry")
    input_lists = [value for value in inputs.values() if isinstance(value, list)]
    max_len = max([len(value) for value in input_lists], default=0)
    mutations: list[tuple[dict, str]] = []
    for ret in returns:
        name = ret.get("name") or "ret"
        if name not in output:
            continue
        type_text = ret.get("type", "")
        value = output[name]
        candidates: list[tuple[Any, str]] = []
        stripped_type = normalize_value_type(type_text)
        registry_def = registry_entry(registry, stripped_type)
        tuple_types = _split_tuple_type_parts(stripped_type)
        if registry_def is not None and registry_def["kind"] == "enum":
            for variant in registry_def["variants"]:
                if variant != value:
                    candidates.append((variant, f"enum_variant_{variant}"))
        elif registry_def is not None and isinstance(value, list) and len(value) == len(registry_def["fields"]):
            for index, (item, (fname, field_type)) in enumerate(zip(value, registry_def["fields"])):
                nested_context = {
                    "returns": [{"name": "element", "type": field_type}],
                    "type_registry": registry,
                }
                for nested, nested_label in mutate_output_values(nested_context, inputs, {"element": item}):
                    candidate = list(value)
                    candidate[index] = nested["element"]
                    candidates.append((candidate, f"struct_{fname}_{nested_label}"))
        elif tuple_types and isinstance(value, list) and len(value) == len(tuple_types):
            for index, (element, element_type) in enumerate(zip(value, tuple_types)):
                nested_context = {"returns": [{"name": "element", "type": element_type}]}
                nested_mutations = mutate_output_values(
                    nested_context,
                    inputs,
                    {"element": element},
                )
                for nested, nested_label in nested_mutations:
                    candidate = list(value)
                    candidate[index] = nested["element"]
                    candidates.append((candidate, f"tuple_{index}_{nested_label}"))
        # Vec/list first — must not treat Vec<bool> as scalar bool.
        elif is_nested_vec_type(type_text) and isinstance(value, list):
            seq = list(value)
            inner_ty = re.sub(
                r"^(?:Vec|Seq)\s*<\s*(.*)\s*>\s*$",
                r"\1",
                stripped_type,
            )
            candidates.extend([
                (seq[:-1], "nested_vec_drop_last"),
                (seq + [[]], "nested_vec_append_empty"),
            ])
            if seq and isinstance(seq[0], list):
                nested_context = {"returns": [{"name": "element", "type": inner_ty}]}
                for nested, label in mutate_output_values(nested_context, inputs, {"element": seq[0]}):
                    candidate = list(seq)
                    candidate[0] = nested["element"]
                    candidates.append((candidate, f"nested_vec_first_{label}"))
        elif isinstance(value, list) or is_vec_type(type_text):
            seq = list(value) if isinstance(value, list) else []
            inner_ty = re.sub(
                r"^(?:Vec|Seq)\s*<\s*(.*)\s*>\s*$",
                r"\1",
                normalize_value_type(type_text),
            )
            bool_elems = is_bool_type(inner_ty) or (bool(seq) and all(isinstance(x, bool) for x in seq))
            # Vec<char> 元素以 ordinal/单字符表示，不能按字符串元素变异。
            str_elems = (not is_char_type(inner_ty)) and (
                _is_string_like_type(inner_ty)
                or (bool(seq) and all(isinstance(x, str) for x in seq))
            )
            float_elems = is_float_type(inner_ty) or (
                bool(seq)
                and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in seq)
                and any(isinstance(x, float) for x in seq)
            )
            if bool_elems:
                append_val: Any = False
            elif str_elems:
                append_val = ""
            elif float_elems:
                append_val = 0.0
            else:
                append_val = 0
            if bool_elems and seq:
                replace_val: Any = not seq[0]
            elif str_elems:
                replace_val = "zz" if (not seq or seq[0] != "zz") else "yy"
            elif float_elems:
                replace_val = 999.0
            else:
                replace_val = True if bool_elems else 999
            candidates.extend(
                [
                    (seq[:-1], "vec_drop_last"),
                    (seq + [append_val], "vec_append"),
                    (list(reversed(seq)), "vec_reverse"),
                    ([replace_val] + seq[1:] if seq else [replace_val], "vec_replace_first"),
                ]
            )
            if bool_elems and seq:
                flipped = list(seq)
                flipped[0] = not flipped[0]
                candidates.append((flipped, "vec_bool_flip_first"))
        elif _is_string_like_type(type_text) or (isinstance(value, str) and not is_char_type(type_text)):
            s = value if isinstance(value, str) else ""
            raw_string_candidates = [
                (s + "x", "str_append_char"),
                (s[:-1] if s else "x", "str_drop_last"),
                ("", "str_empty"),
                (s.swapcase(), "str_case_swap"),
                (s[::-1], "str_reverse"),
                ("y" + s, "str_prepend_char"),
            ]
            seen_strings = {s}
            for cand, label in raw_string_candidates:
                if cand not in seen_strings:
                    seen_strings.add(cand)
                    candidates.append((cand, label))
        elif isinstance(value, bool) or is_bool_type(type_text):
            candidates.append((not bool(value), "bool_flip"))
        elif is_char_type(type_text):
            code = int(value) if isinstance(value, int) else (ord(value) if isinstance(value, str) and len(value) == 1 else ord("a"))
            candidates.extend(
                [
                    (code + 1, "char_next"),
                    (code - 1 if code > 0 else code + 2, "char_prev"),
                    (ord("a"), "char_a"),
                    (ord("z"), "char_z"),
                    (ord("0"), "char_digit"),
                ]
            )
        elif isinstance(value, float) or is_float_type(type_text):
            base = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
            candidates.extend(
                [
                    (base + 1.0, "float_plus_one"),
                    (base - 1.0, "float_minus_one"),
                    (-base if base != 0 else 1.0, "float_negate"),
                    (0.0, "float_zero"),
                    (1.0, "float_one"),
                ]
            )
        elif isinstance(value, int):
            raw_values = [
                (value + 1, "off_by_one_plus"),
                (value - 1, "off_by_one_minus"),
                (0, "zero_output"),
                (max_len, "len_sentinel"),
                (max_len + 1, "out_of_bounds_sentinel"),
            ]
            if input_lists:
                first = input_lists[0]
                if value != 0:
                    raw_values.append((0, "first_index"))
                if len(first) > 1:
                    raw_values.append((len(first) - 1, "last_index"))
            if is_unsigned_type(type_text):
                raw_values = [(candidate, label) for candidate, label in raw_values if candidate >= 0]
            candidates.extend(raw_values)
        elif isinstance(value, str) or normalize_value_type(type_text) in {"String", "str"}:
            text = str(value)
            candidates.extend(
                [
                    (text + "x", "str_append"),
                    (text[:-1] if text else "x", "str_drop_last"),
                    ("", "str_empty"),
                    ("a", "str_singleton"),
                ]
            )
        elif value is None or (isinstance(value, dict) and "Some" in value):
            if value is None:
                candidates.append((0, "option_none_to_some"))
            else:
                candidates.append((None, "option_some_to_none"))
                inner = value.get("Some") if isinstance(value, dict) else None
                if isinstance(inner, int):
                    candidates.append(({"Some": inner + 1}, "option_inner_plus"))
                elif isinstance(inner, bool):
                    candidates.append(({"Some": not inner}, "option_inner_flip"))

        for candidate, label in candidates:
            if candidate == value:
                continue
            label = _search_output_mutation_label(context, inputs, output, name, candidate, label)
            mutated = dict(output)
            mutated[name] = list(candidate) if isinstance(candidate, list) else candidate
            mutations.append((mutated, label))

    deduped: list[tuple[dict, str]] = []
    seen: set[str] = set()
    for mutated, label in mutations:
        key = json.dumps(mutated, sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append((mutated, label))
    return deduped


def _parse_benchmark_scalar(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "true":
            return True
        if stripped == "false":
            return False
        if stripped == "()":
            return None
        if re.fullmatch(r"-?\d+", stripped):
            try:
                return int(stripped)
            except ValueError:
                pass
        if re.fullmatch(r"-?\d+\.\d+", stripped):
            try:
                return float(stripped)
            except ValueError:
                pass
        if stripped.startswith("Some(") and stripped.endswith(")"):
            return {"Some": _parse_benchmark_scalar(stripped[5:-1])}
        if stripped == "None":
            return None
    return value


def _parse_benchmark_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, tuple):
                    parsed = list(parsed)
                return _parse_benchmark_value(parsed)
            except (SyntaxError, ValueError):
                return stripped
        if stripped.startswith("(") and stripped.endswith(")"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, tuple):
                    return [_parse_benchmark_value(item) for item in parsed]
                return _parse_benchmark_value(parsed)
            except (SyntaxError, ValueError):
                return stripped
        # Preserve the original scalar text so char/String inputs such as a
        # space or newline are not collapsed to an empty string. The scalar
        # parser still strips internally when recognizing numbers and bools.
        return _parse_benchmark_scalar(value)
    if isinstance(value, list):
        return [_parse_benchmark_value(item) for item in value]
    if isinstance(value, tuple):
        return [_parse_benchmark_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _parse_benchmark_value(item) for key, item in value.items()}
    return value


def _parse_benchmark_value_typed(value: Any, type_text: str) -> Any:
    """Parse a stored benchmark value using the declared Rust type.

    char/字符串类型（含 Vec/数组元素）保留原文本，不做“数字字符串→int”转换：
    旧格式语料把 Vec<char> 存成 "['a', '1']" 这样的 repr 文本，无类型解析会把
    数字字符 '1' 变成 int 1，下游按码点渲染成 '\\u{0001}'（RQ7 S5 假缺陷根因）。
    新语料的 char 码点（int）与其他类型保持原行为；未识别类型回退无类型解析。
    """
    stripped_type = normalize_value_type(type_text)
    if not stripped_type:
        return _parse_benchmark_value(value)
    if is_char_type(stripped_type) or _is_string_like_type(stripped_type):
        # str 保留字面内容（含 "5" 等数字样式）；int 是 char 的码点约定。
        return value
    is_sequence_type = (
        is_vec_type(stripped_type)
        or stripped_type.startswith("[")
        or stripped_type.startswith("Seq")
    )
    if isinstance(value, str) and is_sequence_type:
        text = value.strip()
        if text.startswith(("[", "(")) and text.endswith(("]", ")")):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return _parse_benchmark_value(value)
            if isinstance(parsed, tuple):
                parsed = list(parsed)
            return _parse_benchmark_value_typed(parsed, stripped_type)
        return _parse_benchmark_value(value)
    if isinstance(value, (list, tuple)) and is_sequence_type:
        inner = _extract_vec_element_type(stripped_type)
        return [_parse_benchmark_value_typed(item, inner) for item in value]
    return _parse_benchmark_value(value)


def _parse_benchmark_inputs(context: dict, raw_inputs: Any) -> Optional[dict]:
    """Parse stored inputs without turning scalar text/char values into numbers."""
    if not isinstance(raw_inputs, dict):
        return None
    parameter_types = {
        str(param.get("name") or ""): str(param.get("type") or "")
        for param in context.get("parameters") or []
        if param.get("name")
    }
    parsed: dict[str, Any] = {}
    for name, value in raw_inputs.items():
        type_text = parameter_types.get(str(name), "")
        if type_text:
            parsed[str(name)] = _parse_benchmark_value_typed(value, type_text)
        else:
            parsed[str(name)] = _parse_benchmark_value(value)
    return parsed


def _return_payload_from_benchmark(context: dict, raw_value: Any) -> dict:
    returns = list(context.get("returns") or [])
    if len(returns) == 1 and not isinstance(raw_value, dict):
        return_type = str(returns[0].get("type") or "")
        return {
            returns[0].get("name")
            or "ret": _parse_benchmark_value_typed(raw_value, return_type)
        }
    value = _parse_benchmark_value(raw_value)
    return_names = [ret.get("name") or f"ret_{index}" for index, ret in enumerate(returns)]
    if isinstance(value, dict) and return_names and all(name in value for name in return_names):
        return {name: value[name] for name in return_names}
    if len(returns) == 1:
        return {returns[0].get("name") or "ret": value}
    if isinstance(value, list) and len(value) == len(returns):
        return {
            ret.get("name") or f"ret_{index}": value[index]
            for index, ret in enumerate(returns)
        }
    if returns:
        return {returns[0].get("name") or "ret": value}
    return {"result": value}


def strict_wrong_output_decision(evaluation: dict, requires_accepted: Optional[bool]) -> dict:
    """Require P and not Q; rejection of P alone is not output rejection.

    The pair decision proves P and Q, or its negation. Together with an
    independent proof of P, the latter proves not Q. Unproved directions
    remain unknown rather than being interpreted as counterexamples.
    """
    result = dict(evaluation)
    accepted = result.get("accepted")
    if accepted is True:
        success, reason = False, "wrong_output_accepted"
    elif requires_accepted is False:
        success, reason = False, "wrong_input_rejected"
    elif requires_accepted is True and accepted is False:
        success, reason = True, "admitted_input_output_rejected"
    elif requires_accepted is None and accepted is False:
        success, reason = None, "input_admission_unresolved"
    else:
        success, reason = None, result.get("reason") or "output_rejection_unresolved"
    result.update(
        success=success, requires_accepted=requires_accepted,
        pair_reason=evaluation.get("reason"), reason=reason,
        check_policy="admitted_input_and_rejected_output_v1",
    )
    return result


def score_io_cases(path: str, suite: dict, kind: str) -> dict:
    category = (suite.get("category_status") or {}).get(kind) or {}
    cases = [
        case for case in suite.get("cases", [])
        if case.get("kind") == kind and case.get("status") == "validated"
    ]
    function_name = str(suite.get("function") or "")
    # The reference-selected task is authoritative. Never fall back to a
    # helper when the generated file lacks that exact function.
    context = context_for_suite_function(path, function_name, fallback=False)
    if context is None:
        return _target_mismatch_score(
            suite, kind, "target_function_not_found",
            expected_function=function_name,
        )
    generated_target = _target_descriptor_for_generated(path, function_name, context)
    reference_target = suite.get("target") or {}
    if generated_target is None:
        return _target_mismatch_score(
            suite, kind, "generated_target_not_found",
            expected_function=function_name,
        )
    # Category status describes the generation target, not whether existing
    # validated cases can be scored (a blocked category can contain 1--4 cases).
    if not cases and category.get("state") in {"not_applicable", "blocked"}:
        return {
            "status": "not_available",
            "score": None,
            "reason": f"category_{category.get('state')}",
            "category_status": category,
            "target": reference_target or generated_target,
        }
    context = _observable_io_context(context, generated_target)
    reference_parameters = list(reference_target.get("parameters") or [])
    generated_parameters = list(generated_target.get("parameters") or [])
    reference_outputs = _observable_output_fields(reference_target)
    generated_outputs = _observable_output_fields(generated_target)
    if not cases:
        return {
            "status": "not_available",
            "score": None,
            "reason": f"no_validated_{kind}_cases",
            "suite": suite,
        }

    parameter_signature_issue = _positional_signature_issue(
        reference_parameters, generated_parameters, category="parameter",
    )
    output_signature_issue = _positional_signature_issue(
        reference_outputs, generated_outputs, category="output",
    )
    preflight_details: dict[str, dict[str, Any]] = {}

    if kind in ("positive", "negative"):
        batch_cases = []
        batch_details: dict[str, dict[str, Any]] = {}
        negative_requires: dict[str, Optional[bool]] = {}
        signature_issue = parameter_signature_issue or output_signature_issue
        if signature_issue:
            for i in range(len(cases)):
                batch_details[f"sc_{i}"] = {
                    "accepted": None,
                    "engine": "io_signature_preflight",
                    **signature_issue,
                }
        else:
            for i, case in enumerate(cases):
                key = f"sc_{i}"
                output = case.get("mutated_output", {}) if kind == "negative" else case.get("output", {})
                mapped_inputs = _remap_payload_by_position(
                    case.get("inputs", {}), reference_parameters, generated_parameters,
                )
                mapped_output = _remap_payload_by_position(
                    output, reference_outputs, generated_outputs,
                )
                if mapped_inputs is None or mapped_output is None:
                    batch_details[key] = {
                        "accepted": None,
                        "reason": "payload_mapping_failed",
                        "engine": "io_payload_preflight",
                    }
                    continue
                batch_cases.append({"inputs": mapped_inputs, "output": mapped_output, "key": key})
            batch_details.update(batch_verus_contract_decide_detailed(context, batch_cases))
            if kind == "negative":
                negative_requires = batch_verus_requires_decide(context, batch_cases)
    elif kind == "invalid":
        batch_cases = []
        batch_result = {f"sc_{i}": None for i in range(len(cases))}
        if parameter_signature_issue:
            for i in range(len(cases)):
                preflight_details[f"sc_{i}"] = {
                    "accepted": None,
                    "engine": "io_signature_preflight",
                    **parameter_signature_issue,
                }
        else:
            for i, case in enumerate(cases):
                key = f"sc_{i}"
                mapped_inputs = _remap_payload_by_position(
                    case.get("inputs", {}), reference_parameters, generated_parameters,
                )
                if mapped_inputs is None:
                    preflight_details[key] = {
                        "accepted": None,
                        "reason": "payload_mapping_failed",
                        "engine": "io_payload_preflight",
                    }
                    continue
                batch_cases.append({"inputs": mapped_inputs, "key": key})
            batch_result.update(batch_verus_requires_decide(context, batch_cases))
        runtime_cases = [
            case for case in batch_cases
            if batch_result.get(str(case.get("key"))) is not False
        ]
        runtime_result = _run_generated_invalid_inputs(path, function_name, runtime_cases)
    else:
        batch_result = {}
        runtime_result = {}

    passed = 0
    failed = 0
    unknown = 0
    details: list[dict] = []
    oracle_counts: dict[str, dict[str, int]] = {}
    unknown_reason_summary: dict[str, int] = {}
    for i, case in enumerate(cases):
        key = f"sc_{i}"
        if kind in ("positive", "negative"):
            evaluation = dict(batch_details.get(key) or {
                "accepted": None,
                "engine": "verus_contract_dual_proof",
                "reason": "verification_unresolved",
            })
            outcome = evaluation.get("accepted")
            if kind == "negative":
                evaluation = strict_wrong_output_decision(evaluation, negative_requires.get(key))
                success = evaluation["success"]
            elif outcome is None:
                success = None
            else:
                success = outcome is True
        elif kind == "invalid":
            outcome = batch_result.get(key)
            preflight = preflight_details.get(key)
            runtime = runtime_result.get(key) or {"status": "UNKNOWN", "reason": "not_run"}
            runtime_status = str(runtime.get("status") or "UNKNOWN")
            if preflight is not None:
                success = None
                reason = str(preflight.get("reason") or "verification_unresolved")
                engine = str(preflight.get("engine") or "io_preflight")
            elif outcome is False:
                success: Optional[bool] = True
                reason = "requires_rejected"
                engine = "verus_requires_dual_proof"
            elif runtime_status == "PANIC":
                success = True
                reason = "explicit_runtime_panic"
                engine = "verus_native_runtime"
            elif runtime_status == "OK":
                success = False
                reason = "runtime_ok_without_requires_rejection"
                engine = "verus_requires+native_runtime"
            else:
                success = None
                reason = (
                    "runtime_timeout" if runtime_status == "TIMEOUT"
                    else "runtime_or_requires_unknown"
                )
                engine = "verus_requires+native_runtime"
            evaluation = {
                "accepted": outcome,
                "engine": engine,
                "reason": reason,
                "requires_accepted": outcome,
                "runtime_status": runtime_status,
                "runtime_reason": runtime.get("reason"),
            }
            if preflight and preflight.get("detail"):
                evaluation["reason_detail"] = preflight["detail"]
        else:
            evaluation = {"accepted": None, "reason": "unsupported_case_kind"}
            success = None

        if success is None:
            unknown += 1
            unknown_reason = str(evaluation.get("reason") or "verification_unresolved")
            unknown_reason_summary[unknown_reason] = unknown_reason_summary.get(unknown_reason, 0) + 1
        elif success is True:
            passed += 1
        else:
            failed += 1
        details.append(
            {
                "id": case.get("id"),
                "kind": kind,
                "success": success,
                "evaluation": evaluation,
                "tags": case.get("tags", []),
                "oracle": case.get("oracle"),
                "harness": io_harness_for_case(context, case, kind),
            }
        )
        oracle = str(case.get("oracle") or "unknown")
        counts = oracle_counts.setdefault(oracle, {"passed": 0, "failed": 0, "unknown": 0})
        if success is None:
            counts["unknown"] += 1
        elif success is True:
            counts["passed"] += 1
        else:
            counts["failed"] += 1

    total = len(cases)
    denominator = passed + failed
    # Conservative all-case score: unresolved checks stay visible and contribute
    # zero instead of silently disappearing from plots and correlations.
    score = passed / total
    coverage = denominator / total
    decidable_score = passed / denominator if denominator else None
    status = "ok" if unknown == 0 else "partial"
    result = {
        "status": status,
        "score": score,
        "coverage": coverage,
        "decidable_score": decidable_score,
        "passed": passed,
        "failed": failed,
        "unknown": unknown,
        "unknown_reason_summary": unknown_reason_summary,
        "total": total,
        "evaluated": denominator,
        "target": reference_target or generated_target,
        "category_status": category or None,
        "oracle_summary": oracle_counts,
        "suite": suite,
        "details": details[:50],
        "scoring_policy": "all_validated_cases_unknown_zero",
        "check_policy": (
            "admitted_input_and_rejected_output_v1" if kind == "negative"
            else "accepted_pair_v1" if kind == "positive"
            else "requires_rejection_or_runtime_panic_v1"
        ),
        "note": "SAFE/SpecRL-style concrete I/O evaluation; unresolved cases remain unknown and contribute zero to the all-case score.",
    }
    if denominator == 0 and len(unknown_reason_summary) == 1:
        result["reason"] = next(iter(unknown_reason_summary))
    return result


def load_offline_io_suite(reference_path: str) -> Optional[dict]:
    """按 stem 从 io_suite_dir 加载离线生成的 test.json（generate_io_tests_llm.py 产出）。

    离线套件在生成阶段已用 reference 的 Verus 契约验证过 positive/negative/invalid，
    因此这里直接采信，不再对 reference 重跑 Verus。所有 case 标为 status="validated"。

    返回 None 表示：未配置 io_suite_dir / 找不到该 stem 的套件 / source_hash 不匹配
    （即离线所用 reference 与当前 ground 不是同一份文件）——调用方据此 fail-loud。
    """
    suite_root = _load_io_suite_dir_from_config()
    if suite_root is None:
        return None

    # 分层布局：<suite_root>/<benchmark>/<task>/，与 data/io
    # 及 scripts/evaluation/evaluate_analysis_dataset.py 的 _io_dir_for_reference 一致。
    # benchmark = reference 文件的父目录名（verified/<benchmark>/<task>.rs）。
    ref_path_obj = Path(reference_path)
    benchmark = ref_path_obj.parent.name
    io_dir = Path(suite_root) / benchmark / ref_path_obj.stem
    test_path = io_dir / "test.json"
    meta_path = io_dir / "meta.json"
    if not test_path.exists():
        return None
    try:
        raw_cases = json.loads(test_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw_cases, list):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        meta = {}

    strict_schema = int(meta.get("schema_version", 0) or 0) >= 3
    authoritative = authoritative_target_for_path(reference_path)
    if strict_schema:
        if (
            meta.get("validation_complete") is not True
            or authoritative is None
            or meta.get("function") != authoritative.get("function")
        ):
            return None
        target = authoritative.get("context")
    elif meta.get("function"):
        target = context_for_suite_function(
            reference_path, str(meta["function"]), fallback=False,
        )
    else:
        target = choose_io_target_context(reference_path)
    if target is None:
        return None

    target_descriptor = _target_descriptor_from_meta(meta, target)
    if strict_schema and not target_signatures_match(authoritative, target_descriptor):
        return None
    observable_target = _observable_io_context(target, target_descriptor)

    # 校验"同一份文件"：离线生成时的 reference 必须与当前 ground 契约一致
    expected_hash = meta.get("source_hash")
    if not expected_hash or expected_hash != source_hash(reference_path):
        return None

    cases: list[dict] = []
    positive_count = 0
    invalid_count = 0
    raw_audit = meta.get("case_audit") or {}
    if strict_schema and not isinstance(raw_audit, dict):
        return None
    invalid_entries = raw_audit.get("invalid") or []
    positive_entries = raw_audit.get("positive") or []
    negative_entries = raw_audit.get("negative") or []
    if strict_schema and not all(
        isinstance(entries, list)
        for entries in (invalid_entries, positive_entries, negative_entries)
    ):
        return None
    invalid_audit = {
        str(item.get("case_key")): item
        for item in invalid_entries
        if isinstance(item, dict) and item.get("case_key")
    }
    positive_audit = {
        str(item.get("case_key")): item
        for item in positive_entries
        if isinstance(item, dict) and item.get("case_key")
    }
    negative_audit = {
        str(item.get("case_key")): item
        for item in negative_entries
        if isinstance(item, dict) and item.get("case_key")
    }
    used_audit_keys = {kind: set() for kind in ("positive", "negative", "invalid")}
    for raw in raw_cases:
        if not isinstance(raw, dict) or not isinstance(raw.get("input"), dict):
            if strict_schema:
                return None
            continue
        parsed_inputs = _parse_benchmark_inputs(target, raw.get("input"))
        if not isinstance(parsed_inputs, dict):
            if strict_schema:
                return None
            continue
        inputs = typed_input_payload(target, parsed_inputs)
        if inputs is None:
            if strict_schema:
                return None
            continue
        tags = tag_input_values(inputs)
        expected = raw.get("expected")

        if expected == "INVALID_INPUT":
            invalid_count += 1
            audit_key = _audit_case_key({"input": raw.get("input"), "kind": "invalid"})
            audit = invalid_audit.get(audit_key, {})
            audit_oracle = audit.get("oracle")
            if strict_schema and (
                audit.get("state") != "validated"
                or audit.get("function") != target.get("function")
                or audit_oracle not in {"reference_runtime_panic", "reference_requires_rejection"}
                or (
                    audit_oracle == "reference_requires_rejection"
                    and audit.get("engine") != "verus_requires_dual_proof"
                )
                or (
                    audit_oracle == "reference_runtime_panic"
                    and audit.get("engine") != "verus_native_runtime"
                )
                or audit_key in used_audit_keys["invalid"]
                or (
                    audit_oracle == "reference_requires_rejection"
                    and audit.get("requires_verdict") is not False
                )
                or (
                    audit_oracle == "reference_runtime_panic"
                    and audit.get("runtime_status") != "PANIC"
                )
            ):
                return None
            used_audit_keys["invalid"].add(audit_key)
            oracle = (
                "offline_reference_runtime_panic"
                if audit_oracle in {"runtime_panic", "reference_runtime_panic"}
                else "offline_reference_requires_rejection"
            )
            cases.append({
                "id": f"invalid_{invalid_count:03d}",
                "kind": "invalid",
                "inputs": inputs,
                "tags": sorted(set(tags) | {"requires_boundary", "offline"}),
                "oracle": oracle,
                "status": "validated",
            })
            continue

        output = _return_payload_from_benchmark(observable_target, expected)
        typed_output = typed_output_payload(observable_target, output)
        if typed_output is None:
            if strict_schema:
                return None
            continue
        positive_key = _audit_case_key({"input": raw.get("input"), "expected": expected})
        positive_detail = positive_audit.get(positive_key, {})
        if strict_schema and (
            positive_detail.get("state") != "validated"
            or positive_detail.get("function") != target.get("function")
            or positive_detail.get("oracle") != "reference_execution_exact"
            or positive_detail.get("engine")
            != "verus_native_runtime+verus_requires_dual_proof"
            or positive_detail.get("requires_verdict") is not True
            or positive_detail.get("runtime_status") != "OK"
            or positive_key in used_audit_keys["positive"]
        ):
            return None
        used_audit_keys["positive"].add(positive_key)
        positive_count += 1
        positive_id = f"pos_{positive_count:03d}"
        pos_tags = sorted(set(tags) | {"normal", "offline"})
        cases.append({
            "id": positive_id,
            "kind": "positive",
            "inputs": inputs,
            "output": typed_output,
            "tags": pos_tags,
            "oracle": "offline_reference_contract_acceptance",
            "status": "validated",
        })

        unexpected = raw.get("unexpected")
        if not isinstance(unexpected, list):
            if strict_schema:
                return None
            continue
        neg_index = 0
        for item in unexpected:
            mutated_output = _return_payload_from_benchmark(observable_target, item)
            typed_mutated_output = typed_output_payload(observable_target, mutated_output)
            if typed_mutated_output is None:
                if strict_schema:
                    return None
                continue
            negative_key = _audit_case_key({"input": raw.get("input"), "unexpected": item})
            negative_detail = negative_audit.get(negative_key, {})
            if strict_schema and (
                negative_detail.get("state") != "validated"
                or negative_detail.get("function") != target.get("function")
                or negative_detail.get("oracle") != "reference_contract_rejection"
                or negative_detail.get("engine") != "verus_contract_dual_proof"
                or negative_detail.get("contract_verdict") is not False
                or negative_key in used_audit_keys["negative"]
            ):
                return None
            used_audit_keys["negative"].add(negative_key)
            neg_index += 1
            cases.append({
                "id": f"neg_{positive_count:03d}_{neg_index}",
                "kind": "negative",
                "base_case": positive_id,
                "inputs": inputs,
                "mutated_output": typed_mutated_output,
                "tags": sorted(set(pos_tags) | {"unexpected_output", "offline"}),
                "oracle": "offline_reference_rejection",
                "status": "validated",
            })

    category_status = meta.get("category_status") if isinstance(meta.get("category_status"), dict) else {}
    if strict_schema:
        loaded_counts = {
            "positive": positive_count,
            "negative": sum(1 for case in cases if case.get("kind") == "negative"),
            "invalid": invalid_count,
        }
        audit_entries = {
            "positive": positive_entries,
            "negative": negative_entries,
            "invalid": invalid_entries,
        }
        for audit_kind, count in loaded_counts.items():
            category = category_status.get(audit_kind)
            if (
                int(meta.get(f"{audit_kind}_count", -1)) != count
                or not isinstance(category, dict)
                or int(category.get("count", -1)) != count
                or len(audit_entries[audit_kind]) != count
                or len(used_audit_keys[audit_kind]) != count
            ):
                return None
    if not cases and not category_status:
        return None
    return {
        "status": "ok",
        "function": target.get("function"),
        "target": target_descriptor,
        "category_status": category_status,
        "source_hash": source_hash(reference_path),
        "generator": {
            "backend": "offline_io_suite",
            "io_dir": str(io_dir),
            "note": (
                "Loaded pre-generated offline IO suite (generate_io_tests_llm.py). "
                "Cases were validated against the reference contract at generation time and are "
                "trusted here; only the submitted (generated) spec is scored with Verus."
            ),
        },
        "cases": cases,
        "summary": {
            "validated_positive": positive_count,
            "validated_negative": sum(1 for c in cases if c.get("kind") == "negative"),
            "validated_invalid": invalid_count,
            "offline_meta": {
                k: meta.get(k)
                for k in ("schema_version", "function", "return_type", "runner", "status", "category_status")
                if k in meta
            },
        },
    }


def io_metric_pair(generated_rs_path: str, ground_rs_path: str, kind: str) -> dict:
    suite = load_offline_io_suite(ground_rs_path)
    if suite is None:
        return not_available_metric("offline_io_suite_unavailable", source="io_test_pipeline")
    alignment = resolve_pair_target(
        generated_rs_path,
        ground_rs_path,
        preferred_name=suite.get("function"),
    )
    if alignment.get("status") != "ok":
        generated = _target_mismatch_score(suite, kind, str(alignment.get("reason") or "target_mismatch"))
        generated["target_alignment"] = alignment
        return {
            "generated": generated,
            "target": suite.get("target"),
            "target_alignment": alignment,
            "suite_summary": suite.get("summary", {}),
        }
    suite = {**suite, "target_alignment": alignment}
    generated = score_io_cases(generated_rs_path, suite, kind)
    return {
        "generated": generated,
        "target": suite.get("target"),
        "target_alignment": alignment,
        "suite_summary": suite.get("summary", {}),
    }


__all__ = [
    "DEFAULT_IO_INVALID_CANDIDATES",
    "DEFAULT_IO_POSITIVE_CANDIDATES",
    "io_metric_pair",
    "load_offline_io_suite",
    "score_io_cases",
    "set_io_suite_dir",
]
