#!/usr/bin/env python3
"""离线 IO 测试用例生成器（LLM 驱动输入 + verus --no-verify --compile 运行 + Verus 判定）.

对每个参考 .rs 文件生成 test.json/meta.json，与 data/io
格式兼容，可被 metrics_rebuild/share/io_cases.py 的 load_offline_io_suite 复用。

每题目标：>= per_kind 个 positive / negative / invalid（默认各 5 个）。

流程：
  1. 解析 exec 函数 + 契约（parameters/returns/requires/ensures）
  2. LLM 生成候选输入（不可用时启发式兜底）
  3. 构造 Verus harness：参考代码原样保留在 verus! 块内（移除原 fn main），
     块外写 fn main + 类型化输出格式化 + catch_unwind 调用循环
  4. verus --no-verify --compile -C panic=unwind 编译运行：
     OK+输出 => positive 候选；PANIC => panic-invalid
  5. Verus requires 检查：违反 => requires-invalid（Verus 不可用时跳过）
  6. 变异 positive 输出 + Verus ensures 检查：违反 => negative（Verus 不可用时不采信）
  7. 各类不足 per_kind 时扩大 budget 重试一轮；仍不足则记录到 meta

positive 的 expected 输出来自 Verus 编译产出的二进制实际运行结果，不使用 LLM 猜测，
也不回退到 strip_verus_syntax + rustc 路径。

用法：
  python scripts/io/generate_io_tests_llm.py --input-dir <ref_rs_dir> --output-dir <out_dir>
  python scripts/io/generate_io_tests_llm.py --input <file.rs> --output-dir <out_dir>
"""

from __future__ import annotations

import argparse
import json
import ast
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# 让脚本能从项目根目录运行时 import metrics_rebuild 和 scripts.*
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics_rebuild.share.contract_eval import (  # noqa: E402
    batch_verus_contract_check_detailed,
    batch_verus_requires_check,
    contract_evaluation,
    typed_input_payload,
    typed_output_payload,
)
from metrics_rebuild.share.functions import (  # noqa: E402
    authoritative_target_for_path,
    strength_contexts_for_path,
)
from metrics_rebuild.share.io_cases import (  # noqa: E402
    _parse_benchmark_value,
    _return_payload_from_benchmark,
    generate_boundary_candidate_inputs,
    generate_candidate_inputs,
    generate_invalid_cases_from_reference_requires,
    generate_requires_satisfying_candidate_inputs,
    llm_candidate_inputs,
    llm_panic_candidate_inputs,
    mutate_output_values,
    source_hash,
    tag_input_values,
)
from metrics_rebuild.share.paths import CONFIG_PATH  # noqa: E402
from metrics_rebuild.share.text import read_text  # noqa: E402
from metrics_rebuild.share.type_defs import (  # noqa: E402
    parse_type_definitions,
    registry_entry,
    resolve_alias_text,
)
from metrics_rebuild.share.verus_runner import get_verus_binary  # noqa: E402
from metrics_rebuild.share.io_harness import (  # noqa: E402
    FuncInfo,
    _find_matching,
    _read_type_until_boundary,
    _skip_optional_generics,
    array_type_parts,
    extract_function,
    format_input_json,
    format_value_rust,
    parse_results,
    preferred_io_function_name,
    runtime_type_support_issue,
)

DEFAULT_PER_KIND = 5
DEFAULT_CANDIDATE_BUDGET = 16
DEFAULT_CANDIDATE_CAP = 24
DEFAULT_NEGATIVES_PER_POSITIVE = 3
COMPILE_TIMEOUT = 120  # seconds for verus --compile
RUN_TIMEOUT = 8  # seconds per case (legit fns finish in ms; catches infinite loops)
META_SCHEMA_VERSION = 2


def _canonical_key(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _normalized_requires_texts(context: dict) -> list[str]:
    texts: list[str] = []
    for clause in context.get("requires") or []:
        text = str(clause.get("text") or clause.get("normalized") or "")
        text = re.sub(r"\bold\s*\(\s*([A-Za-z_]\w*)\s*\)", r"\1", text)
        text = re.sub(r"\s*@\s*", "", text)
        text = re.sub(r"\s*\.\s*", ".", text)
        text = re.sub(r"\s*\(\s*\)", "()", text)
        text = re.sub(r"\s+", " ", text).strip().rstrip(",;")
        if text:
            texts.append(text)
    return texts


def _normalized_ensures_texts(context: dict) -> list[str]:
    texts: list[str] = []
    for clause in context.get("ensures") or []:
        text = str(clause.get("text") or clause.get("normalized") or "")
        text = re.sub(r"\bold\s*\(\s*([A-Za-z_]\w*)\s*\)", r"\1", text)
        text = re.sub(r"\s*@\s*", "", text)
        text = re.sub(r"\s*\.\s*", ".", text)
        text = re.sub(r"\s*\(\s*\)", "()", text)
        text = re.sub(r"\s+", " ", text).strip().rstrip(",;")
        if text:
            texts.append(text)
    return texts


def _invalid_target_info(target: Target, per_kind: int) -> dict:
    """Estimate a sound quota for finite, recognizable invalid domains."""
    texts = _normalized_requires_texts(target.context)
    if not texts:
        return {"target": 0, "reason": "runtime_panic_only", "mode": "requires_or_panic"}

    param_types = {p.name: p.rust_type.strip().lstrip("&").replace("mut ", "", 1).strip() for p in target.func.params}
    if all(text.lower() == "true" or re.search(r"==>\s*true$", text, re.IGNORECASE) for text in texts):
        return {"target": 0, "reason": "tautological_requires", "mode": "requires_or_panic"}

    runtime_bound_patterns = (
        r"[A-Za-z_]\w*\.len\(\)\s*<=\s*(?:u32|i32|usize)\s*::\s*MAX",
        r"-\s*[A-Za-z_]\w*\.len\(\)\s*>=\s*(?:i32|isize)\s*::\s*MIN",
    )
    if all(any(re.fullmatch(pattern, text) for pattern in runtime_bound_patterns) for text in texts):
        return {"target": 0, "reason": "runtime_type_bounds", "mode": "requires_or_panic"}

    if len(texts) == 1:
        text = texts[0]
        helper_call = re.fullmatch(r"([A-Za-z_]\w*)\s*\(.*\)", text)
        if helper_call:
            helper_name = re.escape(helper_call.group(1))
            helper_match = re.search(
                rf"\bspec\s+fn\s+{helper_name}\s*\([^)]*\)\s*(?:->\s*bool\s*)?\{{(?P<body>[^{{}}]*)\}}",
                target.raw_code,
                flags=re.DOTALL,
            )
            if helper_match:
                body = re.sub(r"/\*.*?\*/|//[^\n]*", " ", helper_match.group("body"), flags=re.DOTALL)
                body = re.sub(r"\s+", " ", body).strip()
                if body == "true":
                    return {"target": 0, "reason": "tautological_requires", "mode": "requires_or_panic"}
                if re.fullmatch(r"[A-Za-z_]\w*\.len\(\)\s*(?:>|>=)\s*(?:0|1)", body):
                    return {"target": min(per_kind, 1), "reason": "finite_invalid_domain", "mode": "requires_or_panic"}

        match = re.fullmatch(r"([A-Za-z_]\w*)\.len\(\)\s*(?:>|>=|!=)\s*(?:0|1)", text)
        if match and re.search(r"(?:>|!=)\s*0$|>=\s*1$", text):
            return {"target": min(per_kind, 1), "reason": "finite_invalid_domain", "mode": "requires_or_panic"}

        match = re.fullmatch(r"([A-Za-z_]\w*)\s*(>=|>)\s*(-?\d+)", text)
        if match:
            name, op, raw_bound = match.groups()
            ty = param_types.get(name, "")
            if ty.startswith("u") or ty == "usize":
                bound = int(raw_bound) + (1 if op == ">" else 0)
                capacity = max(0, bound)
                return {
                    "target": min(per_kind, capacity),
                    "reason": "finite_invalid_domain" if capacity else "tautological_requires",
                    "mode": "requires_or_panic",
                }

        match = re.fullmatch(r"(-?\d+)\s*(<=|<)\s*([A-Za-z_]\w*)", text)
        if match:
            raw_bound, op, name = match.groups()
            ty = param_types.get(name, "")
            if ty.startswith("u") or ty == "usize":
                capacity = int(raw_bound) + (1 if op == "<" else 0)
                return {
                    "target": min(per_kind, max(0, capacity)),
                    "reason": "finite_invalid_domain" if capacity else "tautological_requires",
                    "mode": "requires_or_panic",
                }

        match = re.fullmatch(r"(-?\d+)\s*<=\s*([A-Za-z_]\w*)\s*<=\s*[A-Za-z_]\w*\s*::\s*MAX", text)
        if match:
            lower, name = match.groups()
            ty = param_types.get(name, "")
            if ty.startswith("u") or ty == "usize":
                return {
                    "target": min(per_kind, max(0, int(lower))),
                    "reason": "finite_invalid_domain",
                    "mode": "requires_or_panic",
                }

        if re.fullmatch(r"[A-Za-z_]\w*\s*!=\s*[A-Za-z_]\w*\s*::\s*(?:MIN|MAX)", text):
            return {"target": min(per_kind, 1), "reason": "finite_invalid_domain", "mode": "requires_or_panic"}

        if re.fullmatch(r"([A-Za-z_]\w*)\.len\(\)\s*>=\s*0", text):
            return {"target": 0, "reason": "tautological_requires", "mode": "requires_or_panic"}
        match = re.fullmatch(r"([A-Za-z_]\w*)\.len\(\)\s*<=\s*(\d+)", text)
        if match and int(match.group(2)) + 1 > 12:
            return {"target": 0, "reason": "unconstructable_invalid_domain", "mode": "requires_or_panic"}
        if re.search(r"\.len\(\)\s*[<]=?\s*(?:u32|i32)\s*::\s*MAX", text):
            return {"target": 0, "reason": "unconstructable_invalid_domain", "mode": "requires_or_panic"}
        if re.search(r"\.len\(\)\s*<\s*0x[0-9A-Fa-f_]+", text):
            return {"target": 0, "reason": "unconstructable_invalid_domain", "mode": "requires_or_panic"}

    return {"target": per_kind, "reason": None, "mode": "requires_or_panic"}


def _has_observable_output(target: Target) -> bool:
    returns = target.context.get("returns") or []
    has_runtime_return = any(str(ret.get("type") or "").strip() != "()" for ret in returns)
    return has_runtime_return or any(param.is_mut_ref for param in target.func.params)


def _unconstructable_precondition_reason(target: Target) -> Optional[str]:
    for text in _normalized_requires_texts(target.context):
        match = re.match(r"forall\s*\|\s*([A-Za-z_]\w*)[^|]*\|(?P<body>.*)", text, flags=re.DOTALL)
        if not match:
            continue
        binder = match.group(1)
        body = match.group("body")
        if re.search(rf"\[\s*{re.escape(binder)}\s*\]", body) and "==>" not in body:
            return "unbounded_sequence_precondition"
    return None


def _positive_target_info(target: Target, per_kind: int) -> dict:
    unconstructable = _unconstructable_precondition_reason(target)
    if unconstructable:
        return {"target": 0, "reason": unconstructable}
    if not target.context.get("ensures"):
        return {"target": 0, "reason": "no_ensures"}
    if not _has_observable_output(target):
        return {"target": 0, "reason": "no_observable_output"}
    if not target.func.params:
        return {"target": 1, "reason": "finite_input_domain"}
    return {"target": per_kind, "reason": None}


def _negative_target_info(target: Target, per_kind: int) -> dict:
    unconstructable = _unconstructable_precondition_reason(target)
    if unconstructable:
        return {"target": 0, "reason": unconstructable}
    if not target.context.get("ensures"):
        return {"target": 0, "reason": "no_ensures"}
    if not _has_observable_output(target):
        return {"target": 0, "reason": "no_observable_output"}
    texts = _normalized_ensures_texts(target.context)
    if texts and all(text.lower() == "true" or re.search(r"==>\s*true$", text, re.IGNORECASE) for text in texts):
        return {"target": 0, "reason": "tautological_ensures"}
    return {"target": per_kind, "reason": None}


def _category_entry(count: int, target: int, reason: Optional[str] = None, *, blocked: bool = False) -> dict:
    if blocked:
        state = "blocked"
    elif target == 0:
        state = "not_applicable"
    elif count >= target:
        state = "complete"
    else:
        state = "partial"
    return {"target": target, "count": count, "state": state, "reason": reason}


def _category_status(target: Target, per_kind: int, positive: int, negative: int, invalid: int, *, blocked_reason: Optional[str] = None) -> dict:
    invalid_info = _invalid_target_info(target, per_kind)
    negative_info = _negative_target_info(target, per_kind)
    positive_info = _positive_target_info(target, per_kind)
    blocked = blocked_reason is not None
    return {
        "positive": _category_entry(
            positive,
            positive_info["target"],
            blocked_reason or positive_info["reason"],
            blocked=blocked,
        ),
        "negative": _category_entry(negative, negative_info["target"], blocked_reason or negative_info["reason"], blocked=blocked),
        "invalid": {
            **_category_entry(invalid, invalid_info["target"], blocked_reason or invalid_info["reason"], blocked=blocked),
            "mode": invalid_info["mode"],
        },
    }


def _has_runtime_panic_hazard(target: Target) -> bool:
    name_match = re.search(rf"\bfn\s+{re.escape(target.func.name)}\s*\(", target.raw_code)
    source = target.raw_code
    if name_match:
        body_start = target.raw_code.find("{", name_match.end())
        if body_start >= 0:
            depth = 1
            cursor = body_start + 1
            while cursor < len(target.raw_code) and depth > 0:
                if target.raw_code[cursor] == "{":
                    depth += 1
                elif target.raw_code[cursor] == "}":
                    depth -= 1
                cursor += 1
            source = target.raw_code[body_start:cursor]
    return bool(re.search(
        r"\[[^\]]+\]|\.unwrap\s*\(|\.expect\s*\(|(?<![/])/(?![/])|%|\.subrange\s*\(|\.split_at\s*\(",
        source,
    ))


def _load_existing_meta(output_dir: Path) -> dict:
    try:
        data = json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _merge_audit(existing: dict, current: dict) -> dict:
    merged: dict[str, list[dict]] = {}
    for kind in ("positive", "negative", "invalid"):
        by_key: dict[str, dict] = {}
        current_keys = {
            str(item["case_key"])
            for item in (current.get(kind) or [])
            if isinstance(item, dict) and item.get("case_key")
        }
        # Existing audited provenance wins over generic reconstruction during
        # append/supplement runs, but only for cases that still exist.  Keeping
        # audit entries for deleted/reclassified cases makes meta.json disagree
        # with test.json and can preserve a stale INVALID_INPUT oracle.
        for item in [*(current.get(kind) or []), *(existing.get(kind) or [])]:
            if (
                isinstance(item, dict)
                and item.get("case_key")
                and str(item["case_key"]) in current_keys
            ):
                by_key[str(item["case_key"])] = item
        merged[kind] = list(by_key.values())
    return merged


def _build_case_audit(
    test_cases: list[dict],
    *,
    invalid_details: Sequence[dict] = (),
    negative_details: Sequence[dict] = (),
) -> dict:
    invalid_by_input = {
        _canonical_key(item.get("input")): item
        for item in invalid_details
        if isinstance(item, dict) and isinstance(item.get("input"), dict)
    }
    negative_by_value: dict[str, dict] = {}
    for item in negative_details:
        if not isinstance(item, dict):
            continue
        key = _canonical_key({"inputs": item.get("inputs"), "output": item.get("mutated_output")})
        negative_by_value[key] = item

    audit = {"positive": [], "negative": [], "invalid": []}
    for case in test_cases:
        input_value = case.get("input")
        if case.get("expected") == "INVALID_INPUT":
            detail = invalid_by_input.get(_canonical_key(input_value), {})
            kind_detail = str(detail.get("kind_detail") or "legacy_unknown")
            oracle = (
                "runtime_panic" if kind_detail == "panic"
                else "requires_violation" if kind_detail == "requires_violation"
                else "legacy_unknown"
            )
            audit["invalid"].append({
                "case_key": _canonical_key({"input": input_value, "kind": "invalid"}),
                "oracle": oracle,
                "source": detail.get("source") or ("runtime" if oracle == "runtime_panic" else "requires_generator" if oracle == "requires_violation" else "legacy"),
                "requires_verdict": detail.get("requires_verdict", False if oracle == "requires_violation" else None),
                "runtime_status": "panic" if oracle == "runtime_panic" else detail.get("runtime_status", "not_run"),
            })
            continue

        positive_key = _canonical_key({"input": input_value, "expected": case.get("expected")})
        audit["positive"].append({
            "case_key": positive_key,
            "source": "reference_execution",
            "requires_verdict": True,
            "runtime_status": "ok",
        })
        for unexpected in case.get("unexpected") or []:
            raw_input = _parse_stored_input(input_value)
            detail_key = _canonical_key({"inputs": raw_input, "output": unexpected})
            detail = negative_by_value.get(detail_key, {})
            audit["negative"].append({
                "case_key": _canonical_key({"input": input_value, "unexpected": unexpected}),
                "base_case_key": positive_key,
                "mutation": detail.get("label") or "contract_mutation",
                "contract_verdict": False,
                "verification_reason": detail.get("verification_reason") or "verification_failed",
            })
    return audit


def _enrich_meta_v2(
    meta: dict,
    target: Target,
    test_cases: list[dict],
    *,
    per_kind: int,
    invalid_details: Sequence[dict] = (),
    negative_details: Sequence[dict] = (),
    existing_meta: Optional[dict] = None,
) -> dict:
    positive, negative, invalid = _case_counts(test_cases)
    blocked_reason = None
    if str(meta.get("status", "")).startswith("unsupported") or meta.get("failure_category") in {
        "no_candidate_inputs", "compile_or_run_failed", "no_runtime_results",
    }:
        blocked_reason = str(meta.get("failure_category") or meta.get("status"))
    categories = _category_status(
        target, per_kind, positive, negative, invalid, blocked_reason=blocked_reason,
    )
    if categories["negative"]["state"] == "partial":
        negative_log = meta.get("negative_log") or {}
        if negative_log.get("skipped_no_verus"):
            categories["negative"]["reason"] = "verus_unknown"
        elif negative_log.get("verus_checked"):
            categories["negative"]["reason"] = "insufficient_verified_mutations"
        else:
            categories["negative"]["reason"] = "no_violating_mutation"
    if categories["invalid"]["state"] == "partial" and not categories["invalid"].get("reason"):
        categories["invalid"]["reason"] = "insufficient_validated_invalids"

    if meta.get("status") == "no_valid_cases" and positive == 0:
        for kind in ("positive", "negative"):
            if categories[kind]["target"] > 0 and categories[kind]["count"] == 0:
                categories[kind].update({
                    "state": "blocked",
                    "reason": "no_validated_runtime_cases",
                })

    terminal_states = {"complete", "not_applicable", "blocked"}
    if meta.get("status") == "partial" and all(
        categories[kind]["state"] in terminal_states
        for kind in ("positive", "negative", "invalid")
    ):
        meta["status"] = "ok"

    current_audit = _build_case_audit(
        test_cases,
        invalid_details=invalid_details,
        negative_details=negative_details,
    )
    old_audit = (existing_meta or {}).get("case_audit") or {}
    meta.update({
        "schema_version": META_SCHEMA_VERSION,
        "category_status": categories,
        "case_audit": _merge_audit(old_audit, current_audit),
        "attempt_summary": {
            "candidate_count": int(meta.get("candidate_count", 0) or 0),
            "positive": meta.get("positive_select_log") or {},
            "negative": meta.get("negative_log") or {},
            "invalid": meta.get("invalid_log") or {},
            "llm": meta.get("llm_meta") or {},
        },
    })
    return meta


def _write_terminal_meta(output_dir: Path, source: str, status: str, reason: str, per_kind: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = {
        kind: _category_entry(0, per_kind, reason, blocked=True)
        for kind in ("positive", "negative", "invalid")
    }
    categories["invalid"]["mode"] = "requires_or_panic"
    meta = {
        "schema_version": META_SCHEMA_VERSION,
        "source": source,
        "source_hash": source_hash(source),
        "function": None,
        "status": status,
        "failure_category": reason,
        "positive_count": 0,
        "negative_count": 0,
        "invalid_count": 0,
        "category_status": categories,
        "case_audit": {"positive": [], "negative": [], "invalid": []},
        "attempt_summary": {},
        "notes": [reason],
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    return meta


def _load_verus_path_from_config() -> Optional[str]:
    """从 config.yaml 读取 verus_path；避免导入评估入口造成额外依赖。"""
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

_INT_RUST_TYPES = {
    "i8", "i16", "i32", "i64", "i128", "isize",
    "u8", "u16", "u32", "u64", "u128", "usize",
    "int", "nat",
}
_FLOAT_RUST_TYPES = {"f32", "f64"}

# Sentinel for parse failure (distinguishes from valid None for Option::None / ())
_PARSE_FAIL = object()


@dataclass
class Target:
    func: FuncInfo
    context: dict
    raw_code: str


_VERUS_MATH_RUNTIME_TYPE_RE = re.compile(r"\b(?:int|nat)\b")


def _has_verus_math_runtime_type(func: FuncInfo) -> bool:
    """目标签名是否含 Verus 数学类型 int/nat（exec 侧无法直接构造其值）。

    这类任务不再整题跳过：native harness 会把参考代码改写成 i128/u128 运行副本
    （见 rewrite_verus_math_types / build_verus_harness）。
    """
    if _VERUS_MATH_RUNTIME_TYPE_RE.search(func.return_type or ""):
        return True
    return any(_VERUS_MATH_RUNTIME_TYPE_RE.search(p.rust_type or "") for p in func.params)


_MATH_INT_SUFFIX_RE = re.compile(r"(\d)\s*int\b")
_MATH_NAT_SUFFIX_RE = re.compile(r"(\d)\s*nat\b")
_MATH_INT_WORD_RE = re.compile(r"(?<![A-Za-z0-9_])int(?![A-Za-z0-9_])")
_MATH_NAT_WORD_RE = re.compile(r"(?<![A-Za-z0-9_])nat(?![A-Za-z0-9_])")


def rewrite_math_type_text(type_text: str) -> str:
    """把类型/片段文本中的 int/nat 直接替换为 i128/u128（含数字面量后缀）。"""
    text = _MATH_INT_SUFFIX_RE.sub(r"\1i128", type_text)
    text = _MATH_NAT_SUFFIX_RE.sub(r"\1u128", text)
    text = _MATH_INT_WORD_RE.sub("i128", text)
    text = _MATH_NAT_WORD_RE.sub("u128", text)
    return text


_SPEC_PROOF_FN_RE = re.compile(r"\b(?:spec|proof)\s+fn\b")
_EXEC_FN_HEADER_RE = re.compile(r"\bfn\s+\w+")
_EXEC_LET_RE = re.compile(r"\blet\s+(?:ghost\s+|tracked\s+)?(?:mut\s+)?[A-Za-z_]\w*\s*:\s*")

_GENERIC_MONO_TYPE = "i32"


def _generic_binder_names(code: str, fn_name: str) -> list[str]:
    """目标函数声明的泛型类型参数名（跳过 lifetime / const 泛型）。"""
    match = re.search(rf"\bfn\s+{re.escape(fn_name)}\s*<", code)
    if not match:
        return []
    open_idx = match.end() - 1
    close_idx = _find_matching(code, open_idx, "<", ">")
    if close_idx < 0:
        return []
    binders: list[str] = []
    for part in _split_tuple_types(code[open_idx + 1:close_idx]):
        part = part.strip()
        if not part or part.startswith("'") or part.startswith("const "):
            continue
        name_match = re.match(r"([A-Za-z_]\w*)", part)
        if name_match:
            binders.append(name_match.group(1))
    return binders


def _substitute_generic_binders(text: str, binders: Sequence[str]) -> str:
    for binder in binders:
        text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(binder)}(?![A-Za-z0-9_])", _GENERIC_MONO_TYPE, text)
    return text


def _monomorphize_source(code: str, binders: Sequence[str]) -> str:
    """把源码中的泛型参数单态化为具体类型（仅用于 native 运行副本）。

    先删除声明这些 binder 的 `<...>` 段（fn 名后的泛型列表）以及提及 binder 的
    turbofish 调用段（`::<T>`，否则替换后会残留多余的泛型实参），再做全词替换。
    """
    removals: list[tuple[int, int]] = []
    demoted_fns: set[str] = set()
    for match in re.finditer(r"\bfn\s+(\w+)\s*(<)", code):
        open_idx = match.end(2) - 1
        close_idx = _find_matching(code, open_idx, "<", ">")
        if close_idx < 0:
            continue
        segment = code[open_idx:close_idx + 1]
        if any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(b)}(?![A-Za-z0-9_])", segment) for b in binders):
            removals.append((open_idx, close_idx + 1))
            demoted_fns.add(match.group(1))
    # 被去泛型化的函数，其调用点的 turbofish（如 f::<i32>(..)）必须一并删除。
    for name in demoted_fns:
        for match in re.finditer(rf"\b{re.escape(name)}\s*(::\s*<)", code):
            open_idx = match.end(1) - 1
            close_idx = _find_matching(code, open_idx, "<", ">")
            if close_idx < 0:
                continue
            removals.append((match.start(1), close_idx + 1))
    for start, end in sorted(removals, reverse=True):
        code = code[:start] + code[end:]
    return _substitute_generic_binders(code, binders)


def _mask_spec_proof_items(code: str) -> list[tuple[int, int]]:
    """返回 spec fn / proof fn 完整定义（含体）的跨度，改写时跳过。

    体起点探测只跟踪圆/方括号深度：`<`/`>` 也是比较运算符（requires 子句里的
    `i <= j` 会让尖括号计数错乱），而泛型参数里不会出现 `{`，忽略尖括号是安全的。
    """
    spans: list[tuple[int, int]] = []
    for match in _SPEC_PROOF_FN_RE.finditer(code):
        start = match.start()
        depth_paren = depth_bracket = 0
        body_start = -1
        i = match.end()
        while i < len(code):
            ch = code[i]
            if ch == '(':
                depth_paren += 1
            elif ch == ')':
                depth_paren -= 1
            elif ch == '[':
                depth_bracket += 1
            elif ch == ']':
                depth_bracket -= 1
            elif depth_paren == 0 and depth_bracket == 0:
                if ch == '{':
                    body_start = i
                    break
                if ch == ';':
                    body_start = -2
                    i += 1
                    break
            i += 1
        if body_start == -2:
            spans.append((start, i))
            continue
        if body_start < 0:
            spans.append((start, len(code)))
            continue
        end = _find_matching(code, body_start, "{", "}")
        spans.append((start, end + 1 if end >= 0 else len(code)))
    return spans


def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


def rewrite_verus_math_types(code: str) -> str:
    """把 exec 位置的 int/nat 改写成 i128/u128（仅用于 --no-verify --compile 的运行副本）。

    int/nat 的函数**定义**能过 verus --compile，但 exec harness 无法构造其值
    （`3int` 字面量在 Rust 侧是 invalid suffix）。只改写 exec 位置：

    - exec fn 的参数列表与返回类型（含命名返回元组）；
    - exec 体内的 `let x: T = ...` 类型标注（跳过 `let ghost/tracked`）。

    ghost 代码（spec/proof fn、量词 binder、requires/ensures/invariant 子句、
    `as int` 转换）保持 int/nat：spec 语境下 i128/u128 与 int/nat 可混合比较，
    而 `Seq::index/subrange` 等内建操作硬性要求 int 下标，全量替换会破坏它们。
    运行侧算术溢出会 panic（debug 溢出检查），由 catch_unwind 捕获为 PANIC。
    """
    skip_spans = _mask_spec_proof_items(code)
    replacements: list[tuple[int, int, str]] = []

    for match in _EXEC_FN_HEADER_RE.finditer(code):
        # 签名改写不依赖跨度掩码：直接看 fn 前面的词是否 spec/proof
        # （掩码的体探测是启发式的，误吞会让 exec 签名漏改）。
        prefix_words = code[max(0, match.start() - 30):match.start()].split()[-2:]
        if any(word in ("spec", "proof") for word in prefix_words):
            continue
        params_start = _skip_optional_generics(code, match.end())
        if params_start >= len(code) or code[params_start] != "(":
            continue
        params_end = _find_matching(code, params_start, "(", ")")
        if params_end < 0:
            continue
        segment = code[params_start + 1:params_end]
        rewritten = rewrite_math_type_text(segment)
        if rewritten != segment:
            replacements.append((params_start + 1, params_end, rewritten))
        cursor = params_end + 1
        while cursor < len(code) and code[cursor].isspace():
            cursor += 1
        if code.startswith("->", cursor):
            type_start = cursor + 2
            type_text, type_end = _read_type_until_boundary(code, type_start)
            segment = code[type_start:type_end]
            rewritten = rewrite_math_type_text(segment)
            if rewritten != segment:
                replacements.append((type_start, type_end, rewritten))

    for match in _EXEC_LET_RE.finditer(code):
        if _in_spans(match.start(), skip_spans):
            continue
        if re.search(r"\b(?:ghost|tracked)\b", match.group(0)):
            continue
        type_start = match.end()
        type_text, type_end = _read_type_until_boundary(code, type_start)
        # `let x: T = ...` 的类型段止于 `=`（boundary 会停在 `=` / `;` 之前）。
        eq = code.find("=", type_start)
        semi = code.find(";", type_start)
        limit = min(x for x in (eq, semi, len(code)) if x >= 0)
        type_end = min(type_end, limit)
        segment = code[type_start:type_end]
        rewritten = rewrite_math_type_text(segment)
        if rewritten != segment:
            replacements.append((type_start, type_end, rewritten))

    for start, end, rewritten in sorted(replacements, reverse=True):
        code = code[:start] + rewritten + code[end:]
    return code


def _unsupported_runtime_type_issues(func: FuncInfo, registry: Optional[dict] = None) -> list[dict]:
    issues: list[dict] = []
    for param in func.params:
        issue = runtime_type_support_issue(param.rust_type, registry)
        if issue:
            issues.append({
                "role": "parameter",
                "name": param.name,
                "type": param.rust_type,
                "reason": issue,
            })
    if func.return_type and func.return_type != "()":
        issue = runtime_type_support_issue(func.return_type, registry)
        if issue:
            issues.append({
                "role": "return",
                "name": func.return_name or "result",
                "type": func.return_type,
                "reason": issue,
            })
    return issues


# ---------------------------------------------------------------------------
# 解析阶段
# ---------------------------------------------------------------------------

def _context_from_func(func: FuncInfo) -> dict:
    """从 FuncInfo 构造无 spec 的 context（fallback，当 strength_contexts 匹配不到时用）."""
    params = [{"name": p.name, "type": p.rust_type} for p in func.params]
    returns: list[dict] = []
    if func.return_type and func.return_type != "()":
        returns = [{"name": func.return_name or "ret", "type": func.return_type}]
    return {
        "function": func.name,
        "parameters": params,
        "returns": returns,
        "requires": [],
        "ensures": [],
        "has_contract": False,
        "mode": "exec",
        "modifiers": [],
        "has_signature_spec": False,
        "spec_preamble": "",
    }


def _context_with_func_returns(context: dict, func: FuncInfo) -> dict:
    if context.get("returns") or not func.return_type or func.return_type == "()":
        return context
    patched = dict(context)
    patched["returns"] = [{"name": func.return_name or "result", "type": func.return_type}]
    return patched


def resolve_target(reference_path: str, function_name: Optional[str] = None) -> Optional[Target]:
    code = read_text(reference_path)
    if function_name is None:
        authoritative = authoritative_target_for_path(reference_path)
        function_name = str(authoritative.get("function") or "") if authoritative else None
    func = extract_function(code, function_name)
    if func is None:
        return None
    context: Optional[dict] = None
    for ctx in strength_contexts_for_path(reference_path):
        if ctx.get("function") == func.name:
            context = ctx
            break
    if context is None:
        context = _context_from_func(func)
    else:
        context = _context_with_func_returns(context, func)
    context.setdefault("type_registry", parse_type_definitions(code))
    return Target(func=func, context=context, raw_code=code)


def resolve_existing_target(
    reference_path: str,
    test_cases: Sequence[dict],
    preferred_name: Optional[str] = None,
) -> Optional[Target]:
    """Resolve the function whose parameter names match an existing IO suite."""
    input_names: Optional[set[str]] = None
    for case in test_cases:
        raw_input = case.get("input") if isinstance(case, dict) else None
        if isinstance(raw_input, dict):
            input_names = set(raw_input)
            break
    if input_names is None:
        return resolve_target(reference_path, preferred_name)

    candidate_names: list[str] = []
    if preferred_name:
        candidate_names.append(preferred_name)
    candidate_names.extend(
        str(context.get("function"))
        for context in strength_contexts_for_path(reference_path)
        if context.get("function")
    )
    fallback = preferred_io_function_name(read_text(reference_path))
    if fallback:
        candidate_names.append(fallback)
    seen: set[str] = set()
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        target = resolve_target(reference_path, name)
        if target is not None and {param.name for param in target.func.params} == input_names:
            return target
    return resolve_target(reference_path, preferred_name)


# ---------------------------------------------------------------------------
# 参考代码预处理：移除原 fn main
# ---------------------------------------------------------------------------

def remove_fn_main(code: str) -> str:
    """移除所有 `fn main(...)` 定义（平衡括号匹配）.

    处理空体和非空体 main，以及在 verus! 块内或块外的情况。
    """
    result = code
    pattern = re.compile(r'\bfn\s+main\s*\(')
    while True:
        m = pattern.search(result)
        if not m:
            break
        start = m.start()
        i = m.end()
        # 找到函数体的开括号 {
        while i < len(result) and result[i] != '{':
            i += 1
        if i >= len(result):
            break
        depth = 1
        j = i + 1
        while j < len(result) and depth > 0:
            if result[j] == '{':
                depth += 1
            elif result[j] == '}':
                depth -= 1
            j += 1
        # 从 start 删到 j（含整个 fn main {...}）
        result = result[:start] + result[j:]
    return result


# ---------------------------------------------------------------------------
# 类型化输出：Rust 格式化表达式生成
# ---------------------------------------------------------------------------

def _norm_rust_type(t: str) -> str:
    """归一化 Rust/Verus 类型：剥离 & / &mut（&str 保留 &），保留 int/nat。"""
    t = str(t).strip()
    if t.startswith("&"):
        rest = t[1:].strip()
        if rest.startswith("mut "):
            rest = rest[4:].strip()
        if rest in ("str", "'static str"):
            return "&" + rest  # 保留 &str
        t = rest
    if t.startswith("mut "):
        t = t[4:].strip()
    return t


def _extract_generic_inner(t: str, prefix: str) -> str:
    """从 `Vec<...>` / `Option<...>` / `Seq<...>` 提取内部类型（平衡尖括号）."""
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
    return t[start:i - 1].strip()


def _split_tuple_types(t: str) -> list[str]:
    """从 `(T1, T2, T3)` 提取元素类型列表."""
    inner = t[1:-1] if t.startswith("(") and t.endswith(")") else t
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in inner:
        if ch in "<(":
            depth += 1
            cur += ch
        elif ch in ">)":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def gen_fmt_expr(type_text: str, var: str, *, value_is_ref: bool = False, registry: Optional[dict] = None) -> str:
    """生成 Rust 表达式，把 `var`（类型 type_text）格式化成类型化字符串."""
    if registry:
        type_text = resolve_alias_text(registry, str(type_text))
    t = _norm_rust_type(type_text)
    value = f"*{var}" if value_is_ref else var
    if t == "bool":
        return f'(if {value} {{"true".to_string()}} else {{"false".to_string()}})'
    if t == "char":
        # 单字符也可能含换行等控制符，需转义以免破坏 RESULT 行
        return f'escape_typed_char({value})'
    if t == "String":
        return f'escape_typed_str({var})' if value_is_ref else f'escape_typed_str(&{var})'
    if t in ("str", "&str", "'static str", "&'static str"):
        return f'escape_typed_str({var})'
    if t in ("int", "nat"):
        return f'format!("{{:?}}", {value})'
    if t in _INT_RUST_TYPES:
        return f'format!("{{}}", {value})'
    if t in _FLOAT_RUST_TYPES:
        return f'format!("{{}}", {value})'
    if t == "()" or t == "":
        return '"null".to_string()'
    if t.startswith("Option<"):
        inner = _extract_generic_inner(t, "Option")
        inner_fmt = gen_fmt_expr(inner, "x", registry=registry)
        return f'(match {var} {{ Some(x) => {inner_fmt}, None => "null".to_string() }})'
    array = array_type_parts(t)
    if t.startswith("Vec<") or t.startswith("Seq<") or array is not None:
        if array is not None:
            inner = array[0]
        else:
            prefix = t[:t.index("<")]
            inner = _extract_generic_inner(t, prefix)
        if inner == "char":
            # 输出 unicode ordinals 列表，loader 的 coerce_value_for_type(Vec<char>) 接受 int 列表
            return (f'format!("[{{}}]", {var}.iter().map(|c| (*c as u32).to_string())'
                    f'.collect::<Vec<String>>().join(", "))')
        inner_fmt = gen_fmt_expr(inner, "x", value_is_ref=True, registry=registry)
        return (f'format!("[{{}}]", {var}.iter().map(|x| {inner_fmt})'
                f'.collect::<Vec<String>>().join(", "))')
    if t.startswith("(") and t.endswith(")"):
        elem_types = _split_tuple_types(t)
        tuple_var = f"(*{var})" if value_is_ref else var
        parts = [gen_fmt_expr(et, f'{tuple_var}.{i}', registry=registry) for i, et in enumerate(elem_types)]
        return f'format!("[{{}}]", vec![{", ".join(parts)}].join(", "))'
    entry = registry_entry(registry, t)
    if entry is not None:
        if entry["kind"] == "enum":
            arms = ", ".join(
                f'{t}::{variant} => "{variant}".to_string()'
                for variant in entry["variants"]
            )
            return f"(match {var} {{ {arms} }})"
        field_var = f"(*{var})" if value_is_ref else var
        parts = [
            gen_fmt_expr(field_type, f"{field_var}.{fname}", registry=registry)
            for fname, field_type in entry["fields"]
        ]
        return f'format!("[{{}}]", vec![{", ".join(parts)}].join(", "))'
    # fallback: Debug（罕见类型）
    return f'format!("{{:?}}", {var})'


# 始终注入到 harness 的转义辅助函数（String/char 输出可能含换行等控制符）
_ESCAPE_HELPERS = r'''fn escape_typed_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

fn escape_typed_char(c: char) -> String {
    let mut out = String::new();
    match c {
        '\\' => out.push_str("\\\\"),
        '\n' => out.push_str("\\n"),
        '\r' => out.push_str("\\r"),
        '\t' => out.push_str("\\t"),
        _ => out.push(c),
    }
    out
}
'''


def gen_fmt_fn(ret_type: str, registry: Optional[dict] = None) -> str:
    """生成 `fn fmt_typed(v: T) -> String { ... }` 源码."""
    raw = str(ret_type).strip()
    is_ref = raw.startswith("&") and raw not in ("&str", "&'static str")
    t = _norm_rust_type(raw)
    body = gen_fmt_expr(t, "v", value_is_ref=is_ref, registry=registry)
    param = raw if is_ref else ("&str" if t == "str" else t)
    return f'fn fmt_typed(v: {param}) -> String {{ {body} }}'


def _input_type_annotation(ptype: str) -> str:
    """返回输入局部变量的类型标注；str/slice 由 Vec 或字面量承载，不能直接标注。"""
    ptype = ptype.strip()
    if ptype in ("str", "'static str") or ptype.startswith("["):
        return ""
    return f": {ptype}"


# ---------------------------------------------------------------------------
# Verus harness 构造
# ---------------------------------------------------------------------------

def build_verus_harness(target: Target, candidate_inputs: list[dict]) -> str:
    """构造 Verus harness 源码：参考代码 verus! 块 + 块外 main + 类型化输出."""
    func = target.func
    raw_code = target.raw_code
    registry = target.context.get("type_registry") if isinstance(target.context, dict) else None
    ref_code = remove_fn_main(raw_code).rstrip()

    # 泛型目标：native 副本单态化为具体类型（i32）运行。
    generic_binders = _generic_binder_names(raw_code, func.name)
    if generic_binders:
        ref_code = _monomorphize_source(ref_code, generic_binders)

    # int/nat 签名任务：整份参考代码与 harness 侧类型统一改写为 i128/u128 运行。
    math_rewrite = _has_verus_math_runtime_type(func)
    if math_rewrite:
        ref_code = rewrite_verus_math_types(ref_code)

    def runtime_type(type_text: str) -> str:
        if generic_binders:
            type_text = _substitute_generic_binders(type_text, generic_binders)
        return rewrite_math_type_text(type_text) if math_rewrite else type_text

    ret_type_norm = _norm_rust_type(func.return_type)
    is_void = ret_type_norm in ("()", "")
    mut_params = [p for p in func.params if p.is_mut_ref]

    # 生成格式化函数
    fmt_fns: list[str] = []
    if is_void and mut_params:
        for mp in mut_params:
            mp_type = _norm_rust_type(runtime_type(mp.rust_type))
            body = gen_fmt_expr(mp_type, "v", registry=registry)
            fmt_fns.append(f'fn fmt_mut_{mp.name}(v: &{mp_type}) -> String {{ {body} }}')
    else:
        fmt_fns.append(gen_fmt_fn(runtime_type(func.return_type), registry=registry))

    # 生成 main：从命令行参数取 target_idx，只运行该 case（避免一个慢 case 拖死整批）
    lines: list[str] = ["use std::panic;", "", _ESCAPE_HELPERS.rstrip()]
    lines.extend(fmt_fns)
    lines.extend([
        "",
        "fn main() {",
        "    let args: Vec<String> = std::env::args().collect();",
        "    let target_idx: usize = match args.get(1).and_then(|s| s.parse::<usize>().ok()) {",
        "        Some(i) => i,",
        "        None => return,",
        "    };",
    ])

    for idx, inp in enumerate(candidate_inputs):
        var_decls: list[str] = []
        call_args: list[str] = []
        mut_var_names: list[tuple[str, str]] = []
        for p in func.params:
            val = inp.get(p.name, 0)
            ptype = runtime_type(p.rust_type).replace("&mut ", "").replace("&", "").strip()
            var_name = f"p_{p.name}_{idx}"
            rust_val = format_value_rust(val, ptype, registry)
            mut_kw = "mut " if p.is_mut_ref else ""
            type_annotation = _input_type_annotation(ptype)
            var_decls.append(f"        let {mut_kw}{var_name}{type_annotation} = {rust_val};")
            if p.is_mut_ref and p.is_slice:
                call_args.append(f"&mut {var_name}[..]")
                mut_var_names.append((p.name, var_name))
            elif p.is_mut_ref:
                call_args.append(f"&mut {var_name}")
                mut_var_names.append((p.name, var_name))
            elif p.is_ref and p.is_slice:
                call_args.append(f"&{var_name}[..]")
            elif p.is_ref:
                call_args.append(f"&{var_name}")
            else:
                call_args.append(var_name)
        lines.append(f"    if target_idx == {idx} {{")
        lines.extend(var_decls)
        args_str = ", ".join(call_args)

        if is_void and mut_var_names:
            lines.append(
                f'        match panic::catch_unwind(panic::AssertUnwindSafe(|| {{ {func.name}({args_str}); }})) {{'
            )
            fmt_parts = [f'fmt_mut_{pname}(&{vname})' for pname, vname in mut_var_names]
            if len(fmt_parts) == 1:
                output_expr = fmt_parts[0]
            else:
                output_expr = f'format!("{{}}", vec![{", ".join(fmt_parts)}].join("|"))'
            lines.append(f'            Ok(_) => println!("RESULT:{idx}:OK:{{}}", {output_expr}),')
            lines.append(f'            Err(_) => println!("RESULT:{idx}:PANIC"),')
            lines.append(f'        }}')
        else:
            lines.append(
                f'        match panic::catch_unwind(panic::AssertUnwindSafe(|| {func.name}({args_str}))) {{'
            )
            lines.append(f'            Ok(r) => println!("RESULT:{idx}:OK:{{}}", fmt_typed(r)),')
            lines.append(f'            Err(_) => println!("RESULT:{idx}:PANIC"),')
            lines.append(f'        }}')
        lines.append(f"    }}")

    lines.append("}")
    return ref_code + "\n\n" + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Verus 编译运行
# ---------------------------------------------------------------------------

def _find_harness_binary(work_dir: str) -> Optional[str]:
    """定位 verus --compile 产出的可执行文件.

    首选源文件 stem 命名的 `harness`；若某版本命名不同则回退到 work_dir 下
    的可执行普通文件（排除源码/中间产物）。
    """
    preferred = os.path.join(work_dir, "harness")
    if os.path.isfile(preferred):
        return preferred
    skip_ext = {".rs", ".d", ".rlib", ".rmeta", ".pdb", ".o"}
    candidates: list[str] = []
    for entry in os.listdir(work_dir):
        path = os.path.join(work_dir, entry)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(entry)[1].lower() in skip_ext:
            continue
        if os.access(path, os.X_OK):
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates)[0]


def verus_compile_and_run(
    harness_code: str, work_dir: str, verus_bin: str, num_cases: int,
) -> tuple[dict, str]:
    """用 verus --no-verify --compile 编译 harness，逐 case 运行产出二进制.

    每个 case 单独跑一个子进程（带超时），避免一个慢/死循环 case 拖死整批。
    返回 (results, reason)，results: {idx: (status, value)}，status ∈ {OK, PANIC, TIMEOUT}。
    编译失败时 results 为空，reason 记原因。
    """
    src_path = os.path.join(work_dir, "harness.rs")
    with open(src_path, "w", encoding="utf-8") as f:
        f.write(harness_code)

    cmd = [verus_bin, "--no-verify", "--compile", "-C", "panic=unwind", src_path]
    try:
        cp = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=COMPILE_TIMEOUT, check=False, cwd=work_dir,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {}, f"compile_exception:{type(exc).__name__}"
    if cp.returncode != 0:
        snippet = (cp.stderr or cp.stdout or "")[-400:]
        return {}, f"compile_failed:{snippet}"

    # 二进制产出在源文件所在目录，名字通常 = 源文件 stem（harness）
    bin_path = _find_harness_binary(work_dir)
    if bin_path is None:
        return {}, "binary_not_found"

    results: dict[int, tuple[str, str]] = {}
    for idx in range(num_cases):
        try:
            rp = subprocess.run(
                [bin_path, str(idx)], capture_output=True, text=True,
                timeout=RUN_TIMEOUT, check=False, cwd=work_dir,
            )
        except subprocess.TimeoutExpired:
            results[idx] = ("TIMEOUT", "")
            continue
        except FileNotFoundError:
            return results, "binary_disappeared"
        parsed = parse_results(rp.stdout or "")
        if idx in parsed:
            results[idx] = parsed[idx]
        else:
            results[idx] = ("UNKNOWN", "")
    return results, "ok"


# ---------------------------------------------------------------------------
# rustc 运行阶段（替换为 verus --compile）
# ---------------------------------------------------------------------------

@dataclass
class RunOutcome:
    ok: list[tuple[dict, str]]  # (typed_inputs, typed_output_string)
    panic: list[dict]  # typed_inputs that panicked
    runtime_invalid: list[tuple[dict, str]]  # (typed_inputs, kind_detail) for TIMEOUT / UNKNOWN
    failure_reason: str  # 编译/运行失败原因（ok 和 panic 都为空时用）
    diagnostics: dict


def run_verus_compile_batch(
    target: Target,
    candidate_inputs: list[dict],
    work_dir: str,
    verus_bin: str,
) -> RunOutcome:
    """编译运行 Verus harness，返回 OK 输出与 panic 输入."""
    if not candidate_inputs:
        return RunOutcome(ok=[], panic=[], runtime_invalid=[], failure_reason="no_candidate_inputs",
                          diagnostics={"compile_reason": "no_candidate_inputs", "raw_result_count": 0})
    harness = build_verus_harness(target, candidate_inputs)
    results, reason = verus_compile_and_run(harness, work_dir, verus_bin, len(candidate_inputs))
    diagnostics = {
        "compile_reason": reason,
        "raw_result_count": len(results),
        "unknown_result_count": sum(1 for status, _value in results.values() if status == "UNKNOWN"),
        "typed_input_rejected_count": 0,
    }
    if not results:
        return RunOutcome(ok=[], panic=[], runtime_invalid=[], failure_reason=reason, diagnostics=diagnostics)
    ok: list[tuple[dict, str]] = []
    panic: list[dict] = []
    runtime_invalid: list[tuple[dict, str]] = []
    for idx, raw_inp in enumerate(candidate_inputs):
        if idx not in results:
            continue
        status, value = results[idx]
        typed = typed_input_payload(target.context, raw_inp)
        if typed is None:
            diagnostics["typed_input_rejected_count"] += 1
            continue
        if status == "OK":
            ok.append((typed, value))
        elif status == "PANIC":
            panic.append(typed)
        elif status in {"TIMEOUT", "UNKNOWN"}:
            runtime_invalid.append((typed, status.lower()))
    failure_reason = reason if not ok and not panic and not runtime_invalid else ""
    return RunOutcome(ok=ok, panic=panic, runtime_invalid=runtime_invalid,
                      failure_reason=failure_reason, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# 类型化输出解析（harness 字符串 -> Python 值）
# ---------------------------------------------------------------------------

def _unescape_typed(s: str) -> str:
    """反转义 harness 输出中的 \\n / \\r / \\t / \\\\."""
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == 'n':
                out.append('\n'); i += 2
            elif nxt == 'r':
                out.append('\r'); i += 2
            elif nxt == 't':
                out.append('\t'); i += 2
            elif nxt == '\\':
                out.append('\\'); i += 2
            else:
                out.append(s[i]); i += 1
        else:
            out.append(s[i]); i += 1
    return ''.join(out)


def _split_list_elements(s: str) -> Optional[list[str]]:
    """把 harness 输出的 `[a, b, ...]` 拆成元素字符串（平衡括号）。非 `[...]` 返回 None."""
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    inner = s[1:-1].strip()
    if not inner:
        return []
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in inner:
        if ch in "[(":
            depth += 1
            cur += ch
        elif ch in "])":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def parse_typed_value(value_str: str, ret_type: str, registry: Optional[dict] = None) -> Any:
    """把 harness 输出的类型化字符串解析成 Python 值。失败返回 _PARSE_FAIL."""
    if registry:
        ret_type = resolve_alias_text(registry, str(ret_type))
    t = _norm_rust_type(ret_type)
    s = value_str
    if t == "bool":
        if s == "true":
            return True
        if s == "false":
            return False
        return _PARSE_FAIL
    if t == "char":
        unescaped = _unescape_typed(s)
        return unescaped if len(unescaped) == 1 else _PARSE_FAIL
    if t == "String" or t in ("str", "&str", "'static str", "&'static str"):
        return _unescape_typed(s)
    if t in _INT_RUST_TYPES:
        try:
            return int(s)
        except ValueError:
            return _PARSE_FAIL
    if t in _FLOAT_RUST_TYPES:
        try:
            return float(s)
        except ValueError:
            return _PARSE_FAIL
    if t == "()" or t == "":
        return None
    if t.startswith("Option<"):
        if s == "null":
            return None
        inner = _extract_generic_inner(t, "Option")
        return parse_typed_value(s, inner, registry)
    array = array_type_parts(t)
    if t.startswith("Vec<") or t.startswith("Seq<") or array is not None:
        if array is not None:
            inner_type = array[0]
        else:
            prefix = t[:t.index("<")]
            inner_type = _extract_generic_inner(t, prefix)
        elems = _split_list_elements(s)
        if elems is None:
            return _PARSE_FAIL
        # Vec<char> 的 harness 输出为 unicode ordinals（int），见 gen_fmt_expr；按整数解析
        if _norm_rust_type(inner_type) == "char":
            inner_type = "u32"
        result: list = []
        for elem in elems:
            val = parse_typed_value(elem, inner_type, registry)
            if val is _PARSE_FAIL:
                return _PARSE_FAIL
            result.append(val)
        return result
    if t.startswith("(") and t.endswith(")"):
        elem_types = _split_tuple_types(t)
        elems = _split_list_elements(s)
        if elems is None or len(elems) != len(elem_types):
            return _PARSE_FAIL
        result = []
        for elem, elem_type in zip(elems, elem_types):
            val = parse_typed_value(elem, elem_type, registry)
            if val is _PARSE_FAIL:
                return _PARSE_FAIL
            result.append(val)
        return result
    entry = registry_entry(registry, t)
    if entry is not None:
        if entry["kind"] == "enum":
            return s if s in entry["variants"] else _PARSE_FAIL
        elems = _split_list_elements(s)
        if elems is None or len(elems) != len(entry["fields"]):
            return _PARSE_FAIL
        result = []
        for elem, (_fname, field_type) in zip(elems, entry["fields"]):
            val = parse_typed_value(elem, field_type, registry)
            if val is _PARSE_FAIL:
                return _PARSE_FAIL
            result.append(val)
        return result
    return s


def _mut_params_from_func(func: Optional[FuncInfo]) -> list[dict]:
    if func is None:
        return []
    return [
        {"name": param.name, "type": param.rust_type}
        for param in func.params
        if param.is_mut_ref
    ]


def _negative_context(target: Target) -> dict:
    if target.context.get("returns"):
        return target.context
    mut_params = _mut_params_from_func(target.func)
    if not mut_params:
        return target.context
    patched = dict(target.context)
    patched["returns"] = [
        {
            "name": param["name"],
            "type": re.sub(r"^&\s*mut\s+", "", str(param["type"])).strip(),
        }
        for param in mut_params
    ]
    patched["_mutable_post_state_names"] = [param["name"] for param in mut_params]
    return patched


def _parse_typed_output_to_dict(context: dict, output_str: str, func: Optional[FuncInfo] = None) -> Optional[dict]:
    """把 harness 输出的类型化字符串解析回 {return_name: typed_value}."""
    registry = context.get("type_registry")
    returns = list(context.get("returns") or [])
    if not returns:
        mut_params = _mut_params_from_func(func) or [
            param for param in context.get("parameters") or []
            if str(param.get("type", "")).strip().startswith("&mut")
        ]
        if not mut_params:
            return None
        raw_parts = output_str.split("|") if len(mut_params) > 1 else [output_str]
        if len(raw_parts) != len(mut_params):
            return None
        result: dict[str, Any] = {}
        for param, raw in zip(mut_params, raw_parts):
            name = param.get("name")
            if not name:
                return None
            param_type = param.get("type", "")
            val = parse_typed_value(raw, param_type, registry)
            if val is _PARSE_FAIL:
                return None
            result[name] = val
        return result
    if len(returns) == 1:
        ret = returns[0]
        name = ret.get("name") or "ret"
        val = parse_typed_value(output_str, ret.get("type", ""), registry)
        if val is _PARSE_FAIL:
            return None
        return {name: val}
    # 多返回值（tuple）
    elems = _split_list_elements(output_str)
    if elems is None or len(elems) != len(returns):
        return None
    result: dict[str, Any] = {}
    for ret, item in zip(returns, elems):
        name = ret.get("name") or "ret"
        val = parse_typed_value(item, ret.get("type", ""), registry)
        if val is _PARSE_FAIL:
            return None
        result[name] = val
    return result


def format_typed_output_for_json(context: dict, output_dict: dict, func: Optional[FuncInfo] = None) -> Any:
    """把 typed dict {return_name: value} 转成 test.json 用的 JSON 值."""
    returns = list(context.get("returns") or [])
    if not returns:
        mut_params = _mut_params_from_func(func) or [
            param for param in context.get("parameters") or []
            if str(param.get("type", "")).strip().startswith("&mut")
        ]
        if len(mut_params) == 1:
            return output_dict.get(mut_params[0].get("name"))
        if mut_params:
            return {param.get("name"): output_dict.get(param.get("name")) for param in mut_params}
        return None
    if len(returns) == 1:
        name = returns[0].get("name") or "ret"
        return output_dict.get(name)
    return [output_dict.get(ret.get("name") or "ret") for ret in returns]


# ---------------------------------------------------------------------------
# invalid 生成
# ---------------------------------------------------------------------------

def collect_invalid_cases(
    target: Target,
    panic_inputs: list[dict],
    runtime_invalid_inputs: list[tuple[dict, str]],
    existing_inputs: list[dict],
    per_kind: int,
) -> tuple[list[dict], dict]:
    """合并 panic-invalid 与 requires-invalid，去重到 per_kind 个."""
    seen_keys: set[str] = set()
    for inp in existing_inputs:
        seen_keys.add(json.dumps(inp, ensure_ascii=False, sort_keys=True))

    invalid_cases: list[dict] = []
    method_log = {"panic": 0, "timeout_invalid": 0, "unknown_invalid": 0, "requires_violation": 0}

    # 1) panic 类
    for inp in panic_inputs:
        key = json.dumps(inp, ensure_ascii=False, sort_keys=True)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        invalid_cases.append({
            "input": inp,
            "kind_detail": "panic",
            "source": "runtime",
            "requires_verdict": None,
            "runtime_status": "panic",
            "tags": sorted(set(tag_input_values(inp)) | {"runtime_panic"}),
        })
        method_log["panic"] += 1
        if len(invalid_cases) >= per_kind:
            return invalid_cases, method_log

    # 2) timeout / unknown are diagnostics, not evidence that the input is
    # invalid.  Keep counts for auditability but never emit INVALID_INPUT.
    for inp, detail in runtime_invalid_inputs:
        if detail == "timeout":
            method_log["timeout_invalid"] += 1
        else:
            method_log["unknown_invalid"] += 1

    # 3) requires 违反类（用 io_cases 的 LLM + 启发式 + Verus 验证）
    requires = target.context.get("requires") or []
    if requires and len(invalid_cases) < per_kind:
        supplement, supplement_meta = generate_invalid_cases_from_reference_requires(
            target.context,
            existing_cases=[{"inputs": inp} for inp in existing_inputs + [c["input"] for c in invalid_cases]],
            budget=max(per_kind * 4, 20),
            positive_inputs=existing_inputs,
        )
        for case in supplement:
            inp = case.get("inputs")
            if not isinstance(inp, dict):
                continue
            key = json.dumps(inp, ensure_ascii=False, sort_keys=True)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            invalid_cases.append({
                "input": inp,
                "kind_detail": "requires_violation",
                "source": case.get("source") or "requires_generator",
                "requires_verdict": False,
                "runtime_status": "not_run",
                "tags": case.get("tags") or sorted(set(tag_input_values(inp)) | {"requires_boundary"}),
            })
            method_log["requires_violation"] += 1
            if len(invalid_cases) >= per_kind:
                break

    return invalid_cases, method_log


# ---------------------------------------------------------------------------
# negative 生成
# ---------------------------------------------------------------------------

def _negative_detail_with_local_fallback(
    detail: dict,
    context: dict,
    inputs: dict,
    mutated_output: dict,
) -> dict:
    """Classify a mutation locally only when Verus leaves it unresolved.

    ``ensures_not_satisfied`` means the concrete evaluator understood every
    relevant clause and found a false postcondition. Unknown expressions and
    requires failures remain unresolved and are never emitted as negatives.
    """
    if detail.get("accepted") is not None:
        return detail
    local = contract_evaluation(context, inputs, mutated_output, strict=True)
    if local.get("accepted") is False and local.get("reason") == "ensures_not_satisfied":
        return {
            "accepted": False,
            "reason": "local_contract_ensures_not_satisfied",
            "engine": "python_contract",
        }
    return detail


def _negative_verdicts(context: dict, pending: list[tuple]) -> dict[str, dict]:
    """Resolve obvious concrete rejections before invoking the Verus batch."""
    verdicts: dict[str, dict] = {}
    unresolved: list[dict] = []
    for index, item in enumerate(pending):
        inputs = item[1]
        mutated_output = item[2]
        key = f"mut_{index}"
        local = contract_evaluation(context, inputs, mutated_output, strict=True)
        if local.get("accepted") is False and local.get("reason") == "ensures_not_satisfied":
            verdicts[key] = {
                "accepted": False,
                "reason": "local_contract_ensures_not_satisfied",
                "engine": "python_contract",
            }
        else:
            unresolved.append({"inputs": inputs, "output": mutated_output, "key": key})
    if unresolved:
        verdicts.update(batch_verus_contract_check_detailed(context, unresolved))
    return verdicts


def collect_negative_cases(
    target: Target,
    positives: list[tuple[dict, dict, str]],
    per_kind: int,
    negatives_per_positive: int,
) -> tuple[list[dict], dict]:
    """对每个 positive 变异输出，用 Verus ensures 检查确认违反；Verus 不可用时跳过（不采信）."""
    if not positives:
        return [], {"verus_checked": 0, "skipped_no_verus": 0}

    negative_context = _negative_context(target)
    pending: list[tuple[int, dict, dict, dict, str]] = []
    for pos_idx, (inputs, output_dict, _output_str) in enumerate(positives):
        for mutated_output, label in mutate_output_values(negative_context, inputs, output_dict):
            typed_mut = typed_output_payload(negative_context, mutated_output)
            if typed_mut is not None:
                pending.append((pos_idx, inputs, typed_mut, mutated_output, label))

    negatives: list[dict] = []
    neg_counts: dict[int, int] = {}
    method_log = {"verus_checked": 0, "skipped_no_verus": 0, "accepted_mutations": []}

    ensures = negative_context.get("ensures") or []
    if not ensures or not pending:
        method_log["skipped_no_verus"] = len(pending)
        return negatives, method_log

    verdicts = _negative_verdicts(negative_context, pending)
    method_log["verification_reasons"] = {}
    for i, (pos_idx, inputs, typed_mut, mutated_output, label) in enumerate(pending):
        if neg_counts.get(pos_idx, 0) >= negatives_per_positive:
            continue
        detail = verdicts.get(f"mut_{i}") or {"accepted": None, "reason": "verification_unresolved"}
        detail = _negative_detail_with_local_fallback(
            detail, negative_context, inputs, mutated_output,
        )
        accepted = detail.get("accepted")
        reason = str(detail.get("reason") or "verification_unresolved")
        method_log["verification_reasons"][reason] = method_log["verification_reasons"].get(reason, 0) + 1
        if accepted is False:
            method_log["verus_checked"] += 1
            negatives.append({
                "base_index": pos_idx,
                "inputs": inputs,
                "mutated_output": typed_mut,
                "label": label,
                "verification_reason": reason,
            })
            neg_counts[pos_idx] = neg_counts.get(pos_idx, 0) + 1
            if len(negatives) >= per_kind:
                return negatives, method_log
        elif accepted is None:
            method_log["skipped_no_verus"] += 1

    return negatives, method_log


# ---------------------------------------------------------------------------
# positive 筛选
# ---------------------------------------------------------------------------

def select_positives(
    target: Target,
    ok_results: list[tuple[dict, str]],
    per_kind: int,
) -> tuple[list[tuple[dict, dict, str]], dict]:
    """从 OK 结果中筛 positive：要求 Verus requires 满足（Verus 不可用则放行）."""
    positives: list[tuple[dict, dict, str]] = []
    log = {"requires_rejected": 0, "requires_unchecked": 0}

    if not ok_results:
        return positives, log

    requires = target.context.get("requires") or []
    verdicts: dict[str, Optional[bool]] = {}
    if requires:
        batch = [{"inputs": inp, "key": f"req_{i}"} for i, (inp, _v) in enumerate(ok_results)]
        verdicts = batch_verus_requires_check(target.context, batch)

    for i, (inputs, typed_output_str) in enumerate(ok_results):
        if requires:
            accepted = verdicts.get(f"req_{i}")
            if accepted is False:
                log["requires_rejected"] += 1
                continue
            if accepted is None:
                log["requires_unchecked"] += 1
        output_dict = _parse_typed_output_to_dict(target.context, typed_output_str, target.func)
        if output_dict is None:
            # 无法解析输出：跳过（不采信，符合 no_fallback）
            log.setdefault("output_unparseable", 0)
            log["output_unparseable"] = log.get("output_unparseable", 0) + 1
            continue
        positives.append((inputs, output_dict, typed_output_str))
        if len(positives) >= per_kind:
            break
    return positives, log


def _resolve_candidate_cap(per_kind: int, candidate_cap: Optional[int] = None) -> int:
    if candidate_cap is not None and candidate_cap > 0:
        return candidate_cap
    return max(per_kind * 4, DEFAULT_CANDIDATE_BUDGET, DEFAULT_CANDIDATE_CAP)


def _gather_candidate_inputs(
    target: Target,
    output_dir: Path,
    per_kind: int,
    *,
    append_existing: bool,
    candidate_cap: int,
    attempt: int = 0,
) -> tuple[list[dict], dict, list[dict], list[dict], list[dict]]:
    candidate_budget = max(per_kind * 3, DEFAULT_CANDIDATE_BUDGET)
    if attempt > 0:
        candidate_budget = max(candidate_budget, per_kind * 5)
    llm_inputs, llm_meta = llm_candidate_inputs(
        target.context, budget=candidate_budget, attempt=attempt,
    )
    requires_inputs = generate_requires_satisfying_candidate_inputs(
        target.context, budget=candidate_budget * (2 if attempt > 0 else 1),
    )
    heuristic_inputs = generate_candidate_inputs(
        target.context, budget=candidate_budget * (2 if attempt > 0 else 1),
    )
    seen: set[str] = set()
    existing_input_keys = _existing_input_keys(output_dir, target) if append_existing else set()
    candidate_inputs: list[dict] = []
    for inp in [*llm_inputs, *requires_inputs, *heuristic_inputs]:
        key = _typed_input_key(target, inp)
        if key in seen:
            continue
        if key in existing_input_keys:
            continue
        seen.add(key)
        candidate_inputs.append(inp)
        if len(candidate_inputs) >= candidate_cap * (2 if attempt > 0 else 1):
            break
    return candidate_inputs, llm_meta, llm_inputs, requires_inputs, heuristic_inputs


def _positive_cases_from_working(target: Target, working_cases: list[dict]) -> list[tuple[int, dict, dict]]:
    """返回 (working_index, typed_inputs, output_dict) 列表，用于 negative 补齐。"""
    positives: list[tuple[int, dict, dict]] = []
    for idx, case in enumerate(working_cases):
        if case.get("expected") == "INVALID_INPUT":
            continue
        parsed_inputs = _parse_benchmark_value(case.get("input"))
        if not isinstance(parsed_inputs, dict):
            continue
        inputs = typed_input_payload(target.context, parsed_inputs)
        if inputs is None:
            continue
        negative_context = _negative_context(target)
        if target.context.get("returns"):
            output_dict = _return_payload_from_benchmark(negative_context, case.get("expected"))
        else:
            mut_returns = negative_context.get("returns") or []
            parsed_output = _parse_benchmark_value(case.get("expected"))
            if len(mut_returns) == 1:
                output_dict = {mut_returns[0].get("name"): parsed_output}
            elif isinstance(parsed_output, dict):
                output_dict = {
                    ret.get("name"): parsed_output.get(ret.get("name"))
                    for ret in mut_returns
                }
            else:
                continue
        positives.append((idx, inputs, output_dict))
    return positives


def _supplement_negatives_in_working(
    target: Target,
    working_cases: list[dict],
    need_neg: int,
    negatives_per_positive: int,
) -> tuple[int, dict]:
    """在 working_cases 上就地追加 negative（unexpected），返回新增数量。"""
    if need_neg <= 0:
        return 0, {"verus_checked": 0, "skipped_no_verus": 0}

    positives = _positive_cases_from_working(target, working_cases)
    if not positives:
        return 0, {"verus_checked": 0, "skipped_no_verus": 0}

    negative_context = _negative_context(target)
    pending: list[tuple[int, dict, dict, dict, str]] = []
    for case_idx, inputs, output_dict in positives:
        unexpected = working_cases[case_idx].get("unexpected")
        if not isinstance(unexpected, list):
            unexpected = []
        if len(unexpected) >= negatives_per_positive:
            continue
        for mutated_output, label in mutate_output_values(negative_context, inputs, output_dict):
            typed_mut = typed_output_payload(negative_context, mutated_output)
            if typed_mut is not None:
                pending.append((case_idx, inputs, typed_mut, mutated_output, label))

    method_log = {"verus_checked": 0, "skipped_no_verus": 0, "accepted_mutations": []}
    ensures = negative_context.get("ensures") or []
    if not ensures or not pending:
        method_log["skipped_no_verus"] = len(pending)
        return 0, method_log

    verdicts = _negative_verdicts(negative_context, pending)
    method_log["verification_reasons"] = {}
    added = 0
    per_case_added: dict[int, int] = {}
    for i, (case_idx, _inputs, typed_mut, mutated_output, _label) in enumerate(pending):
        if added >= need_neg:
            break
        unexpected = working_cases[case_idx].get("unexpected")
        if not isinstance(unexpected, list):
            unexpected = []
        if len(unexpected) >= negatives_per_positive:
            continue
        if per_case_added.get(case_idx, 0) >= negatives_per_positive:
            continue
        detail = verdicts.get(f"mut_{i}") or {"accepted": None, "reason": "verification_unresolved"}
        detail = _negative_detail_with_local_fallback(
            detail, negative_context, _inputs, mutated_output,
        )
        accepted = detail.get("accepted")
        reason = str(detail.get("reason") or "verification_unresolved")
        method_log["verification_reasons"][reason] = method_log["verification_reasons"].get(reason, 0) + 1
        if accepted is False:
            method_log["verus_checked"] += 1
            new_val = format_typed_output_for_json(target.context, mutated_output, target.func)
            expected = working_cases[case_idx].get("expected")
            updated_unexpected = _dedupe_unexpected(
                expected, [*unexpected, new_val],
            )
            if len(updated_unexpected) == len(unexpected):
                continue
            working_cases[case_idx]["unexpected"] = updated_unexpected
            method_log["accepted_mutations"].append({
                "inputs": _inputs,
                "mutated_output": new_val,
                "label": _label,
                "verification_reason": reason,
            })
            per_case_added[case_idx] = per_case_added.get(case_idx, 0) + 1
            added += 1
        elif accepted is None:
            method_log["skipped_no_verus"] += 1
    return added, method_log


def _fill_invalid_gap(
    target: Target,
    output_dir: Path,
    need_inv: int,
    per_kind: int,
    verus_bin: str,
    *,
    verbose: bool,
    existing_positive_inputs: list[dict],
    existing_invalid_inputs: list[dict],
    used_input_keys: set[str],
) -> tuple[list[dict], dict]:
    """invalid 不足时：LLM panic + 边界候选 + requires 违反，尽量补到 need_inv 个。"""
    invalid_cases: list[dict] = []
    invalid_meta: dict = {
        "panic": 0, "timeout_invalid": 0, "unknown_invalid": 0, "requires_violation": 0,
    }
    if need_inv <= 0:
        return invalid_cases, invalid_meta

    existing_test_inputs = [
        _parse_stored_input(c.get("input"))
        for c in _load_existing_cases(output_dir)
        if isinstance(c.get("input"), dict)
    ]
    panic_budget = max(need_inv * 4, per_kind * 3, 24)
    if _has_runtime_panic_hazard(target):
        panic_inputs, panic_llm_meta = llm_panic_candidate_inputs(
            target.context, source_code=target.raw_code, budget=panic_budget,
        )
    else:
        panic_inputs, panic_llm_meta = [], {"status": "skipped", "reason": "no_static_panic_hazard"}
    if len(panic_inputs) < need_inv * 2 and panic_llm_meta.get("status") == "ok":
        retry_inputs, _ = llm_panic_candidate_inputs(
            target.context, source_code=target.raw_code, budget=panic_budget, attempt=1,
        )
        seen_panic = {json.dumps(inp, ensure_ascii=False, sort_keys=True) for inp in panic_inputs}
        for inp in retry_inputs:
            key = json.dumps(inp, ensure_ascii=False, sort_keys=True)
            if key not in seen_panic:
                seen_panic.add(key)
                panic_inputs.append(inp)

    boundary_inputs = generate_boundary_candidate_inputs(
        target.context, budget=max(need_inv * 4, per_kind * 3, 24),
    )
    extra_panic: list[dict] = []
    for inp in [*panic_inputs, *boundary_inputs]:
        key = json.dumps(inp, ensure_ascii=False, sort_keys=True)
        if key in used_input_keys:
            continue
        used_input_keys.add(key)
        extra_panic.append(inp)
        if len(extra_panic) >= panic_budget * 2:
            break

    panic_accepted = 0
    req_accepted = 0
    if extra_panic:
        with tempfile.TemporaryDirectory() as work_dir_panic:
            run_panic = run_verus_compile_batch(target, extra_panic, work_dir_panic, verus_bin)
        if run_panic.panic or run_panic.runtime_invalid:
            more_invalid, sub_log = collect_invalid_cases(
                target, run_panic.panic, run_panic.runtime_invalid,
                existing_positive_inputs + existing_invalid_inputs + [c["input"] for c in invalid_cases],
                need_inv,
            )
            panic_accepted = len(more_invalid)
            invalid_cases.extend(more_invalid)
            for k, v in sub_log.items():
                invalid_meta[k] = invalid_meta.get(k, 0) + v

        requires = target.context.get("requires") or []
        if requires and len(invalid_cases) < need_inv:
            req_verdicts = batch_verus_requires_check(
                target.context,
                [{"inputs": inp, "key": f"panic_req_{i}"} for i, inp in enumerate(extra_panic)],
            )
            req_seen = {json.dumps(inp, ensure_ascii=False, sort_keys=True) for inp in existing_invalid_inputs}
            req_seen |= {json.dumps(inp, ensure_ascii=False, sort_keys=True) for inp in existing_positive_inputs}
            req_seen |= {json.dumps(inp, ensure_ascii=False, sort_keys=True) for inp in existing_test_inputs}
            for i, inp in enumerate(extra_panic):
                if len(invalid_cases) >= need_inv:
                    break
                if req_verdicts.get(f"panic_req_{i}") is not False:
                    continue
                key = json.dumps(inp, ensure_ascii=False, sort_keys=True)
                if key in req_seen:
                    continue
                req_seen.add(key)
                invalid_cases.append({
                    "input": inp,
                    "kind_detail": "requires_violation",
                    "source": "panic_candidate_requires_check",
                    "requires_verdict": False,
                    "runtime_status": "ok",
                    "tags": sorted(set(tag_input_values(inp)) | {"requires_boundary", "panic_candidate"}),
                })
                req_accepted += 1

    requires = target.context.get("requires") or []
    if requires and len(invalid_cases) < need_inv:
        supplement, _supp_meta = generate_invalid_cases_from_reference_requires(
            target.context,
            existing_cases=[
                {"inputs": c["input"]} for c in invalid_cases
            ] + [{"inputs": inp} for inp in existing_positive_inputs]
            + [{"inputs": inp} for inp in existing_test_inputs],
            budget=max(need_inv * 6, per_kind * 6, 36),
            positive_inputs=existing_positive_inputs,
        )
        for case in supplement:
            if len(invalid_cases) >= need_inv:
                break
            inp = case.get("inputs")
            if not isinstance(inp, dict):
                continue
            invalid_cases.append({
                "input": inp,
                "kind_detail": "requires_violation",
                "source": case.get("source") or "requires_generator",
                "requires_verdict": False,
                "runtime_status": "not_run",
                "tags": case.get("tags") or sorted(set(tag_input_values(inp)) | {"requires_boundary"}),
            })
            req_accepted += 1

    invalid_meta["llm_panic_candidates"] = len(extra_panic)
    invalid_meta["llm_panic_accepted"] = panic_accepted
    invalid_meta["panic_req_accepted"] = req_accepted
    if panic_llm_meta.get("status"):
        invalid_meta["llm_panic_status"] = panic_llm_meta.get("status")
    return invalid_cases[:need_inv], invalid_meta


def _write_working_cases(
    output_dir: Path,
    target: Target,
    working_cases: list[dict],
    *,
    source: str,
    status: str,
    notes: list[str],
    positives_meta: dict,
    invalid_meta: dict,
    negative_meta: dict,
    llm_meta: dict,
    candidate_count: int,
    per_kind: int = DEFAULT_PER_KIND,
    invalid_details: Sequence[dict] = (),
) -> dict:
    existing_meta = _load_existing_meta(output_dir)
    accepted_mutations = list(negative_meta.get("accepted_mutations") or [])
    positive_count, negative_count, invalid_count = _case_counts(working_cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(working_cases, f, indent=2, ensure_ascii=False, default=str)

    meta = {
        "source": source,
        "source_hash": source_hash(source),
        "function": target.func.name,
        "parameters": [{"name": p.name, "type": p.rust_type} for p in target.func.params],
        "returns": target.context.get("returns") or [],
        "return_type": target.func.return_type,
        "status": status,
        "positive_count": positive_count,
        "invalid_count": invalid_count,
        "negative_count": negative_count,
        "runner": "verus_native_compile",
        "compile_command": "verus --no-verify --compile -C panic=unwind",
        "candidate_count": candidate_count,
        "positive_select_log": positives_meta,
        "invalid_log": invalid_meta,
        "negative_log": {k: v for k, v in negative_meta.items() if k != "accepted_mutations"},
        "llm_meta": {
            k: v for k, v in llm_meta.items()
            if k in {"status", "accepted_candidates", "raw_candidates", "cached", "llm"}
        },
        "notes": notes,
        "supplement_gaps": True,
    }
    formatted_invalid_details = [
        {**item, "input": format_input_json(item["input"], target.func)}
        for item in invalid_details
        if isinstance(item, dict) and isinstance(item.get("input"), dict)
    ]
    _enrich_meta_v2(
        meta, target, working_cases,
        per_kind=per_kind,
        invalid_details=formatted_invalid_details,
        negative_details=accepted_mutations,
        existing_meta=existing_meta,
    )
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    return meta


def supplement_gaps(
    reference_path: str,
    output_dir: Path,
    per_kind: int,
    negatives_per_positive: int,
    verus_bin: str,
    verbose: bool = False,
    candidate_cap: Optional[int] = None,
) -> dict:
    """按缺口定向补齐 positive/negative/invalid，保留已有 test.json 用例。"""
    name = Path(reference_path).stem
    output_dir = Path(output_dir)
    cap = _resolve_candidate_cap(per_kind, candidate_cap)

    if not (output_dir / "test.json").exists():
        return process_one(
            reference_path, output_dir, per_kind, negatives_per_positive, verus_bin,
            verbose=verbose, append_existing=False, candidate_cap=cap,
        )

    working_cases = [dict(c) for c in _load_existing_cases(output_dir)]
    existing_meta = _load_existing_meta(output_dir)
    target = resolve_existing_target(
        reference_path, working_cases, str(existing_meta.get("function") or "") or None,
    )
    if target is None:
        return {"name": name, "status": "no_function"}
    pos, neg, inv = _case_counts(working_cases)
    positive_target = _positive_target_info(target, per_kind)["target"]
    invalid_target = _invalid_target_info(target, per_kind)["target"]
    negative_target = _negative_target_info(target, per_kind)["target"]
    need_pos = max(0, positive_target - pos)
    need_neg = max(0, negative_target - neg)
    need_inv = max(0, invalid_target - inv)

    if need_pos == 0 and need_neg == 0 and need_inv == 0:
        if verbose:
            print(f"  [{name}] complete pos={pos} neg={neg} inv={inv}")
        return _existing_result(output_dir, name, "ok")

    unsupported_issues = _unsupported_runtime_type_issues(target.func, target.context.get("type_registry"))
    if unsupported_issues:
        if verbose:
            print(f"  [{name}] skip unsupported_runtime_type")
        return _existing_result(output_dir, name, "unsupported_runtime_type")

    notes: list[str] = [f"supplement_gaps: need_pos={need_pos} need_neg={need_neg} need_inv={need_inv}"]
    positives_meta: dict = {}
    invalid_meta: dict = {"panic": 0, "timeout_invalid": 0, "unknown_invalid": 0, "requires_violation": 0}
    negative_meta: dict = {"verus_checked": 0, "skipped_no_verus": 0}
    llm_meta: dict = {"status": "skipped", "reason": "supplement_only"}
    candidate_count = 0

    existing_positive_inputs = [
        inp for _idx, inp, _out in _positive_cases_from_working(target, working_cases)
    ]
    existing_invalid_inputs = [
        _parse_stored_input(c.get("input"))
        for c in working_cases if c.get("expected") == "INVALID_INPUT"
    ]
    used_input_keys = {
        json.dumps(_parse_stored_input(c.get("input")), ensure_ascii=False, sort_keys=True)
        for c in working_cases if isinstance(c.get("input"), dict)
    }

    if need_pos > 0:
        candidate_inputs, llm_meta, llm_inputs, requires_inputs, heuristic_inputs = _gather_candidate_inputs(
            target, output_dir, per_kind, append_existing=True, candidate_cap=cap,
        )
        candidate_count = len(candidate_inputs)
        new_positives: list[tuple[dict, dict, str]] = []
        if candidate_inputs:
            with tempfile.TemporaryDirectory() as work_dir:
                run = run_verus_compile_batch(target, candidate_inputs, work_dir, verus_bin)
            new_positives, positives_meta = select_positives(target, run.ok, need_pos)
            positives_meta["requires_candidate_count"] = len(requires_inputs)
            positives_meta["llm_candidate_count"] = len(llm_inputs)
            positives_meta["heuristic_candidate_count"] = len(heuristic_inputs)
            for inputs, output_dict, _out_str in new_positives:
                working_cases.append({
                    "input": format_input_json(inputs, target.func),
                    "expected": format_typed_output_for_json(target.context, output_dict, target.func),
                    "unexpected": [],
                })
                existing_positive_inputs.append(inputs)
                used_input_keys.add(json.dumps(inputs, ensure_ascii=False, sort_keys=True))
            notes.append(f"added_positives={len(new_positives)}")
        else:
            notes.append("no_new_positive_candidates")

        # Symmetric with process_one: if still short on positives, retry with higher
        # temperature LLM + expanded requires/heuristic seeds.
        pos_now, _, _ = _case_counts(working_cases)
        still_need_pos = max(0, positive_target - pos_now)
        if still_need_pos > 0:
            retry_inputs, retry_llm_meta, retry_llm, retry_req, retry_heur = _gather_candidate_inputs(
                target, output_dir, per_kind, append_existing=True, candidate_cap=cap, attempt=1,
            )
            if retry_llm_meta.get("status") == "ok":
                llm_meta = retry_llm_meta
            if retry_inputs:
                candidate_count += len(retry_inputs)
                with tempfile.TemporaryDirectory() as work_dir:
                    run_retry = run_verus_compile_batch(target, retry_inputs, work_dir, verus_bin)
                more_positives, retry_meta = select_positives(target, run_retry.ok, still_need_pos)
                positives_meta = {
                    **positives_meta,
                    **{f"retry_{k}": v for k, v in retry_meta.items()},
                    "retry_requires_candidate_count": len(retry_req),
                    "retry_llm_candidate_count": len(retry_llm),
                    "retry_heuristic_candidate_count": len(retry_heur),
                }
                for inputs, output_dict, _out_str in more_positives:
                    working_cases.append({
                        "input": format_input_json(inputs, target.func),
                        "expected": format_typed_output_for_json(target.context, output_dict, target.func),
                        "unexpected": [],
                    })
                    existing_positive_inputs.append(inputs)
                    used_input_keys.add(json.dumps(inputs, ensure_ascii=False, sort_keys=True))
                notes.append(f"added_positives_retry={len(more_positives)}")

    if need_neg > 0:
        neg_added, negative_meta = _supplement_negatives_in_working(
            target, working_cases, need_neg, negatives_per_positive,
        )
        notes.append(f"added_negatives={neg_added}")

    if need_inv > 0:
        new_invalid, invalid_meta = _fill_invalid_gap(
            target, output_dir, need_inv, per_kind, verus_bin,
            verbose=verbose,
            existing_positive_inputs=existing_positive_inputs,
            existing_invalid_inputs=existing_invalid_inputs,
            used_input_keys=used_input_keys,
        )
        for inv in new_invalid:
            working_cases.append({
                "input": format_input_json(inv["input"], target.func),
                "expected": "INVALID_INPUT",
                "unexpected": [],
            })
        notes.append(f"added_invalid={len(new_invalid)}")
        if verbose and invalid_meta.get("llm_panic_candidates"):
            print(
                f"  [{name}] invalid 补齐: 候选={invalid_meta.get('llm_panic_candidates')} "
                f"panic接受={invalid_meta.get('llm_panic_accepted', 0)} "
                f"requires接受={invalid_meta.get('panic_req_accepted', 0)}"
            )

    pos, neg, inv = _case_counts(working_cases)
    status = "ok" if pos >= positive_target else ("partial" if pos > 0 else "no_valid_cases")
    meta = _write_working_cases(
        output_dir, target, working_cases,
        source=reference_path, status=status, notes=notes,
        positives_meta=positives_meta, invalid_meta=invalid_meta,
        negative_meta=negative_meta, llm_meta=llm_meta,
        candidate_count=candidate_count,
        per_kind=per_kind,
        invalid_details=new_invalid if need_inv > 0 else (),
    )
    if verbose:
        print(f"  [{name}] supplement pos={meta['positive_count']} neg={meta['negative_count']} "
              f"inv={meta['invalid_count']} status={meta['status']}")
    return {
        "name": name,
        "status": meta["status"],
        "positive": meta["positive_count"],
        "negative": meta["negative_count"],
        "invalid": meta["invalid_count"],
    }


def refresh_existing_meta(reference_path: str, output_dir: Path, per_kind: int) -> dict:
    """Recompute schema-v2 counts/category targets without compiling or generating cases."""
    name = Path(reference_path).stem
    output_dir = Path(output_dir)
    existing_meta = _load_existing_meta(output_dir)
    test_cases = _load_existing_cases(output_dir)
    target = resolve_existing_target(
        reference_path, test_cases, str(existing_meta.get("function") or "") or None,
    )
    if target is None:
        return {"name": name, "status": "no_function"}

    positive, negative, invalid = _case_counts(test_cases)
    meta = dict(existing_meta)
    meta.update({
        "source": reference_path,
        "source_hash": source_hash(reference_path),
        "function": target.func.name,
        "positive_count": positive,
        "negative_count": negative,
        "invalid_count": invalid,
    })
    _enrich_meta_v2(
        meta,
        target,
        test_cases,
        per_kind=per_kind,
        existing_meta=existing_meta,
    )
    with open(output_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False, default=str)
    return {
        "name": name,
        "status": meta.get("status", "error"),
        "positive": positive,
        "negative": negative,
        "invalid": invalid,
    }


# ---------------------------------------------------------------------------
# 单文件主流程
# ---------------------------------------------------------------------------

def process_one(
    reference_path: str,
    output_dir: Path,
    per_kind: int,
    negatives_per_positive: int,
    verus_bin: str,
    verbose: bool = False,
    append_existing: bool = False,
    candidate_cap: Optional[int] = None,
) -> dict:
    name = Path(reference_path).stem
    target = resolve_target(reference_path)
    if target is None:
        _write_terminal_meta(output_dir, reference_path, "no_function", "no_function", per_kind)
        return {"name": name, "status": "no_function"}

    unsupported_issues = _unsupported_runtime_type_issues(target.func, target.context.get("type_registry"))
    if unsupported_issues:
        if append_existing and (output_dir / "test.json").exists():
            return _existing_result(output_dir, name, "unchanged_unsupported_runtime_type")
        notes = ["unsupported_runtime_type: native harness cannot construct or print at least one IO type"]
        _write_output(output_dir, target, [], [], [], notes, "unsupported_runtime_type",
                      positives_meta={},
                      invalid_meta={"panic": 0, "timeout_invalid": 0, "unknown_invalid": 0, "requires_violation": 0},
                      negative_meta={"verus_checked": 0, "skipped_no_verus": 0},
                      llm_meta={"status": "skipped", "reason": "unsupported_runtime_type"},
                      candidate_count=0, source=reference_path,
                      failure_category="unsupported_runtime_type",
                      unsupported_type_issues=unsupported_issues, per_kind=per_kind)
        if verbose:
            print(f"  [{name}] unsupported_runtime_type")
        return {"name": name, "status": "unsupported_runtime_type", "positive": 0, "negative": 0, "invalid": 0}

    invalid_target = _invalid_target_info(target, per_kind)["target"]
    negative_target = _negative_target_info(target, per_kind)["target"]
    invalid_generation_goal = invalid_target
    if invalid_target == 0 and _has_runtime_panic_hazard(target):
        # Runtime-only invalids are opportunistic: confirm at most one panic,
        # but do not make it a completion quota.
        invalid_generation_goal = 1

    candidate_budget = max(per_kind * 3, DEFAULT_CANDIDATE_BUDGET)
    cap = _resolve_candidate_cap(per_kind, candidate_cap)
    # 有 requires 时，LLM 是主策略（能理解 spec fn 语义）；启发式作为兜底补充。
    has_requires = bool(target.context.get("requires"))

    candidate_inputs, llm_meta, llm_inputs, requires_inputs, heuristic_inputs = _gather_candidate_inputs(
        target, output_dir, per_kind, append_existing=append_existing, candidate_cap=cap,
    )

    if not candidate_inputs:
        if append_existing and (output_dir / "test.json").exists():
            # 有 requires 且已有 invalid 不足时，补充违反 requires 的 invalid。
            # append-existing 模式下已有 positive 可能覆盖候选空间导致 candidate_inputs 空，
            # 但 invalid 补齐（generate_invalid_cases_from_reference_requires）不依赖新 candidate，
            # 可基于已有 case 去重生成新的违反 requires 输入。
            requires = target.context.get("requires") or []
            if requires:
                existing_cases = _load_existing_cases(output_dir)
                existing_invalid = [c for c in existing_cases if c.get("expected") == "INVALID_INPUT"]
                if len(existing_invalid) < invalid_target:
                    existing_other_inputs = [
                        _parse_stored_input(c.get("input"))
                        for c in existing_cases if c.get("expected") != "INVALID_INPUT"
                    ]
                    supplement, _supp_meta = generate_invalid_cases_from_reference_requires(
                        target.context,
                        existing_cases=[
                            {"inputs": _parse_stored_input(c.get("input"))} for c in existing_invalid
                        ] + [{"inputs": inp} for inp in existing_other_inputs],
                        budget=max(per_kind * 6, 30),
                        positive_inputs=existing_other_inputs,
                    )
                    invalid_cases = [
                        {
                            "input": s["inputs"],
                            "kind_detail": "requires_violation",
                            "source": s.get("source") or "requires_generator",
                            "requires_verdict": False,
                            "runtime_status": "not_run",
                            "tags": s.get("tags")
                            or sorted(set(tag_input_values(s["inputs"])) | {"requires_boundary"}),
                        }
                        for s in supplement
                        if isinstance(s.get("inputs"), dict)
                    ]
                    if invalid_cases:
                        inv_meta = {
                            "panic": 0, "timeout_invalid": 0, "unknown_invalid": 0,
                            "requires_violation": len(invalid_cases),
                            "panic_req_accepted": len(invalid_cases),
                            "supplement_only": True,
                        }
                        meta = _write_output(
                            output_dir, target, [], invalid_cases, [], [], "ok",
                            positives_meta={}, invalid_meta=inv_meta,
                            negative_meta={"verus_checked": 0, "skipped_no_verus": 0},
                            llm_meta={"status": "skipped", "reason": "no_candidate_inputs_invalid_supplement"},
                            candidate_count=0, source=reference_path, append_existing=True,
                            per_kind=per_kind,
                        )
                        if verbose:
                            print(f"  [{name}] invalid 补齐(无新候选): +{len(invalid_cases)} "
                                  f"pos={meta['positive_count']} neg={meta['negative_count']} inv={meta['invalid_count']}")
                        return {
                            "name": name, "status": meta["status"],
                            "positive": meta["positive_count"], "negative": meta["negative_count"],
                            "invalid": meta["invalid_count"],
                        }
            return _existing_result(output_dir, name, "unchanged_no_candidate_inputs")
        meta = _write_output(
            output_dir, target, [], [], [], ["no_candidate_inputs"], "no_valid_cases",
            positives_meta={},
            invalid_meta={"panic": 0, "timeout_invalid": 0, "unknown_invalid": 0, "requires_violation": 0},
            negative_meta={"verus_checked": 0, "skipped_no_verus": 0},
            llm_meta=llm_meta,
            candidate_count=0,
            source=reference_path,
            failure_category="no_candidate_inputs",
            per_kind=per_kind,
        )
        return {
            "name": name,
            "status": meta["status"],
            "positive": 0,
            "negative": 0,
            "invalid": 0,
        }

    # 2) verus --compile 运行
    notes: list[str] = []
    with tempfile.TemporaryDirectory() as work_dir:
        run = run_verus_compile_batch(target, candidate_inputs, work_dir, verus_bin)

    if not run.ok and not run.panic and not run.runtime_invalid:
        if append_existing and (output_dir / "test.json").exists():
            return _existing_result(output_dir, name, "unchanged_no_runtime_results")
        # 编译/运行失败：不回退（no_fallback），记录原因
        notes.append(f"verus_compile_failed: {run.failure_reason}")
        status = "no_runtime_results" if run.failure_reason == "ok" else "no_valid_cases"
        failure_category = "no_runtime_results" if status == "no_runtime_results" else "compile_or_run_failed"
        _write_output(output_dir, target, [], [], [], notes, status,
                      positives_meta={},
                      invalid_meta={"panic": 0, "timeout_invalid": 0, "unknown_invalid": 0, "requires_violation": 0},
                      negative_meta={"verus_checked": 0, "skipped_no_verus": 0},
                      llm_meta=llm_meta, candidate_count=len(candidate_inputs), source=reference_path,
                      failure_category=failure_category,
                      runtime_diagnostics=run.diagnostics, per_kind=per_kind)
        if verbose:
            print(f"  [{name}] COMPILE/RUN FAILED: {run.failure_reason}")
        return {"name": name, "status": status, "reason": run.failure_reason}

    # 3) positive 筛选
    positives, positives_meta = select_positives(target, run.ok, per_kind)
    positives_meta["requires_candidate_count"] = len(requires_inputs)
    positives_meta["llm_candidate_count"] = len(llm_inputs)
    positives_meta["heuristic_candidate_count"] = len(heuristic_inputs)

    # 4) invalid
    existing_for_invalid = [inp for (inp, _o, _s) in positives]
    invalid_cases, invalid_meta = collect_invalid_cases(
        target, run.panic, run.runtime_invalid, existing_for_invalid, invalid_generation_goal
    )

    # 5) negative is generated after the positive retry, so newly recovered
    # positives can immediately serve as mutation bases.
    negatives: list[dict] = []
    negative_meta: dict = {"verus_checked": 0, "skipped_no_verus": 0}

    # 6) 兜底补齐：positive 不足时，先用 LLM（attempt=1，高温度）再扩启发式 budget
    if len(positives) < per_kind:
        extra_keys = {json.dumps(inp, ensure_ascii=False, sort_keys=True) for inp in candidate_inputs}
        more_inputs: list[dict] = []

        # 6a) LLM 第二轮（attempt=1，temperature=0.7，不同缓存键，生成更多样的候选）
        more_llm_inputs, _ = llm_candidate_inputs(
            target.context, budget=candidate_budget * 2, attempt=1
        )
        for inp in more_llm_inputs:
            k = json.dumps(inp, ensure_ascii=False, sort_keys=True)
            if k not in extra_keys:
                more_inputs.append(inp)
                extra_keys.add(k)

        # 6b) 扩大启发式 budget（补充多样性）
        more_requires_inputs = generate_requires_satisfying_candidate_inputs(
            target.context, budget=candidate_budget * 2
        )
        for inp in [*more_requires_inputs, *generate_candidate_inputs(target.context, budget=candidate_budget * 2)]:
            k = json.dumps(inp, ensure_ascii=False, sort_keys=True)
            if k not in extra_keys:
                more_inputs.append(inp)
                extra_keys.add(k)

        extra = more_inputs[:candidate_budget * 2]
        if extra:
            with tempfile.TemporaryDirectory() as work_dir3:
                run2 = run_verus_compile_batch(target, extra, work_dir3, verus_bin)
            more_positives, _ = select_positives(target, run2.ok, per_kind - len(positives))
            for inp, out_dict, out_str in more_positives:
                if len(positives) >= per_kind:
                    break
                positives.append((inp, out_dict, out_str))
            # 合并 panic 到 invalid（如果还有空间）
            if run2.panic:
                more_invalid, _ = collect_invalid_cases(
                    target, run2.panic,
                    run2.runtime_invalid,
                    existing_for_invalid + [c["input"] for c in invalid_cases],
                    max(0, invalid_generation_goal - len(invalid_cases)),
                )
                invalid_cases.extend(more_invalid)

    if negative_target > 0:
        negatives, negative_meta = collect_negative_cases(
            target, positives, negative_target, negatives_per_positive
        )

    # 6.5) invalid 兜底补齐：invalid 不足时，用 LLM 分析函数体生成可能触发 panic
    #      的候选输入并运行验证，收集运行时 panic 作为 invalid（不依赖显式 requires）。
    #      LLM 候选不足时重试一轮（高温度），并补充启发式边界候选提升覆盖。
    if len(invalid_cases) < invalid_generation_goal:
        # 加载已有 test.json 的 case（append-existing 模式），用于去重，
        # 避免生成的 invalid/panic 候选与已有 case 重复导致 _merge_test_cases 丢弃。
        existing_test_inputs: list[dict] = []
        if append_existing and (output_dir / "test.json").exists():
            existing_test_inputs = [
                _parse_stored_input(c.get("input"))
                for c in _load_existing_cases(output_dir)
                if isinstance(c.get("input"), dict)
            ]
        used_input_keys = {
            json.dumps(inp, ensure_ascii=False, sort_keys=True)
            for inp in candidate_inputs + [c["input"] for c in invalid_cases]
            + [inp for (inp, _o, _s) in positives] + existing_test_inputs
        }

        # 6.5a) LLM panic 候选（attempt=0）
        panic_budget = max(max(1, invalid_generation_goal) * 3, 18)
        if _has_runtime_panic_hazard(target):
            panic_inputs, panic_llm_meta = llm_panic_candidate_inputs(
                target.context, source_code=target.raw_code, budget=panic_budget,
            )
        else:
            panic_inputs, panic_llm_meta = [], {"status": "skipped", "reason": "no_static_panic_hazard"}
        # 6.5b) LLM 候选不足时重试一轮（attempt=1，高温度提升多样性）
        if len(panic_inputs) < max(1, invalid_generation_goal) * 2 and panic_llm_meta.get("status") == "ok":
            retry_inputs, _ = llm_panic_candidate_inputs(
                target.context, source_code=target.raw_code, budget=panic_budget, attempt=1,
            )
            seen_panic = {json.dumps(inp, ensure_ascii=False, sort_keys=True) for inp in panic_inputs}
            for inp in retry_inputs:
                key = json.dumps(inp, ensure_ascii=False, sort_keys=True)
                if key not in seen_panic:
                    seen_panic.add(key)
                    panic_inputs.append(inp)

        # 6.5c) 启发式边界候选补充（不依赖 LLM，覆盖空数组/极值/长度不匹配等）
        boundary_inputs = generate_boundary_candidate_inputs(
            target.context, budget=max(max(1, invalid_generation_goal) * 3, 18),
        )

        # 合并去重：LLM 优先，启发式补充
        extra_panic: list[dict] = []
        for inp in [*panic_inputs, *boundary_inputs]:
            key = json.dumps(inp, ensure_ascii=False, sort_keys=True)
            if key in used_input_keys:
                continue
            used_input_keys.add(key)
            extra_panic.append(inp)
            if len(extra_panic) >= panic_budget * 2:
                break

        panic_accepted = 0
        req_accepted = 0
        if extra_panic:
            # 6.5d-1) 跑 verus --compile 收集运行时 panic（数组越界、除零、unwrap on None 等）
            with tempfile.TemporaryDirectory() as work_dir_panic:
                run_panic = run_verus_compile_batch(target, extra_panic, work_dir_panic, verus_bin)
            if run_panic.panic or run_panic.runtime_invalid:
                more_invalid, _ = collect_invalid_cases(
                    target, run_panic.panic, run_panic.runtime_invalid,
                    [inp for (inp, _o, _s) in positives] + [c["input"] for c in invalid_cases],
                    max(0, invalid_generation_goal - len(invalid_cases)),
                )
                panic_accepted = len(more_invalid)
                invalid_cases.extend(more_invalid)
            # 6.5d-2) 有 requires 时，用 batch_verus_requires_check 验证候选是否违反 requires。
            #         verus --no-verify --compile 不在运行时检查 requires，故违反 requires 但
            #         body 有 bounds check 的输入不会 panic，需用 spec 层 requires check 判定。
            requires = target.context.get("requires") or []
            if requires and len(invalid_cases) < invalid_generation_goal:
                req_cases = [
                    {"inputs": inp, "key": f"panic_req_{i}"}
                    for i, inp in enumerate(extra_panic)
                ]
                req_verdicts = batch_verus_requires_check(target.context, req_cases)
                req_seen = {
                    json.dumps(c["input"], ensure_ascii=False, sort_keys=True)
                    for c in invalid_cases
                }
                req_seen |= {
                    json.dumps(inp, ensure_ascii=False, sort_keys=True)
                    for inp in [p[0] for p in positives]
                }
                req_seen |= {
                    json.dumps(inp, ensure_ascii=False, sort_keys=True)
                    for inp in existing_test_inputs
                }
                for i, inp in enumerate(extra_panic):
                    if len(invalid_cases) >= invalid_generation_goal:
                        break
                    if req_verdicts.get(f"panic_req_{i}") is not False:
                        continue
                    key = json.dumps(inp, ensure_ascii=False, sort_keys=True)
                    if key in req_seen:
                        continue
                    req_seen.add(key)
                    invalid_cases.append({
                        "input": inp,
                        "kind_detail": "requires_violation",
                        "source": "panic_candidate_requires_check",
                        "requires_verdict": False,
                        "runtime_status": "ok",
                        "tags": sorted(set(tag_input_values(inp)) | {"requires_boundary", "panic_candidate"}),
                    })
                    req_accepted += 1
        # 6.5d-3) 有 requires 且仍不足时，独立调用 generate_invalid_cases_from_reference_requires
        #         以更大 budget 补充违反 requires 的候选（LLM invalid + 启发式 + batch 验证）。
        #         不依赖 extra_panic：append-existing 模式下已有 positive 可能过滤光 panic 候选，
        #         但 requires 违反候选由 generate_invalid_cases_from_reference_requires 内部去重，
        #         能在已有 invalid 基础上生成新的违反输入。
        requires = target.context.get("requires") or []
        if requires and len(invalid_cases) < invalid_generation_goal:
            supplement, _supp_meta = generate_invalid_cases_from_reference_requires(
                target.context,
                existing_cases=[
                    {"inputs": c["input"]} for c in invalid_cases
                ] + [{"inputs": inp} for inp in [p[0] for p in positives]]
                + [{"inputs": inp} for inp in existing_test_inputs],
                budget=max(per_kind * 6, 30),
                positive_inputs=[p[0] for p in positives],
            )
            for case in supplement:
                if len(invalid_cases) >= invalid_generation_goal:
                    break
                inp = case.get("inputs")
                if not isinstance(inp, dict):
                    continue
                invalid_cases.append({
                    "input": inp,
                    "kind_detail": "requires_violation",
                    "source": case.get("source") or "requires_generator",
                    "requires_verdict": False,
                    "runtime_status": "not_run",
                    "tags": case.get("tags") or sorted(set(tag_input_values(inp)) | {"requires_boundary"}),
                })
                req_accepted += 1
        invalid_meta["llm_panic_candidates"] = len(extra_panic)
        invalid_meta["llm_panic_accepted"] = panic_accepted
        invalid_meta["panic_req_accepted"] = req_accepted
        panic_status = panic_llm_meta.get("status")
        if panic_status:
            invalid_meta["llm_panic_status"] = panic_status
        if verbose and (panic_accepted or req_accepted or extra_panic):
            print(f"  [{name}] invalid 补齐: 候选={len(extra_panic)} panic接受={panic_accepted} requires接受={req_accepted}")

    # 7) 写出
    positive_target = _positive_target_info(target, per_kind)["target"]
    status = "ok" if len(positives) >= positive_target else ("partial" if positives else "no_valid_cases")
    if status != "ok":
        notes.append(f"positive_count={len(positives)} < target={positive_target}")

    meta = _write_output(
        output_dir, target, positives, invalid_cases, negatives, notes, status,
        positives_meta=positives_meta, invalid_meta=invalid_meta,
        negative_meta=negative_meta, llm_meta=llm_meta,
        candidate_count=len(candidate_inputs), source=reference_path,
        append_existing=append_existing,
        per_kind=per_kind,
    )

    if verbose:
        print(
            f"  [{name}] positive={meta['positive_count']} "
            f"negative={meta['negative_count']} invalid={meta['invalid_count']} status={meta['status']}"
        )
    return {
        "name": name,
        "status": meta["status"],
        "positive": meta["positive_count"],
        "negative": meta["negative_count"],
        "invalid": meta["invalid_count"],
    }


def _load_existing_cases(output_dir: Path) -> list[dict]:
    test_path = output_dir / "test.json"
    try:
        data = json.loads(test_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [item for item in data if isinstance(item, dict)]


def _parse_stored_input(stored_input: Any) -> Any:
    """把 format 后存入 test.json 的 input 转回 raw，用于和新生成的候选统一去重。

    format_value_json 把 Vec/list 转成 str(val)（如 `[1, 2, 3]` → `"[1, 2, 3]"`），
    导致已有 input（str）与生成候选（raw list）的 json key 不一致，去重失效。
    这里用 ast.literal_eval 把形如 list/dict 的字符串还原。
    """
    if not isinstance(stored_input, dict):
        return stored_input
    raw: dict[str, Any] = {}
    for k, v in stored_input.items():
        if isinstance(v, str) and v.strip().startswith(("[", "{", "(")):
            try:
                raw[k] = ast.literal_eval(v)
                continue
            except (ValueError, SyntaxError):
                pass
        raw[k] = v
    return raw


def _existing_input_keys(output_dir: Path, target: Target) -> set[str]:
    keys: set[str] = set()
    for case in _load_existing_cases(output_dir):
        raw_input = case.get("input")
        if isinstance(raw_input, dict):
            keys.add(_typed_input_key(target, raw_input))
    return keys


def _typed_input_key(target: Target, raw_input: dict) -> str:
    """Canonicalize stored and generated inputs through the declared Rust types."""
    parsed = _parse_stored_input(raw_input)
    typed = typed_input_payload(target.context, parsed) if isinstance(parsed, dict) else None
    payload = typed if typed is not None else parsed
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _case_identity(case: dict) -> str:
    payload = {
        "input": case.get("input"),
        "expected": case.get("expected"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _dedupe_unexpected(expected: Any, values: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[str] = set()
    expected_key = json.dumps(expected, ensure_ascii=False, sort_keys=True, default=str)
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen or key == expected_key:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _merge_test_cases(existing: list[dict], new_cases: list[dict]) -> tuple[list[dict], int]:
    merged: list[dict] = []
    by_identity: dict[str, dict] = {}
    for case in [*existing, *new_cases]:
        if not isinstance(case.get("input"), dict) or "expected" not in case:
            continue
        identity = _case_identity(case)
        unexpected = case.get("unexpected")
        if not isinstance(unexpected, list):
            unexpected = []
        if identity in by_identity:
            prior = by_identity[identity]
            prior_unexpected = prior.get("unexpected")
            if not isinstance(prior_unexpected, list):
                prior_unexpected = []
            prior["unexpected"] = _dedupe_unexpected(
                prior.get("expected"),
                [*prior_unexpected, *unexpected],
            )
            continue
        copied = {
            "input": case.get("input"),
            "expected": case.get("expected"),
            "unexpected": _dedupe_unexpected(case.get("expected"), unexpected),
        }
        by_identity[identity] = copied
        merged.append(copied)
    return merged, max(0, len(merged) - len(existing))


def _case_counts(test_cases: list[dict]) -> tuple[int, int, int]:
    positive_count = 0
    invalid_count = 0
    negative_count = 0
    for case in test_cases:
        if case.get("expected") == "INVALID_INPUT":
            invalid_count += 1
            continue
        positive_count += 1
        unexpected = case.get("unexpected")
        if isinstance(unexpected, list):
            negative_count += len(unexpected)
    return positive_count, negative_count, invalid_count


def _existing_result(output_dir: Path, name: str, fallback_status: str) -> dict:
    meta_path = output_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        meta = {}
    test_cases = _load_existing_cases(output_dir)
    positive_count, negative_count, invalid_count = _case_counts(test_cases)
    return {
        "name": name,
        "status": meta.get("status") or fallback_status,
        "positive": int(meta.get("positive_count", positive_count) or positive_count),
        "negative": int(meta.get("negative_count", negative_count) or negative_count),
        "invalid": int(meta.get("invalid_count", invalid_count) or invalid_count),
    }


def _is_already_complete(output_dir: Path, per_kind: int) -> tuple[bool, dict]:
    """检查 output_dir/meta.json 是否显示 positive/negative/invalid 三类均已 >= per_kind.

    返回 (already_complete, counts)。meta.json 不存在或不可解析时返回 (False, {})，
    以便调用方对缺失题正常生成。
    """
    meta_path = output_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, {}
    category_status = meta.get("category_status")
    if isinstance(category_status, dict):
        terminal_states = {"complete", "not_applicable", "blocked"}
        states = [
            (category_status.get(kind) or {}).get("state")
            for kind in ("positive", "negative", "invalid")
        ]
        counts = {
            kind: int((category_status.get(kind) or {}).get("count", 0) or 0)
            for kind in ("positive", "negative", "invalid")
        }
        return (all(state in terminal_states for state in states), counts)
    positive = int(meta.get("positive_count", 0) or 0)
    negative = int(meta.get("negative_count", 0) or 0)
    invalid = int(meta.get("invalid_count", 0) or 0)
    counts = {"positive": positive, "negative": negative, "invalid": invalid}
    return (positive >= per_kind and negative >= per_kind and invalid >= per_kind), counts


def _write_output(
    output_dir: Path,
    target: Target,
    positives: list[tuple[dict, dict, str]],
    invalid_cases: list[dict],
    negatives: list[dict],
    notes: list[str],
    status: str,
    *,
    positives_meta: dict,
    invalid_meta: dict,
    negative_meta: dict,
    llm_meta: dict,
    candidate_count: int,
    source: str,
    failure_category: Optional[str] = None,
    unsupported_type_issues: Optional[list[dict]] = None,
    runtime_diagnostics: Optional[dict] = None,
    append_existing: bool = False,
    per_kind: int = DEFAULT_PER_KIND,
) -> dict:
    """写 test.json + meta.json."""
    existing_meta = _load_existing_meta(output_dir) if append_existing else {}
    # 重新组装 test.json（与 process_one 中的逻辑一致）
    test_cases: list[dict] = []
    unexpected_by_pos: dict[int, list] = {}
    for neg in negatives:
        bi = neg.get("base_index")
        if bi is None:
            continue
        unexpected_by_pos.setdefault(bi, []).append(
            format_typed_output_for_json(target.context, neg["mutated_output"], target.func)
        )
    for pos_idx, (inputs, output_dict, _output_str) in enumerate(positives):
        expected_json = format_typed_output_for_json(target.context, output_dict, target.func)
        unexpected_vals = unexpected_by_pos.get(pos_idx, [])
        deduped: list = []
        seen_u: set[str] = set()
        for u in unexpected_vals:
            key = json.dumps(u, default=str, sort_keys=True)
            if key in seen_u:
                continue
            if key == json.dumps(expected_json, default=str, sort_keys=True):
                continue
            seen_u.add(key)
            deduped.append(u)
        test_cases.append({
            "input": format_input_json(inputs, target.func),
            "expected": expected_json,
            "unexpected": deduped,
        })
    for inv in invalid_cases:
        test_cases.append({
            "input": format_input_json(inv["input"], target.func),
            "expected": "INVALID_INPUT",
            "unexpected": [],
        })

    append_added = 0
    if append_existing:
        existing_cases = _load_existing_cases(output_dir)
        test_cases, append_added = _merge_test_cases(existing_cases, test_cases)
        if append_added:
            notes = [*notes, f"append_existing: added {append_added} new unique case(s)"]
        else:
            notes = [*notes, "append_existing: no new unique cases"]

    positive_count, negative_count, invalid_count = _case_counts(test_cases)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False, default=str)

    meta = {
        "source": source,
        "source_hash": source_hash(source),
        "function": target.func.name,
        "parameters": [{"name": p.name, "type": p.rust_type} for p in target.func.params],
        "returns": target.context.get("returns") or [],
        "return_type": target.func.return_type,
        "status": status,
        "positive_count": positive_count,
        "invalid_count": invalid_count,
        "negative_count": negative_count,
        "runner": "verus_native_compile",
        "compile_command": "verus --no-verify --compile -C panic=unwind",
        "candidate_count": candidate_count,
        "positive_select_log": positives_meta,
        "invalid_log": invalid_meta,
        "negative_log": negative_meta,
        "llm_meta": {
            k: v for k, v in llm_meta.items()
            if k in {"status", "accepted_candidates", "raw_candidates", "cached", "llm"}
        },
        "notes": notes,
    }
    if append_existing:
        meta["append_existing"] = True
        meta["append_added_cases"] = append_added
    if failure_category:
        meta["failure_category"] = failure_category
    if unsupported_type_issues:
        meta["unsupported_type_issues"] = unsupported_type_issues
    if runtime_diagnostics:
        meta["runtime_diagnostics"] = runtime_diagnostics
    formatted_invalid_details = [
        {**item, "input": format_input_json(item["input"], target.func)}
        for item in invalid_cases
        if isinstance(item, dict) and isinstance(item.get("input"), dict)
    ]
    formatted_negative_details = [
        {
            **item,
            "mutated_output": format_typed_output_for_json(
                target.context, item.get("mutated_output") or {}, target.func,
            ),
        }
        for item in negatives
        if isinstance(item, dict)
    ]
    _enrich_meta_v2(
        meta, target, test_cases,
        per_kind=per_kind,
        invalid_details=formatted_invalid_details,
        negative_details=formatted_negative_details,
        existing_meta=existing_meta,
    )
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _gather_inputs(args: argparse.Namespace, output_base: Path) -> list[tuple[Path, Path]]:
    """返回 (输入 .rs, 该文件的输出目录) 对。

    - --input <file>          : 单文件         -> output_base/<stem>/
    - --input-dir <dir>       : 扁平一层        -> output_base/<stem>/
    - --input-root <root>     : 分层 <bench>/*.rs -> output_base/<bench>/<stem>/
                                （与 data/references 及打分时的
                                 <suite_root>/<benchmark>/<task>/ 布局一致）
    """
    if args.input:
        p = Path(args.input)
        return [(p, output_base / p.stem)]
    if args.input_root:
        root = Path(args.input_root)
        if not root.exists():
            print(f"Error: {root} does not exist", file=sys.stderr)
            sys.exit(1)
        pairs: list[tuple[Path, Path]] = []
        for p in sorted(root.glob("*/*.rs")):
            benchmark = p.parent.name
            pairs.append((p, output_base / benchmark / p.stem))
        return pairs
    in_dir = Path(args.input_dir)
    if not in_dir.exists():
        print(f"Error: {in_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    return [(p, output_base / p.stem) for p in sorted(in_dir.glob("*.rs"))]


def _summary_stats(results: list[dict]) -> dict[str, int]:
    stats: dict[str, int] = {}
    for result in results:
        status = str(result.get("status") or "error")
        stats[status] = stats.get(status, 0) + 1
    return stats


def _write_summary(output_base: Path, results: list[dict], *, append_existing: bool) -> dict[str, int]:
    summary_path = output_base / "summary.json"
    by_name: dict[str, dict] = {}
    existing_order: list[str] = []
    if append_existing:
        try:
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_summary = {}
        existing_results = (
            existing_summary.get("results")
            if isinstance(existing_summary, dict) and isinstance(existing_summary.get("results"), list)
            else []
        )
        for result in existing_results:
            if not isinstance(result, dict) or not result.get("name"):
                continue
            name = str(result["name"])
            if name not in by_name:
                existing_order.append(name)
            by_name[name] = result

    by_name.update({
        str(result.get("name")): result
        for result in results
        if isinstance(result, dict) and result.get("name")
    })
    # Meta/test files are authoritative.  Re-read them instead of carrying
    # stale counts forward from a previous summary merge.
    for meta_path in sorted(output_base.glob("**/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        name = meta_path.parent.name
        test_cases = _load_existing_cases(meta_path.parent)
        positive, negative, invalid = _case_counts(test_cases)
        if not test_cases:
            positive = int(meta.get("positive_count", 0) or 0)
            negative = int(meta.get("negative_count", 0) or 0)
            invalid = int(meta.get("invalid_count", 0) or 0)
        by_name[name] = {
            "name": name,
            "status": meta.get("status") or "error",
            "positive": positive,
            "negative": negative,
            "invalid": invalid,
        }
    existing_names = set(existing_order)
    ordered_names = [name for name in existing_order if name in by_name]
    ordered_names.extend(sorted(name for name in by_name if name not in existing_names))
    merged_results = [by_name[name] for name in ordered_names] if append_existing else [
        by_name[name] for name in sorted(by_name)
    ]

    stats = _summary_stats(merged_results)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "results": merged_results}, f, indent=2, ensure_ascii=False, default=str)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-driven offline IO test case generator (verus --compile backend)")
    parser.add_argument("--input", type=str, help="single .rs reference file")
    parser.add_argument("--input-dir", type=str, help="flat directory of .rs reference files")
    parser.add_argument("--input-root", type=str, help="layered root <benchmark>/<task>.rs (e.g. data/references); output preserves the benchmark layer")
    parser.add_argument("--output-dir", type=str, required=True, help="output directory root")
    parser.add_argument("--per-kind", type=int, default=DEFAULT_PER_KIND, help="target cases per kind (positive/negative/invalid)")
    parser.add_argument("--negatives-per-positive", type=int, default=DEFAULT_NEGATIVES_PER_POSITIVE)
    parser.add_argument("--parallel", type=int, default=1, help="parallel workers (threads)")
    parser.add_argument("--filter", type=str, default=None, help="only process files matching this substring")
    parser.add_argument("--verus-path", type=str, default=None, help="Verus binary path; defaults to config.yaml verus_path")
    parser.add_argument("--append-existing", action="store_true", help="append unique generated cases to existing test.json instead of replacing it")
    parser.add_argument("--skip-complete", action=argparse.BooleanOptionalAction, default=True,
                        help="skip files whose existing meta.json already has positive/negative/invalid all >= per-kind (default: on; use --no-skip-complete to disable)")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="disable LLM candidate calls and use only deterministic generators",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.no_llm:
        def _offline_candidates(*_args, **_kwargs):
            return [], {"status": "skipped", "reason": "no_llm_cli"}

        global llm_candidate_inputs, llm_panic_candidate_inputs
        llm_candidate_inputs = _offline_candidates
        llm_panic_candidate_inputs = _offline_candidates

    if not args.input and not args.input_dir and not args.input_root:
        parser.error("one of --input / --input-dir / --input-root is required")

    verus_bin = args.verus_path or _load_verus_path_from_config() or get_verus_binary()
    from metrics_rebuild.share.config import set_verus_binary
    set_verus_binary(verus_bin)
    if not Path(verus_bin).is_file() and shutil.which(verus_bin) is None:
        print(f"Warning: verus binary not found at {verus_bin}", file=sys.stderr)

    output_base = Path(args.output_dir)
    pairs = _gather_inputs(args, output_base)
    if args.filter:
        pairs = [(p, out) for (p, out) in pairs if args.filter in p.stem]

    results: list[dict] = []
    if args.skip_complete:
        filtered_pairs: list[tuple[Path, Path]] = []
        skipped_pairs: list[tuple[Path, Path]] = []
        for p, out_dir in pairs:
            complete, _counts = _is_already_complete(out_dir, args.per_kind)
            if complete:
                skipped_pairs.append((p, out_dir))
                continue
            filtered_pairs.append((p, out_dir))
        if skipped_pairs:
            print(f"Skipping {len(skipped_pairs)} already-complete file(s) "
                  f"(positive/negative/invalid all >= {args.per_kind})")
            for p, out_dir in skipped_pairs:
                existing = _existing_result(out_dir, p.stem, "ok")
                results.append(existing)
                if args.verbose:
                    print(f"  [skip] {p.stem}: positive={existing['positive']} "
                          f"negative={existing['negative']} invalid={existing['invalid']}")
        pairs = filtered_pairs

    print(f"Processing {len(pairs)} file(s) -> {output_base}")
    print(f"Verus binary: {verus_bin}")

    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    process_one,
                    str(p),
                    out_dir,
                    args.per_kind,
                    args.negatives_per_positive,
                    verus_bin,
                    args.verbose,
                    args.append_existing,
                ): p
                for p, out_dir in pairs
            }
            for future in as_completed(futures):
                try:
                    r = future.result()
                except Exception as exc:
                    r = {"status": "error", "reason": str(exc)[:300]}
                results.append(r)
    else:
        for p, out_dir in pairs:
            try:
                r = process_one(
                    str(p),
                    out_dir,
                    args.per_kind,
                    args.negatives_per_positive,
                    verus_bin,
                    args.verbose,
                    args.append_existing,
                )
            except Exception as exc:
                r = {"status": "error", "reason": str(exc)[:300]}
            results.append(r)
            if args.verbose:
                print(f"  -> {r.get('status')}")

    output_base.mkdir(parents=True, exist_ok=True)
    # --skip-complete 模式下只处理了部分题目，summary 需合并已有结果以保持完整
    summary_merge = args.append_existing or args.skip_complete
    stats = _write_summary(output_base, results, append_existing=summary_merge)

    print("\n=== Summary ===")
    print(f"Total: {sum(stats.values())}")
    for k, v in sorted(stats.items()):
        if v:
            print(f"  {k}: {v}")

    print(f"\nSummary written to {output_base / 'summary.json'}")


if __name__ == "__main__":
    main()
