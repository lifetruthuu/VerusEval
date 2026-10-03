from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

from metrics_rebuild.share.clauses import extract_clauses_from_text
from metrics_rebuild.share.functions import (
    FunctionInfo,
    extract_functions,
    function_blocks_for_path,
    function_declaration_prefix,
    spec_fn_blocks_for_path,
)
from metrics_rebuild.share.text import read_text


def source_hash(path: str) -> str:
    return hashlib.sha256(read_text(path).encode("utf-8", errors="replace")).hexdigest()[:16]


def code_part_before_comment(line: str) -> str:
    state = "normal"
    i = 0
    while i < len(line):
        ch = line[i]
        nxt = line[i + 1] if i + 1 < len(line) else ""
        if state == "normal":
            if ch == '"':
                state = "string"
            elif ch == "'":
                state = "char"
            elif ch == "/" and nxt == "/":
                return line[:i]
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
    return line


def brace_delta(line: str) -> int:
    clean = code_part_before_comment(line)
    return clean.count("{") - clean.count("}")


def llm_contract_expr_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().rstrip(",")


def llm_contract_block_for_function(function: FunctionInfo) -> str:
    signature = re.sub(r"\s+", " ", function_declaration_prefix(function.header)).strip().rstrip(",")
    clauses = [
        clause
        for clause in extract_clauses_from_text(function.header)
        if clause.kind in {"requires", "ensures", "default_ensures"}
    ]
    lines = [signature] if signature else [function.name]
    current_kind: Optional[str] = None
    for clause in clauses:
        expr = llm_contract_expr_text(clause.text)
        if not expr:
            continue
        if clause.kind != current_kind:
            lines.append(f"    {clause.kind}")
            current_kind = clause.kind
        lines.append(f"        {expr},")
    return "\n".join(lines)


def llm_contracts_text_for_path(path: str, function_name: Optional[str] = None) -> str:
    blocks: list[str] = []
    for function in extract_functions(path):
        if function.name == "main" or function.mode != "exec":
            continue
        if function_name and function.name != function_name:
            continue
        block = llm_contract_block_for_function(function).strip()
        if block:
            blocks.append(block)
    return "\n\n".join(blocks)


def llm_contract_blocks_for_path(path: str) -> list[dict]:
    blocks: list[dict] = []
    for function in extract_functions(path):
        if function.name == "main" or function.mode != "exec":
            continue
        clauses = [
            {
                "kind": clause.kind,
                "text": llm_contract_expr_text(clause.text),
                "normalized": clause.normalized,
            }
            for clause in extract_clauses_from_text(function.header)
            if clause.kind in {"requires", "ensures", "default_ensures"}
        ]
        contract_text = llm_contract_block_for_function(function).strip()
        if not contract_text and not clauses:
            continue
        blocks.append(
            {
                "function": function.name,
                "signature": re.sub(
                    r"\s+",
                    " ",
                    function_declaration_prefix(function.header),
                ).strip().rstrip(","),
                "contract": contract_text,
                "clauses": clauses,
                "has_contract": bool(clauses),
            }
        )
    return blocks


def compact_llm_code_text(text: str, *, max_chars: int = 8000) -> str:
    compact = re.sub(r"\n{3,}", "\n\n", str(text or "").strip())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 80].rstrip() + "\n/* truncated for LLM judge */"


def llm_executable_body_lines(body: str) -> list[str]:
    kept: list[str] = []
    proof_depth = 0
    assert_by_depth = 0
    contract_pending = False
    contract_keyword_re = re.compile(
        r"\b(requires|ensures|default_ensures|returns|recommends|opens_invariants|no_unwind|invariant|invariant_except_break|decreases)\b"
    )

    for line in str(body or "").splitlines():
        stripped = line.strip()
        code = code_part_before_comment(line).rstrip()
        if proof_depth > 0:
            proof_depth += brace_delta(line)
            if proof_depth <= 0:
                proof_depth = 0
            continue
        if assert_by_depth > 0:
            assert_by_depth += brace_delta(line)
            if assert_by_depth <= 0:
                assert_by_depth = 0
            continue
        if contract_pending:
            if "{" in code:
                brace_index = code.find("{")
                kept.append(f"{line[: len(line) - len(line.lstrip())]}{code[brace_index:].strip()}")
                contract_pending = False
            continue

        if not stripped:
            continue
        if re.search(r"\bproof\s*\{", code):
            proof_depth = max(brace_delta(line), 0)
            continue
        if "assert" in code or "assume" in code:
            if "by" in code and "{" in code:
                assert_by_depth = max(brace_delta(line), 0)
            continue

        keyword_match = contract_keyword_re.search(code)
        if keyword_match:
            prefix = code[: keyword_match.start()].rstrip()
            if prefix:
                kept.append(prefix)
            suffix = code[keyword_match.end() :]
            if "{" in suffix:
                brace_index = suffix.find("{")
                kept.append(f"{line[: len(line) - len(line.lstrip())]}{suffix[brace_index:].strip()}")
            else:
                contract_pending = True
            continue

        kept.append(code)
    return kept


def executable_body_view_for_llm(header: str, body: str) -> str:
    signature = re.sub(r"\s+", " ", function_declaration_prefix(header)).strip().rstrip(",")
    kept = llm_executable_body_lines(body)
    body_text = "\n".join(kept).strip()
    if not body_text:
        body_text = str(body or "").strip()
    return compact_llm_code_text(f"{signature} {{\n{body_text}\n}}")


def llm_code_blocks_for_path(path: str) -> list[dict]:
    blocks: list[dict] = []
    for block in function_blocks_for_path(path):
        body_view = executable_body_view_for_llm(
            str(block.get("header") or ""),
            str(block.get("body") or ""),
        )
        blocks.append(
            {
                "function": block.get("function"),
                "signature": re.sub(
                    r"\s+",
                    " ",
                    function_declaration_prefix(str(block.get("header") or "")),
                ).strip().rstrip(","),
                "implementation": body_view,
                "start_line": block.get("start_line"),
                "body_start_line": block.get("body_start_line"),
            }
        )
    return blocks


def spec_code_alignment_contexts(
    spec_rs_path: str,
    code_rs_path: str,
) -> tuple[list[dict], dict]:
    spec_blocks = llm_contract_blocks_for_path(spec_rs_path)
    code_blocks = llm_code_blocks_for_path(code_rs_path)
    code_by_name: dict[str, dict] = {
        str(block.get("function")): block
        for block in code_blocks
        if block.get("function")
    }
    contexts: list[dict] = []
    missing_code_functions: list[str] = []
    for spec in spec_blocks:
        function_name = str(spec.get("function") or "")
        code = code_by_name.get(function_name)
        if code is None:
            missing_code_functions.append(function_name)
            contexts.append({"function": function_name, "spec": spec, "code": None})
            continue
        contexts.append({"function": function_name, "spec": spec, "code": code})

    spec_function_names = {
        str(block.get("function"))
        for block in spec_blocks
        if block.get("function")
    }
    extra_code_functions = [
        str(block.get("function"))
        for block in code_blocks
        if block.get("function") and str(block.get("function")) not in spec_function_names
    ]
    metadata = {
        "spec_functions": [block.get("function") for block in spec_blocks],
        "code_functions": [block.get("function") for block in code_blocks],
        "matched_functions": [
            item.get("function")
            for item in contexts
            if item.get("code") is not None
        ],
        "missing_code_functions": missing_code_functions,
        "extra_code_functions": extra_code_functions,
        "spec_function_count": len(spec_blocks),
        "code_function_count": len(code_blocks),
        "matched_function_count": sum(1 for item in contexts if item.get("code") is not None),
    }
    return contexts, metadata


def llm_spec_code_judge_prompt(rs_path: str) -> list[dict]:
    payload = {
        "task": (
            "You are given one complete Verus .rs file, including spec, code, and proof. "
            "Use the full file content directly to judge whether its specifications and executable code are intent-consistent. "
            "Penalize missing behavior, wrong or over-strong constraints, vacuous/trivial contracts, and mismatched intent. "
            "Return JSON only."
        ),
        "score_definition": (
            "1.0 means the specification intent and executable behavior are consistent; "
            "0.0 means it is mostly unrelated, vacuous, or contradictory."
        ),
        "inputs": read_text(rs_path),
        "output_schema": {
            "score": 0.0,
            "verdict": "aligned|missing_constraints|too_strong|too_weak|incorrect|incomparable|vacuous|unknown",
            "reasoning": "concise explanation",
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a Verus intent-consistency judge. Return one JSON object matching the requested schema. Do not include markdown."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def _spec_fn_context_for_path(path: str) -> Optional[str]:
    blocks = spec_fn_blocks_for_path(path)
    if not blocks:
        return None
    texts = [re.sub(r"\s+", " ", b["text"]).strip() for b in blocks]
    return "\n\n".join(texts)


def llm_judge_prompt(
    generated_rs_path: str,
    ground_rs_path: str,
    function_name: Optional[str] = None,
) -> list[dict]:
    generated_contracts = llm_contracts_text_for_path(generated_rs_path, function_name)
    ground_contracts = llm_contracts_text_for_path(ground_rs_path, function_name)
    generated_spec_fns = _spec_fn_context_for_path(generated_rs_path)
    ground_spec_fns = _spec_fn_context_for_path(ground_rs_path)
    payload = {
        "target_function": function_name,
        "rubric": (
            "Compare generated Verus specifications against reference Verus specifications. "
            "Use function signatures and their requires/ensures/default_ensures contracts shown below. "
            "When spec_fn_definitions are provided, use them to understand the meaning of predicates "
            "referenced in the contracts — spec fn define helper predicates that may appear in "
            "requires/ensures clauses. "
            "Ignore any implementation or proof text not shown. "
            "Focus on behavioral intent, not formatting. "
            "Reward equivalent constraints, penalize missing postconditions, over-strong preconditions, "
            "vacuous/trivial clauses, and unrelated contract text. Return JSON only."
        ),
        "score_definition": "1.0 means semantically equivalent or stronger in the intended behavior; 0.0 means mostly unrelated or vacuous.",
        "generated": {
            **({"spec_fn_definitions": generated_spec_fns} if generated_spec_fns else {}),
            "contracts": generated_contracts,
        },
        "reference": {
            **({"spec_fn_definitions": ground_spec_fns} if ground_spec_fns else {}),
            "contracts": ground_contracts,
        },
        "output_schema": {
            "score": 0.0,
            "verdict": "equivalent|missing_constraints|too_strong|too_weak|incomparable|vacuous|unknown",
            "missing_constraints": ["short description"],
            "extra_or_overstrong_constraints": ["short description"],
            "vacuity_risks": ["short description"],
            "reasoning": "concise explanation",
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict Verus specification judge. "
                "Return a single JSON object matching the requested schema. Do not include markdown."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


__all__ = [
    "llm_judge_prompt",
    "llm_spec_code_judge_prompt",
    "source_hash",
    "spec_code_alignment_contexts",
]
