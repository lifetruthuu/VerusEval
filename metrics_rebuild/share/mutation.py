from __future__ import annotations

import hashlib
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence

from metrics_rebuild.share.functions import (
    extract_functions,
    function_blocks_for_path,
    function_parameters_from_header,
)
from metrics_rebuild.share.contract_eval import normalize_type_key
from metrics_rebuild.share.text import read_text, strip_comments
from metrics_rebuild.share.verus_runner import (
    DEFAULT_VERUS_TIMEOUT_SECONDS,
    VerusRun,
    run_verus,
    verus_run_to_dict,
)

DEFAULT_MUTANT_TIMEOUT_SECONDS = 12
DEFAULT_MAX_MUTANTS = 30

MUTATION_FAMILY_BUDGETS = {
    "ROR": 5,
    "AOR": 5,
    "COR_LOR": 5,
    "UOI_UOD": 5,
    "SVR": 5,
    "BOUNDARY": 5,
}

MUTATION_PATTERNS: tuple[dict, ...] = (
    {"pattern": re.compile(r"(?<![=!<>])==(?!>)"), "replacements": ("!=", "<", "<=", ">", ">="), "operator": "ROR", "family": "ROR"},
    {"pattern": re.compile(r"!="), "replacements": ("==", "<", "<=", ">", ">="), "operator": "ROR", "family": "ROR"},
    {"pattern": re.compile(r"<="), "replacements": ("<", ">", ">="), "operator": "ROR", "family": "ROR"},
    {"pattern": re.compile(r">="), "replacements": (">", "<", "<="), "operator": "ROR", "family": "ROR"},
    {"pattern": re.compile(r"(?<![<>=!])<(?![=<])"), "replacements": ("<=", ">", ">="), "operator": "ROR", "family": "ROR"},
    {"pattern": re.compile(r"(?<![<>=!\-])>(?!=)"), "replacements": (">=", "<", "<="), "operator": "ROR", "family": "ROR"},
    {"pattern": re.compile(r"(?<![+\-*/%])\+(?![+=])"), "replacements": ("-", "*"), "operator": "AOR", "family": "AOR"},
    {"pattern": re.compile(r"(?<![-=])-(?![>=-])"), "replacements": ("+",), "operator": "AOR", "family": "AOR"},
    {"pattern": re.compile(r"(?<![*/])\*(?![=])"), "replacements": ("+", "-", "/"), "operator": "AOR", "family": "AOR"},
    {"pattern": re.compile(r"(?<![/])/(?![/=*])"), "replacements": ("*", "%"), "operator": "AOR", "family": "AOR"},
    {"pattern": re.compile(r"(?<![%])%(?!=)"), "replacements": ("/", "*"), "operator": "AOR", "family": "AOR"},
    {"pattern": re.compile(r"&&"), "replacements": ("||",), "operator": "COR_LOR", "family": "COR_LOR"},
    {"pattern": re.compile(r"\|\|"), "replacements": ("&&",), "operator": "COR_LOR", "family": "COR_LOR"},
    # BOUNDARY: source-token analogues of IO mutate_output_values scalar/bool flips.
    # Only rewrite *literals* (0/1/-1/true/false), never type names like Vec<bool>.
    {"pattern": re.compile(r"\b0\b"), "replacements": ("1", "-1"), "operator": "BOUNDARY", "family": "BOUNDARY"},
    {"pattern": re.compile(r"\b1\b"), "replacements": ("0", "2", "-1"), "operator": "BOUNDARY", "family": "BOUNDARY"},
    {"pattern": re.compile(r"(?<![A-Za-z0-9_])-1\b"), "replacements": ("0", "1"), "operator": "BOUNDARY", "family": "BOUNDARY"},
    {"pattern": re.compile(r"\btrue\b"), "replacements": ("false",), "operator": "BOUNDARY", "family": "BOUNDARY"},
    {"pattern": re.compile(r"\bfalse\b"), "replacements": ("true",), "operator": "BOUNDARY", "family": "BOUNDARY"},
)


GENERIC_TYPE_NAMES = {
    "Array",
    "Box",
    "Map",
    "Option",
    "Result",
    "Seq",
    "Set",
    "Vec",
}


def simple_mutation_kill_rate(
    *,
    original_success: Optional[bool],
    original_verification: dict,
    mutants_total: int,
    killed_mutants: int = 0,
    survived_mutants: int = 0,
    unknown_mutants: int = 0,
    invalid_mutants: int = 0,
    max_mutants: Optional[int] = None,
    examples: Sequence[dict] = (),
) -> dict:
    if original_success is not True:
        return {
            "status": "skipped",
            "score": None,
            "reason": "original_file_does_not_verify",
            "original_verification": dict(original_verification),
            "mutants_total": 0,
            "valid_mutants": 0,
            "scored_mutants": 0,
            "killed_mutants": 0,
            "survived_mutants": 0,
            "unknown_mutants": 0,
            "invalid_mutants": 0,
        }
    if mutants_total == 0:
        return {
            "status": "no_mutants",
            "score": None,
            "original_verification": dict(original_verification),
            "mutants_total": 0,
            "valid_mutants": 0,
            "scored_mutants": 0,
            "killed_mutants": 0,
            "survived_mutants": 0,
            "unknown_mutants": 0,
            "invalid_mutants": 0,
        }

    valid_mutants = killed_mutants + survived_mutants
    scored_mutants = valid_mutants + unknown_mutants
    kill_rate = killed_mutants / scored_mutants if scored_mutants else None
    result = {
        "status": "no_valid_mutants" if scored_mutants == 0 else "partial" if (unknown_mutants or invalid_mutants) else "ok",
        "score": kill_rate,
        "mutation_kill_rate": kill_rate,
        "original_verification": dict(original_verification),
        "mutants_total": mutants_total,
        "valid_mutants": valid_mutants,
        "scored_mutants": scored_mutants,
        "killed_mutants": killed_mutants,
        "survived_mutants": survived_mutants,
        "unknown_mutants": unknown_mutants,
        "invalid_mutants": invalid_mutants,
        "examples": list(examples),
    }
    if max_mutants is not None:
        result["max_mutants"] = max_mutants
    return result


def _brace_delta(line: str) -> int:
    clean = strip_comments(line)
    return clean.count("{") - clean.count("}")


def _code_part_before_comment(line: str) -> str:
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


def _identifier_before(text: str, pos: int) -> str:
    i = pos - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    end = i + 1
    while i >= 0 and (text[i].isalnum() or text[i] == "_"):
        i -= 1
    return text[i + 1 : end]


def _looks_like_generic_type_open(text: str, pos: int) -> bool:
    if pos > 0 and text[pos - 1] == ":":
        return True
    name = _identifier_before(text, pos)
    if not name:
        return False
    if name in GENERIC_TYPE_NAMES:
        return True
    return name[0].isupper()


def _generic_angle_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    stack: list[int] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "<":
            if stack or _looks_like_generic_type_open(text, i):
                stack.append(i)
        elif ch == ">" and stack:
            start = stack.pop()
            spans.append((start, i + 1))
        i += 1
    return spans


def _position_in_spans(pos: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


def _should_skip_operator_match(original: str, start: int, generic_angle_spans: Sequence[tuple[int, int]]) -> bool:
    return original in {"<", ">"} and _position_in_spans(start, generic_angle_spans)


def mutation_skip_lines(lines: Sequence[str]) -> set[int]:
    skip: set[int] = set()
    exec_depth = 0
    pending_exec_fn = False
    spec_or_proof_fn_depth = 0
    pending_spec_or_proof_fn = False
    loop_contract_pending = False
    assert_by_depth = 0
    proof_block_depth = 0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        code_part = _code_part_before_comment(line)
        ends_signature = ";" in code_part and "{" not in code_part

        if spec_or_proof_fn_depth > 0:
            skip.add(idx)
            spec_or_proof_fn_depth += _brace_delta(line)
            if spec_or_proof_fn_depth <= 0:
                spec_or_proof_fn_depth = 0
            continue

        if pending_spec_or_proof_fn:
            skip.add(idx)
            if "{" in line:
                spec_or_proof_fn_depth = max(_brace_delta(line), 0)
                pending_spec_or_proof_fn = False
            elif ends_signature:
                pending_spec_or_proof_fn = False
            continue

        if pending_exec_fn:
            skip.add(idx)
            if "{" in line:
                exec_depth = max(_brace_delta(line), 0)
                pending_exec_fn = False
            elif ends_signature:
                pending_exec_fn = False
            continue

        if exec_depth == 0:
            skip.add(idx)
            if re.search(r"\b(spec(?:\s*\(\s*checked\s*\))?|proof)\s+fn\b", line):
                delta = _brace_delta(line)
                if "{" in line:
                    spec_or_proof_fn_depth = max(delta, 0)
                elif not ends_signature:
                    pending_spec_or_proof_fn = True
            elif re.search(r"\bfn\s+[A-Za-z_][A-Za-z0-9_]*", line):
                if "{" in line:
                    exec_depth = max(_brace_delta(line), 0)
                elif not ends_signature:
                    pending_exec_fn = True
            continue

        if assert_by_depth > 0:
            skip.add(idx)
            assert_by_depth += _brace_delta(line)
            if assert_by_depth <= 0:
                assert_by_depth = 0
            exec_depth += _brace_delta(line)
            if exec_depth <= 0:
                exec_depth = 0
            continue

        if proof_block_depth > 0:
            skip.add(idx)
            proof_block_depth += _brace_delta(line)
            if proof_block_depth <= 0:
                proof_block_depth = 0
            exec_depth += _brace_delta(line)
            if exec_depth <= 0:
                exec_depth = 0
            continue

        if loop_contract_pending:
            skip.add(idx)
            if "{" in line:
                loop_contract_pending = False
            exec_depth += _brace_delta(line)
            if exec_depth <= 0:
                exec_depth = 0
            continue

        should_skip = False
        if (
            not stripped
            or stripped in {"{", "}", "};"}
            or stripped.startswith("#")
            or stripped.startswith("use ")
            or stripped.startswith("verus!")
            or re.search(
                r"\b(requires|ensures|default_ensures|returns|recommends|opens_invariants|no_unwind|invariant|invariant_except_break|decreases)\b",
                line,
            )
        ):
            should_skip = True

        if re.search(r"\b(invariant|invariant_except_break|decreases)\b", line):
            loop_contract_pending = "{" not in line

        if re.search(r"\bproof\s*\{", line):
            proof_block_depth = max(_brace_delta(line), 0)
            should_skip = True

        if "assert" in line or "assume" in line:
            should_skip = True
            if "by" in line and "{" in line:
                assert_by_depth = max(_brace_delta(line), 0)

        if should_skip:
            skip.add(idx)

        exec_depth += _brace_delta(line)
        if exec_depth <= 0:
            exec_depth = 0

    return skip


def _implementation_body_lines_for_path(path: str) -> Optional[set[int]]:
    try:
        blocks = function_blocks_for_path(path)
    except Exception:
        return None
    if not blocks:
        return None

    body_lines: set[int] = set()
    for block in blocks:
        try:
            start_line = int(block.get("body_start_line") or 0) + 1
        except (TypeError, ValueError):
            continue
        body_text = str(block.get("body") or "")
        end_line = start_line + len(body_text.splitlines())
        for line_number in range(start_line, end_line + 1):
            body_lines.add(line_number - 1)
    return body_lines


def _mutation_skip_lines_with_body_scope(lines: Sequence[str], body_lines: set[int]) -> set[int]:
    skip: set[int] = set(range(len(lines))) - set(body_lines)
    loop_contract_pending = False
    assert_by_depth = 0
    proof_block_depth = 0

    for idx, line in enumerate(lines):
        if idx not in body_lines:
            continue
        stripped = line.strip()

        if assert_by_depth > 0:
            skip.add(idx)
            assert_by_depth += _brace_delta(line)
            if assert_by_depth <= 0:
                assert_by_depth = 0
            continue

        if proof_block_depth > 0:
            skip.add(idx)
            proof_block_depth += _brace_delta(line)
            if proof_block_depth <= 0:
                proof_block_depth = 0
            continue

        if loop_contract_pending:
            skip.add(idx)
            if "{" in line:
                loop_contract_pending = False
            continue

        should_skip = False
        if (
            not stripped
            or stripped in {"{", "}", "};"}
            or stripped.startswith("#")
            or stripped.startswith("use ")
            or stripped.startswith("verus!")
            or re.search(
                r"\b(requires|ensures|default_ensures|returns|recommends|opens_invariants|no_unwind|invariant|invariant_except_break|decreases)\b",
                line,
            )
        ):
            should_skip = True

        if re.search(r"\b(invariant|invariant_except_break|decreases)\b", line):
            loop_contract_pending = "{" not in line

        if re.search(r"\bproof\s*\{", line):
            proof_block_depth = max(_brace_delta(line), 0)
            should_skip = True

        if "assert" in line or "assume" in line:
            should_skip = True
            if "by" in line and "{" in line:
                assert_by_depth = max(_brace_delta(line), 0)

        if should_skip:
            skip.add(idx)

    return skip


def _mutant_sort_key(mutant: dict) -> str:
    stable_key = mutant.get("stable_key")
    if stable_key:
        return str(stable_key)
    seed = "|".join(
        str(mutant.get(key, ""))
        for key in ("family", "operator", "line", "column", "original", "replacement")
    )
    return hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()


def _mutation_executable_line_ordinals(lines: Sequence[str], skip_lines: set[int]) -> dict[int, int]:
    ordinals: dict[int, int] = {}
    ordinal = 0
    for idx, line in enumerate(lines):
        if idx in skip_lines:
            continue
        if not _code_part_before_comment(line).strip():
            continue
        ordinal += 1
        ordinals[idx] = ordinal
    return ordinals


def _mutation_stable_key(
    *,
    family: str,
    operator: str,
    executable_line_ordinal: Optional[int],
    column: int,
    original: str,
    replacement: str,
) -> str:
    seed = (
        f"{family}:{operator}:exec_line={executable_line_ordinal}:"
        f"column={column}:{original}->{replacement}"
    )
    return hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()[:24]


def _annotate_mutation_stable_keys(
    mutants: Sequence[dict],
    lines: Sequence[str],
    skip_lines: set[int],
) -> list[dict]:
    ordinals = _mutation_executable_line_ordinals(lines, skip_lines)
    annotated: list[dict] = []
    for mutant in mutants:
        item = dict(mutant)
        try:
            line_idx = int(item.get("line", 0)) - 1
        except (TypeError, ValueError):
            line_idx = -1
        executable_line_ordinal = ordinals.get(line_idx)
        item["executable_line_ordinal"] = executable_line_ordinal
        if 0 <= line_idx < len(lines):
            item["source_code_line"] = _code_part_before_comment(lines[line_idx]).rstrip()
        item["stable_key"] = _mutation_stable_key(
            family=str(item.get("family") or "OTHER"),
            operator=str(item.get("operator") or ""),
            executable_line_ordinal=executable_line_ordinal,
            column=int(item.get("column") or 0),
            original=str(item.get("original") or ""),
            replacement=str(item.get("replacement") or ""),
        )
        annotated.append(item)
    return annotated


def _mutation_token_name(token: str) -> str:
    return {
        "==": "eq",
        "!=": "neq",
        "<": "lt",
        "<=": "lte",
        ">": "gt",
        ">=": "gte",
        "+": "plus",
        "-": "minus",
        "*": "mul",
        "/": "div",
        "%": "mod",
        "&&": "and",
        "||": "or",
        "true": "true",
        "false": "false",
        "0": "zero",
        "1": "one",
        "2": "two",
        "-1": "minus_one",
    }.get(str(token), re.sub(r"[^A-Za-z0-9]+", "_", str(token)).strip("_") or "empty")


def _mutation_description(original: str, replacement: str) -> str:
    return f"{_mutation_token_name(original)}_to_{_mutation_token_name(replacement)}"


def _mutants_by_family(mutants: Sequence[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for mutant in mutants:
        grouped[str(mutant.get("family") or "OTHER")].append(mutant)
    return grouped


def sample_mutants_by_family(mutants: Sequence[dict], max_mutants: int) -> list[dict]:
    grouped = _mutants_by_family(mutants)
    selected: list[dict] = []
    selected_ids: set[str] = set()
    budgets = dict(MUTATION_FAMILY_BUDGETS)
    if max_mutants != sum(budgets.values()):
        scale = max_mutants / sum(budgets.values())
        budgets = {family: max(1, int(round(count * scale))) for family, count in budgets.items()}

    sorted_candidates = {
        family: sorted(grouped[family], key=_mutant_sort_key)
        for family in sorted(grouped)
    }
    offsets: dict[str, int] = {family: 0 for family in sorted_candidates}

    for family, candidates in sorted_candidates.items():
        budget = budgets.get(family, 1)
        for mutant in candidates[:budget]:
            offsets[family] += 1
            key = str(mutant.get("id"))
            if key not in selected_ids:
                selected.append(mutant)
                selected_ids.add(key)

    if len(selected) < max_mutants:
        progress = True
        while len(selected) < max_mutants and progress:
            progress = False
            for family in sorted(sorted_candidates):
                candidates = sorted_candidates[family]
                while offsets[family] < len(candidates):
                    mutant = candidates[offsets[family]]
                    offsets[family] += 1
                    key = str(mutant.get("id"))
                    if key in selected_ids:
                        continue
                    selected.append(mutant)
                    selected_ids.add(key)
                    progress = True
                    break
                if len(selected) >= max_mutants:
                    break
    return selected[:max_mutants]


def _line_mutant(
    *,
    lines: Sequence[str],
    line_idx: int,
    start: int,
    end: int,
    replacement: str,
    original: str,
    family: str,
    operator: str,
    description: str,
) -> Optional[dict]:
    line = lines[line_idx]
    mutated_line = line[:start] + replacement + line[end:]
    if mutated_line == line:
        return None
    mutated_lines = list(lines)
    mutated_lines[line_idx] = mutated_line
    text = "".join(mutated_lines)
    mutant_id = hashlib.sha256(
        f"{family}:{operator}:{line_idx + 1}:{start}:{original}->{replacement}".encode("utf-8", errors="replace")
    ).hexdigest()[:16]
    return {
        "id": mutant_id,
        "family": family,
        "operator": operator,
        "description": description,
        "line": line_idx + 1,
        "column": start + 1,
        "original": original,
        "replacement": replacement,
        "text": text,
        "confidence": "normal",
    }


def _function_param_type_groups(path: str) -> dict[str, list[str]]:
    """Group exec-fn parameters by normalized type for SVR.

    Uses the same normalize_type_key as IO type helpers so `Vec<bool>` never
    collapses into `bool`, and `&T` matches `T` for grouping.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for function in extract_functions(path):
        if function.mode != "exec" or function.name == "main":
            continue
        for param in function_parameters_from_header(function.header):
            name = param.get("name")
            type_text = normalize_type_key(str(param.get("type") or ""))
            if name and type_text:
                groups[type_text].append(str(name))
    return {key: values for key, values in groups.items() if len(values) >= 2}


def _generate_svr_mutants(path: str, lines: Sequence[str], skip_lines: set[int]) -> list[dict]:
    mutants: list[dict] = []
    try:
        blocks = function_blocks_for_path(path)
    except Exception:
        blocks = []
    if not blocks:
        groups = _function_param_type_groups(path)
        blocks = [
            {
                "body_start_line": 0,
                "body": "\n".join(lines),
                "parameters": [
                    {"name": name, "type": type_text}
                    for type_text, names in groups.items()
                    for name in names
                ],
            }
        ]

    for block in blocks:
        groups: dict[str, list[str]] = defaultdict(list)
        for param in block.get("parameters") or []:
            name = param.get("name")
            type_text = re.sub(r"\s+", "", str(param.get("type") or ""))
            if name and type_text:
                groups[type_text].append(str(name))
        groups = {key: values for key, values in groups.items() if len(values) >= 2}
        if not groups:
            continue
        start_line = int(block.get("body_start_line") or 0) + 1
        end_line = start_line + len(str(block.get("body") or "").splitlines())
        for line_idx, line in enumerate(lines):
            line_number = line_idx + 1
            if line_idx in skip_lines or not (start_line <= line_number <= end_line):
                continue
            code_part = _code_part_before_comment(line)
            for names in groups.values():
                for source in names:
                    for target in names:
                        if source == target:
                            continue
                        pattern = re.compile(rf"\b{re.escape(source)}\b")
                        for match in pattern.finditer(code_part):
                            mutant = _line_mutant(
                                lines=lines,
                                line_idx=line_idx,
                                start=match.start(),
                                end=match.end(),
                                replacement=target,
                                original=source,
                                family="SVR",
                                operator="SVR",
                                description=f"svr_{source}_to_{target}",
                            )
                            if mutant is not None:
                                mutants.append(mutant)
    return mutants


def _generate_uoi_uod_mutants(
    lines: Sequence[str],
    skip_lines: set[int],
    allowed_lines: Optional[set[int]] = None,
) -> list[dict]:
    mutants: list[dict] = []
    condition_pattern = re.compile(r"\b(if|while)\s+(!?)\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*\(\))?)")
    for line_idx, line in enumerate(lines):
        if line_idx in skip_lines or (allowed_lines is not None and line_idx not in allowed_lines):
            continue
        code_part = _code_part_before_comment(line)
        for match in condition_pattern.finditer(code_part):
            keyword, bang, expr = match.groups()
            start = match.start(2) if bang else match.start(3)
            end = match.end(2) if bang else match.start(3)
            replacement = "" if bang else "!"
            original = "!" if bang else ""
            mutant = _line_mutant(
                lines=lines,
                line_idx=line_idx,
                start=start,
                end=end,
                replacement=replacement,
                original=original or expr,
                family="UOI_UOD",
                operator="UOD" if bang else "UOI",
                description=f"{'delete' if bang else 'insert'}_logical_not_in_{keyword}_condition",
            )
            if mutant is not None:
                mutants.append(mutant)
    return mutants


def generate_all_implementation_mutants(path: str) -> list[dict]:
    text = read_text(path)
    lines = text.splitlines(keepends=True)
    body_lines = _implementation_body_lines_for_path(path)
    skip_lines = (
        _mutation_skip_lines_with_body_scope(lines, body_lines)
        if body_lines is not None
        else mutation_skip_lines(lines)
    )
    mutants: list[dict] = []

    for line_idx, line in enumerate(lines):
        if line_idx in skip_lines or (body_lines is not None and line_idx not in body_lines):
            continue
        code_part = _code_part_before_comment(line)
        for config in MUTATION_PATTERNS:
            pattern = config["pattern"]
            for match in pattern.finditer(code_part):
                start, end = match.span()
                if start == end:
                    continue
                original = match.group(0)
                if _should_skip_operator_match(original, start, _generic_angle_spans(code_part)):
                    continue
                for replacement in config["replacements"]:
                    if replacement == original:
                        continue
                    mutant = _line_mutant(
                        lines=lines,
                        line_idx=line_idx,
                        start=start,
                        end=end,
                        replacement=str(replacement),
                        original=original,
                        family=str(config["family"]),
                        operator=str(config["operator"]),
                        description=_mutation_description(original, str(replacement)),
                    )
                    if mutant is not None:
                        mutants.append(mutant)

    mutants.extend(_generate_uoi_uod_mutants(lines, skip_lines, body_lines))
    mutants.extend(_generate_svr_mutants(path, lines, skip_lines))

    deduped: list[dict] = []
    seen: set[str] = set()
    for mutant in mutants:
        key = str(mutant.get("id"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(mutant)
    return _annotate_mutation_stable_keys(deduped, lines, skip_lines)


def generate_simple_mutants(path: str, max_mutants: int = DEFAULT_MAX_MUTANTS) -> list[dict]:
    return sample_mutants_by_family(generate_all_implementation_mutants(path), max_mutants)


def mutation_outcome_from_runs(run: VerusRun, frontend_run: Optional[VerusRun] = None) -> str:
    """Classify a mutant using a full run plus an optional frontend rerun."""
    if run.success is True:
        return "survived"
    outcome = verus_run_to_dict(run).get("outcome_status")
    if outcome in {"timeout", "tool_error", "resource_exhausted"}:
        return "unknown"
    if frontend_run is None:
        return "unknown"
    frontend_outcome = verus_run_to_dict(frontend_run).get("outcome_status")
    if frontend_run.success is True:
        return "killed"
    if frontend_outcome in {"timeout", "tool_error", "resource_exhausted"}:
        return "unknown"
    return "invalid_mutant"


def mutation_counter_dict(counter: Counter) -> dict:
    families = sorted(set(MUTATION_FAMILY_BUDGETS) | set(counter))
    return {family: int(counter.get(family, 0)) for family in families}


def mutation_kill_rate_for_path(path: str) -> dict:
    original = run_verus(path, no_verify=False, timeout_seconds=DEFAULT_VERUS_TIMEOUT_SECONDS)
    original_dict = verus_run_to_dict(original)
    if original.success is not True:
        return simple_mutation_kill_rate(
            original_success=original.success,
            original_verification=original_dict,
            mutants_total=0,
        )

    mutants = generate_simple_mutants(path, DEFAULT_MAX_MUTANTS)
    if not mutants:
        return simple_mutation_kill_rate(
            original_success=original.success,
            original_verification=original_dict,
            mutants_total=0,
        )

    killed = survived = unknown = invalid = 0
    killed_by_family: Counter = Counter()
    survived_by_family: Counter = Counter()
    unknown_by_family: Counter = Counter()
    invalid_by_family: Counter = Counter()
    examples: list[dict] = []
    source_name = Path(path).name
    with tempfile.TemporaryDirectory(prefix="spec_metrics_mutants_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for idx, mutant in enumerate(mutants):
            mutant_path = tmpdir_path / f"mutant_{idx}_{source_name}"
            mutant_path.write_text(mutant["text"], encoding="utf-8")
            run = run_verus(
                str(mutant_path),
                no_verify=False,
                timeout_seconds=DEFAULT_MUTANT_TIMEOUT_SECONDS,
            )
            run_dict = verus_run_to_dict(run)
            frontend_run: Optional[VerusRun] = None
            if run.success is not True and run_dict.get("outcome_status") not in {
                "timeout",
                "tool_error",
                "resource_exhausted",
            }:
                frontend_run = run_verus(
                    str(mutant_path),
                    no_verify=True,
                    timeout_seconds=DEFAULT_MUTANT_TIMEOUT_SECONDS,
                )
            outcome = mutation_outcome_from_runs(run, frontend_run)
            if outcome == "survived":
                survived += 1
            elif outcome == "invalid_mutant":
                invalid += 1
            elif outcome == "killed":
                killed += 1
            else:
                unknown += 1
            family = str(mutant.get("family") or "OTHER")
            if outcome == "killed":
                killed_by_family[family] += 1
            elif outcome == "survived":
                survived_by_family[family] += 1
            elif outcome == "invalid_mutant":
                invalid_by_family[family] += 1
            else:
                unknown_by_family[family] += 1
            if len(examples) < 20:
                frontend_dict = verus_run_to_dict(frontend_run) if frontend_run is not None else None
                examples.append(
                    {
                        "id": mutant.get("id"),
                        "family": mutant.get("family"),
                        "operator": mutant.get("operator"),
                        "line": mutant["line"],
                        "column": mutant.get("column"),
                        "description": mutant["description"],
                        "original": mutant["original"],
                        "replacement": mutant["replacement"],
                        "outcome": outcome,
                        "verification_success": run.success,
                        "verus_status": run.status,
                        "verus_outcome_status": run_dict.get("outcome_status"),
                        "frontend_status": frontend_dict.get("status") if frontend_dict else None,
                        "frontend_outcome_status": frontend_dict.get("outcome_status") if frontend_dict else None,
                        "mutated_code": mutant["text"],
                    }
                )

    candidate_mutants = generate_all_implementation_mutants(path)
    mutants_by_family = Counter(str(mutant.get("family") or "OTHER") for mutant in mutants)
    candidate_mutants_by_family = Counter(str(mutant.get("family") or "OTHER") for mutant in candidate_mutants)
    result = simple_mutation_kill_rate(
        original_success=original.success,
        original_verification=original_dict,
        mutants_total=len(mutants),
        killed_mutants=killed,
        survived_mutants=survived,
        unknown_mutants=unknown,
        invalid_mutants=invalid,
        max_mutants=DEFAULT_MAX_MUTANTS,
        examples=examples,
    )
    result.update(
        {
            "method": "stratified_verus_backed_implementation_mutation",
            "mutation_framework_version": "implementation_stratified_v3",
            "candidate_mutants_total": len(candidate_mutants),
            "sampled_from_candidates": len(mutants),
            "mutants_by_family": mutation_counter_dict(mutants_by_family),
            "candidate_mutants_by_family": mutation_counter_dict(candidate_mutants_by_family),
            "killed_by_family": mutation_counter_dict(killed_by_family),
            "survived_by_family": mutation_counter_dict(survived_by_family),
            "unknown_by_family": mutation_counter_dict(unknown_by_family),
            "invalid_by_family": mutation_counter_dict(invalid_by_family),
            "families": sorted(MUTATION_FAMILY_BUDGETS),
            "operator_families": {
                "ROR": "relational operator replacement",
                "AOR": "arithmetic operator replacement",
                "COR_LOR": "conditional/logical operator replacement",
                "UOI_UOD": "unary logical operator insertion/deletion",
                "SVR": "same-type function-parameter variable replacement",
                "BOUNDARY": "boundary constants and boolean replacement",
            },
        }
    )
    result["note"] = (
        "Implementation mutation kill rate with stratified AOR/ROR/COR/UOI/SVR/boundary operators. "
        "A failed full run is counted as killed only after --no-verify confirms that the mutant is frontend-valid. "
        "score = killed / (killed + survived + unknown); invalid mutants are excluded, while "
        "unknown/timeouts remain in the denominator as not demonstrably killed."
    )
    return result


def generated_self_spec_robustness(
    generated_rs_path: str,
    ground_rs_path: Optional[str] = None,
) -> dict:
    generated = mutation_kill_rate_for_path(generated_rs_path)
    generated.update(
        {
            "method": "generated_self_spec_robustness_mutation",
            "reference_filter": {"status": "not_used"},
            "note": (
                "Generated-spec self robustness: mutates executable implementation lines in the generated file "
                "and measures whether the generated specification rejects those implementation perturbations. "
                "This metric does not use a GT/reference oracle filter."
            ),
        }
    )
    ground = {
        "status": "not_applicable",
        "score": None,
        "reason": "reference_not_used",
    }
    if ground_rs_path is not None:
        ground["path"] = ground_rs_path
    return {"generated": generated, "ground": ground, "delta": None}


__all__ = [
    "DEFAULT_MAX_MUTANTS",
    "DEFAULT_MUTANT_TIMEOUT_SECONDS",
    "MUTATION_FAMILY_BUDGETS",
    "generate_all_implementation_mutants",
    "generate_simple_mutants",
    "generated_self_spec_robustness",
    "mutation_kill_rate_for_path",
    "sample_mutants_by_family",
    "simple_mutation_kill_rate",
]
