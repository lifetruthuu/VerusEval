from __future__ import annotations

import json
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from metrics_rebuild.share.clauses import (
    CLAUSE_BOUNDARY_KINDS,
    EXPRESSION_CLAUSE_KINDS,
    STOPWORDS,
    _is_probable_clause_body_open,
    _span_index_after_offset,
    _token_depth_update,
    _top_level,
    extract_clauses,
    find_matching_brace,
    find_matching_paren,
    split_top_level_comma_spans,
)
from metrics_rebuild.share.text import (
    normalize_expr,
    normalized_tokens,
    read_text,
    strip_comments,
    token_spans,
)
from metrics_rebuild.share.verus_runner import (
    DEFAULT_VERUS_TIMEOUT_SECONDS,
    VerusRun,
    run_verus,
    verus_run_to_dict,
    verus_runtime_fingerprint,
)

REDUNDANCY_DELETE_AND_REVERIFY_KINDS = {
    "invariant",
    "invariant_except_break",
    "assert",
    "decreases",
}
NEAR_DUPLICATE_THRESHOLD = 0.85
DEFAULT_HOUDINI_TIMEOUT_SECONDS = 12
DEFAULT_HOUDINI_MAX_ATTEMPTS = 60

_HOUDINI_REDUNDANCY_CACHE: dict[tuple, dict] = {}


@dataclass(frozen=True)
class ClauseRemovalCandidate:
    id: int
    kind: str
    text: str
    normalized: str
    line: int
    removal_start: int
    removal_end: int


def _map_current_candidates_to_original(
    current: Sequence[ClauseRemovalCandidate],
    original: Sequence[ClauseRemovalCandidate],
    removed_ids: set[int],
) -> tuple[list[tuple[ClauseRemovalCandidate, ClauseRemovalCandidate]], bool]:
    """Match surviving occurrences in order without collapsing duplicates."""
    remaining = [candidate for candidate in original if candidate.id not in removed_ids]
    pairs: list[tuple[ClauseRemovalCandidate, ClauseRemovalCandidate]] = []
    cursor = 0
    for candidate in current:
        matched_index = next(
            (
                index
                for index in range(cursor, len(remaining))
                if remaining[index].kind == candidate.kind
                and remaining[index].normalized == candidate.normalized
            ),
            None,
        )
        if matched_index is None:
            return pairs, False
        pairs.append((candidate, remaining[matched_index]))
        cursor = matched_index + 1
    return pairs, len(pairs) == len(current)


def _skip_whitespace_offset(text: str, offset: int) -> int:
    while offset < len(text) and text[offset].isspace():
        offset += 1
    return offset


def _assert_tail_end_offset(clean: str, spans: Sequence, idx: int, fallback_offset: int) -> int:
    if idx >= len(spans) or spans[idx].text != "by":
        cursor = _skip_whitespace_offset(clean, fallback_offset)
        if cursor < len(clean) and clean[cursor] == ";":
            return cursor + 1
        return fallback_offset

    cursor = _skip_whitespace_offset(clean, spans[idx].end)
    if cursor >= len(clean):
        return spans[idx].end

    if clean[cursor] == "{":
        block_close = find_matching_brace(clean, cursor)
        if block_close is None:
            return spans[idx].end
        end = _skip_whitespace_offset(clean, block_close + 1)
        if end < len(clean) and clean[end] == ";":
            return end + 1
        return block_close + 1

    if clean[cursor] == "(":
        tactic_close = find_matching_paren(clean, cursor)
        if tactic_close is None:
            return spans[idx].end
        semicolon = clean.find(";", tactic_close)
        if semicolon != -1:
            return semicolon + 1
        return tactic_close + 1

    return spans[idx].end


def _line_number_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def extract_removal_candidates_from_clean(clean: str) -> list[ClauseRemovalCandidate]:
    spans = token_spans(clean)
    candidates: list[ClauseRemovalCandidate] = []
    candidate_id = 0
    i = 0

    def add_candidate(kind: str, text: str, normalized: str, start: int, end: int, line_offset: int) -> None:
        nonlocal candidate_id
        if not normalized or start >= end:
            return
        candidates.append(
            ClauseRemovalCandidate(
                id=candidate_id,
                kind=kind,
                text=text.strip(),
                normalized=normalized,
                line=_line_number_for_offset(clean, line_offset),
                removal_start=start,
                removal_end=end,
            )
        )
        candidate_id += 1

    while i < len(spans):
        tok = spans[i].text
        if tok in EXPRESSION_CLAUSE_KINDS or tok == "no_unwind":
            keyword_start = spans[i].start
            group_start = spans[i].end
            depths = {"paren": 0, "bracket": 0, "brace": 0}
            j = i + 1
            while j < len(spans):
                next_tok = spans[j].text
                if _top_level(depths):
                    if next_tok in CLAUSE_BOUNDARY_KINDS:
                        break
                    if next_tok == ";":
                        break
                    if next_tok == "{" and _is_probable_clause_body_open(
                        clean,
                        group_start,
                        spans[j].start,
                    ):
                        break
                _token_depth_update(next_tok, depths)
                j += 1

            group_end = spans[j].start if j < len(spans) else len(clean)
            parts = split_top_level_comma_spans(clean[group_start:group_end], group_start)
            if tok == "no_unwind" and not parts:
                add_candidate("no_unwind", "true", "true", keyword_start, group_end, keyword_start)
            else:
                for part_idx, (part, part_start, part_end) in enumerate(parts):
                    normalized = normalize_expr(part)
                    if not normalized:
                        continue
                    # The span that ends the clause group must stop at the last
                    # part (or its trailing comma), not at the next keyword:
                    # eating the whitespace in between glues the previous token
                    # to the keyword (``i as intdecreases``) and the deletion
                    # fails to parse instead of being classified.
                    trailing_comma = clean.find(",", part_end, group_end) if part_idx == len(parts) - 1 else -1
                    last_end = trailing_comma + 1 if trailing_comma != -1 else part_end
                    if len(parts) == 1:
                        removal_start, removal_end = keyword_start, last_end
                    elif part_idx == 0:
                        removal_start, removal_end = part_start, parts[part_idx + 1][1]
                    elif part_idx == len(parts) - 1:
                        removal_start = part_start if trailing_comma != -1 else parts[part_idx - 1][2]
                        removal_end = last_end
                    else:
                        removal_start, removal_end = part_start, parts[part_idx + 1][1]
                    add_candidate(tok, part, normalized, removal_start, removal_end, part_start)
            i = j
            continue

        if tok == "assert":
            if i + 1 < len(spans) and spans[i + 1].text == "!":
                i += 1
                continue

            keyword_start = spans[i].start
            cursor = _skip_whitespace_offset(clean, spans[i].end)
            if cursor < len(clean) and clean[cursor] == "(":
                close = find_matching_paren(clean, cursor)
                if close is not None:
                    part = clean[cursor + 1 : close].strip()
                    normalized = normalize_expr(part)
                    after_close_idx = _span_index_after_offset(spans, close)
                    end = _assert_tail_end_offset(clean, spans, after_close_idx, close + 1)
                    add_candidate("assert", part, normalized, keyword_start, end, cursor)
                    i = _span_index_after_offset(spans, end)
                    continue

            start = spans[i].end
            depths = {"paren": 0, "bracket": 0, "brace": 0}
            j = i + 1
            while j < len(spans):
                next_tok = spans[j].text
                if _top_level(depths):
                    if next_tok in {"by", ";"}:
                        break
                    if next_tok == "{" and _is_probable_clause_body_open(
                        clean,
                        start,
                        spans[j].start,
                    ):
                        break
                _token_depth_update(next_tok, depths)
                j += 1
            end_expr = spans[j].start if j < len(spans) else len(clean)
            part = clean[start:end_expr].strip()
            normalized = normalize_expr(part)
            if j < len(spans) and spans[j].text == "by":
                removal_end = _assert_tail_end_offset(clean, spans, j, end_expr)
            elif j < len(spans) and spans[j].text == ";":
                removal_end = spans[j].end
            else:
                removal_end = end_expr
            add_candidate("assert", part, normalized, keyword_start, removal_end, start)
            i = _span_index_after_offset(spans, removal_end)
            continue

        i += 1

    return candidates


def extract_removal_candidates_from_text(text: str) -> tuple[str, list[ClauseRemovalCandidate]]:
    clean = strip_comments(text)
    return clean, extract_removal_candidates_from_clean(clean)


def duplicate_clause_rate_for_path(path: str) -> dict:
    clause_list = [clause.normalized for clause in extract_clauses(path)]
    counts = Counter(clause_list)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    duplicate_values = [
        {"clause": clause, "count": count}
        for clause, count in counts.items()
        if count > 1
    ]
    return {
        "score": duplicates / len(clause_list) if clause_list else 0.0,
        "clauses_total": len(clause_list),
        "duplicate_instances": duplicates,
        "duplicate_groups": len(duplicate_values),
        "duplicates": duplicate_values[:50],
    }


def _token_set_for_clause(clause: str) -> set[str]:
    return {
        tok
        for tok in normalized_tokens(clause)
        if tok not in {",", ";", "(", ")", "[", "]", "{", "}"} and tok not in STOPWORDS
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def near_duplicate_clause_rate_for_path(path: str) -> dict:
    clause_list = [clause.normalized for clause in extract_clauses(path)]
    token_sets = [_token_set_for_clause(clause) for clause in clause_list]
    near_duplicate_indexes: set[int] = set()
    examples: list[dict] = []
    limit = min(len(clause_list), 1000)
    for i in range(limit):
        for j in range(i + 1, limit):
            if clause_list[i] == clause_list[j]:
                continue
            similarity = _jaccard(token_sets[i], token_sets[j])
            if similarity >= NEAR_DUPLICATE_THRESHOLD:
                near_duplicate_indexes.add(i)
                near_duplicate_indexes.add(j)
                if len(examples) < 20:
                    examples.append(
                        {
                            "left": clause_list[i],
                            "right": clause_list[j],
                            "jaccard": similarity,
                        }
                    )
    return {
        "score": len(near_duplicate_indexes) / len(clause_list) if clause_list else 0.0,
        "clauses_total": len(clause_list),
        "near_duplicate_clauses": len(near_duplicate_indexes),
        "threshold": NEAR_DUPLICATE_THRESHOLD,
        "pair_scan_limit": limit,
        "examples": examples,
    }


def redundancy_rate(exact_result: dict, near_result: dict) -> dict:
    score = max(float(exact_result.get("score", 0.0)), float(near_result.get("score", 0.0)))
    return {
        "score": score,
        "exact_duplicate_rate": exact_result.get("score"),
        "near_duplicate_rate": near_result.get("score"),
        "clauses_total": exact_result.get("clauses_total", 0),
    }


def copy_jsonable(value: dict) -> dict:
    return json.loads(json.dumps(value, ensure_ascii=False))


def candidate_to_dict(
    candidate: ClauseRemovalCandidate,
    *,
    outcome: Optional[str] = None,
    run: Optional[VerusRun] = None,
    frontend_run: Optional[VerusRun] = None,
) -> dict:
    payload = {
        "id": candidate.id,
        "kind": candidate.kind,
        "line": candidate.line,
        "text": candidate.text,
        "normalized": candidate.normalized,
        "removal_start": candidate.removal_start,
        "removal_end": candidate.removal_end,
    }
    if outcome is not None:
        payload["outcome"] = outcome
    if run is not None:
        payload["verus_outcome_status"] = verus_run_to_dict(run).get("outcome_status")
        payload["verus_status"] = run.status
        payload["verification_success"] = run.success
        payload["verified"] = run.verified
        payload["errors"] = run.errors
        payload["elapsed_seconds"] = run.elapsed_seconds
    if frontend_run is not None:
        frontend = verus_run_to_dict(frontend_run)
        payload["frontend_check"] = {
            "mode": "no_verify",
            "status": frontend.get("status"),
            "outcome_status": frontend.get("outcome_status"),
            "success": frontend.get("success"),
            "elapsed_seconds": frontend.get("elapsed_seconds"),
        }
    return payload


def remove_candidate_text(text: str, candidate: ClauseRemovalCandidate) -> str:
    return text[: candidate.removal_start] + text[candidate.removal_end :]


def remove_candidates_text(
    text: str, candidates: Sequence[ClauseRemovalCandidate]
) -> str:
    """Remove several clauses from ``text`` in a single splice.

    Candidate removal spans are half-open ``[removal_start, removal_end)`` offsets
    into the *same* ``text``. Sibling parts of one clause group overlap by
    construction (see :func:`extract_removal_candidates_from_clean`), so we merge
    overlapping/adjacent intervals and splice them out in descending order —
    otherwise an earlier splice would invalidate later offsets, or a pair of
    overlapping spans would corrupt the text.

    The merged result is not guaranteed to be syntactically valid (deleting every
    part of ``invariant a, b, c`` leaves an empty ``invariant``); callers rerun
    Verus on the result and fall back on failure, so no pre-validation is done.
    """
    spans = sorted(
        {
            (candidate.removal_start, candidate.removal_end)
            for candidate in candidates
            if candidate.removal_start < candidate.removal_end
        }
    )
    if not spans:
        return text
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    result = text
    for start, end in reversed(merged):
        result = result[:start] + result[end:]
    return result


def run_verus_on_text(text: str, tmpdir: Path, filename: str, timeout_seconds: int) -> VerusRun:
    path = tmpdir / filename
    path.write_text(text, encoding="utf-8")
    return run_verus(str(path), no_verify=False, timeout_seconds=timeout_seconds)


def run_verus_frontend_on_text(
    text: str, tmpdir: Path, filename: str, timeout_seconds: int,
) -> VerusRun:
    """Run Verus frontend checks without discharging proof obligations."""
    path = tmpdir / filename
    path.write_text(text, encoding="utf-8")
    return run_verus(str(path), no_verify=True, timeout_seconds=timeout_seconds)


def _needs_frontend_confirmation(run: VerusRun) -> bool:
    """Whether a normal full-run failure needs ``--no-verify`` disambiguation."""
    outcome = verus_run_to_dict(run).get("outcome_status")
    return run.success is False and outcome in {"parse_error", "verification_failed"}


def _classify_removal_run(
    run: VerusRun,
    frontend_run: Optional[VerusRun] = None,
) -> str:
    """Classify a delete-and-reverify result as removable/necessary/unknown.

    A successful full run proves the clause removable. For a normal full-run
    failure, a successful ``--no-verify`` run establishes that parsing, typing,
    name resolution, and Verus frontend checks all passed; the remaining full
    failure is therefore a proof failure and the clause is necessary for the
    current proof. Frontend failures, timeouts, resource exhaustion, and tool
    errors remain unknown.
    """
    outcome = verus_run_to_dict(run).get("outcome_status")
    if outcome == "verified":
        return "removable"
    if (
        _needs_frontend_confirmation(run)
        and frontend_run is not None
        and frontend_run.success is True
    ):
        return "necessary"
    return "unknown"


def houdini_redundancy_for_path(
    path: str,
    *,
    max_attempts: int = DEFAULT_HOUDINI_MAX_ATTEMPTS,
    timeout_seconds: int = DEFAULT_HOUDINI_TIMEOUT_SECONDS,
) -> dict:
    source_path = Path(path)
    source_text = read_text(path)
    clean_text, all_original_candidates = extract_removal_candidates_from_text(source_text)
    original_candidates = [
        candidate
        for candidate in all_original_candidates
        if candidate.kind in REDUNDANCY_DELETE_AND_REVERIFY_KINDS
    ]
    skipped_contract_clauses = [
        candidate_to_dict(candidate)
        for candidate in all_original_candidates
        if candidate.kind not in REDUNDANCY_DELETE_AND_REVERIFY_KINDS
    ]
    candidates_total = len(original_candidates)
    all_clauses_total = len(all_original_candidates)
    attempt_budget = max(max_attempts, 3 * candidates_total + 4)
    runtime_info = verus_runtime_fingerprint()
    try:
        stat = source_path.stat()
        source_identity = (str(source_path.resolve()), stat.st_mtime_ns, stat.st_size)
    except OSError:
        source_identity = (str(source_path), 0, 0)
    cache_key = (
        *source_identity,
        json.dumps(runtime_info, sort_keys=True, ensure_ascii=False),
        timeout_seconds,
        attempt_budget,
    )
    cached = _HOUDINI_REDUNDANCY_CACHE.get(cache_key)
    if cached is not None:
        return copy_jsonable(cached)

    original_run = run_verus(path, no_verify=False, timeout_seconds=DEFAULT_VERUS_TIMEOUT_SECONDS)
    original_verification = verus_run_to_dict(original_run)

    if original_run.success is not True:
        result = {
            "status": "skipped",
            "score": None,
            "reason": "original_file_does_not_verify",
            "method": "delete_clause_and_reverify",
            "original_verification": original_verification,
            "clauses_total": candidates_total,
            "all_clauses_total": all_clauses_total,
            "skipped_contract_clauses": skipped_contract_clauses[:100],
            "removable_clause_candidates": 0,
            "necessary_clause_candidates": 0,
            "unknown_clause_candidates": 0,
            "minimal_size": candidates_total,
            "max_attempts": attempt_budget,
            "timeout_seconds": timeout_seconds,
            "verus_runtime": runtime_info,
        }
        return copy_jsonable(result)

    if candidates_total == 0:
        result = {
            "status": "no_testable_clauses",
            "score": 0.0,
            "method": "delete_clause_and_reverify",
            "original_verification": original_verification,
            "clauses_total": 0,
            "all_clauses_total": all_clauses_total,
            "skipped_contract_clauses": skipped_contract_clauses[:100],
            "removable_clause_candidates": 0,
            "necessary_clause_candidates": 0,
            "unknown_clause_candidates": 0,
            "minimal_size": 0,
            "max_attempts": attempt_budget,
            "timeout_seconds": timeout_seconds,
            "verus_runtime": runtime_info,
        }
        _HOUDINI_REDUNDANCY_CACHE[cache_key] = result
        return copy_jsonable(result)

    independent_removable: list[dict] = []
    independent_necessary = 0
    independent_unknown = 0
    independent_results: list[dict] = []
    independent_examples: list[dict] = []
    removable_candidates: list[ClauseRemovalCandidate] = []
    attempts = 0
    frontend_checks = 0
    source_name = source_path.name or "target.rs"
    with tempfile.TemporaryDirectory(prefix="spec_metrics_redundancy_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Phase 1 — independent leave-one-out over the pristine text. Doubles as the
        # per-clause diagnostic and identifies the removable set R for Phase 2.
        for idx, candidate in enumerate(original_candidates):
            if attempts >= attempt_budget:
                break
            mutated_text = remove_candidate_text(clean_text, candidate)
            run = run_verus_on_text(
                mutated_text, tmpdir_path, f"delta_{idx}_{source_name}", timeout_seconds
            )
            attempts += 1
            frontend_run = None
            if _needs_frontend_confirmation(run) and attempts < attempt_budget:
                frontend_run = run_verus_frontend_on_text(
                    mutated_text,
                    tmpdir_path,
                    f"delta_frontend_{idx}_{source_name}",
                    timeout_seconds,
                )
                attempts += 1
                frontend_checks += 1
            outcome = _classify_removal_run(run, frontend_run)
            item = candidate_to_dict(
                candidate,
                outcome=outcome,
                run=run,
                frontend_run=frontend_run,
            )
            independent_results.append(item)
            if len(independent_examples) < 20:
                independent_examples.append(item)
            if outcome == "removable":
                independent_removable.append(item)
                removable_candidates.append(candidate)
            elif outcome == "necessary":
                independent_necessary += 1
            else:
                independent_unknown += 1

        removed_clauses: list[dict] = []
        removed_ids: set[int] = set()
        current_text = clean_text
        optimistic_hit = False
        occurrence_mapping_complete = True

        # Phase 2 — optimistic joint removal of the whole independently-removable
        # set in a single reverify. If it holds, every clause in R is jointly
        # redundant (the residual program was directly verified).
        if removable_candidates and attempts < attempt_budget:
            optimistic_text = remove_candidates_text(clean_text, removable_candidates)
            opt_run = run_verus_on_text(
                optimistic_text, tmpdir_path, f"optimistic_{source_name}", timeout_seconds
            )
            attempts += 1
            if opt_run.success is True:
                removed_clauses.extend(
                    candidate_to_dict(candidate, outcome="removed", run=opt_run)
                    for candidate in removable_candidates
                )
                removed_ids.update(candidate.id for candidate in removable_candidates)
                current_text = optimistic_text
                optimistic_hit = True

        # Phase 3 — greedy minimization to a fixpoint from wherever we are. When the
        # optimistic pass fired this just confirms the fixpoint and picks up any
        # clause that became removable once R was gone; otherwise it is the full
        # sound one-at-a-time minimization. Re-parsing after each commit keeps
        # offsets valid.
        greedy_before = len(removed_clauses)
        # If Phase 1 found no removable clause, the source text is unchanged and
        # a greedy scan would only repeat the exact same full runs.
        changed = bool(removable_candidates)
        while changed and attempts < attempt_budget:
            changed = False
            _, current_candidates = extract_removal_candidates_from_text(current_text)
            current_candidates = [
                candidate
                for candidate in current_candidates
                if candidate.kind in REDUNDANCY_DELETE_AND_REVERIFY_KINDS
            ]
            if not current_candidates:
                break
            current_pairs, mapped = _map_current_candidates_to_original(
                current_candidates,
                original_candidates,
                removed_ids,
            )
            occurrence_mapping_complete = occurrence_mapping_complete and mapped
            for candidate, original_candidate in current_pairs:
                if attempts >= attempt_budget:
                    break
                mutated_text = remove_candidate_text(current_text, candidate)
                run = run_verus_on_text(
                    mutated_text,
                    tmpdir_path,
                    f"greedy_{len(removed_clauses)}_{source_name}",
                    timeout_seconds,
                )
                attempts += 1
                if _classify_removal_run(run) == "removable":
                    removed_clauses.append(
                        candidate_to_dict(original_candidate, outcome="removed", run=run)
                    )
                    removed_ids.add(original_candidate.id)
                    current_text = mutated_text
                    changed = True
                    break
        greedy_hit = len(removed_clauses) > greedy_before

    _, final_candidates = extract_removal_candidates_from_text(current_text)
    final_candidates = [
        candidate
        for candidate in final_candidates
        if candidate.kind in REDUNDANCY_DELETE_AND_REVERIFY_KINDS
    ]
    final_pairs, final_mapped = _map_current_candidates_to_original(
        final_candidates,
        original_candidates,
        removed_ids,
    )
    occurrence_mapping_complete = occurrence_mapping_complete and final_mapped
    final_original_candidates = [original for _current, original in final_pairs]

    removed_count = len(removed_clauses)
    definitive_ids = {
        item["id"]
        for item in independent_results
        if item.get("outcome") in {"removable", "necessary"}
    }
    original_by_id = {candidate.id: candidate for candidate in original_candidates}
    unknown_ids = set(original_by_id) - definitive_ids - removed_ids
    unknown_count = len(unknown_ids)
    resolved_total = candidates_total - unknown_count
    score = removed_count / candidates_total if candidates_total else 0.0
    coverage = resolved_total / candidates_total if candidates_total else 0.0
    cap_reached = attempts >= attempt_budget
    strategy = "+".join(
        part for part, hit in (("optimistic", optimistic_hit), ("greedy", greedy_hit)) if hit
    ) or "none"

    independent_attempted = len(independent_results)
    result = {
        "status": "partial" if (unknown_count or cap_reached or not occurrence_mapping_complete) else "ok",
        "score": score,
        "coverage": coverage,
        "method": "delete_clause_and_reverify",
        "strategy": strategy,
        "engine": "verus",
        "original_verification": original_verification,
        "clauses_total": candidates_total,
        "all_clauses_total": all_clauses_total,
        "candidate_clauses": [candidate_to_dict(candidate) for candidate in original_candidates[:50]],
        "skipped_contract_clauses": skipped_contract_clauses[:100],
        "removable_clause_candidates": removed_count,
        "necessary_clause_candidates": resolved_total - removed_count,
        "unknown_clause_candidates": unknown_count,
        "minimal_size": len(final_candidates),
        "remaining_clauses": [candidate_to_dict(candidate) for candidate in final_original_candidates[:50]],
        "removed_clauses": removed_clauses[:50],
        "unknown_clauses": [candidate_to_dict(original_by_id[candidate_id]) for candidate_id in sorted(unknown_ids)[:50]],
        "independent_delta_pass": {
            "attempted": independent_attempted,
            "removable": len(independent_removable),
            "necessary": independent_necessary,
            "unknown": independent_unknown,
            "score": len(independent_removable) / independent_attempted if independent_attempted else 0.0,
            "results": independent_results[:100],
            "examples": independent_examples,
        },
        "attempts": attempts,
        "frontend_checks": frontend_checks,
        "max_attempts": attempt_budget,
        "timeout_seconds": timeout_seconds,
        "verus_runtime": runtime_info,
        "note": (
            "Conservative redundancy check: proof-support clauses (assert/invariant/decreases) are "
            "deleted from a temporary file and Verus is rerun; requires/ensures are skipped because "
            "delete-only checks can falsely mark weakened specs as redundant. A failed full Verus run "
            "is followed by --no-verify: frontend success confirms a proof failure (necessary), while "
            "frontend failure or an abnormal full-run outcome remains unknown. The removable set is "
            "first removed jointly in one reverify (optimistic), then a greedy pass minimizes to a "
            "fixpoint. score = jointly-removed / all proof-support candidates; timeouts and "
            "frontend-invalid or unattempted deletions count as unknown in the denominator but not "
            "the numerator. Unknown candidates or an exhausted attempt budget produce status=partial."
        ),
    }
    if result["status"] == "ok" and unknown_count == 0:
        _HOUDINI_REDUNDANCY_CACHE[cache_key] = result
    return copy_jsonable(result)


def spec_redundancy_for_path(path: str) -> dict:
    houdini = houdini_redundancy_for_path(path)
    exact = duplicate_clause_rate_for_path(path)
    near = near_duplicate_clause_rate_for_path(path)
    houdini["exact_duplicate_rate"] = exact.get("score")
    houdini["near_duplicate_rate"] = near.get("score")
    houdini["textual_redundancy_proxy"] = redundancy_rate(exact, near)
    return houdini


__all__ = [
    "ClauseRemovalCandidate",
    "DEFAULT_HOUDINI_MAX_ATTEMPTS",
    "DEFAULT_HOUDINI_TIMEOUT_SECONDS",
    "NEAR_DUPLICATE_THRESHOLD",
    "REDUNDANCY_DELETE_AND_REVERIFY_KINDS",
    "candidate_to_dict",
    "duplicate_clause_rate_for_path",
    "extract_removal_candidates_from_text",
    "houdini_redundancy_for_path",
    "near_duplicate_clause_rate_for_path",
    "remove_candidate_text",
    "remove_candidates_text",
    "run_verus_on_text",
    "spec_redundancy_for_path",
]
