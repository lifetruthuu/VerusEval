#!/usr/bin/env python
"""Build a schema-v3 IO corpus using only machine-validated evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics_rebuild.share.contract_eval import (  # noqa: E402
    batch_verus_contract_decide_detailed,
    batch_verus_requires_decide,
    contract_evaluation,
    typed_input_payload,
    typed_output_payload,
)
from metrics_rebuild.share.functions import authoritative_target_for_path  # noqa: E402
from metrics_rebuild.share.io_cases import (  # noqa: E402
    _audit_case_key,
    _return_payload_from_benchmark,
    generate_boundary_candidate_inputs,
    generate_candidate_inputs,
    generate_requires_satisfying_candidate_inputs,
    generate_requires_violating_from_positives,
    mutate_output_values,
    source_hash,
)
from metrics_rebuild.share.io_harness import format_input_json  # noqa: E402
from metrics_rebuild.share.verus_runner import get_verus_binary, set_verus_binary  # noqa: E402
from scripts.io import generate_io_tests_llm as generate  # noqa: E402

SCHEMA_VERSION = 3
KINDS = ("positive", "negative", "invalid")


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _typed_input(target: generate.Target, raw_input: Any) -> Optional[dict]:
    if not isinstance(raw_input, dict):
        return None
    parsed = generate._parse_stored_input(raw_input)
    if not isinstance(parsed, dict):
        return None
    return typed_input_payload(target.context, parsed)


def _input_key(value: dict) -> str:
    return _canonical(value)


def _observable_context(target: generate.Target) -> dict:
    return generate._negative_context(target)


def _typed_stored_output(target: generate.Target, raw_output: Any) -> Optional[dict]:
    context = _observable_context(target)
    payload = _return_payload_from_benchmark(context, raw_output)
    return typed_output_payload(context, payload)


def _typed_runtime_output(target: generate.Target, raw_output: str) -> Optional[dict]:
    parsed = generate._parse_typed_output_to_dict(target.context, raw_output, target.func)
    if parsed is None:
        return None
    return typed_output_payload(_observable_context(target), parsed)


def _json_output(target: generate.Target, output: dict) -> Any:
    return generate.format_typed_output_for_json(target.context, output, target.func)


def _requires_verdicts(target: generate.Target, inputs: Sequence[dict]) -> dict[str, Optional[bool]]:
    if not target.context.get("requires"):
        return {_input_key(item): True for item in inputs}
    unique_inputs = _dedupe_inputs(inputs)
    batch = [
        {"inputs": item, "key": f"req_{index}"}
        for index, item in enumerate(unique_inputs)
    ]
    raw = batch_verus_requires_decide(target.context, batch)
    verdicts = {
        _input_key(item): raw.get(f"req_{index}")
        for index, item in enumerate(unique_inputs)
    }
    # The local evaluator is only a disagreement detector.  A locally proven
    # contradiction never becomes evidence by itself; instead, re-run that one
    # normalized input in its own Verus harness so a neighbouring batch error
    # cannot be attributed to the wrong candidate.
    for index, item in enumerate(unique_inputs):
        key = _input_key(item)
        batch_verdict = verdicts.get(key)
        local_verdict = contract_evaluation(
            target.context, item, strict=True,
        ).get("requires_ok")
        if (
            isinstance(batch_verdict, bool)
            and isinstance(local_verdict, bool)
            and batch_verdict != local_verdict
        ):
            singleton_key = f"req_single_{index}"
            singleton = batch_verus_requires_decide(
                target.context,
                [{"inputs": item, "key": singleton_key}],
            )
            verdicts[key] = singleton.get(singleton_key)
    return verdicts


def _runtime_verdicts(
    target: generate.Target,
    inputs: Sequence[dict],
    verus_bin: str,
) -> tuple[dict[str, dict], dict]:
    if not inputs:
        return {}, {"compile_reason": "not_needed", "raw_result_count": 0}
    with tempfile.TemporaryDirectory() as work_dir:
        harness = generate.build_verus_harness(target, list(inputs))
        results, reason = generate.verus_compile_and_run(
            harness, work_dir, verus_bin, len(inputs),
        )
    outcomes: dict[str, dict] = {}
    for index, inputs_value in enumerate(inputs):
        status, output = results.get(index, ("UNKNOWN", ""))
        outcomes[_input_key(inputs_value)] = {
            "status": status,
            "output": output,
            "reason": reason,
        }
    return outcomes, {
        "compile_reason": reason,
        "raw_result_count": len(results),
        "panic_count": sum(1 for item in outcomes.values() if item["status"] == "PANIC"),
        "timeout_count": sum(1 for item in outcomes.values() if item["status"] == "TIMEOUT"),
        "unknown_count": sum(1 for item in outcomes.values() if item["status"] == "UNKNOWN"),
    }


def _dedupe_inputs(values: Sequence[dict], limit: Optional[int] = None) -> list[dict]:
    result: list[dict] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, dict):
            continue
        key = _input_key(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if limit is not None and len(result) >= limit:
            break
    return result


def _category_entry(count: int, target: int, reason: Optional[str]) -> dict:
    if target == 0:
        return {"target": 0, "count": count, "state": "not_applicable", "reason": reason}
    if count >= target:
        return {"target": target, "count": count, "state": "complete", "reason": reason}
    return {
        "target": target,
        "count": count,
        "state": "blocked",
        "reason": reason or "strict_deterministic_exhausted",
    }


def _terminal_meta(
    source: Path,
    target: Optional[generate.Target],
    status: str,
    reason: str,
    old_function: Optional[str],
    per_kind: int,
) -> dict:
    function = target.func.name if target else None
    parameters = (
        [{"name": item.name, "type": item.rust_type} for item in target.func.params]
        if target else []
    )
    returns = list(target.context.get("returns") or []) if target else []
    return {
        "schema_version": SCHEMA_VERSION,
        "validation_complete": True,
        "source": str(source),
        "source_hash": source_hash(str(source)),
        "function": function,
        "old_function": old_function,
        "target_selection": "unavailable",
        "parameters": parameters,
        "returns": returns,
        "return_type": target.func.return_type if target else "()",
        "status": status,
        "failure_category": reason,
        "positive_count": 0,
        "negative_count": 0,
        "invalid_count": 0,
        "category_status": {
            kind: {"target": per_kind, "count": 0, "state": "blocked", "reason": reason}
            for kind in KINDS
        },
        "case_audit": {kind: [] for kind in KINDS},
    }


def _audit_task(
    source: Path,
    old_dir: Path,
    output_dir: Path,
    quarantine_path: Path,
    *,
    per_kind: int,
    candidate_budget: int,
    verus_bin: str,
    resume: bool,
) -> dict:
    existing_output_meta = _load_json(output_dir / "meta.json", {})
    if (
        resume
        and existing_output_meta.get("schema_version") == SCHEMA_VERSION
        and existing_output_meta.get("validation_complete") is True
        and existing_output_meta.get("source_hash") == source_hash(str(source))
    ):
        strict = existing_output_meta.get("strict_validation") or {}
        quarantine = _load_json(quarantine_path, {})
        quarantine_cases = quarantine.get("cases") if isinstance(quarantine, dict) else []
        return {
            "name": source.stem,
            "benchmark": source.parent.name,
            "status": existing_output_meta.get("status", "unknown"),
            "positive": int(existing_output_meta.get("positive_count", 0) or 0),
            "negative": int(existing_output_meta.get("negative_count", 0) or 0),
            "invalid": int(existing_output_meta.get("invalid_count", 0) or 0),
            "quarantined": len(quarantine_cases) if isinstance(quarantine_cases, list) else 0,
            "wrong_target": bool(strict.get("legacy_target_mismatch")),
            "resumed": True,
        }

    old_meta = _load_json(old_dir / "meta.json", {})
    old_cases = _load_json(old_dir / "test.json", [])
    if not isinstance(old_cases, list):
        old_cases = []
    old_function = str(old_meta.get("function") or "") or None
    authoritative = authoritative_target_for_path(str(source))
    target = generate.resolve_target(str(source))
    quarantined: list[dict] = []
    if authoritative is None or target is None:
        quarantined = [
            {"origin": "legacy", "reason": "authoritative_target_not_found", "case": case}
            for case in old_cases
        ]
        meta = _terminal_meta(
            source, None, "no_function", "authoritative_target_not_found", old_function, per_kind,
        )
        _write_json_atomic(output_dir / "test.json", [])
        _write_json_atomic(output_dir / "meta.json", meta)
        if quarantined:
            _write_json_atomic(quarantine_path, {
                "schema_version": 1,
                "source": str(source),
                "old_function": old_function,
                "authoritative_function": None,
                "cases": quarantined,
            })
        return {
            "name": source.stem,
            "benchmark": source.parent.name,
            "status": meta["status"],
            "positive": 0,
            "negative": 0,
            "invalid": 0,
            "quarantined": len(quarantined),
            "wrong_target": False,
            "resumed": False,
        }

    wrong_target = bool(old_function and old_function != target.func.name)
    candidate_cases = [] if wrong_target else old_cases
    if wrong_target:
        quarantined.extend(
            {"origin": "legacy", "reason": "wrong_target", "old_function": old_function, "case": case}
            for case in old_cases
        )

    # int/nat 数学类型不再整题跳过：build_verus_harness 会改写成 i128/u128 运行副本。
    unsupported_reason = (
        "unsupported_runtime_type"
        if generate._unsupported_runtime_type_issues(target.func, target.context.get("type_registry"))
        else None
    )
    if unsupported_reason:
        quarantined.extend(
            {"origin": "legacy", "reason": unsupported_reason, "case": case}
            for case in candidate_cases
        )
        meta = _terminal_meta(
            source, target, unsupported_reason, unsupported_reason, old_function, per_kind,
        )
        meta["target_selection"] = authoritative.get("selection")
        meta["marked_function"] = authoritative.get("marked_function")
        meta["strict_validation"] = {
            "legacy_target_mismatch": wrong_target,
            "legacy_case_count": len(old_cases),
            "quarantined_count": len(quarantined),
            "llm_used": False,
        }
        _write_json_atomic(output_dir / "test.json", [])
        _write_json_atomic(output_dir / "meta.json", meta)
        if quarantined:
            _write_json_atomic(quarantine_path, {
                "schema_version": 1,
                "source": str(source),
                "old_function": old_function,
                "authoritative_function": target.func.name,
                "cases": quarantined,
            })
        return {
            "name": source.stem,
            "benchmark": source.parent.name,
            "status": meta["status"],
            "positive": 0,
            "negative": 0,
            "invalid": 0,
            "quarantined": len(quarantined),
            "wrong_target": wrong_target,
            "resumed": False,
        }

    input_by_key: dict[str, dict] = {}
    old_positive_by_input: dict[str, list[dict]] = {}
    old_invalid_by_input: dict[str, list[dict]] = {}
    for case in candidate_cases:
        raw_input = case.get("input") if isinstance(case, dict) else None
        typed = _typed_input(target, raw_input)
        if typed is None:
            quarantined.append({"origin": "legacy", "reason": "input_type_mismatch", "case": case})
            continue
        key = _input_key(typed)
        input_by_key.setdefault(key, typed)
        if case.get("expected") == "INVALID_INPUT":
            old_invalid_by_input.setdefault(key, []).append(case)
        else:
            old_positive_by_input.setdefault(key, []).append(case)

    existing_inputs = list(input_by_key.values())
    requires = _requires_verdicts(target, existing_inputs)
    runnable = [item for item in existing_inputs if requires.get(_input_key(item)) is not False]
    runtime, runtime_diag = _runtime_verdicts(target, runnable, verus_bin)

    validated_positive: dict[str, dict] = {}
    invalid_cases: dict[str, dict] = {}
    invalid_audit: dict[str, dict] = {}

    def accept_runtime_positive(inputs: dict, runtime_item: dict, origin: str) -> Optional[dict]:
        if requires.get(_input_key(inputs)) is not True or runtime_item.get("status") != "OK":
            return None
        output = _typed_runtime_output(target, str(runtime_item.get("output") or ""))
        if output is None:
            return None
        return {
            "input": format_input_json(inputs, target.func),
            "expected": _json_output(target, output),
            "unexpected": [],
            "_typed_input": inputs,
            "_typed_output": output,
            "_origin": origin,
        }

    for key, cases in old_positive_by_input.items():
        inputs = input_by_key[key]
        runtime_item = runtime.get(key, {"status": "UNKNOWN", "reason": "not_run"})
        actual = accept_runtime_positive(inputs, runtime_item, "legacy_revalidated")
        for case in cases:
            expected = _typed_stored_output(target, case.get("expected"))
            if actual is None:
                quarantined.append({
                    "origin": "legacy", "reason": "positive_not_machine_validated",
                    "requires_verdict": requires.get(key), "runtime": runtime_item, "case": case,
                })
                continue
            if expected != actual["_typed_output"]:
                quarantined.append({
                    "origin": "legacy", "reason": "positive_output_mismatch",
                    "observed_output": actual["expected"], "case": case,
                })
                continue
            validated_positive.setdefault(key, actual)

    for key, cases in old_invalid_by_input.items():
        inputs = input_by_key[key]
        runtime_item = runtime.get(key, {"status": "UNKNOWN", "reason": "not_run"})
        if requires.get(key) is False:
            oracle = "reference_requires_rejection"
            engine = "verus_requires_dual_proof"
        elif runtime_item.get("status") == "PANIC":
            oracle = "reference_runtime_panic"
            engine = "verus_native_runtime"
        else:
            for case in cases:
                quarantined.append({
                    "origin": "legacy", "reason": "invalid_oracle_not_proven",
                    "requires_verdict": requires.get(key), "runtime": runtime_item, "case": case,
                })
            continue
        stored = {"input": format_input_json(inputs, target.func), "expected": "INVALID_INPUT", "unexpected": []}
        invalid_cases.setdefault(key, stored)
        invalid_audit[key] = {"oracle": oracle, "engine": engine, "runtime": runtime_item}

    positive_info = generate._positive_target_info(target, per_kind)
    invalid_info = generate._invalid_target_info(target, per_kind)
    negative_info = generate._negative_target_info(target, per_kind)

    # Deterministic supplement: every seed is rechecked by Verus and runtime.
    if len(validated_positive) < positive_info["target"] or len(invalid_cases) < invalid_info["target"]:
        positive_inputs = [item["_typed_input"] for item in validated_positive.values()]
        generated_inputs = _dedupe_inputs([
            *generate_requires_satisfying_candidate_inputs(target.context, budget=candidate_budget),
            *generate_candidate_inputs(target.context, budget=candidate_budget),
            *generate_boundary_candidate_inputs(target.context, budget=candidate_budget),
            *generate_requires_violating_from_positives(
                target.context, positive_inputs, budget=candidate_budget,
            ),
        ], limit=candidate_budget * 4)
        generated_inputs = [
            item for item in generated_inputs
            if _input_key(item) not in input_by_key
            and typed_input_payload(target.context, item) is not None
        ]
        generated_inputs = [typed_input_payload(target.context, item) for item in generated_inputs]
        generated_inputs = [item for item in generated_inputs if isinstance(item, dict)]
        generated_requires = _requires_verdicts(target, generated_inputs)
        generated_runnable = [
            item for item in generated_inputs
            if generated_requires.get(_input_key(item)) is not False
        ]
        generated_runtime, generated_runtime_diag = _runtime_verdicts(
            target, generated_runnable, verus_bin,
        )
        runtime_diag = {"legacy": runtime_diag, "supplement": generated_runtime_diag}
        for inputs in generated_inputs:
            key = _input_key(inputs)
            verdict = generated_requires.get(key)
            runtime_item = generated_runtime.get(key, {"status": "UNKNOWN", "reason": "not_run"})
            if verdict is True and runtime_item.get("status") == "OK" and len(validated_positive) < positive_info["target"]:
                output = _typed_runtime_output(target, str(runtime_item.get("output") or ""))
                if output is not None:
                    validated_positive[key] = {
                        "input": format_input_json(inputs, target.func),
                        "expected": _json_output(target, output),
                        "unexpected": [],
                        "_typed_input": inputs,
                        "_typed_output": output,
                        "_origin": "deterministic_supplement",
                    }
            if len(invalid_cases) < invalid_info["target"]:
                if verdict is False:
                    oracle = "reference_requires_rejection"
                    engine = "verus_requires_dual_proof"
                elif runtime_item.get("status") == "PANIC":
                    oracle = "reference_runtime_panic"
                    engine = "verus_native_runtime"
                else:
                    continue
                invalid_cases[key] = {
                    "input": format_input_json(inputs, target.func),
                    "expected": "INVALID_INPUT",
                    "unexpected": [],
                }
                invalid_audit[key] = {"oracle": oracle, "engine": engine, "runtime": runtime_item}

    # Validate every legacy negative, then deterministically mutate if needed.
    pending_negatives: list[dict] = []
    for key, positive in validated_positive.items():
        for old_case in old_positive_by_input.get(key, []):
            for unexpected in old_case.get("unexpected") or []:
                output = _typed_stored_output(target, unexpected)
                if output is None:
                    quarantined.append({
                        "origin": "legacy", "reason": "negative_output_type_mismatch",
                        "input": old_case.get("input"), "unexpected": unexpected,
                    })
                    continue
                pending_negatives.append({
                    "input_key": key, "inputs": positive["_typed_input"], "output": output,
                    "json_output": unexpected, "origin": "legacy_revalidated",
                })
    for key, positive in validated_positive.items():
        if len(pending_negatives) >= max(negative_info["target"] * 6, 20):
            break
        for output, label in mutate_output_values(
            _observable_context(target), positive["_typed_input"], positive["_typed_output"],
        ):
            typed = typed_output_payload(_observable_context(target), output)
            if typed is None:
                continue
            pending_negatives.append({
                "input_key": key, "inputs": positive["_typed_input"], "output": typed,
                "json_output": _json_output(target, typed), "origin": "deterministic_mutation",
                "mutation": label,
            })

    deduped_pending: list[dict] = []
    seen_negative: set[str] = set()
    for item in pending_negatives:
        key = _canonical({"input": item["inputs"], "output": item["output"]})
        if key in seen_negative or item["output"] == validated_positive[item["input_key"]]["_typed_output"]:
            continue
        seen_negative.add(key)
        deduped_pending.append(item)
    contract_batch = [
        {"inputs": item["inputs"], "output": item["output"], "key": f"neg_{index}"}
        for index, item in enumerate(deduped_pending)
    ]
    # Outputs are typed against the observable context, which exposes `&mut`
    # post-states as returns; deciding against target.context would leave those
    # tasks with an empty return list and no provable rejection.
    contract_verdicts = batch_verus_contract_decide_detailed(
        _observable_context(target), contract_batch,
    )
    negative_audit: list[dict] = []
    negative_count = 0
    for index, item in enumerate(deduped_pending):
        if item["origin"] != "legacy_revalidated" and negative_count >= negative_info["target"]:
            continue
        detail = contract_verdicts.get(f"neg_{index}") or {"accepted": None, "reason": "unknown"}
        if detail.get("accepted") is not False:
            if item["origin"] == "legacy_revalidated":
                quarantined.append({
                    "origin": "legacy", "reason": "negative_rejection_not_proven",
                    "input": validated_positive[item["input_key"]]["input"],
                    "unexpected": item["json_output"], "contract": detail,
                })
            continue
        base = validated_positive[item["input_key"]]
        unexpected = base["unexpected"]
        encoded = _canonical(item["json_output"])
        if encoded in {_canonical(value) for value in unexpected}:
            continue
        unexpected.append(item["json_output"])
        negative_count += 1
        negative_audit.append({
            "case_key": _audit_case_key({"input": base["input"], "unexpected": item["json_output"]}),
            "base_case_key": _audit_case_key({"input": base["input"], "expected": base["expected"]}),
            "state": "validated",
            "function": target.func.name,
            "oracle": "reference_contract_rejection",
            "engine": "verus_contract_dual_proof",
            "contract_verdict": False,
            "verification_reason": detail.get("reason"),
            "mutation": item.get("mutation") or "legacy_mutation",
        })

    live_cases = []
    positive_audit_list = []
    for positive in validated_positive.values():
        raw = {key: value for key, value in positive.items() if not key.startswith("_")}
        live_cases.append(raw)
        positive_audit_list.append({
            "case_key": _audit_case_key({"input": raw["input"], "expected": raw["expected"]}),
            "state": "validated",
            "function": target.func.name,
            "oracle": "reference_execution_exact",
            "engine": "verus_native_runtime+verus_requires_dual_proof",
            "requires_verdict": True,
            "runtime_status": "OK",
            "source": positive["_origin"],
        })
    live_cases.extend(invalid_cases.values())
    invalid_audit_list = []
    for key, case in invalid_cases.items():
        detail = invalid_audit[key]
        invalid_audit_list.append({
            "case_key": _audit_case_key({"input": case["input"], "kind": "invalid"}),
            "state": "validated",
            "function": target.func.name,
            "oracle": detail["oracle"],
            "engine": detail["engine"],
            "requires_verdict": False if detail["oracle"] == "reference_requires_rejection" else None,
            "runtime_status": "PANIC" if detail["oracle"] == "reference_runtime_panic" else "not_run",
        })

    positive_count = len(validated_positive)
    invalid_count = len(invalid_cases)
    category_status = {
        "positive": _category_entry(positive_count, positive_info["target"], positive_info.get("reason")),
        "negative": _category_entry(negative_count, negative_info["target"], negative_info.get("reason")),
        "invalid": {
            **_category_entry(invalid_count, invalid_info["target"], invalid_info.get("reason")),
            "mode": "requires_or_explicit_panic",
        },
    }
    meta = {
        "schema_version": SCHEMA_VERSION,
        "validation_complete": True,
        "source": str(source),
        "source_hash": source_hash(str(source)),
        "function": target.func.name,
        "old_function": old_function,
        "target_selection": authoritative.get("selection"),
        "marked_function": authoritative.get("marked_function"),
        "parameters": [{"name": item.name, "type": item.rust_type} for item in target.func.params],
        "returns": list(target.context.get("returns") or []),
        "return_type": target.func.return_type,
        "status": "ok",
        "runner": "verus_native_runtime+verus_dual_contract_proof",
        "positive_count": positive_count,
        "negative_count": negative_count,
        "invalid_count": invalid_count,
        "category_status": category_status,
        "case_audit": {
            "positive": positive_audit_list,
            "negative": negative_audit,
            "invalid": invalid_audit_list,
        },
        "strict_validation": {
            "legacy_target_mismatch": wrong_target,
            "legacy_case_count": len(old_cases),
            "quarantined_count": len(quarantined),
            "runtime": runtime_diag,
            "candidate_budget": candidate_budget,
            "llm_used": False,
        },
    }
    _write_json_atomic(output_dir / "test.json", live_cases)
    _write_json_atomic(output_dir / "meta.json", meta)
    if quarantined:
        _write_json_atomic(quarantine_path, {
            "schema_version": 1,
            "source": str(source),
            "old_function": old_function,
            "authoritative_function": target.func.name,
            "cases": quarantined,
        })
    elif quarantine_path.exists():
        quarantine_path.unlink()
    return {
        "name": source.stem,
        "benchmark": source.parent.name,
        "status": meta["status"],
        "positive": positive_count,
        "negative": negative_count,
        "invalid": invalid_count,
        "quarantined": len(quarantined),
        "wrong_target": wrong_target,
        "resumed": False,
    }


def _write_summary(output_root: Path, quarantine_root: Path, results: Sequence[dict]) -> None:
    statuses = Counter(str(item.get("status") or "unknown") for item in results)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "tasks": len(results),
        "stats": dict(sorted(statuses.items())),
        "wrong_target_tasks": sum(bool(item.get("wrong_target")) for item in results),
        "quarantined_cases": sum(int(item.get("quarantined", 0) or 0) for item in results),
        "results": sorted(results, key=lambda item: (str(item.get("benchmark")), str(item.get("name")))),
    }
    _write_json_atomic(output_root / "summary.json", summary)
    _write_json_atomic(quarantine_root.parent / "summary.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", default="data/references")
    parser.add_argument("--source-suite", default="data/io")
    parser.add_argument("--output-suite", default="runs/io/revalidated")
    parser.add_argument("--quarantine-root", default="runs/io/quarantine")
    parser.add_argument("--per-kind", type=int, default=5)
    parser.add_argument("--candidate-budget", type=int, default=24)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="only revalidate these source stems; repeat for multiple tasks",
    )
    parser.add_argument(
        "--task-file",
        default=None,
        help="file with one source stem per line (merged with --task; '#' lines ignored)",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verus-path", default=None)
    args = parser.parse_args()

    # Strict v3 never calls an LLM, even if credentials happen to be present.
    def offline_candidates(*_args, **_kwargs):
        return [], {"status": "skipped", "reason": "strict_no_llm"}

    generate.llm_candidate_inputs = offline_candidates
    generate.llm_panic_candidate_inputs = offline_candidates

    verus_bin = args.verus_path or generate._load_verus_path_from_config() or get_verus_binary()
    set_verus_binary(verus_bin)
    reference_root = Path(args.reference_root)
    source_suite = Path(args.source_suite)
    output_suite = Path(args.output_suite)
    quarantine_root = Path(args.quarantine_root)
    sources = sorted(reference_root.glob("*/*.rs"))
    if args.benchmark:
        sources = [item for item in sources if item.parent.name == args.benchmark]
    requested_tasks = set(args.task)
    if args.task_file:
        for line in Path(args.task_file).read_text(encoding="utf-8").splitlines():
            stem = line.strip()
            if stem and not stem.startswith("#"):
                requested_tasks.add(stem)
    if requested_tasks:
        sources = [item for item in sources if item.stem in requested_tasks]
    if args.limit:
        sources = sources[: args.limit]
    print(f"strict revalidation: {len(sources)} tasks, parallel={args.parallel}, verus={verus_bin}")

    def run(source: Path) -> dict:
        benchmark = source.parent.name
        return _audit_task(
            source,
            source_suite / benchmark / source.stem,
            output_suite / benchmark / source.stem,
            quarantine_root / benchmark / f"{source.stem}.json",
            per_kind=args.per_kind,
            candidate_budget=args.candidate_budget,
            verus_bin=verus_bin,
            resume=args.resume,
        )

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        futures = {executor.submit(run, source): source for source in sources}
        completed = 0
        for future in as_completed(futures):
            source = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "name": source.stem,
                    "benchmark": source.parent.name,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            completed += 1
            if completed % 10 == 0 or result.get("status") == "error":
                print(f"[{completed}/{len(sources)}] {result.get('benchmark')}/{result.get('name')}: {result.get('status')}")
    _write_summary(output_suite, quarantine_root, results)
    errors = [item for item in results if item.get("status") == "error"]
    print(json.dumps({
        "tasks": len(results),
        "errors": len(errors),
        "wrong_target_tasks": sum(bool(item.get("wrong_target")) for item in results),
        "quarantined_cases": sum(int(item.get("quarantined", 0) or 0) for item in results),
    }, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
