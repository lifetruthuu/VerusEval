"""Eight-operator summaries: confirmed changes only, fixed denominators, explicit unknowns."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from rq3_io_common import ROOT, CATEGORIES, digest, file_hash, read_csv, read_json, stem, write_csv, write_json
from rq3_mutations import OPERATORS


def classify_metric(node, category):
    details = node.get("details") or []
    semantic, unproved, expression = [], [], []
    for case in details:
        ev = case.get("evaluation") or {}
        if case.get("success") is False:
            if category != "invalid" or ev.get("requires_accepted") is True:
                semantic.append(case["id"])
            else:
                unproved.append(case["id"])
        reason = str(ev.get("reason") or "")
        detail = str(ev.get("reason_detail") or "")
        if reason == "contract_expression_undefined" or "recommendation not met" in detail.lower():
            expression.append(case["id"])
    all_passed = bool(details) and all(c.get("success") is True for c in details)
    state = ("behavior_detected" if semantic else "diagnostic_only" if unproved or expression
             else "missed" if all_passed else "unresolved")
    return {"state": state, "semantic_cases": semantic, "unproved_failure_cases": unproved,
            "expression_diagnostic_cases": expression, "all_passed": all_passed,
            "unresolved_cases": sum(c.get("success") is None for c in details),
            "case_count": len(details),
            "accepted_pre_but_panic": sum((c.get("evaluation") or {}).get("requires_accepted") is True
                                          and (c.get("evaluation") or {}).get("runtime_status") == "PANIC" for c in details)}


def choose_confirmed(rows, confirmations):
    chosen = {}
    for row in sorted(rows, key=lambda r: r.get("candidate_rank", 0)):
        if confirmations.get(row["mutant_id"], {}).get("status") == "confirmed":
            chosen.setdefault((row["base_key"], row["operator"]), row)
    return list(chosen.values())


def analyze(output, results):
    output, results = Path(output), Path(results) / "variants"
    inventory = read_json(output / "inventory.json")
    config = read_json(output / "config.json")
    candidates = [r for r in inventory if r["operator"] != "IDENTITY"]
    confirmations, audit, hashes = {}, [], {}
    for row in candidates:
        path = output / "confirm" / (stem(row["mutant_id"]) + ".json")
        record = read_json(path) if path.exists() else {"status": "not_attempted"}
        expected_key = digest({"row": row, "stage": "confirm", "execution": config["execution_key"]})
        if record.get("cache_key") != expected_key:
            record = {"status": "not_attempted", "reason": "missing_or_stale_confirmation"}
        confirmations[row["mutant_id"]] = record
        if record["status"] != "not_attempted":
            hashes[str(path.relative_to(ROOT))] = file_hash(path)
        audit.append({k: row[k] for k in ("mutant_id", "base_id", "base_key", "task_id", "operator", "direction", "candidate_rank")}
                     | {"status": record["status"], "reason": record.get("reason", ""),
                        "difference_witness_source": (record.get("difference_witness") or {}).get("source", ""),
                        "nondegenerate_witness_source": (record.get("nondegenerate_witness") or {}).get("source", "")})
    selected = choose_confirmed(candidates, confirmations)
    retained, readings, witnesses = [], [], []
    pending_io = errors = 0
    for row in selected:
        confirmation = confirmations[row["mutant_id"]]
        path = output / "io" / (stem(row["mutant_id"]) + ".json")
        evaluated = read_json(path) if path.exists() else {"status": "missing"}
        if evaluated.get("cache_key") != digest({"row": row, "stage": "io", "execution": config["execution_key"]}):
            evaluated = {"status": "missing"}
        if path.exists():
            hashes[str(path.relative_to(ROOT))] = file_hash(path)
        pending_io += evaluated["status"] == "missing"
        errors += evaluated["status"] == "error"
        states = {}
        record = {k: row[k] for k in ("mutant_id", "base_id", "base_key", "task_id", "workflow", "model", "shot", "operator", "direction", "changed_text", "replacement_text", "proposal_source", "mutant_path")}
        record["evaluation_status"] = evaluated["status"]
        for category in CATEGORIES:
            node = evaluated.get("metrics", {}).get(category, {})
            result = classify_metric(node, category)
            states[category] = result["state"]
            record[category + "_state"] = result["state"]
            readings.append({**record, "category": category, **result})
        state = ("behavior_detected" if "behavior_detected" in states.values()
                 else "diagnostic_only" if "diagnostic_only" in states.values()
                 else "missed" if all(v == "missed" for v in states.values()) else "unresolved")
        record["union_state"] = state
        retained.append(record)
        witnesses.append({"mutant_id": row["mutant_id"], "operator": row["operator"],
                          "difference": confirmation["difference_witness"],
                          "nondegenerate": confirmation["nondegenerate_witness"],
                          "required_implication": confirmation["required_implication"],
                          "confirmation_path": str((output / "confirm" / (stem(row["mutant_id"]) + ".json")).relative_to(ROOT))})
    availability = read_csv(output / "operator_availability.csv")
    confirmation_rows, matrix = [], []
    remaining = []
    for operator, direction in OPERATORS.items():
        entries = [r for r in audit if r["operator"] == operator]
        available = [r for r in availability if r["operator"] == operator]
        counts = Counter(r["status"] for r in entries)
        group = [r for r in retained if r["operator"] == operator]
        confirmation_rows.append({"operator": operator, "direction": direction,
                                  "selected_bases": len(available),
                                  "applicable_bases": sum(int(r["available_candidates"]) > 0 for r in available),
                                  "budgeted_candidates": len(entries), "attempted_candidates": len(entries) - counts["not_attempted"],
                                  "retained": len(group), **{s: counts[s] for s in
                                  ("confirmed", "equivalent", "verification_failed", "unconfirmed", "invalid_edit", "error", "not_attempted")}})
        for category in (*CATEGORIES, "union"):
            counts = Counter(r[category + "_state"] for r in group)
            matrix.append({"operator": operator, "direction": direction, "category": category, "n": len(group),
                           "tasks": len({r["task_id"] for r in group}),
                           **{s: counts[s] for s in ("behavior_detected", "diagnostic_only", "missed", "unresolved")},
                           "behavior_detection_rate": counts["behavior_detected"] / len(group) if group else None,
                           "all_flag_rate": (counts["behavior_detected"] + counts["diagnostic_only"]) / len(group) if group else None})
    for available in availability:
        group = [r for r in audit if r["base_key"] == available["base_key"] and r["operator"] == available["operator"]]
        if not any(r["status"] == "confirmed" for r in group) and any(r["status"] == "not_attempted" for r in group):
            remaining.append((available["base_key"], available["operator"]))
    # Invalid source edits are explicitly rejected candidates, not infrastructure errors.
    errors += sum(r["status"] == "error" for r in audit)
    for name, rows in (("confirmation_audit.csv", audit), ("operator_selection.csv", confirmation_rows),
                       ("retained_variants.csv", retained), ("io_pair_readings.csv", readings), ("detection_matrix.csv", matrix)):
        write_csv(results / name, rows)
    write_json(results / "witnesses.json", witnesses)
    source_paths = [output / name for name in ("inventory.json", "inventory_summary.json", "operator_availability.csv", "config.json")]
    summary = {"status": "complete" if not remaining and not pending_io and not errors else "incomplete",
               "experiment_root": str(output.relative_to(ROOT)), "selected_bases": sum(r["operator"] == "IDENTITY" for r in inventory),
               "retained_variants": len(retained), "retained_tasks": len({r["task_id"] for r in retained}),
               "pending_confirmation_groups": len(remaining), "pending_io": pending_io, "errors": errors,
               "verification_release": "0.2025.09.25.04e8687", "policy": {
                   "population": "First confirmed nondegenerate candidate per sampled base and operator; equivalent edits excluded",
                   "semantic_confirmation": "Whole-program acceptance, directional implication, verified difference and nondegeneracy witnesses",
                   "postcondition_witnesses": "Require common legal input; implication itself has no precondition",
                   "denominator": "All retained variants, including unresolved I/O; identical across three kinds",
                   "diagnostics": "Expression diagnostics and unproved invalid-input failures separate from behavioral counterexamples",
                   "invalid_inputs": "Existing requires-rejection-or-panic protocol; no change to RQ1/RQ2",
                   "missing_detection": "Missed only when every relevant fixed case proves passing"},
               "operator_selection": confirmation_rows,
               "union_states": dict(Counter(r["union_state"] for r in retained)),
               "source_sha256": {**hashes, **{str(p.relative_to(ROOT)): file_hash(p) for p in source_paths}},
               "execution_key": config["execution_key"], "analysis_code_sha256": file_hash(Path(__file__))}
    write_json(results / "summary.json", summary)
    print({k: summary[k] for k in ("status", "retained_variants", "retained_tasks", "pending_confirmation_groups", "pending_io", "errors", "union_states")}, flush=True)
    return summary


if __name__ == "__main__":
    import argparse
    from rq3_io_common import OUTPUT, RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    analyze(args.output, args.results)
