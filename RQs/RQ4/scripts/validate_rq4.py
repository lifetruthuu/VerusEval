"""Check RQ4 populations, source hashes, review records, and verifier evidence."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from build_rq4 import KEYS, ROOT, RQ4, REFERENCE_SCREEN, read_csv, source, write_json


def check_source(record):
    assert source(ROOT / record["path"])["sha256"] == record["sha256"], record["path"]


def main():
    original = {r["sample_id"]: r for r in read_csv(ROOT / "data/evaluation/rq1_artifact_outcomes.csv")
                if r["verification_stage"] == "accepted"}
    members = read_csv(RQ4 / "membership.csv")
    assert {r["sample_id"] for r in members} == original.keys()
    assert all(all(r[k] == original[r["sample_id"]][k] for k in KEYS) for r in members)
    flagged = [r for r in members if any(r[k] == "invalid" for k in KEYS)]
    combinations = read_csv(RQ4 / "combinations.csv")
    observed = Counter(tuple(r[k] == "invalid" for k in KEYS) for r in flagged)
    assert observed == {tuple(r[k] == "True" for k in KEYS): int(r["artifacts"]) for r in combinations}
    clauses = json.loads((RQ4 / "single_flag_clauses.json").read_text())
    assert {r["sample_id"] for r in clauses} == {r["sample_id"] for r in flagged if sum(r[k] == "invalid" for k in KEYS) == 1}
    for row in read_csv(RQ4 / "clause_summary.csv"):
        subset = [r for r in clauses if r["direction"] == row["direction"]]
        localized = [r for r in subset if r["status"] == "localized_proof_rejection"]
        assert len(subset) == int(row["single_flag"])
        assert len(localized) == int(row["one_rejected_clause"]) + int(row["multiple_rejected_clauses"])
        assert all(r["passed_clauses"] + r["rejected_clauses"] == r["clauses"] for r in localized)
    screen = read_csv(REFERENCE_SCREEN / "screen_tasks.csv")
    assert len(screen) == len({r["task_id"] for r in screen}) == 762
    selected = [r for r in screen if r["selected"] == "True"]
    reviews = read_csv(ROOT / "data/evidence/reference_review/candidate_reviews.csv")
    assert len(selected) == len(reviews) == len({r["task_id"] for r in reviews}) == 61
    assert {r["task_id"] for r in reviews} == {r["task_id"] for r in selected}
    screened = {r["task_id"]: r for r in selected}
    disagreements = 0
    for row in reviews:
        task = row["task_id"]
        agreement = all(row[f"reviewer_a_{key}"] == row[f"reviewer_b_{key}"]
                        for key in ("reference_defect", "primary"))
        assert row["initial_agreement"] == str(agreement), task
        assert row["expert_adjudicated"] == str(not agreement), task
        assert screened[task]["final_reference_defect"] == row["final_reference_defect"], task
        assert screened[task]["category"] == row["final_primary"], task
        if not agreement:
            assert row["expert_rationale"], task
            disagreements += 1
    assert disagreements == 4
    assert all((r["selected"] == "True") == any(r[k] == "True" for k in ("S0", "S1", "S2", "S3", "S4")) for r in screen)
    for row in read_csv(REFERENCE_SCREEN / "screen_signals.csv"):
        subset = [r for r in screen if r[row["signal"]] == "True"]
        assert len(subset) == int(row["tasks"])
        assert len(subset) == sum(int(row[k]) for k in ("candidate", "no_candidate", "not_reviewed"))
    candidates = {r["task_id"] for r in selected if r["final_reference_defect"] == "yes"}
    current = [r for r in flagged if r["task_id"] not in candidates]
    sensitivity = read_csv(REFERENCE_SCREEN / "sensitivity.csv")
    assert len(sensitivity) == 8
    assert {r["cohort"] for r in sensitivity} == {"All flagged", "Exclude current candidate tasks"}
    for row in sensitivity:
        if row["cohort"] == "Exclude current candidate tasks":
            assert len(current) == int(row["denominator"])
            assert sum(r[row["direction"]] == "invalid" for r in current) == int(row["artifacts"])
    for name in ("manifest.json", "paper_manifest.json"):
        manifest = json.loads((RQ4 / name).read_text())
        check_source(manifest["script"])
        for record in manifest.get("sources", {}).values():
            check_source(record)
        for record in manifest.get("inputs", []) + manifest.get("outputs", []):
            check_source(record)
    cases = json.loads((ROOT / "data/evidence/clause_cases/summary.json").read_text())
    assert cases["examples"] == 4
    check_source(cases["script"])
    for case in cases["cases"]:
        for key in ("generated", "reference", "record"):
            check_source(case[key])
        field = "pair" if "output" in case["witness"] else "pre"
        decisions = case["decisions"]
        assert {decisions["reference"][field], decisions["generated"][field]} == {True, False}
        if field == "pair":
            assert all(x["pre"] is True for x in decisions.values())
    repair_root = ROOT / "data/evidence/reference_repair/comparisons"
    repair = json.loads((repair_root / "summary.json").read_text())
    for entry in repair["sources"]:
        check_source(entry)
    repair_records = [json.loads(p.read_text()) for p in sorted((repair_root / "artifacts").glob("*.json"))]
    assert {r["sample_id"] for r in repair_records} == {r["sample_id"] for r in members if r["task_id"] == repair["task_id"]}
    for record in repair_records:
        check_source(record["generated"])
        check_source(record["record"])
        assert all(record["before"][key]["status"] == original[record["sample_id"]][key] for key in KEYS)
        for stage in ("before", "after"):
            for key, check in record[stage].items():
                expected_kind = "requires" if key.startswith("pre_") else "ensures"
                assert all(k == expected_kind for k in check["obligation_context"]["antecedent_kinds"])
        assert all(r["passed"] for r in record["io"])
        assert all(r["input_admitted"] is True for r in record["io"] if r["kind"] == "negative")
    assert sum(all(c["holds"] is True for c in r["after"].values()) for r in repair_records) == repair["corrected_equivalent"]
    repair_table = {row["sample_id"]: row for row in read_csv(RQ4 / "reference_repair.csv")}
    assert set(repair_table) == {record["sample_id"] for record in repair_records}
    for record in repair_records:
        row = repair_table[record["sample_id"]]
        for stage in ("before", "after"):
            assert all(row[f"{stage}_{key}"] == record[stage][key]["status"] for key in KEYS)
            assert (row[f"{stage}_equivalent"] == "True") == all(record[stage][key]["holds"] is True for key in KEYS)
        assert (row["io_all_passed"] == "True") == all(case["passed"] for case in record["io"])
    remaining = json.loads((repair_root / "remaining_flag_explanation.json").read_text())
    assert remaining["unconditional_L2"] == "invalid" and remaining["legal_domain_L2"]["holds"] is True
    for key in ("script", "generated", "reference"):
        check_source(remaining[key])
    runs = 0
    for directory in (ROOT / "data/evidence/clause_cases", repair_root):
        for log in directory.rglob("runs.jsonl"):
            for line in log.read_text().splitlines():
                record = json.loads(line)
                check_source({"path": record["harness"], "sha256": record["sha256"]})
                assert (record.get("verus_release") == "0.2025.09.25.04e8687" or
                        "verus-04e8687" in record["command"][0])
                assert record["returncode"] in {0, 1} or record.get("timeout")
                runs += 1
    result = {"status": "passed", "accepted": len(members), "flagged": len(flagged),
              "single_flag": len(clauses), "screened_references": len(selected),
              "current_candidate_references": len(candidates), "verified_witness_cases": 4,
              "repair_artifacts": len(repair_records), "corrected_equivalent": repair["corrected_equivalent"],
              "archived_verus_runs_checked": runs, "validator": source(Path(__file__))}
    write_json(RQ4 / "validation.json", result)
    write_json(REFERENCE_SCREEN / "validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
