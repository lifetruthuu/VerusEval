"""Independently audit RQ3 populations, confirmations, fixed denominators, and evidence hashes."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from rq3_io_common import ROOT, OUTPUT, RESULTS, CATEGORIES, METRICS, digest, file_hash, read_csv, read_json, stem, write_json


def validate(output, results, check_proofs=True):
    output, results = Path(output), Path(results)
    natural = ROOT / "RQs/RQ3/results/natural"
    natural_summary = read_json(natural / "summary.json")
    summary = read_json(results / "variants/summary.json")
    assert summary["status"] == "complete", "Experiment is not complete"
    hashed = {}
    def check(relative, expected):
        if relative not in hashed:
            hashed[relative] = file_hash(ROOT / relative)
        assert hashed[relative] == expected, relative
    inventory_summary = read_json(output / "inventory_summary.json")
    for record in (natural_summary, summary, inventory_summary):
        for path, expected in record["source_sha256"].items():
            check(path, expected)
    # Recompute the natural conditional counts from source CSVs, independently of profiles.
    source = read_csv(ROOT / "data/evaluation/rq1_artifact_outcomes.csv")
    accepted = [r for r in source if r["verification_stage"] == "accepted"]
    for scope in natural_summary["overall"]:
        rows = []
        for r in accepted:
            totals = [sum(int(r[f"io_{kind}_{state}"]) for state in ("passed", "failed", "unresolved")) for kind in ("correct", "wrong", "invalid")]
            if scope["scope"] == "all_three_nonempty" and not all(totals):
                continue
            rows.append(r)
        passing = [r for r in rows if sum(int(r[f"io_{k}_passed"]) for k in ("correct", "wrong", "invalid")) > 0
                   and not any(int(r[f"io_{k}_{s}"]) for k in ("correct", "wrong", "invalid") for s in ("failed", "unresolved"))]
        assert len(rows) == scope["accepted"]
        assert len(passing) == scope["io_passed"]
        assert sum(any(int(r[f"io_{k}_failed"]) for k in ("correct", "wrong", "invalid")) for r in rows) == scope["io_failed"]
        assert sum(not any(int(r[f"io_{k}_failed"]) for k in ("correct", "wrong", "invalid")) and
                   any(int(r[f"io_{k}_unresolved"]) for k in ("correct", "wrong", "invalid")) for r in rows) == scope["io_unresolved"]
        assert sum(not any(int(r[f"io_{k}_{s}"]) for k in ("correct", "wrong", "invalid") for s in ("passed", "failed", "unresolved")) for r in rows) == scope["io_unavailable"]
        for metric, directions in (("soundness", ("pre_ref_to_gen", "post_gen_to_ref")),
                                   ("completeness", ("pre_gen_to_ref", "post_ref_to_gen")),
                                   ("equivalence", ("pre_ref_to_gen", "post_gen_to_ref", "pre_gen_to_ref", "post_ref_to_gen"))):
            assert sum(all(r[k] == "valid" for k in directions) for r in passing) == scope[metric + "_proved_given_io_pass"]
            assert sum(any(r[k] == "invalid" for k in directions) for r in passing) == scope[metric + "_rejected_given_io_pass"]
    inventory = read_json(output / "inventory.json")
    by_id = {r["mutant_id"]: r for r in inventory}
    assert len(by_id) == len(inventory)
    bases = [r for r in inventory if r["operator"] == "IDENTITY"]
    assert len(bases) == inventory_summary["selected_bases"] == summary["selected_bases"]
    assert len(inventory) - len(bases) == inventory_summary["candidates"]
    for row in bases:
        check(row["source_result_path"], row["source_result_sha256"])
        check(row["source_generated_path"], row["base_sha256"])
        evidence = read_json(output / "base_evidence" / (row["base_key"] + ".json"))
        original = read_json(ROOT / row["source_result_path"])
        assert evidence["metrics"] == {kind: original["metrics"][metric]["generated"] for kind, metric in METRICS.items()}
        assert evidence["target_evaluation"] == original["target_evaluation"]
        suite = read_json(ROOT / row["suite_path"])
        assert digest(suite) == row["suite_sha256"]
        for kind in CATEGORIES:
            cases = evidence["metrics"][kind]["details"]
            assert cases and all(c["success"] is True for c in cases)
            assert {c["id"] for c in cases} == {c["id"] for c in suite["cases"] if c["kind"] == kind and c.get("status") == "validated"}
    config = read_json(output / "config.json")
    retained = read_csv(results / "variants/retained_variants.csv")
    assert len(retained) == summary["retained_variants"]
    seen = set()
    proof_runs = witnesses = 0
    for entry in retained:
        row = by_id[entry["mutant_id"]]
        pair = (row["base_key"], row["operator"])
        assert pair not in seen
        seen.add(pair)
        for label, column in (("base", "base_clean_path"), ("mutant", "mutant_path"), ("reference", "reference_path")):
            check(row[column], row[label + "_sha256"])
        check(row["source_result_path"], row["source_result_sha256"])
        assert digest(read_json(ROOT / row["suite_path"])) == row["suite_sha256"]
        confirmation = read_json(output / "confirm" / (stem(row["mutant_id"]) + ".json"))
        assert confirmation["cache_key"] == digest({"row": row, "stage": "confirm", "execution": config["execution_key"]})
        assert confirmation["status"] == "confirmed" and confirmation["verification"]["success"] is True
        assert confirmation["direction_checks"][confirmation["required_implication"]]["holds"] is True
        for earlier in inventory:
            if (earlier["base_key"], earlier["operator"]) != pair or earlier.get("candidate_rank", 0) >= row["candidate_rank"]:
                continue
            previous = read_json(output / "confirm" / (stem(earlier["mutant_id"]) + ".json"))
            assert previous["cache_key"] == digest({"row": earlier, "stage": "confirm", "execution": config["execution_key"]})
            assert previous["status"] != "confirmed", "Did not retain the first confirmed candidate"
        w, n = confirmation["difference_witness"], confirmation["nondegenerate_witness"]
        post, strong = row["direction"].startswith("post_"), row["direction"].endswith("strengthening")
        if post:
            assert all(c["base_pre"] is True and c["mutant_pre"] is True for c in (w, n))
            assert w["base_contract"] is strong and w["mutant_contract"] is (not strong)
            assert n["mutant_contract"] is strong
        else:
            assert w["base_pre"] is strong and w["mutant_pre"] is (not strong)
            assert n["mutant_pre"] is strong
        witnesses += 2
        # All three kinds use exactly the same source case IDs as the eligible base.
        evaluated = read_json(output / "io" / (stem(row["mutant_id"]) + ".json"))
        assert evaluated["cache_key"] == digest({"row": row, "stage": "io", "execution": config["execution_key"]})
        assert evaluated["status"] == "completed"
        base = read_json(output / "base_evidence" / (row["base_key"] + ".json"))
        states = []
        for kind in CATEGORIES:
            node = evaluated["metrics"][kind]
            cases = node.get("details") or []
            assert {c["id"] for c in cases} == {c["id"] for c in base["metrics"][kind]["details"]}, (row["mutant_id"], kind, "case IDs")
            assert len(cases) == len({c["id"] for c in cases})
            counts = Counter("passed" if c["success"] is True else "failed" if c["success"] is False else "unknown" for c in cases)
            assert all(int(node.get(k, 0)) == counts[k] for k in ("passed", "failed", "unknown"))
            assert node["total"] == len(cases)
            assert abs(node["score"] - counts["passed"] / len(cases)) < 1e-12
            behavioral = any(c["success"] is False and (kind != "invalid" or c["evaluation"].get("requires_accepted") is True) for c in cases)
            if entry[kind + "_state"] == "behavior_detected":
                assert behavioral
            elif entry[kind + "_state"] == "missed":
                assert all(c["success"] is True for c in cases)
            states.append(entry[kind + "_state"])
        expected = ("behavior_detected" if "behavior_detected" in states else "diagnostic_only" if "diagnostic_only" in states
                    else "missed" if all(s == "missed" for s in states) else "unresolved")
        assert entry["union_state"] == expected
        if check_proofs:
            for stage in ("confirm", "io"):
                runs_path = output / stage / stem(row["mutant_id"]) / "runs.jsonl"
                import json
                runs = [json.loads(line) for line in runs_path.read_text().splitlines()]
                for run in runs:
                    check(run["source"], run["source_sha256"])
                proof_runs += len(runs)
    matrix = read_csv(results / "variants/detection_matrix.csv")
    for cell in matrix:
        subset = [r for r in retained if r["operator"] == cell["operator"]]
        counts = Counter(r[cell["category"] + "_state"] for r in subset)
        assert int(cell["n"]) == len(subset)
        assert sum(int(cell[k]) for k in ("behavior_detected", "diagnostic_only", "missed", "unresolved")) == len(subset)
        assert all(int(cell[k]) == counts[k] for k in ("behavior_detected", "diagnostic_only", "missed", "unresolved"))
    result = {"status": "complete", "natural_accepted_checked": len(accepted), "base_records_checked": len(bases), "confirmed_variants_checked": len(retained),
              "witnesses_checked": witnesses, "verifier_run_records_checked": proof_runs,
              "source_files_hashed": len(hashed), "matrix_cells_checked": len(matrix),
              "method": "Source-CSV recomputation, witness truth conditions, source/case identity, per-case scoring, fixed-denominator reconstruction, recorded proof-source hashes",
              "summary_sha256": file_hash(results / "variants/summary.json"), "validator_sha256": file_hash(Path(__file__))}
    write_json(results / "validation.json", result)
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    validate(args.output, args.results)
