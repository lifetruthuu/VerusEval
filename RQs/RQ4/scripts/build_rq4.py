"""Rebuild RQ4 diagnostics and REFERENCE_SCREEN reference screening from merged records.

Historical labels provide preliminary evidence. Screened references without a
historical label use the agreed labels of the 2026-10-01 reference review.
No historical corrections are overlaid on the merged target outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RQ4 = ROOT / "RQs/RQ4/results"
REFERENCE_SCREEN = ROOT / "RQs/RQ4/results/reference_screen"
DIRECTIONS = (
    ("pre_ref_to_gen", "Precondition stronger", "Soundness", "L1", "precondition_clause_reliability_rate"),
    ("post_gen_to_ref", "Postcondition weaker", "Soundness", "L2", "postcondition_clause_reliability_rate"),
    ("pre_gen_to_ref", "Precondition weaker", "Completeness", "L3", "precondition_clause_completeness_rate"),
    ("post_ref_to_gen", "Postcondition stronger", "Completeness", "L4", "postcondition_clause_completeness_rate"),
)
KEYS = [d[0] for d in DIRECTIONS]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def source(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def normalized(clauses):
    return [" ".join(c.split()) for c in clauses] or ["true"]


def summarize_reference_repair():
    """Expose the archived repair comparisons alongside the screening results."""
    directory = ROOT / "data/evidence/reference_repair/comparisons"
    records = [json.loads(p.read_text()) for p in sorted((directory / "artifacts").glob("*.json"))]
    rows = []
    for record in records:
        row = {"sample_id": record["sample_id"]}
        for stage in ("before", "after"):
            row.update({f"{stage}_{key}": record[stage][key]["status"] for key in KEYS})
            row[f"{stage}_equivalent"] = all(record[stage][key]["holds"] is True for key in KEYS)
        row["io_all_passed"] = all(case["passed"] for case in record["io"])
        rows.append(row)
    write_csv(RQ4 / "reference_repair.csv", rows)
    write_json(RQ4 / "reference_repair_summary.json", {
        "artifacts": len(rows),
        "original_equivalent": sum(row["before_equivalent"] for row in rows),
        "corrected_equivalent": sum(row["after_equivalent"] for row in rows),
        "io_all_passed": sum(row["io_all_passed"] for row in rows),
        "source": source(directory / "summary.json"),
    })
    write_json(RQ4 / "case_summary.json", json.loads(
        (ROOT / "data/evidence/clause_cases/summary.json").read_text()))


def target_probe(record, side, kind, target):
    nodes = record.get("metrics", {}).get("trivial_spec_ratio", {}).get(side, {}).get(
        "details", {}).get(kind, {}).get("functions", [])
    return next((n for n in nodes if n.get("function") == target), {})


def locate_clauses(row, record):
    key = next(k for k in KEYS if row[k] == "invalid")
    metric = next(d[4] for d in DIRECTIONS if d[0] == key)
    target = record["target_evaluation"]["target_function"]
    node = next((n for n in record["metrics"].get(metric, {}).get("generated", {}).get(
        "functions", []) if n.get("function") == target), {})
    clauses = node.get("details", [])
    kind = "requires" if key.startswith("pre_") else "ensures"
    probe = "precondition_falsity" if kind == "requires" else "postcondition_truth"
    ant_side, con_side = (("ground", "generated") if key.endswith("ref_to_gen")
                          else ("generated", "ground"))
    ant = target_probe(record, ant_side, probe, target).get(kind)
    con = target_probe(record, con_side, probe, target).get(kind)
    reason = ""
    contexts = [c.get("implication_check", {}).get("obligation_context", {}) for c in clauses]
    if not clauses or ant is None or con is None:
        reason = "missing_target_clause_context"
    elif normalized([c.get("clause", {}).get("text", "") for c in clauses]) != normalized(con):
        reason = "consequent_coverage_mismatch"
    elif any(normalized(c.get("antecedent", [])) != normalized(ant) for c in contexts):
        reason = "antecedent_mismatch"
    elif any(any(k != kind for k in c.get("antecedent_kinds", [])) for c in contexts):
        reason = "different_premise_policy"
    elif any(normalized(c.get("consequent", [])) != normalized([n["clause"]["text"]])
             for c, n in zip(contexts, clauses)):
        reason = "consequent_context_mismatch"
    states = [c.get("implication_check", {}).get("status", "unknown") for c in clauses]
    if not reason and any(s not in {"valid", "invalid"} for s in states):
        reason = "unresolved_clause_checks"
    if not reason and "invalid" not in states:
        reason = "no_rejected_clause_in_saved_checks"
    rejected = [c["clause"]["text"] for c, s in zip(clauses, states) if s == "invalid"]
    return {
        "sample_id": row["sample_id"], "task_id": row["task_id"], "direction": key,
        "target_function": target, "result_path": row["result_path"],
        "status": reason or "localized_proof_rejection", "clauses": len(clauses),
        "rejected_clauses": len(rejected), "passed_clauses": states.count("valid"),
        "antecedent": ant, "consequent": con, "rejected_text": rejected,
        "obligation_hashes": [c.get("implication_check", {}).get("obligation_sha256") for c in clauses],
    }


def main():
    paths = {name: ROOT / path for name, path in {
        "outcomes": "data/evaluation/rq1_artifact_outcomes.csv",
        "index": "data/evaluation/artifact_index.csv",
        "catalog": "data/evaluation/target_functions.csv",
        "historical_labels": "data/evidence/reference_review/prior_labels.csv",
        "review_labels": "data/evidence/reference_review/labels_final.csv",
        "historical_annotations": "data/evidence/reference_review/prior_annotations.csv",
    }.items()}
    data = {name: read_csv(path) for name, path in paths.items()}
    index = {r["sample_id"]: r for r in data["index"]}
    catalog = {r["task_id"]: r for r in data["catalog"]}
    labels = {r["task_id"]: r for r in data["historical_labels"]}
    review = {r["task_id"]: r for r in data["review_labels"]}
    assert not review.keys() & labels.keys()
    # Both annotators of a label agree on the defect decision and the primary label.
    agreement = {r["task_id"]: r["defect_agree"] == r["primary_agree"] == "True" for r in data["historical_annotations"]}
    assert agreement.keys() == labels.keys()
    agreement.update({task: True for task in review})
    accepted = [r for r in data["outcomes"] if r["verification_stage"] == "accepted"]
    assert len({r["sample_id"] for r in accepted}) == len(accepted)
    grouped = defaultdict(list)
    for row in accepted:
        grouped[row["task_id"]].append(row)
    ref_samples, ref_hits, ref_available = Counter(), defaultdict(set), set()
    reference_evidence, records, clauses, evidence_index = [], [], [], []

    def read_record(sample):
        entry = index[sample]
        path = ROOT / entry["result_path"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["result_sha256"], sample
        evidence_index.append({"sample_id": sample, "result_path": entry["result_path"],
                               "result_sha256": entry["result_sha256"]})
        return json.loads(raw)

    def harvest_reference(task, sample, record):
        if ref_samples[task] >= 3:
            return
        ref_samples[task] += 1
        target = catalog[task]["target_function"]
        for probe, flag in (("precondition_falsity", "always_false"),
                            ("postcondition_truth", "always_true")):
            node = target_probe(record, "ground", probe, target)
            if node:
                ref_available.add(task)
            if node.get(flag) is True:
                ref_hits[task].add(probe)
            reference_evidence.append({"task_id": task, "sample_id": sample,
                                       "target_function": target, "probe": probe,
                                       "status": node.get("status", "unavailable"),
                                       "proved": node.get(flag) is True,
                                       "result_path": index[sample]["result_path"]})

    for row in accepted:
        sample, task = row["sample_id"], row["task_id"]
        record = read_record(sample)
        target = record["target_evaluation"]
        assert target["analysis_eligible"] and target["verification_stage"] == "accepted"
        assert all(target["directions"][k] == row[k] for k in KEYS), sample
        assert target["target_function"] == catalog[task]["target_function"], sample
        row["result_path"] = index[sample]["result_path"]
        row["flags"] = sum(row[k] == "invalid" for k in KEYS)
        row["state"] = "flagged" if row["flags"] else (
            "equivalent" if all(row[k] == "valid" for k in KEYS) else "unresolved")
        trivial = target["triviality"]
        row["S1"] = trivial["postcondition_truth"] == "unfavorable" and row["post_gen_to_ref"] == "valid"
        row["S2"] = trivial["precondition_falsity"] == "unfavorable"
        harvest_reference(task, sample, record)
        if row["flags"] == 1:
            clauses.append(locate_clauses(row, record))
        records.append(row)
    # References of tasks with fewer than three accepted artifacts are still screened.
    used = {r["sample_id"] for r in accepted}
    for sample in sorted(index):
        task = sample.split("|")[-1]
        if task in catalog and ref_samples[task] < 3 and sample not in used:
            harvest_reference(task, sample, read_record(sample))
    reference_sources = []
    for task, row in catalog.items():
        candidates = [ROOT / row["reference_path"],
                      ROOT / "data/generation/Y" / row["benchmark"] / row["reference_file"],
                      ROOT / "data/evidence/contract_variants/references" / row["benchmark"] / row["reference_file"]]
        matches = [p for p in candidates if p.is_file()
                   and source(p)["sha256"] == row["reference_sha256"]]
        reference_sources.append({"task_id": task, "original_path": row["reference_path"],
                                  "sha256": row["reference_sha256"],
                                  "available_copy": str(matches[0].relative_to(ROOT)) if matches else "",
                                  "status": "hash_matched_copy" if matches else "recorded_probe_only"})

    flagged = [r for r in records if r["state"] == "flagged"]
    assert all(r[k] in {"valid", "invalid"} for r in flagged for k in KEYS)
    combos = Counter(tuple(r[k] == "invalid" for k in KEYS) for r in flagged)
    combination_rows = [{**dict(zip(KEYS, combo)), "artifacts": count,
                         "share": count / len(flagged), "diagnostics": sum(combo)}
                        for combo, count in combos.most_common()]
    direction_rows = [{"key": k, "diagnostic": name, "group": group, "obligation": obligation,
                       "artifacts": sum(r[k] == "invalid" for r in flagged),
                       "denominator": len(flagged),
                       "share": sum(r[k] == "invalid" for r in flagged) / len(flagged),
                       "single_flag": sum(r[k] == "invalid" and r["flags"] == 1 for r in flagged)}
                      for k, name, group, obligation, _ in DIRECTIONS]
    clause_counts = []
    for key in KEYS:
        subset = [r for r in clauses if r["direction"] == key]
        localized = [r for r in subset if r["status"] == "localized_proof_rejection"]
        clause_counts.append({"direction": key, "single_flag": len(subset),
                              "localized": len(localized),
                              "one_rejected_clause": sum(r["rejected_clauses"] == 1 for r in localized),
                              "multiple_rejected_clauses": sum(r["rejected_clauses"] > 1 for r in localized),
                              "not_localized": len(subset) - len(localized)})
    summary4 = {"accepted": len(records), "states": dict(Counter(r["state"] for r in records)),
                "flagged_tasks": len({r["task_id"] for r in flagged}), "directions": direction_rows,
                "flag_counts": dict(Counter(r["flags"] for r in flagged)),
                "post_both": sum(r["post_gen_to_ref"] == r["post_ref_to_gen"] == "invalid" for r in flagged),
                "clause_localization": clause_counts,
                "localization_statuses": dict(Counter(r["status"] for r in clauses))}
    write_csv(RQ4 / "membership.csv", records)
    write_csv(RQ4 / "directions.csv", direction_rows)
    write_csv(RQ4 / "combinations.csv", combination_rows)
    write_json(RQ4 / "single_flag_clauses.json", clauses)
    write_csv(RQ4 / "clause_summary.csv", clause_counts)
    write_json(RQ4 / "summary.json", summary4)

    screen = []
    for task in catalog:
        rows = grouped[task]
        s3 = [r for r in rows if r["post_gen_to_ref"] == "valid" and r["post_ref_to_gen"] == "invalid"]
        s4 = [r for r in rows if r["pre_gen_to_ref"] == "valid" and r["pre_ref_to_gen"] == "invalid"]
        signals = {"S0": bool(ref_hits[task]), "S1": any(r["S1"] for r in rows),
                   "S2": any(r["S2"] for r in rows)}
        for name, subset in (("S3", s3), ("S4", s4)):
            signals[name] = (len(rows) >= 6 and len(subset) / len(rows) >= .8
                             and len({r["sample_id"].split("|")[1] for r in subset}) >= 3)
        label = labels.get(task) or review.get(task, {})
        label_source = "historical" if task in labels else "review_20261001" if task in review else ""
        screen.append({"task_id": task, "accepted": len(rows), **signals,
                       "selected": any(signals.values()), "s3_artifacts": len(s3), "s4_artifacts": len(s4),
                       "preliminary_label": label.get("final_reference_defect", "not_reviewed"),
                       "category": label.get("final_primary", ""),
                       "label_layer": label.get("label_layer", "not_reviewed"),
                       "label_source": label_source})
    selected = [r for r in screen if r["selected"]]
    signals = []
    for signal in ("S0", "S1", "S2", "S3", "S4", "selected"):
        subset = [r for r in screen if r[signal]]
        signals.append({"signal": signal, "tasks": len(subset),
                        "candidate": sum(r["preliminary_label"] == "yes" for r in subset),
                        "no_candidate": sum(r["preliminary_label"] == "no" for r in subset),
                        "not_reviewed": sum(r["preliminary_label"] == "not_reviewed" for r in subset)})
    candidates = {t for t, r in labels.items() if r["final_reference_defect"] == "yes"}
    current_candidates = {r["task_id"] for r in selected if r["preliminary_label"] == "yes"}
    sensitivity = []
    for name, subset in (("All flagged", flagged),
                         ("Exclude current candidate tasks", [r for r in flagged if r["task_id"] not in current_candidates]),
                         ("Exclude historical candidate tasks", [r for r in flagged if r["task_id"] not in candidates])):
        for key, title, *_ in DIRECTIONS:
            n = sum(r[key] == "invalid" for r in subset)
            sensitivity.append({"cohort": name, "direction": key, "diagnostic": title,
                                "artifacts": n, "denominator": len(subset), "share": n / len(subset)})
    review_summary = {"references": len(catalog), "references_with_probe": len(ref_available),
                "selected": len(selected), "signals": signals,
                "selected_label_layers": dict(Counter(r["label_layer"] for r in selected)),
                "selected_candidate_categories": dict(Counter(r["category"] for r in selected if r["preliminary_label"] == "yes")),
                "selected_label_sources": dict(Counter(r["label_source"] or "none" for r in selected)),
                "selected_annotator_agreement": sum(agreement.get(r["task_id"], False) for r in selected),
                "selected_disagreements_human_reviewed": sum(
                    not agreement[r["task_id"]] and r["label_layer"] == "human_review"
                    for r in selected if r["task_id"] in agreement),
                "historical_labels": len(labels),
                "historical_label_layers": dict(Counter(r["label_layer"] for r in labels.values())),
                "historical_candidates": len(candidates),
                "historical_candidate_categories": dict(Counter(r["final_primary"] for r in labels.values() if r["final_reference_defect"] == "yes")),
                "not_reviewed": [r["task_id"] for r in selected if r["preliminary_label"] == "not_reviewed"],
                "candidate_flagged": sum(r["task_id"] in candidates for r in flagged),
                "sensitivity": sensitivity,
                "scope": "Target-function signals and candidate annotations, combining historical labels and recorded subset review with the 2026-10-01 review of previously unlabeled references."}
    write_csv(REFERENCE_SCREEN / "screen_tasks.csv", screen)
    write_csv(REFERENCE_SCREEN / "screen_signals.csv", signals)
    write_csv(REFERENCE_SCREEN / "reference_probes.csv", reference_evidence)
    write_csv(REFERENCE_SCREEN / "reference_sources.csv", reference_sources)
    write_csv(REFERENCE_SCREEN / "historical_labels.csv", data["historical_labels"])
    write_csv(REFERENCE_SCREEN / "sensitivity.csv", sensitivity)
    write_json(REFERENCE_SCREEN / "summary.json", review_summary)
    manifest = {"sources": {name: source(path) for name, path in paths.items()},
                "script": source(Path(__file__)), "record_count": len(evidence_index),
                "scope": "Merged target-level results; no historical result overlays"}
    write_csv(RQ4 / "record_sources.csv", evidence_index)
    write_json(RQ4 / "manifest.json", manifest)
    write_json(REFERENCE_SCREEN / "manifest.json", manifest)
    summarize_reference_repair()
    print(json.dumps({"rq4": summary4, "reference_screen": review_summary}, indent=2))


if __name__ == "__main__":
    main()
