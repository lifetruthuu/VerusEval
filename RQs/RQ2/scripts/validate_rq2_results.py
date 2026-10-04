"""Validate RQ2 scores, populations, paired intervals and sampled per-file sources.

Called by: python scripts/reproduce.py --rq 2 --output-dir runs/rq2
Uses independent score/aggregation code, without importing the analysis script.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "RQs/RQ2/results"
FORMAL = {
    "Soundness": ("pre_ref_to_gen", "post_gen_to_ref"),
    "Completeness": ("pre_gen_to_ref", "post_ref_to_gen"),
    "Equivalence": ("pre_ref_to_gen", "post_gen_to_ref", "pre_gen_to_ref", "post_ref_to_gen"),
}
TEXT = {"BLEU": "text_bleu", "ROUGE-L": "text_rouge_l", "KeySpecMatch": "text_key_spec_match"}
LLM = {"Spec-code judgement": "llm_spec_code_score", "Spec-spec judgement": "llm_intent_score"}
TRIVIAL = {"No false precondition": "triviality_pre_false_state", "No true postcondition": "triviality_post_true_state"}
IO = {"Correct-I/O acceptance": "io_correct", "Wrong-output rejection": "io_wrong",
      "Invalid-input rejection": "io_invalid"}
IO_NODES = {"io_correct": "correct_io_pass_rate", "io_wrong": "wrong_io_reject_rate",
            "io_invalid": "invalid_test_filtering_rate"}
STAGE_PASS = {"Syntactic validity": {"frontend_failed", "proof_failed", "accepted"},
              "Type validity": {"proof_failed", "accepted"}, "Verifier acceptance": {"accepted"}}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: str) -> float:
    return float(value) if value else np.nan


def equal(actual: float, expected: float, context: object) -> None:
    assert np.isclose(actual, expected, atol=1e-12, rtol=0, equal_nan=True), (context, actual, expected)


def validate() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    assert summary["status"] == "complete"
    for group in ("source_sha256", "code_sha256", "result_sha256"):
        for path, expected in summary[group].items():
            assert sha256(ROOT / path) == expected, path
    source = ROOT / summary["source_root"]
    labels = {r["sample_id"]: r for r in read_csv(source / "artifact_labels.csv")}
    outcomes = {r["sample_id"]: r for r in read_csv(source / "artifact_outcomes.csv")}
    index = {r["sample_id"]: r for r in read_csv(source / "artifact_index.csv")}
    records = read_csv(RESULTS / "rq2_artifact_metrics.csv")
    assert len(records) == len({r["sample_id"] for r in records}) == len(labels) == summary["artifacts"]
    assert {r["sample_id"] for r in records} == labels.keys()
    configurations, audit_samples = defaultdict(dict), {}
    metrics = list(TEXT) + list(LLM) + list(STAGE_PASS) + list(TRIVIAL) + list(FORMAL) + list(IO)
    for record in records:
        sample_id = record["sample_id"]
        label, outcome = labels[sample_id], outcomes[sample_id]
        stage = label["verification_stage"]
        eligible = label["analysis_eligible"] == "True"
        for field in ("workflow", "model", "shot", "task_id", "verification_stage", "analysis_eligible"):
            assert record[field] == label[field], (sample_id, field)
        assert record["exclusion_reason"] == ("" if eligible else "missing_executable_target")
        expected = {name: float(stage in stages) for name, stages in STAGE_PASS.items()}
        expected.update({name: number(label[key]) for name, key in TEXT.items()})
        expected.update({name: number(label[key]) if stage in STAGE_PASS["Type validity"] else np.nan
                         for name, key in LLM.items()})
        expected.update({name: {"favorable": 1, "unfavorable": 0}.get(label[key], np.nan)
                         for name, key in TRIVIAL.items()})
        for name, directions in FORMAL.items():
            states = [outcome[d] for d in directions]
            expected[name] = 0 if any(s == "invalid" for s in states) else 1 if all(s == "valid" for s in states) else np.nan
        for name, key in IO.items():
            counts = [int(outcome[key + suffix]) for suffix in ("_passed", "_failed", "_unresolved")]
            assert min(counts) >= 0 and int(outcome[key + "_skipped"]) == 0
            expected[name] = counts[0] / sum(counts) if sum(counts) else np.nan
        for name in metrics:
            equal(number(record[name]), expected[name] if eligible or name in STAGE_PASS else np.nan,
                  (sample_id, name))
        config = tuple(record[k] for k in ("workflow", "model", "shot"))
        assert record["task_id"] not in configurations[config]
        configurations[config][record["task_id"]] = record
        audit_samples.setdefault((*config, stage, eligible), sample_id)

    profiles = read_csv(RESULTS / "rq2_configuration_profiles.csv")
    for row in profiles:
        members = list(configurations[tuple(row[k] for k in ("workflow", "model", "shot"))].values())
        scores = np.array([number(r[row["metric"]]) for r in members])
        scores = scores[np.isfinite(scores)]
        eligible = len(members) if row["metric"] in STAGE_PASS else sum(r["analysis_eligible"] == "True" for r in members)
        assert len(scores) == int(row["n"])
        assert len(members) == int(row["total"])
        assert eligible == int(row["eligible"])
        assert int(row["missing_score"]) == eligible - len(scores)
        assert int(row["excluded_missing_target"]) == len(members) - eligible
        equal(float(scores.mean()) if scores.size else np.nan, number(row["mean"]), row)
        equal(len(scores) / eligible, number(row["coverage"]), row)

    coverage = read_csv(RESULTS / "rq2_metric_coverage.csv")
    for row in coverage:
        members = [r for r in configurations[tuple(row[k] for k in ("workflow", "model", "shot"))].values()
                   if r["verification_stage"] == row["verification_stage"]]
        scores = [number(r[row["metric"]]) for r in members if r[row["metric"]]]
        assert len(members) == int(row["total"]) and len(scores) == int(row["n"])
        equal(float(np.mean(scores)) if scores else np.nan, number(row["mean"]), row)

    interval_checks = {}
    for name, seed_key in (("rq2_prompt_deltas.csv", "bootstrap_seed"),
                           ("rq2_matched_deltas.csv", "matched_bootstrap_seed"),
                           ("rq2_prompt_frontend_valid_deltas.csv", "frontend_sensitivity_bootstrap_seed")):
        rng = np.random.default_rng(summary[seed_key])
        rows = read_csv(RESULTS / name)
        for row in rows:
            if name == "rq2_matched_deltas.csv":
                left_config, right_config = (tuple(row[prefix + k] for k in ("workflow", "model", "shot"))
                                             for prefix in ("left_", "right_"))
            else:
                left_config, right_config = ((row["workflow"], row["model"], shot) for shot in ("few-shot", "zero-shot"))
            left, right = configurations[left_config], configurations[right_config]
            assert left.keys() == right.keys()
            pairs, frontend_count = [], 0
            for task in sorted(left):
                if name == "rq2_prompt_frontend_valid_deltas.csv":
                    if not all(r["verification_stage"] in STAGE_PASS["Type validity"] and r["analysis_eligible"] == "True"
                               for r in (left[task], right[task])):
                        continue
                    frontend_count += 1
                a, b = number(left[task][row["metric"]]), number(right[task][row["metric"]])
                if np.isfinite(a) and np.isfinite(b):
                    pairs.append((a, b))
            assert len(pairs) == int(row["n_pairs"])
            assert len(left) == int(row["total_tasks"])
            equal(len(pairs) / len(left), number(row["pair_coverage"]), (name, row))
            if "both_frontend_passed_targets" in row:
                assert frontend_count == int(row["both_frontend_passed_targets"])
            if not pairs:
                assert all(not row[k] for k in ("left_mean", "right_mean", "delta", "ci_low", "ci_high"))
                continue
            a, b = np.array(pairs).T
            equal(float(a.mean()), number(row["left_mean"]), row)
            equal(float(b.mean()), number(row["right_mean"]), row)
            equal(float(a.mean() - b.mean()), number(row["delta"]), row)
            draws = rng.integers(len(a), size=(summary["bootstrap_replicates"], len(a)))
            # Resample each member of the pair with the same draws, then subtract means.
            estimates = a[draws].mean(axis=1) - b[draws].mean(axis=1)
            low, high = np.quantile(estimates, (0.025, 0.975))
            equal(float(low), number(row["ci_low"]), row)
            equal(float(high), number(row["ci_high"]), row)
        interval_checks[name] = len(rows)

    # Audit one record per configuration, verification stage and eligibility stratum.
    # The reproduction runner checks all released file hashes before analysis.
    for sample_id in audit_samples.values():
        entry, label, outcome = index[sample_id], labels[sample_id], outcomes[sample_id]
        path = ROOT / entry["result_path"]
        assert sha256(path) == entry["result_sha256"], path
        payload = json.loads(path.read_text())
        target = payload["target_evaluation"]
        assert target["sample_id"] == sample_id
        assert target["analysis_eligible"] == (label["analysis_eligible"] == "True")
        assert target["verification_stage"] == label["verification_stage"]
        for direction in FORMAL["Equivalence"]:
            assert target["directions"][direction] == outcome[direction], (sample_id, direction)
        for field, part in (("triviality_pre_false_state", "precondition_falsity"),
                            ("triviality_post_true_state", "postcondition_truth")):
            assert target["triviality"][part] == label[field]
        for prefix, node_name in IO_NODES.items():
            node = payload["metrics"][node_name]["generated"]
            for node_key, suffix in (("passed", "passed"), ("failed", "failed"), ("unknown", "unresolved")):
                assert int(node.get(node_key, 0)) == int(outcome[f"{prefix}_{suffix}"]), (sample_id, prefix, node_key)
    result = {
        "status": "complete", "summary_sha256": sha256(RESULTS / "summary.json"),
        "validator_sha256": sha256(Path(__file__)), "source_code_result_hashes_verified": True,
        "artifacts_checked": len(records), "metric_scores_checked": len(records) * len(metrics),
        "configuration_profiles_checked": len(profiles), "stage_coverage_rows_checked": len(coverage),
        "paired_intervals_checked": interval_checks, "sampled_per_file_records_checked": len(audit_samples),
        "sampled_sample_ids": list(audit_samples.values()),
        "method": "Independent score encoding from merged labels/outcomes, aggregation from artifact scores, and paired bootstrap via differences of resampled means. Stratified per-file checks cover configuration, stage and eligibility.",
    }
    (RESULTS / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(records):,} artifacts, {len(profiles)} profiles, {sum(interval_checks.values())} paired intervals and {len(audit_samples)} per-file records.")


if __name__ == "__main__":
    validate()
