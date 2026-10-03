"""Compare generation configurations on verifier validity and the quality metrics for RQ2.

Run with: conda run -n wd python RQs/RQ2/scripts/analyze_rq2_configurations.py
Reads the merged data/evaluation tables and checks overlapping scores against final RQ1;
no verifier or LLM calls. Outputs stay under RQs/RQ2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "RQs/RQ1/scripts"))
from analyze_rq1_accepted_profiles import FORMAL, formal_state, io_score, read_csv  # noqa: E402


OUTPUT = ROOT / "RQs/RQ2/results"
SOURCE = ROOT / "data/evaluation"
RQ1 = ROOT / "RQs/RQ1/results"
REPLICATES = 2_000
SEED = 20_260_816
UNJUDGED = {"parse_failed", "frontend_failed"}  # the LLM judge does not run on these artifacts
PASSED = {"syntactic": {"frontend_failed", "proof_failed", "accepted"},
          "type": {"proof_failed", "accepted"}, "acceptance": {"accepted"}}
# Rows follow Table 1. Every metric is coded so that higher is better; a metric enters a
# configuration's mean for every artifact that has a result for it.
METRICS = (
    ("Text similarity", "BLEU", "text", "text_bleu"),
    ("Text similarity", "ROUGE-L", "text", "text_rouge_l"),
    ("Text similarity", "KeySpecMatch", "text", "text_key_spec_match"),
    ("Intent judgement", "Spec-code judgement", "llm", "llm_spec_code_score"),
    ("Intent judgement", "Spec-spec judgement", "llm", "llm_intent_score"),
    ("Verifier validity", "Syntactic validity", "stage", "syntactic"),
    ("Verifier validity", "Type validity", "stage", "type"),
    ("Verifier validity", "Verifier acceptance", "stage", "acceptance"),
    ("Semantic triviality", "No false precondition", "triviality", "triviality_pre_false_state"),
    ("Semantic triviality", "No true postcondition", "triviality", "triviality_post_true_state"),
    ("Specification correctness", "Soundness", "formal", "soundness"),
    ("Specification correctness", "Completeness", "formal", "completeness"),
    ("Specification correctness", "Equivalence", "formal", "equivalence"),
    ("Behavior correctness", "Correct-I/O acceptance", "io", "io_correct"),
    ("Behavior correctness", "Wrong-output rejection", "io", "io_wrong"),
    ("Behavior correctness", "Invalid-input rejection", "io", "io_invalid"),
)


def metric_value(row: dict[str, str], outcome: dict[str, str], kind: str, key: str) -> float:
    stage = row["verification_stage"]
    if kind == "stage":
        return float(stage in PASSED[key])
    if row["analysis_eligible"] != "True":
        return np.nan
    if kind == "text":
        return float(row[key]) if row[key] else np.nan
    if kind == "llm":
        return np.nan if stage in UNJUDGED or not row[key] else float(row[key])
    if kind == "triviality":
        return {"favorable": 1.0, "unfavorable": 0.0}.get(row[key], np.nan)
    if kind == "formal":
        return {"pass": 1.0, "fail": 0.0}.get(formal_state(outcome, FORMAL[key]), np.nan)
    score = io_score(outcome, key)
    return np.nan if score is None else score


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: "" if isinstance(v, float) and not np.isfinite(v) else v
                         for k, v in row.items()} for row in rows)


def path_name(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    keyed = {row["sample_id"]: row for row in rows}
    assert len(keyed) == len(rows), f"Duplicate sample_id in {path}"
    return keyed


def paired_contrast(left: dict[str, float], right: dict[str, float],
                    rng: np.random.Generator, replicates: int) -> dict:
    assert left.keys() == right.keys(), "Configurations do not cover the same task catalog"
    pairs = np.array([(left[task], right[task]) for task in sorted(left)])
    pairs = pairs[np.isfinite(pairs).all(axis=1)]
    diff = pairs[:, 0] - pairs[:, 1]
    low = high = np.nan
    if diff.size:
        boot = diff[rng.integers(len(diff), size=(replicates, len(diff)))].mean(axis=1)
        low, high = np.quantile(boot, (0.025, 0.975))
    return {
        "delta": float(diff.mean()) if diff.size else np.nan,
        "ci_low": float(low), "ci_high": float(high), "n_pairs": len(diff),
        "left_mean": float(pairs[:, 0].mean()) if diff.size else np.nan,
        "right_mean": float(pairs[:, 1].mean()) if diff.size else np.nan,
        "total_tasks": len(left), "pair_coverage": len(diff) / len(left),
    }


def verify_rq1(encoded: list[dict]) -> int:
    summary = json.loads((RQ1 / "summary.json").read_text())
    assert summary["status"] == "complete"
    for name, expected in {**summary["source_sha256"], **summary["result_sha256"]}.items():
        assert sha256(ROOT / name) == expected, f"RQ1 input/result changed: {name}"
    rq1 = unique_rows(RQ1 / "rq1_acceptance_artifacts.csv")
    overlap = {row["sample_id"]: row for row in encoded
               if row["analysis_eligible"] == "True" and row["verification_stage"] in ("accepted", "proof_failed")}
    assert overlap.keys() == rq1.keys()
    for sample_id, row in overlap.items():
        other = rq1[sample_id]
        assert row["task_id"] == other["task_id"]
        assert int(row["verification_stage"] == "accepted") == int(other["accepted"])
        for _, name, kind, _ in METRICS:
            if kind != "stage":
                expected = float(other[name]) if other[name] else np.nan
                assert np.isclose(row[name], expected, rtol=0, atol=1e-12, equal_nan=True), (sample_id, name)
    return len(overlap)


def main(source: Path = SOURCE, output: Path = OUTPUT, replicates: int = REPLICATES, seed: int = SEED) -> None:
    validation_path = source / "provenance/validation.json"
    validation = json.loads(validation_path.read_text())
    assert validation["status"] == "complete"
    labels = unique_rows(source / "artifact_labels.csv")
    outcomes = unique_rows(source / "artifact_outcomes.csv")
    index = unique_rows(source / "artifact_index.csv")
    eligible = unique_rows(source / "rq1_artifact_labels.csv")
    eligible_outcomes = unique_rows(source / "rq1_artifact_outcomes.csv")
    exclusions = unique_rows(source / "provenance/excluded_missing_target.csv")
    assert labels.keys() == outcomes.keys() == index.keys()
    assert len(labels) == validation["counts"]["records"]
    assert eligible.keys() == eligible_outcomes.keys() == labels.keys() - exclusions.keys()
    assert len(exclusions) == validation["excluded_from_analysis"]
    artifacts = [labels[key] for key in sorted(labels)]
    # values[(workflow, model, shot)][metric][task] = value
    values: dict[tuple[str, str, str], dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    encoded = []
    config_rows = defaultdict(list)
    for row in artifacts:
        sample_id = row["sample_id"]
        outcome = outcomes[sample_id]
        assert (row["task_id"], row["verification_stage"]) == (outcome["task_id"], outcome["verification_stage"])
        assert row["verification_stage"] in {"parse_failed", "frontend_failed", "proof_failed", "accepted"}
        assert row["analysis_eligible"] == index[sample_id]["analysis_eligible"]
        assert (row["analysis_eligible"] == "True") == (sample_id in eligible)
        if sample_id in eligible:
            assert row == eligible[sample_id] and outcome == eligible_outcomes[sample_id]
        assert all(int(outcome[k + "_skipped"]) == 0 for k in ("io_correct", "io_wrong", "io_invalid"))
        config = (row["workflow"], row["model"], row["shot"])
        record = {k: row[k] for k in ("sample_id", "workflow", "model", "shot", "task_id",
                                      "verification_stage", "analysis_eligible")}
        record["exclusion_reason"] = "missing_executable_target" if sample_id in exclusions else ""
        assert row["task_id"] not in values[config]["BLEU"], (config, row["task_id"])
        for _, name, kind, key in METRICS:
            value = metric_value(row, outcome, kind, key)
            assert np.isnan(value) or 0 <= value <= 1, (sample_id, name, value)
            values[config][name][row["task_id"]] = record[name] = value
        encoded.append(record)
        config_rows[config].append(record)
    task_ids = {r["task_id"] for r in artifacts}
    assert len(values) == validation["configurations"]
    assert all(set(v["BLEU"]) == task_ids for v in values.values())
    overlap = verify_rq1(encoded)

    profiles, coverage = [], []
    for (workflow, model, shot), metrics in sorted(values.items()):
        members = config_rows[(workflow, model, shot)]
        target_count = sum(r["analysis_eligible"] == "True" for r in members)
        for dimension, name, kind, _ in METRICS:
            scores = np.array(list(metrics[name].values()))
            scored = scores[np.isfinite(scores)]
            population = len(scores) if kind == "stage" else target_count
            identity = {"workflow": workflow, "model": model, "shot": shot, "dimension": dimension, "metric": name}
            profiles.append({**identity, "mean": float(scored.mean()) if scored.size else np.nan,
                             "n": int(scored.size), "total": len(scores), "eligible": population,
                             "excluded_missing_target": len(scores) - population,
                             "missing_score": population - len(scored),
                             "coverage": len(scored) / population if population else np.nan})
            for stage in ("parse_failed", "frontend_failed", "proof_failed", "accepted"):
                stage_rows = [r for r in members if r["verification_stage"] == stage]
                stage_scores = [r[name] for r in stage_rows if np.isfinite(r[name])]
                coverage.append({**identity, "verification_stage": stage,
                                 "total": len(stage_rows), "n": len(stage_scores),
                                 "mean": float(np.mean(stage_scores)) if stage_scores else np.nan})

    # Few-shot minus zero-shot on the same task, workflow, and model, with paired task-bootstrap intervals.
    rng = np.random.default_rng(seed)
    deltas = []
    for workflow, model in sorted({(w, m) for w, m, _ in values}):
        few, zero = values[(workflow, model, "few-shot")], values[(workflow, model, "zero-shot")]
        for dimension, name, _, _ in METRICS:
            deltas.append({"workflow": workflow, "model": model, "dimension": dimension, "metric": name,
                           **paired_contrast(few[name], zero[name], rng, replicates)})

    # Supplement the displayed configuration means with contrasts on shared scored tasks.
    comparisons = [("workflow", ("StarVerus", model, "few-shot"), (workflow, model, "few-shot"))
                   for workflow, model in sorted({(w, m) for w, m, _ in values if w != "StarVerus"})]
    star_models = sorted({m for w, m, _ in values if w == "StarVerus"})
    comparisons += [("model", ("StarVerus", left, "few-shot"), ("StarVerus", right, "few-shot"))
                    for left, right in combinations(star_models, 2)]
    rng = np.random.default_rng(seed + 1)
    matched = []
    for factor, left, right in comparisons:
        for dimension, name, _, _ in METRICS:
            matched.append({"factor": factor,
                            **dict(zip(("left_workflow", "left_model", "left_shot"), left)),
                            **dict(zip(("right_workflow", "right_model", "right_shot"), right)),
                            "dimension": dimension, "metric": name,
                            **paired_contrast(values[left][name], values[right][name], rng, replicates)})

    # I/O sensitivity: both prompts must pass frontend checking and have a target.
    # The main analysis still includes all eligible artifacts with an I/O score.
    rng = np.random.default_rng(seed + 2)
    frontend_deltas = []
    for workflow, model in sorted({(w, m) for w, m, _ in values}):
        few_config, zero_config = (workflow, model, "few-shot"), (workflow, model, "zero-shot")
        common = set.intersection(*(
            {r["task_id"] for r in config_rows[c] if r["verification_stage"] in PASSED["type"]
             and r["analysis_eligible"] == "True"} for c in (few_config, zero_config)
        ))
        for dimension, name, kind, _ in METRICS:
            if kind != "io":
                continue
            few, zero = ({t: v if t in common else np.nan for t, v in values[c][name].items()}
                         for c in (few_config, zero_config))
            frontend_deltas.append({"workflow": workflow, "model": model, "dimension": dimension,
                                    "metric": name, "both_frontend_passed_targets": len(common),
                                    **paired_contrast(few, zero, rng, replicates)})

    outputs = {"rq2_artifact_metrics.csv": encoded, "rq2_configuration_profiles.csv": profiles,
               "rq2_metric_coverage.csv": coverage, "rq2_prompt_deltas.csv": deltas,
               "rq2_matched_deltas.csv": matched,
               "rq2_prompt_frontend_valid_deltas.csv": frontend_deltas}
    for name, rows in outputs.items():
        write_csv(output / name, rows)
    source_paths = [source / name for name in ("artifact_labels.csv", "artifact_outcomes.csv", "artifact_index.csv",
                    "rq1_artifact_labels.csv", "rq1_artifact_outcomes.csv", "provenance/validation.json",
                    "provenance/excluded_missing_target.csv")]
    source_paths += [RQ1 / "summary.json", RQ1 / "rq1_acceptance_artifacts.csv"]
    code_paths = [Path(__file__).resolve(), ROOT / "RQs/RQ1/scripts/analyze_rq1_accepted_profiles.py"]
    summary = {
        "status": "complete", "source_root": path_name(source), "artifacts": len(artifacts),
        "tasks": len(task_ids), "configurations": len(values), "quality_eligible": len(eligible),
        "excluded_missing_target": len(exclusions),
        "verification_stages": dict(Counter(r["verification_stage"] for r in artifacts)),
        "eligible_verification_stages": dict(Counter(r["verification_stage"] for r in eligible.values())),
        "rq1_overlap_artifacts": overlap, "rq1_overlap_metric_scores_checked": overlap * 13,
        "profile_rows": len(profiles), "prompt_rows": len(deltas), "matched_rows": len(matched),
        "bootstrap_replicates": replicates, "bootstrap_seed": seed, "matched_bootstrap_seed": seed + 1,
        "frontend_sensitivity_bootstrap_seed": seed + 2,
        "verus_evidence_release": validation["verus_evidence_release"],
        "policy": {
            "verifier_validity": "All original tasks, including artifacts missing the executable target.",
            "quality": "Exclude missing executable targets; average each metric over every remaining artifact with a score, including earlier verification failures where scored.",
            "llm": "Parse/frontend failures have no judgement; stored zeros for these stages remain missing. Later-stage scores use the same encoding as final RQ1.",
            "formal": "Target directions without precondition premises for postconditions. Any rejected required implication gives zero; all proved gives one; otherwise missing. Rejection alone is not a counterexample.",
            "triviality": "Target-function probes, scored as in final RQ1; a rejected probe does not prove non-triviality.",
            "io": "Passed / (passed + failed + unresolved) over all validated cases; empty categories missing. Wrong-output requires admitted input and rejected output.",
            "evidence": "Merged v5, including the supplementary conclusions already used in final RQ1; no historical overlays reapplied.",
            "pairing": "Match by workflow/model/shot and task_id; retain tasks with both scores for each metric; percentile task-bootstrap intervals.",
            "frontend_sensitivity": "I/O prompt contrasts restricted to tasks where both prompts pass frontend checking and have an executable target; this conditional subset is supplementary.",
            "provenance": "Join rq2_artifact_metrics.csv to the hashed source artifact_index.csv by sample_id for per-file paths and hashes.",
        },
        "source_sha256": {path_name(p): sha256(p) for p in source_paths},
        "code_sha256": {path_name(p): sha256(p) for p in code_paths},
        "result_sha256": {path_name(output / name): sha256(output / name) for name in outputs},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(profiles)} profiles, {len(deltas)} prompt contrasts and {len(matched)} matched contrasts.")
    print(f"Checked all 13 quality scores against final RQ1 for {overlap:,} artifacts; results: {path_name(output)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    main(args.source_root.resolve(), args.output_dir.resolve(), args.replicates, args.seed)
