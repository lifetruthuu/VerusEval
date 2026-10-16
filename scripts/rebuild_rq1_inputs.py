"""Rebuild RQ1 input tables from the released per-file JSON records.

Reads artifact_index.csv and target_functions.csv for record locations and target
metadata. No existing labels/outcomes CSV, historical input, verifier, or LLM is
used. Final target_evaluation states take precedence over legacy metric nodes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ("pre_ref_to_gen", "post_gen_to_ref", "pre_gen_to_ref", "post_ref_to_gen")
IO = {"io_correct": "correct_io_pass_rate", "io_wrong": "wrong_io_reject_rate",
      "io_invalid": "invalid_test_filtering_rate"}
LABEL_COLUMNS = (
    "sample_id", "workflow", "model", "shot", "task_id", "verified_side",
    "formal_reliability_state", "formal_completeness_state", "formal_state",
    "io_correct_state", "io_wrong_state", "io_invalid_state", "llm_intent_state",
    "llm_spec_code_state", "triviality_pre_false_state", "triviality_post_true_state",
    "mutation_state", "redundancy_state", "verification_state", "io_state", "triviality_state",
    "verification_stage", "triviality_form", "llm_intent_verdict", "llm_intent_advisory",
    "llm_spec_code_verdict", "llm_spec_code_status", "mutation_valid_mutants", "mutation_status",
    "redundancy_removable", "redundancy_status", "text_bleu", "text_rouge_l", "text_key_spec_match",
    "io_correct_score", "io_wrong_score", "io_invalid_score", "llm_intent_score", "llm_spec_code_score",
    "mutation_score", "redundancy_score", "redundancy_rate", "formal_source", "formal_unknown_dirs",
    "formal_invalid_dirs", "io_correct_status", "io_wrong_status", "io_invalid_status",
    "spec_node_count", "spec_status", "proof_node_count", "proof_status", "verification_time_seconds",
    "analysis_eligible",
)
OUTCOME_COLUMNS = ("sample_id", "task_id", "verification_stage", *DIRECTIONS,
                   *(f"{kind}_{state}" for kind in IO for state in ("passed", "failed", "unresolved", "skipped")))
OUTPUT_NAMES = ("rq1_artifact_labels.csv", "rq1_artifact_outcomes.csv")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite metric value: {value}")
    return value


def unit(value):
    value = number(value)
    if value is not None:
        if not -1e-12 <= value <= 1 + 1e-12:
            raise ValueError(f"Metric score outside [0, 1]: {value}")
        return min(1.0, max(0.0, value))
    return None


def generated(metrics, name):
    node = metrics[name]
    return node.get("generated", node)


def combine(states, positive="favorable", negative="unfavorable"):
    if negative in states:
        return "unfavorable"
    return "favorable" if all(s == positive for s in states) else "other"


def intent_labels(metric, catalog):
    node, raw = metric.get("generated", {}), metric.get("raw_judge", {})
    verdict = str(node.get("verdict") or raw.get("verdict") or "unknown").lower()
    if verdict == "over_strong":
        verdict = "too_strong"
    alignment = metric.get("target_alignment", {})
    aligned = alignment.get("status") == "ok" and alignment.get("function") == catalog["target_function"]
    completed = (metric.get("status") == node.get("status") == metric.get("llm", {}).get("status") == "ok"
                 and (catalog["selection"] != "manual_source_audit" or aligned))
    score = unit(metric.get("score"))
    state = "other"
    if completed and verdict == "equivalent" and score is not None and abs(score - 1) <= 1e-12:
        state = "favorable"
    elif completed and verdict in {"missing_constraints", "too_strong", "too_weak", "incomparable", "incorrect"}:
        state = "unfavorable"
    advisory = False
    for key in ("missing_constraints", "extra_or_overstrong_constraints", "vacuity_risks"):
        value = node.get(key, raw.get(key))
        advisory |= (any(str(v).strip() for v in value) if isinstance(value, list)
                     else value is not None and bool(str(value).strip()))
    return {"llm_intent_state": state, "llm_intent_verdict": verdict,
            "llm_intent_score": score, "llm_intent_advisory": advisory}


def extract(payload, result_path, catalog):
    target, metrics = payload["target_evaluation"], payload["metrics"]
    sample = target["sample_id"]
    workflow, model, shot, task = sample.split("|")
    stage = target["verification_stage"]
    if stage not in {"parse_failed", "frontend_failed", "proof_failed", "accepted"}:
        raise ValueError(f"Unknown verification stage: {sample}: {stage}")
    directions = target["directions"]
    if any(directions[k] not in {"valid", "invalid", "unknown", "unavailable"} for k in DIRECTIONS):
        raise ValueError(f"Invalid formal direction: {sample}")
    outcome = {"sample_id": sample, "task_id": task, "verification_stage": stage,
               **{k: directions[k] for k in DIRECTIONS}}
    label = {"sample_id": sample, "workflow": workflow, "model": model, "shot": shot, "task_id": task,
             # This records the generation directory, independently of final verification_stage.
             "verified_side": Path(result_path).parent.parent.name == "verified",
             "verification_stage": stage, "verification_state": "favorable" if stage == "accepted" else "unfavorable",
             "analysis_eligible": target["analysis_eligible"], "formal_source": "data/evaluation/target_evaluation",
             "formal_unknown_dirs": sum(directions[k] not in {"valid", "invalid"} for k in DIRECTIONS),
             "formal_invalid_dirs": sum(directions[k] == "invalid" for k in DIRECTIONS)}
    for key, keys in (("formal_reliability_state", DIRECTIONS[:2]),
                      ("formal_completeness_state", DIRECTIONS[2:]), ("formal_state", DIRECTIONS)):
        label[key] = combine([directions[k] for k in keys], "valid", "invalid")
    for kind, name in IO.items():
        node = generated(metrics, name)
        counts = [int(node.get(key, 0)) for key in ("passed", "failed", "unknown")]
        if min(counts) < 0 or sum(counts) != int(node.get("total", 0)):
            raise ValueError(f"Inconsistent I/O counts: {sample}: {kind}")
        passed, failed, unknown = counts
        outcome.update({f"{kind}_{key}": value for key, value in
                        zip(("passed", "failed", "unresolved", "skipped"), (*counts, 0))})
        label[f"{kind}_state"] = "unfavorable" if failed else "favorable" if passed and not unknown else "other"
        label[f"{kind}_score"], label[f"{kind}_status"] = node.get("score"), node.get("status")
    label["io_state"] = ("unfavorable" if any(label[f"{k}_state"] == "unfavorable" for k in IO) else
                         "favorable" if sum(outcome[f"{k}_passed"] for k in IO)
                         and not sum(outcome[f"{k}_unresolved"] for k in IO) else "other")
    states = target["triviality"]
    pre, post = states["precondition_falsity"], states["postcondition_truth"]
    if any(s not in {"favorable", "unfavorable", "other"} for s in (pre, post)):
        raise ValueError(f"Invalid triviality state: {sample}")
    label.update(triviality_pre_false_state=pre, triviality_post_true_state=post,
                 triviality_state=combine([pre, post]))
    spec, proof = metrics["spec_size_complexity_spec_only"], metrics["spec_size_complexity_proof"]
    forms = [name for name, state in (("pre_false", pre), ("post_true", post)) if state == "unfavorable"]
    if spec.get("gen", {}).get("breakdown", {}).get("contract_clauses") == 0:
        forms.append("empty_contract")
    label["triviality_form"] = ";".join(forms) or "none"
    for component in ("bleu", "rouge_l", "key_spec_match"):
        label[f"text_{component}"] = number(metrics["verus_textual_similarity_spec_only"]["components"][component]["score"])
    label.update(intent_labels(metrics["llm_as_judge_intent_consistency"], catalog))
    judge = metrics["llm_as_judge_spec_code_intent_consistency"].get("generated", {})
    verdict, score = judge.get("verdict"), unit(judge.get("score"))
    status = judge.get("llm", {}).get("status") or "missing"
    completed = status == "ok" and bool(verdict) and score is not None
    label.update(llm_spec_code_state=("favorable" if verdict == "aligned" else "unfavorable") if completed else "other",
                 llm_spec_code_verdict=verdict or "unknown", llm_spec_code_status=status, llm_spec_code_score=score)
    mutation = generated(metrics, "mutation_kill_rate")
    valid = int(mutation.get("valid_mutants", 0))
    status = mutation.get("status") or "missing"
    eligible = status in {"ok", "partial"} and valid > 0
    label.update(mutation_state=("unfavorable" if int(mutation.get("survived_mutants", 0)) else
                                 "other" if int(mutation.get("unknown_mutants", 0)) else "favorable") if eligible else "other",
                 mutation_valid_mutants=valid, mutation_status=status, mutation_score=unit(mutation.get("score")))
    redundancy = generated(metrics, "spec_redundancy_rate")
    removable = int(redundancy.get("removable_clause_candidates", 0))
    status, rate = redundancy.get("status") or "missing", unit(redundancy.get("score"))
    label.update(redundancy_state="unfavorable" if removable else "favorable" if status in {"ok", "no_testable_clauses"} else "other",
                 redundancy_removable=removable, redundancy_status=status, redundancy_rate=rate,
                 redundancy_score=None if rate is None else 1.0 - rate)
    for prefix, metric in (("spec", spec), ("proof", proof)):
        label[f"{prefix}_node_count"] = metric.get("gen", {}).get("node_count")
        label[f"{prefix}_status"] = metric.get("status") or "missing"
    timing = generated(metrics, "verification_time")
    label["verification_time_seconds"] = number(timing.get("elapsed_seconds", timing.get("score")))
    return label, outcome


def rebuild(source_root, output_dir):
    source_root, output_dir = source_root.resolve(), output_dir.resolve()
    for name in OUTPUT_NAMES:
        path = output_dir / name
        if path.exists() or path.is_symlink():
            raise ValueError(f"Output already exists; choose a new output directory: {path}")
    index = read_csv(source_root / "data/evaluation/artifact_index.csv")
    catalog = {r["task_id"]: r for r in read_csv(source_root / "data/evaluation/target_functions.csv")}
    labels, outcomes, seen = [], [], set()
    for entry in index:
        sample = entry["sample_id"]
        if sample in seen:
            raise ValueError(f"Duplicate sample ID: {sample}")
        seen.add(sample)
        relative = Path(entry["result_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parent.name != "per_file":
            raise ValueError(f"Invalid per-file path: {relative}")
        payload = json.loads((source_root / relative).read_text(encoding="utf-8"))
        target = payload["target_evaluation"]
        if (target["sample_id"] != sample or type(target["analysis_eligible"]) is not bool
                or str(target["analysis_eligible"]) != entry["analysis_eligible"]):
            raise ValueError(f"Index and JSON disagree: {sample}")
        if not target["analysis_eligible"]:
            continue
        try:
            label, outcome = extract(payload, relative, catalog[sample.split("|")[-1]])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Cannot extract {sample}: {error}") from error
        labels.append(label)
        outcomes.append(outcome)
    if not labels:
        raise ValueError("No analysis-eligible records found.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, fields, rows in zip(OUTPUT_NAMES, (LABEL_COLUMNS, OUTCOME_COLUMNS), (labels, outcomes)):
        with (output_dir / name).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    report = {"input_records": len(index), "analysis_eligible": len(labels), "excluded": len(index) - len(labels),
              "tasks": len({r["task_id"] for r in labels}), "stages": dict(Counter(r["verification_stage"] for r in labels))}
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT, help="Root of the extracted release.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the two new CSV files.")
    args = parser.parse_args()
    try:
        rebuild(args.source_root, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"RQ1 input rebuild failed: {error}\n")


if __name__ == "__main__":
    main()
