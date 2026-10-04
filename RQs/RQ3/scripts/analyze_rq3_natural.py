"""Conditional I/O/reference results on the current RQ1 accepted population."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from rq3_io_common import ROOT, RESULTS, file_hash, read_csv, write_csv, write_json

FORMAL = {
    "soundness": ("pre_ref_to_gen", "post_gen_to_ref"),
    "completeness": ("pre_gen_to_ref", "post_ref_to_gen"),
    "equivalence": ("pre_ref_to_gen", "post_gen_to_ref", "pre_gen_to_ref", "post_ref_to_gen"),
}
IO = ("io_correct", "io_wrong", "io_invalid")


def formal_state(row, directions):
    values = [row[k] for k in directions]
    if "invalid" in values:
        return "rejected"
    return "proved" if all(v == "valid" for v in values) else "unresolved"


def profile(outcome, label):
    counts = {k: tuple(int(outcome[k + "_" + s]) for s in ("passed", "failed", "unresolved")) for k in IO}
    passed, failed, unknown = (sum(c[i] for c in counts.values()) for i in range(3))
    state = ("failed" if failed else "passed" if passed and not unknown
             else "unresolved" if unknown else "unavailable")
    row = {k: label[k] for k in ("sample_id", "task_id", "workflow", "model", "shot")}
    row.update(io_state=state, available_io_kinds=sum(sum(c) > 0 for c in counts.values()),
               io_passed_cases=passed, io_failed_cases=failed, io_unresolved_cases=unknown)
    row.update({k: formal_state(outcome, directions) for k, directions in FORMAL.items()})
    return row


def summarize(rows, group, scope):
    denominator = len(rows)
    passing = [r for r in rows if r["io_state"] == "passed"]
    result = {**group, "scope": scope, "accepted": denominator,
              "io_passed": len(passing), "io_failed": sum(r["io_state"] == "failed" for r in rows),
              "io_unresolved": sum(r["io_state"] == "unresolved" for r in rows),
              "io_unavailable": sum(r["io_state"] == "unavailable" for r in rows),
              "io_pass_rate": len(passing) / denominator if denominator else None}
    for metric in FORMAL:
        counts = Counter(r[metric] for r in passing)
        for status in ("proved", "rejected", "unresolved"):
            result[f"{metric}_{status}_given_io_pass"] = counts[status]
        result[f"{metric}_proved_rate_given_io_pass"] = counts["proved"] / len(passing) if passing else None
    return result


def analyze(source=ROOT / "data/evaluation", output=RESULTS / "natural"):
    source, output = Path(source), Path(output)
    paths = [source / name for name in ("rq1_artifact_outcomes.csv", "rq1_artifact_labels.csv")]
    outcomes = read_csv(paths[0])
    labels = {r["sample_id"]: r for r in read_csv(paths[1])}
    if len(labels) != len(outcomes) or len(outcomes) != 13659 or {r["sample_id"] for r in outcomes} != set(labels):
        raise ValueError("Input IDs are not unique/aligned")
    profiles = [profile(r, labels[r["sample_id"]]) for r in outcomes if r["verification_stage"] == "accepted"]
    # Check the definition against the completed RQ1, including incomplete checks.
    rq1_path = ROOT / "RQs/RQ1/results/screening/accepted_profiles.csv"
    rq1 = {r["sample_id"]: r for r in read_csv(rq1_path)}
    if set(rq1) != {r["sample_id"] for r in profiles}:
        raise ValueError("RQ1 population differs")
    mapping = {"pass": "passed", "fail": "failed", "none": "unresolved"}
    formal_mapping = {"pass": "proved", "fail": "rejected", "unresolved": "unresolved"}
    for r in profiles:
        old = rq1[r["sample_id"]]
        expected = "unavailable" if old["io"] == "none" and not r["available_io_kinds"] else mapping[old["io"]]
        assert r["io_state"] == expected, r["sample_id"]
        assert all(r[k] == formal_mapping[old[k]] for k in FORMAL), r["sample_id"]
    groups = defaultdict(list)
    groups[("ALL", "ALL", "ALL")] = profiles
    for r in profiles:
        groups[(r["workflow"], r["model"], r["shot"])].append(r)
    summaries, cross = [], []
    for key, group in sorted(groups.items()):
        metadata = dict(zip(("workflow", "model", "shot"), key))
        for scope, subset in (
            ("available_cases_rq1", group),
            ("all_three_nonempty", [r for r in group if r["available_io_kinds"] == 3]),
        ):
            row = summarize(subset, metadata, scope)
            row["coverage_of_accepted"] = len(subset) / len(group)
            summaries.append(row)
            for (io, formal), n in sorted(Counter((r["io_state"], r["equivalence"]) for r in subset).items()):
                cross.append({**metadata, "scope": scope, "io_state": io, "equivalence": formal, "n": n})
    # Same-task accepted pairs: descriptive condition differences, not causal effects.
    by_config = {key: {r["task_id"]: r for r in group} for key, group in groups.items() if key[0] != "ALL"}
    contrasts = []
    keys = sorted(by_config)
    pairs = []
    for left in keys:
        for right in keys:
            if left >= right:
                continue
            if left[:2] == right[:2] and left[2] != right[2]:
                l, r = (left, right) if left[2] == "few-shot" else (right, left)
                pairs.append(("prompting", l, r))
            elif left[1:] == right[1:] and left[2] == "few-shot" and "StarVerus" in (left[0], right[0]):
                l, r = (left, right) if left[0] == "StarVerus" else (right, left)
                pairs.append(("workflow", l, r))
            elif left[0] == right[0] == "StarVerus" and left[2] == right[2] == "few-shot":
                pairs.append(("model", left, right))
    for factor, left, right in pairs:
        shared = sorted(set(by_config[left]) & set(by_config[right]))
        for scope in ("io_determinate_both", "io_passed_both_formal_determinate"):
            tasks = [t for t in shared if (
                all(by_config[k][t]["io_state"] in ("passed", "failed") for k in (left, right))
                if scope == "io_determinate_both" else
                all(by_config[k][t]["io_state"] == "passed" and by_config[k][t]["equivalence"] != "unresolved" for k in (left, right))
            )]
            field, good = ("io_state", "passed") if scope == "io_determinate_both" else ("equivalence", "proved")
            n_left = sum(by_config[left][t][field] == good for t in tasks)
            n_right = sum(by_config[right][t][field] == good for t in tasks)
            contrasts.append({"factor": factor, "left": "|".join(left), "right": "|".join(right),
                              "scope": scope, "both_accepted": len(shared), "paired_tasks": len(tasks),
                              "left_success": n_left, "right_success": n_right,
                              "paired_difference": (n_left - n_right) / len(tasks) if tasks else None})
    write_csv(output / "accepted_profiles.csv", profiles)
    write_csv(output / "conditional_rates.csv", summaries)
    write_csv(output / "io_reference_crosstab.csv", cross)
    write_csv(output / "matched_configuration_differences.csv", contrasts)
    overall = [r for r in summaries if r["workflow"] == "ALL"]
    summary = {"status": "complete", "source_root": str(source.relative_to(ROOT)), "accepted": len(profiles),
               "tasks": len({r["task_id"] for r in profiles}), "overall": overall,
               "rq1_states_checked": len(profiles), "configuration_comparisons": len(contrasts),
               "policy": "Primary I/O pass matches RQ1: all available cases pass, no unresolved; empty kinds allowed. Complete three-kind scope is separate. Proof rejection is not a counterexample. Matched differences are descriptive.",
               "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in [*paths, rq1_path]},
               "code_sha256": file_hash(Path(__file__))}
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/evaluation")
    parser.add_argument("--output", type=Path, default=RESULTS / "natural")
    args = parser.parse_args()
    result = analyze(args.source, args.output)
    print({"accepted": result["accepted"], "overall": result["overall"]})
