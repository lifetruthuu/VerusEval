from __future__ import annotations

from metrics_rebuild.share.functions import precondition_satisfiability_for_path
from metrics_rebuild.share.proof_probe import (
    TRIVIALITY_RECOVERY_VERSION,
    probe_postcondition_truth_for_path,
    probe_precondition_falsity_for_path,
)
from metrics_rebuild.share.scoring import single_file_wrapper
from metrics_rebuild.share.smt import precondition_satisfiability
from metrics_rebuild.share.text import read_text
from metrics_rebuild.share.triviality import (
    keyword_blacklist_for_text,
    trivial_clause_rate_for_path,
    vacuous_postcondition_flags_for_path,
)


def _component_detection_state(result: dict) -> str:
    score = result.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool) and score > 0:
        return "detected"
    if result.get("status") == "ok":
        return "not_detected"
    return "undetermined"


def _precondition_satisfiability_for_path(path: str) -> dict:
    return precondition_satisfiability_for_path(path, precondition_satisfiability)


def _legacy_signals_for_path(path: str) -> dict:
    precondition = _precondition_satisfiability_for_path(path)
    blacklist = keyword_blacklist_for_text(read_text(path))
    trivial = trivial_clause_rate_for_path(path)
    ensures_true = vacuous_postcondition_flags_for_path(path)
    return {
        "status": "retained_unused",
        "reason": "superseded_by_proof_probes",
        "precondition_satisfiability": precondition,
        "keyword_blacklist_flags": blacklist,
        "trivial_clause_rate": trivial,
        "vacuous_postcondition_flags": ensures_true,
    }


def _comprehensive_triviality_for_path(path: str) -> dict:
    pre = probe_precondition_falsity_for_path(path)
    post = probe_postcondition_truth_for_path(path)
    retained = _legacy_signals_for_path(path)

    usable_scores = [
        float(score)
        for score, status in (
            (pre.get("score"), pre.get("status")),
            (post.get("score"), post.get("status")),
        )
        if isinstance(score, (int, float)) and not isinstance(score, bool) and status != "not_available"
    ]
    score = (sum(usable_scores) / len(usable_scores)) if usable_scores else None

    statuses = {pre.get("status"), post.get("status")}
    if score is None:
        status = "not_available"
    elif "partial" in statuses or "unavailable" in statuses:
        status = "partial"
    else:
        status = "ok"

    return {
        "status": status,
        "score": score,
        "metric_kind": "trivial_spec_ratio",
        "source_metric": "proof_probe_v1",
        "recovery_version": TRIVIALITY_RECOVERY_VERSION,
        "score_semantics": "badness_score_lower_is_better",
        "aggregation": "proof_probe_pre_false_post_true_equal_weight",
        "components": {
            "precondition_always_false": pre.get("score"),
            "postcondition_always_true": post.get("score"),
        },
        "component_detection_states": {
            "precondition_always_false": _component_detection_state(pre),
            "postcondition_always_true": _component_detection_state(post),
        },
        "weights": {
            "precondition_always_false": 0.5,
            "postcondition_always_true": 0.5,
        },
        "details": {
            "precondition_falsity": pre,
            "postcondition_truth": post,
            "retained_unused": retained,
        },
        "note": (
            "规约平凡性仅由两个 Verus proof fn 探针组成：前置条件永假性用 "
            "`requires P => false`，后置条件永真性用 `true => ensures` 整块判断。"
            "两者等权，分数是 badness，越低越好。无 requires 默认 true（记为非永假），"
            "无 ensures 默认 true（记为平凡）；含 old()/&mut 的后置经前态快照参数改写后正常探测"
            "（old(x) 重写为独立的前态自由变量，与永真语义一致）。"
            "未决探针自动尝试目标规约隔离、触发器修复及后置子句检查；"
            "已完成但被拒绝的证明沿用未检出计分，仍未完成的检查保留未决。"
            "旧的前置可满足性、黑名单、平凡子句率、语法永真信号保留在 details.retained_unused，"
            "不参与评分。"
        ),
    }


def metric_trivial_spec_ratio(generated_rs_path: str, ground_rs_path: str) -> dict:
    result = single_file_wrapper(generated_rs_path, ground_rs_path, _comprehensive_triviality_for_path)
    result["metric_kind"] = "trivial_spec_ratio"
    result["note"] = (
        "规约平凡性比例现在只看两个 Verus proof probe：前置条件永假和后置条件永真。"
        "分数越低越好；旧语义信号仅保留在 details.retained_unused 中供回归对照。"
    )
    return result
