#!/usr/bin/env python3
"""
VerusEval 评估脚本

支持两种模式：
  模式1：评估单个文件对
    python scripts/evaluation/evaluate.py file --generated gen.rs --reference ref.rs -o output_dir

  模式2：评估目录对（文件名一一对应）
    python scripts/evaluation/evaluate.py dir --generated gen_dir/ --reference ref_dir/ -o output_dir --workers 10

评估结果输出到指定目录，包含：
  - summary.json          总体结果摘要
  - per_file/             每个文件的详细评估结果（完整 JSON）
  - scores.csv            所有文件 × 所有指标的分数矩阵
  - statistics.json       汇总统计（均值、中位数、分布）
  - deficit_analysis.json 缺陷分析（RQ1 类型）
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 派生规则指标不在 METRIC_CATALOG 中，只在批量评估和 Web UI 里计算。
DERIVED_RULE_METRIC_LABELS = {
    "precondition_reliability_rate": "前置条件可靠率",
    "postcondition_reliability_rate": "后置条件可靠率",
    "rule_reliability_rate": "规则可靠率",
    "approx_rule_reliability_rate": "规则近似可靠率",
    "precondition_completeness_rate": "前置条件完备率",
    "postcondition_completeness_rate": "后置条件完备率",
    "rule_completeness_rate": "规则完备率",
    "approx_rule_completeness_rate": "规则近似完备率",
    "rule_correctness_rate": "规则正确率",
    "approx_rule_correctness_rate": "规则近似正确率",
}

# 已停用的指标名，保留仅为了给更早生成的结果目录画图时不退化成裸 ID。
LEGACY_METRIC_LABELS = {
    "invariant_counterexample_reject_rate": "不变式反例拒绝率",
    "abstract_state_invariant_coverage": "抽象状态属性覆盖率",
    "runtime_safety_alarm_coverage": "安全性质覆盖率",
    "path_behavior_coverage": "路径行为覆盖率",
    "bug_detection_rate": "Bug检测率",
    "spec_houdini_redundancy_rate": "最小证明集冗余率",
    "spec_clause_redundancy_rate": "规约冗余率",
    "gt_edit_distance": "与GT的编辑距离",
    "generation_time": "生成时间",
    "manual_verification": "人工验证",
    "spec_self_consistency": "规约本身一致性",
    "vacuity_risk": "规约真空性",
    "strength_classification": "规约强度比较",
    "nav_assertion_pass_rate": "NAV断言通过率",
    "counterexample_coverage": "反例覆盖率",
}

def metric_plot_labels() -> Dict[str, str]:
    """基础指标的展示名以 metrics_rebuild/catalog.py 为唯一来源，避免与 v3 图表标签漂移。

    这里沿用本文件对 metrics_rebuild 的惰性导入约定，避免 --help 等路径付出注册表导入成本。
    """
    from metrics_rebuild.catalog import METRIC_CATALOG

    return {
        **LEGACY_METRIC_LABELS,
        **DERIVED_RULE_METRIC_LABELS,
        **{entry.metric_id: entry.display_name for entry in METRIC_CATALOG},
    }

RULE_SOURCE_METRIC_NAMES = (
    "precondition_clause_reliability_rate",
    "postcondition_clause_reliability_rate",
    "precondition_clause_completeness_rate",
    "postcondition_clause_completeness_rate",
)

DERIVED_RULE_METRIC_NAMES = (
    "precondition_reliability_rate",
    "postcondition_reliability_rate",
    "rule_reliability_rate",
    "approx_rule_reliability_rate",
    "precondition_completeness_rate",
    "postcondition_completeness_rate",
    "rule_completeness_rate",
    "approx_rule_completeness_rate",
    "rule_correctness_rate",
    "approx_rule_correctness_rate",
)


def metric_names_with_derived_rule_metrics(base_metric_names: List[str]) -> List[str]:
    names = list(base_metric_names)
    for name in DERIVED_RULE_METRIC_NAMES:
        if name not in names:
            names.append(name)
    return names


def _load_verus_path_from_config() -> Optional[str]:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        return None
    try:
        for line in config_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            if key.strip().lower() == "verus_path":
                path = value.strip().strip("\"' ")
                return path if path else None
    except OSError:
        pass
    return None


def evaluate_single(gen_path: str, ref_path: str, verus_bin: Optional[str] = None) -> Dict[str, Any]:
    import metrics_rebuild as metrics
    if verus_bin:
        metrics.set_verus_binary(verus_bin)

    result = {
        "generated_path": gen_path,
        "reference_path": ref_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": {},
        "errors": [],
    }

    for metric_fn in metrics.METRIC_FUNCTIONS:
        name = metric_fn.__name__.replace("metric_", "")
        try:
            start = time.monotonic()
            raw = metric_fn(gen_path, ref_path)
            elapsed = time.monotonic() - start
            raw["_elapsed_seconds"] = round(elapsed, 3)
            result["metrics"][name] = raw
        except Exception as exc:
            result["metrics"][name] = {
                "status": "exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "_elapsed_seconds": 0,
            }
            result["errors"].append({"metric": name, "error": f"{type(exc).__name__}: {exc}"})

    add_derived_rule_metrics(result)
    return result


def _extract_score(metric_data: dict) -> Optional[float]:
    if "generated" in metric_data and isinstance(metric_data["generated"], dict):
        return metric_data["generated"].get("score")
    return metric_data.get("score")


def _extract_status(metric_data: dict) -> str:
    if "generated" in metric_data and isinstance(metric_data["generated"], dict):
        return metric_data["generated"].get("status", "unknown")
    return metric_data.get("status", "unknown")


def _metric_score_status(metrics: Dict[str, Any], name: str) -> tuple[Optional[float], str]:
    raw = metrics.get(name, {})
    if not isinstance(raw, dict):
        return None, "not_available"
    score = _extract_score(raw)
    status = _extract_status(raw)
    return score, status


def _combine_status(statuses: List[str]) -> str:
    if any(status == "exception" for status in statuses):
        return "exception"
    if any(status == "partial" for status in statuses):
        return "partial"
    if any(status in ("not_available", "unknown", None) for status in statuses):
        return "not_available"
    return "ok"


def _binary_rule_metric(
    *,
    metric_kind: str,
    score: Optional[float],
    status: str,
    source_metrics: List[str],
    note: str,
) -> dict:
    total = 1 if score is not None else 0
    passed = 1 if status == "ok" and score == 1.0 else 0
    failed = 1 if status == "ok" and score == 0.0 and total else 0
    unknown = total - passed - failed
    return {
        "status": status,
        "score": score,
        "passed": passed,
        "failed": failed,
        "unknown": unknown,
        "total": total,
        "metric_kind": metric_kind,
        "source_metrics": source_metrics,
        "method": "derived_from_clause_metric_scores",
        "note": note,
    }


def _continuous_rule_metric(
    *,
    metric_kind: str,
    score: Optional[float],
    status: str,
    source_metrics: List[str],
    note: str,
) -> dict:
    return {
        "status": status,
        "score": score,
        "metric_kind": metric_kind,
        "source_metrics": source_metrics,
        "method": "derived_from_clause_metric_scores",
        "note": note,
    }


def add_derived_rule_metrics(result: dict) -> dict:
    """Add per-file rule-level metrics from the four clause-level scores."""

    metrics = result.setdefault("metrics", {})
    source = {
        name: _metric_score_status(metrics, name)
        for name in RULE_SOURCE_METRIC_NAMES
    }
    pre_rel, pre_rel_status = source["precondition_clause_reliability_rate"]
    post_rel, post_rel_status = source["postcondition_clause_reliability_rate"]
    pre_comp, pre_comp_status = source["precondition_clause_completeness_rate"]
    post_comp, post_comp_status = source["postcondition_clause_completeness_rate"]

    def strict_score(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return 1.0 if value == 1.0 else 0.0

    def mean_score(values: List[Optional[float]]) -> Optional[float]:
        if any(value is None for value in values):
            return None
        return sum(float(value) for value in values if value is not None) / len(values)

    pre_rel_score = strict_score(pre_rel)
    post_rel_score = strict_score(post_rel)
    pre_comp_score = strict_score(pre_comp)
    post_comp_score = strict_score(post_comp)

    rel_status = _combine_status([pre_rel_status, post_rel_status])
    comp_status = _combine_status([pre_comp_status, post_comp_status])
    all_status = _combine_status([pre_rel_status, post_rel_status, pre_comp_status, post_comp_status])

    rule_rel_score = (
        None
        if pre_rel_score is None or post_rel_score is None
        else 1.0 if pre_rel_score == 1.0 and post_rel_score == 1.0 else 0.0
    )
    approx_rel_score = mean_score([pre_rel, post_rel])
    rule_comp_score = (
        None
        if pre_comp_score is None or post_comp_score is None
        else 1.0 if pre_comp_score == 1.0 and post_comp_score == 1.0 else 0.0
    )
    approx_comp_score = mean_score([pre_comp, post_comp])
    rule_correct_score = (
        None
        if rule_rel_score is None or rule_comp_score is None
        else 1.0 if rule_rel_score == 1.0 and rule_comp_score == 1.0 else 0.0
    )
    approx_correct_score = mean_score([approx_rel_score, approx_comp_score])

    derived = {
        "precondition_reliability_rate": _binary_rule_metric(
            metric_kind="precondition_reliability_rate",
            score=pre_rel_score,
            status=pre_rel_status,
            source_metrics=["precondition_clause_reliability_rate"],
            note="Counts this rule as reliable on preconditions iff its precondition clause reliability score is 1.",
        ),
        "postcondition_reliability_rate": _binary_rule_metric(
            metric_kind="postcondition_reliability_rate",
            score=post_rel_score,
            status=post_rel_status,
            source_metrics=["postcondition_clause_reliability_rate"],
            note="Counts this rule as reliable on postconditions iff its postcondition clause reliability score is 1.",
        ),
        "rule_reliability_rate": _binary_rule_metric(
            metric_kind="rule_reliability_rate",
            score=rule_rel_score,
            status=rel_status,
            source_metrics=[
                "precondition_clause_reliability_rate",
                "postcondition_clause_reliability_rate",
            ],
            note="Counts this rule as reliable iff both precondition and postcondition clause reliability scores are 1.",
        ),
        "approx_rule_reliability_rate": _continuous_rule_metric(
            metric_kind="approx_rule_reliability_rate",
            score=approx_rel_score,
            status=rel_status,
            source_metrics=[
                "precondition_clause_reliability_rate",
                "postcondition_clause_reliability_rate",
            ],
            note="Average of precondition and postcondition clause reliability scores for this rule.",
        ),
        "precondition_completeness_rate": _binary_rule_metric(
            metric_kind="precondition_completeness_rate",
            score=pre_comp_score,
            status=pre_comp_status,
            source_metrics=["precondition_clause_completeness_rate"],
            note="Counts this rule as complete on preconditions iff its precondition clause completeness score is 1.",
        ),
        "postcondition_completeness_rate": _binary_rule_metric(
            metric_kind="postcondition_completeness_rate",
            score=post_comp_score,
            status=post_comp_status,
            source_metrics=["postcondition_clause_completeness_rate"],
            note="Counts this rule as complete on postconditions iff its postcondition clause completeness score is 1.",
        ),
        "rule_completeness_rate": _binary_rule_metric(
            metric_kind="rule_completeness_rate",
            score=rule_comp_score,
            status=comp_status,
            source_metrics=[
                "precondition_clause_completeness_rate",
                "postcondition_clause_completeness_rate",
            ],
            note="Counts this rule as complete iff both precondition and postcondition clause completeness scores are 1.",
        ),
        "approx_rule_completeness_rate": _continuous_rule_metric(
            metric_kind="approx_rule_completeness_rate",
            score=approx_comp_score,
            status=comp_status,
            source_metrics=[
                "precondition_clause_completeness_rate",
                "postcondition_clause_completeness_rate",
            ],
            note="Average of precondition and postcondition clause completeness scores for this rule.",
        ),
        "rule_correctness_rate": _binary_rule_metric(
            metric_kind="rule_correctness_rate",
            score=rule_correct_score,
            status=all_status,
            source_metrics=list(RULE_SOURCE_METRIC_NAMES),
            note="Counts this rule as correct iff it is both reliable and complete.",
        ),
        "approx_rule_correctness_rate": _continuous_rule_metric(
            metric_kind="approx_rule_correctness_rate",
            score=approx_correct_score,
            status=all_status,
            source_metrics=list(RULE_SOURCE_METRIC_NAMES),
            note="Average of approximate rule reliability and approximate rule completeness.",
        ),
    }

    metrics.update(derived)
    score_names = RULE_SOURCE_METRIC_NAMES + DERIVED_RULE_METRIC_NAMES
    result["metric_scores"] = {
        name: _metric_score_status(metrics, name)[0]
        for name in score_names
    }
    result["rule_metrics"] = {
        name: _metric_score_status(metrics, name)[0]
        for name in DERIVED_RULE_METRIC_NAMES
    }
    return result


def _worker(args):
    filename, gen_path, ref_path, verus_bin = args
    os.chdir(str(PROJECT_ROOT))
    return filename, evaluate_single(gen_path, ref_path, verus_bin)


def build_statistics(all_results: Dict[str, dict], metric_names: List[str]) -> dict:
    stats = {}
    for mname in metric_names:
        scores = []
        statuses = defaultdict(int)
        for entry in all_results.values():
            raw = entry["metrics"].get(mname, {})
            status = _extract_status(raw)
            statuses[status] += 1
            score = _extract_score(raw)
            if isinstance(score, (int, float)) and math.isfinite(float(score)):
                scores.append(float(score))

        metric_stat = {
            "ok_count": statuses.get("ok", 0),
            "not_available_count": sum(v for k, v in statuses.items() if k not in ("ok", "exception")),
            "exception_count": statuses.get("exception", 0),
            "total": sum(statuses.values()),
            "scored_count": len(scores),
            "unscored_count": sum(statuses.values()) - len(scores),
            "scores_count": len(scores),
        }
        if scores:
            metric_stat.update({
                "mean": round(statistics.mean(scores), 4),
                "median": round(statistics.median(scores), 4),
                "stdev": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "q25": round(sorted(scores)[len(scores) // 4], 4),
                "q75": round(sorted(scores)[3 * len(scores) // 4], 4),
            })
        else:
            metric_stat.update({k: None for k in ["mean", "median", "stdev", "min", "max", "q25", "q75"]})
        stats[mname] = metric_stat
    return stats


def build_deficit_analysis(all_results: Dict[str, dict]) -> dict:
    checks = [
        ("weak_strength", "strength_classification", lambda s: s is not None and s < 0.5),
        ("high_redundancy", "spec_redundancy_rate", lambda s: s is not None and s > 0.3),
        ("low_llm_judge", "llm_as_judge_intent_consistency", lambda s: s is not None and s < 0.5),
        ("low_spec_code_intent", "llm_as_judge_spec_code_intent_consistency", lambda s: s is not None and s < 0.5),
        ("low_nav", "nav_assertion_pass_rate", lambda s: s is not None and s < 0.5),
        ("low_bug_detect", "bug_detection_rate", lambda s: s is not None and s < 0.8),
        ("low_mutation_kill", "mutation_kill_rate", lambda s: s is not None and s < 0.8),
        ("high_vacuity", "vacuity_risk", lambda s: s is not None and s > 0.1),
    ]

    total = len(all_results)
    pass1_total = 0
    pass1_deficit = 0
    deficit_types = defaultdict(int)
    deficit_files = defaultdict(list)

    for filename, entry in all_results.items():
        m = entry["metrics"]
        pass_score = _extract_score(m.get("verification_pass_rate", {}))
        if pass_score != 1.0:
            continue
        pass1_total += 1
        has_deficit = False
        for label, metric, check in checks:
            score = _extract_score(m.get(metric, {}))
            if check(score):
                has_deficit = True
                deficit_types[label] += 1
                deficit_files[label].append(filename)
        if has_deficit:
            pass1_deficit += 1

    pass_fail = total - pass1_total
    return {
        "total_files": total,
        "pass_1_total": pass1_total,
        "pass_0_total": pass_fail,
        "pass_1_with_deficit": pass1_deficit,
        "pass_1_deficit_rate": round(pass1_deficit / pass1_total, 4) if pass1_total > 0 else None,
        "deficit_breakdown": {
            label: {
                "count": deficit_types[label],
                "rate": round(deficit_types[label] / pass1_total, 4) if pass1_total > 0 else None,
                "files": deficit_files[label][:20],
            }
            for label in sorted(deficit_types, key=lambda x: -deficit_types[x])
        },
    }


def write_scores_csv(output_dir: Path, all_results: Dict[str, dict], metric_names: List[str]):
    csv_path = output_dir / "scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["filename"] + [f"{m}_score" for m in metric_names] + [f"{m}_status" for m in metric_names]
        writer.writerow(header)
        for filename in sorted(all_results):
            entry = all_results[filename]
            row = [filename]
            for mname in metric_names:
                raw = entry["metrics"].get(mname, {})
                row.append(_extract_score(raw))
            for mname in metric_names:
                raw = entry["metrics"].get(mname, {})
                row.append(_extract_status(raw))
            writer.writerow(row)


def write_per_file_result(per_file_dir: Path, filename: str, result: dict):
    safe_name = filename.replace("/", "_").replace(".rs", "")
    with open(per_file_dir / f"{safe_name}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)


def write_metric_scores_plot(output_dir: Path, stats: Dict[str, dict], metric_names: List[str]) -> Optional[Path]:
    try:
        mpl_config_dir = Path(tempfile.gettempdir()) / "veruseval_matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        import numpy as np
    except Exception as exc:
        print(f"Warning: failed to import plotting dependencies, skip metric_scores_bar.png: {exc}")
        return None

    preferred_fonts = ["Arial Unicode MS", "PingFang SC", "Heiti TC", "Songti SC", "SimHei", "Noto Sans CJK SC"]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_fonts = [font for font in preferred_fonts if font in available_fonts]
    plt.rcParams["font.family"] = selected_fonts + ["sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    plot_labels = metric_plot_labels()
    labels = [plot_labels.get(name, name) for name in metric_names]
    avg_scores = [stats.get(name, {}).get("mean") for name in metric_names]
    scored_rates = [
        f"{stats.get(name, {}).get('scored_count', stats.get(name, {}).get('scores_count', 0))}/{stats.get(name, {}).get('total', 0)}"
        for name in metric_names
    ]

    colors = []
    for score in avg_scores:
        if score is None:
            colors.append("#cccccc")
        elif score >= 0.9:
            colors.append("#2ecc71")
        elif score >= 0.7:
            colors.append("#3498db")
        elif score >= 0.5:
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    plot_scores = [score if score is not None else 0 for score in avg_scores]
    total_files = max((stats.get(name, {}).get("total", 0) for name in metric_names), default=0)

    fig, ax = plt.subplots(figsize=(26, 12))
    x = np.arange(len(labels))
    bars = ax.bar(x, plot_scores, color=colors, edgecolor="white", linewidth=0.5, width=0.7)

    for bar, score, scored_rate in zip(bars, avg_scores, scored_rates):
        if score is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=26,
                fontweight="bold",
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                scored_rate,
                ha="center",
                va="center",
                fontsize=20,
                color="white",
                fontweight="bold",
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.05,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=26,
                color="#666",
                fontstyle="italic",
                fontweight="bold",
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.01,
                scored_rate,
                ha="center",
                va="bottom",
                fontsize=20,
                color="#999",
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=28, fontweight="bold")
    ax.set_ylabel("平均分", fontsize=30, fontweight="bold")
    ax.set_ylim(0, 1.20)
    ax.tick_params(axis="y", labelsize=24)
    ax.tick_params(axis="x", pad=-2)
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    ax.axhline(y=1.0, color="#bdc3c7", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output_path = output_dir / "metric_scores_bar.png"
    try:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    except Exception as exc:
        plt.close(fig)
        print(f"Warning: failed to save metric score plot to {output_path}: {exc}")
        return None
    plt.close(fig)
    return output_path


def run_evaluation(
    file_pairs: List[tuple],
    output_dir: Path,
    verus_bin: Optional[str],
    num_workers: int,
):
    import metrics_rebuild as metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    per_file_dir = output_dir / "per_file"
    per_file_dir.mkdir(exist_ok=True)

    metric_names = metric_names_with_derived_rule_metrics(
        [fn.__name__.replace("metric_", "") for fn in metrics.METRIC_FUNCTIONS]
    )
    total = len(file_pairs)

    print(f"Evaluating {total} file pair(s), output → {output_dir}")
    print(f"Verus binary: {verus_bin or 'system PATH'}")
    print(f"Workers: {num_workers}")
    print()

    all_results: Dict[str, dict] = {}
    done = 0
    start_time = time.monotonic()

    if total == 1:
        filename, gen_path, ref_path = file_pairs[0]
        result = evaluate_single(gen_path, ref_path, verus_bin)
        all_results[filename] = result
        write_per_file_result(per_file_dir, filename, result)
        done = 1
    else:
        tasks = [(fn, gp, rp, verus_bin) for fn, gp, rp in file_pairs]
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_worker, t): t[0] for t in tasks}
            for future in as_completed(futures):
                filename, result = future.result()
                all_results[filename] = result
                write_per_file_result(per_file_dir, filename, result)
                done += 1
                pct = done / total * 100
                ok = sum(1 for r in result["metrics"].values() if _extract_status(r) == "ok")
                sys.stdout.write(f"\r[{'█' * int(pct / 2.5):<40}] {done}/{total} ({pct:.0f}%) ok={ok}/16")
                sys.stdout.flush()
        print()

    elapsed = time.monotonic() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    # 1. Write per-file detailed results
    for filename, result in all_results.items():
        write_per_file_result(per_file_dir, filename, result)

    # 2. Write scores CSV
    write_scores_csv(output_dir, all_results, metric_names)

    # 3. Build and write statistics
    stats = build_statistics(all_results, metric_names)
    with open(output_dir / "statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    metric_plot_path = write_metric_scores_plot(output_dir, stats, metric_names)

    # 4. Build and write deficit analysis
    deficit = build_deficit_analysis(all_results)
    with open(output_dir / "deficit_analysis.json", "w", encoding="utf-8") as f:
        json.dump(deficit, f, indent=2, ensure_ascii=False)

    # 5. Write summary
    summary = {
        "mode": "file" if total == 1 else "directory",
        "total_files": total,
        "elapsed_seconds": round(elapsed, 1),
        "num_workers": num_workers,
        "verus_binary": verus_bin,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metric_names": metric_names,
        "overall_statistics": stats,
        "deficit_summary": {
            "pass_1_total": deficit["pass_1_total"],
            "pass_1_with_deficit": deficit["pass_1_with_deficit"],
            "pass_1_deficit_rate": deficit["pass_1_deficit_rate"],
        },
        "output_files": {
            "per_file_results": str(per_file_dir),
            "scores_csv": str(output_dir / "scores.csv"),
            "statistics": str(output_dir / "statistics.json"),
            "deficit_analysis": str(output_dir / "deficit_analysis.json"),
            "metric_scores_bar": str(metric_plot_path) if metric_plot_path else None,
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 6. Print summary table
    print(f"\n{'=' * 90}")
    print(f"{'Metric':<40s} {'OK':>4s} {'N/A':>4s} {'Mean':>8s} {'Median':>8s} {'Stdev':>8s}")
    print("-" * 90)
    for mname in metric_names:
        s = stats[mname]
        mean = f"{s['mean']:.3f}" if s["mean"] is not None else "N/A"
        med = f"{s['median']:.3f}" if s["median"] is not None else "N/A"
        std = f"{s['stdev']:.3f}" if s["stdev"] is not None else "N/A"
        print(f"  {mname:<38s} {s['ok_count']:>4d} {s['not_available_count']:>4d} {mean:>8s} {med:>8s} {std:>8s}")

    if deficit["pass_1_total"] > 0:
        print(f"\nPass=1 with deficit: {deficit['pass_1_with_deficit']}/{deficit['pass_1_total']}"
              f" ({deficit['pass_1_deficit_rate']*100:.1f}%)")

    if metric_plot_path:
        print(f"\nMetric score plot saved to {metric_plot_path}")

    print(f"\nResults saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="VerusEval 评估脚本：对 Verus 规约进行多维质量评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 模式1：评估单个文件对
  python scripts/evaluation/evaluate.py file -g gen.rs -r ref.rs -o results/

  # 模式2：评估目录对
  python scripts/evaluation/evaluate.py dir -g exp_dataset/autoverus/generated/ \\
                                  -r exp_dataset/autoverus/reference/ \\
                                  -o results/autoverus/ -w 15
        """,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Mode 1: single file pair
    file_parser = subparsers.add_parser("file", help="评估单个文件对")
    file_parser.add_argument("-g", "--generated", required=True, help="生成规约的 .rs 文件路径")
    file_parser.add_argument("-r", "--reference", required=True, help="参考规约的 .rs 文件路径")

    # Mode 2: directory pair
    dir_parser = subparsers.add_parser("dir", help="评估目录对（文件名一一对应）")
    dir_parser.add_argument("-g", "--generated", required=True, help="生成规约目录")
    dir_parser.add_argument("-r", "--reference", required=True, help="参考规约目录")

    # Common arguments
    for p in [file_parser, dir_parser]:
        p.add_argument("-o", "--output", required=True, help="评估结果输出目录")
        p.add_argument("-w", "--workers", type=int, default=8, help="并行 worker 数量（默认 8）")
        p.add_argument("--verus-path", default=None, help="verus 二进制路径（默认从 config.yaml 读取）")
        p.add_argument("--text-scope", choices=["full", "spec_only"], default="spec_only",
                        help="文本指标范围（默认 spec_only）")

    args = parser.parse_args()

    verus_bin = args.verus_path or _load_verus_path_from_config()
    if verus_bin:
        import metrics_rebuild as metrics
        metrics.set_verus_binary(verus_bin)

    output_dir = Path(args.output)

    if args.mode == "file":
        gen_path = str(Path(args.generated).resolve())
        ref_path = str(Path(args.reference).resolve())
        if not Path(gen_path).exists():
            parser.error(f"生成文件不存在: {gen_path}")
        if not Path(ref_path).exists():
            parser.error(f"参考文件不存在: {ref_path}")
        filename = Path(args.generated).name
        file_pairs = [(filename, gen_path, ref_path)]

    elif args.mode == "dir":
        gen_dir = Path(args.generated)
        ref_dir = Path(args.reference)
        if not gen_dir.is_dir():
            parser.error(f"生成目录不存在: {gen_dir}")
        if not ref_dir.is_dir():
            parser.error(f"参考目录不存在: {ref_dir}")

        gen_files = {f.name for f in gen_dir.iterdir() if f.suffix == ".rs"}
        ref_files = {f.name for f in ref_dir.iterdir() if f.suffix == ".rs"}
        common = sorted(gen_files & ref_files)

        if not common:
            parser.error(f"两个目录中没有同名 .rs 文件。\n  generated: {len(gen_files)} files\n  reference: {len(ref_files)} files")

        only_gen = gen_files - ref_files
        only_ref = ref_files - gen_files
        if only_gen:
            print(f"Warning: {len(only_gen)} files only in generated dir (skipped)")
        if only_ref:
            print(f"Warning: {len(only_ref)} files only in reference dir (skipped)")

        file_pairs = [
            (name, str((gen_dir / name).resolve()), str((ref_dir / name).resolve()))
            for name in common
        ]

    run_evaluation(file_pairs, output_dir, verus_bin, args.workers)


if __name__ == "__main__":
    main()
