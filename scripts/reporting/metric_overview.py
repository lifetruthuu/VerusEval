"""Grouped metric overviews and the documented chart-only scoring policy."""

from __future__ import annotations

import math

import os

import statistics

import tempfile

from pathlib import Path

from typing import Sequence

import csv

METRICS = [
    ("vts_bleu",                                       "  ↳ Spec similarity: BLEU",          "text_quality",        False, False),
    ("vts_rouge_l",                                    "  ↳ Spec similarity: ROUGE-L",       "text_quality",        False, False),
    ("vts_key_spec_match",                             "  ↳ Spec similarity: KeySpecMatch",  "text_quality",        False, False),
    ("trivial_pre_false",                              "  ↳ False precondition",             "spec_triviality",     True,  False),
    ("trivial_post_true",                              "  ↳ True postcondition",             "spec_triviality",     True,  False),
    ("llm_as_judge_intent_consistency",                "Spec-spec intent agreement",             "intent_consistency",  False, False),
    ("llm_as_judge_spec_code_intent_consistency",      "Spec-code intent agreement",         "intent_consistency",  False, False),
    ("parse_rate",                                     "Syntactic validity",                    "formal_validity",     False, False),
    ("type_check_rate",                                "Type validity",                "formal_validity",     False, False),
    ("verification_pass_rate",                         "Verifier acceptance",              "formal_validity",     False, False),
    ("proportion_at_least_gt",                         "Not weaker than reference",                "spec_reliability",    False, False),
    ("postcondition_clause_reliability_rate",          "Postcondition clause soundness",         "spec_reliability",    False, False),
    ("precondition_clause_reliability_rate",           "Precondition clause soundness",         "spec_reliability",    False, False),
    ("precondition_reliability_rate",                  "Precondition soundness",                "spec_reliability",    False, False),
    ("postcondition_reliability_rate",                 "Postcondition soundness",                "spec_reliability",    False, False),
    ("rule_reliability_rate",                          "Rule soundness",                    "spec_reliability",    False, False),
    ("approx_rule_reliability_rate",                   "Approximate rule soundness",                "spec_reliability",    False, False),
    ("wrong_io_reject_rate",                           "Wrong-output rejection",           "io_testing",          False, False),
    ("proportion_at_most_gt",                          "Not stronger than reference",                "spec_completeness",   False, False),
    ("postcondition_clause_completeness_rate",         "Postcondition clause completeness",         "spec_completeness",   False, False),
    ("precondition_clause_completeness_rate",          "Precondition clause completeness",         "spec_completeness",   False, False),
    ("precondition_completeness_rate",                 "Precondition completeness",                "spec_completeness",   False, False),
    ("postcondition_completeness_rate",                "Postcondition completeness",                "spec_completeness",   False, False),
    ("rule_completeness_rate",                         "Rule completeness",                    "spec_completeness",   False, False),
    ("approx_rule_completeness_rate",                  "Approximate rule completeness",                "spec_completeness",   False, False),
    ("correct_io_pass_rate",                           "Correct-I/O acceptance",           "io_testing",          False, False),
    ("invalid_test_filtering_rate",                    "Invalid-input rejection",                "io_testing",          False, False),
    ("rule_correctness_rate",                          "Rule correctness",                    "spec_correctness",    False, False),
    ("approx_rule_correctness_rate",                   "Approximate rule correctness",                "spec_correctness",    False, False),
    ("mutation_kill_rate",                             "Mutation kill rate",                    "robustness",          False, False),
    ("spec_redundancy_rate",                           "Proof redundancy",                  "simplicity",          True,  False),
    ("size_spec_volume",                               "  ↳ Spec complexity: Nodes",  "spec_size",           True,  True),
    ("size_spec_depth",                                "  ↳ Spec complexity: Logical depth","spec_size",           True,  True),
    ("size_spec_quantifier",                           "  ↳ Spec complexity: Quantifiers",  "spec_size",           True,  True),
    ("size_spec_vocabulary",                           "  ↳ Spec complexity: Vocabulary",  "spec_size",           True,  True),
    ("size_proof_volume",                              "  ↳ Proof complexity: Nodes", "spec_size",           True,  True),
    ("size_proof_depth",                               "  ↳ Proof complexity: Logical depth","spec_size",           True,  True),
    ("size_proof_quantifier",                          "  ↳ Proof complexity: Quantifiers", "spec_size",           True,  True),
    ("size_proof_vocabulary",                          "  ↳ Proof complexity: Vocabulary", "spec_size",           True,  True),
    ("verification_time",                              "Verification time",              "time_efficiency",     True,  True),
]

IO_METRIC_IDS = frozenset({
    "wrong_io_reject_rate",
    "correct_io_pass_rate",
    "invalid_test_filtering_rate",
})

UNVERIFIED_NULL_METRIC_IDS = frozenset({
    "mutation_kill_rate",
    "spec_redundancy_rate",
})

METRIC_LOWER_IS_BETTER = {metric_id: lower for metric_id, _label, _dim, lower, _raw in METRICS}

RAW_METRIC_IDS = frozenset(metric_id for metric_id, _label, _dim, _lower, raw in METRICS if raw)

DIRECT_IMPLICATION_METRICS = (
    "postcondition_clause_reliability_rate",
    "precondition_clause_reliability_rate",
    "postcondition_clause_completeness_rate",
    "precondition_clause_completeness_rate",
    "proportion_at_least_gt",
    "proportion_at_most_gt",
)

RULE_SOURCE_IMPLICATION_METRICS = (
    "precondition_clause_reliability_rate",
    "postcondition_clause_reliability_rate",
    "precondition_clause_completeness_rate",
    "postcondition_clause_completeness_rate",
)

DERIVED_IMPLICATION_METRICS = (
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

CONSERVATIVE_IMPLICATION_METRICS = frozenset(
    DIRECT_IMPLICATION_METRICS + DERIVED_IMPLICATION_METRICS
)

DIMENSION_LABEL = {
    "text_quality":      "Text similarity",
    "intent_consistency": "Intent judgement",
    "spec_triviality":   "Semantic triviality",
    "formal_validity":   "Verifier validity",
    "spec_size":         "Specification complexity",
    "spec_reliability":  "Specification soundness",
    "spec_completeness": "Specification completeness",
    "spec_correctness":  "Specification correctness",
    "io_testing":        "I/O testing",
    "robustness":        "Robustness",
    "simplicity":        "Proof simplicity",
    "time_efficiency":   "Time efficiency",
}

DIMENSION_ORDER = [
    "text_quality", "intent_consistency", "spec_triviality", "formal_validity",
    "spec_size", "spec_reliability", "spec_completeness", "spec_correctness",
    "io_testing", "robustness", "simplicity", "time_efficiency",
]

DIMENSION_COLOR = {
    "text_quality":      "#2A78D6",  # blue
    "intent_consistency": "#7B4B94",  # orchid (inserted later)
    "spec_triviality":   "#8C5A2B",  # brown (inserted later)
    "formal_validity":   "#1BAF7A",  # aqua
    "spec_size":         "#7A8793",  # gray
    "spec_reliability":  "#EDA100",  # yellow
    "spec_completeness": "#008300",  # green
    "spec_correctness":  "#4A3AA7",  # violet
    "io_testing":        "#00A1C7",  # cyan
    "robustness":        "#E34948",  # red
    "simplicity":        "#E87BA4",  # magenta
    "time_efficiency":   "#EB6834",  # orange
}

def _setup_matplotlib():
    mpl_config_dir = Path(tempfile.gettempdir()) / "veruseval_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["figure.titleweight"] = "bold"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.edgecolor"] = INK_MUTED
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = GRIDLINE
    plt.rcParams["grid.linewidth"] = 0.6
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    return plt

def _nested(d, *keys):
    """Safely read a nested dict value; return None if any level is missing."""
    for key in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d

def _component_score(component: object):
    if not isinstance(component, dict):
        return None
    if component.get("status") == "degraded":
        return None
    return component.get("score")

def _bleu_component_score(component: object):
    if not isinstance(component, dict):
        return None
    if component.get("status") == "degraded":
        return None
    return component.get("score")

def _first_present(*values: object):
    for value in values:
        if value is not None:
            return value
    return None

def load_scores_many(csv_paths: Sequence[Path]) -> dict:
    """Aggregate score rows from atomic CSVs without materializing a merged file."""

    csv_paths = tuple(Path(path) for path in csv_paths)
    rows = []
    for csv_path in csv_paths:
        with csv_path.open(encoding="utf-8", newline="") as fh:
            rows.extend(
                (row, csv_path.parent.name == "unverified")
                for row in csv.DictReader(fh)
            )

    out = {}
    for mid, _label, _dim, lower_is_better, is_raw in METRICS:
        score_col = f"{mid}_score"
        status_col = f"{mid}_status"
        scores = []
        scored = ok = na = 0
        for row, row_is_unverified in rows:
            status = row.get(status_col, "")
            raw = row.get(score_col, "")
            score = None
            if raw not in (None, "", "nan", "NaN"):
                try:
                    score = float(raw)
                except ValueError:
                    score = None
            if score is not None and not math.isfinite(score):
                score = None
            # Mutation killing and proof redundancy require a verified program.
            # For unverified result groups, show them as N/A even if stale numeric
            # values happen to remain in an older CSV.
            unverified_null = (
                row_is_unverified and mid in UNVERIFIED_NULL_METRIC_IDS
            )
            if unverified_null:
                score = None
            # Charting policy: a verified program with no removable proof-support
            # clauses has a defined proof redundancy rate of zero. Keep the raw
            # evaluation output unchanged; only the plots apply this convention.
            no_testable_proof_clauses = (
                mid == "spec_redundancy_rate"
                and status == "no_testable_clauses"
                and score is None
                and not unverified_null
            )
            if no_testable_proof_clauses:
                score = 0.0
            # Chart-only full-coverage policy: keep I/O and raw counts unchanged,
            # while unresolved non-I/O scores receive the worst bounded value.
            # Lower-is-better metrics therefore receive 1; all others receive 0.
            if (
                score is None
                and mid not in IO_METRIC_IDS
                and not is_raw
                and not unverified_null
            ):
                score = 1.0 if lower_is_better else 0.0
            if score is not None:
                scored += 1
                scores.append(score)
            else:
                na += 1
            if status == "ok" or no_testable_proof_clauses:
                ok += 1
        out[mid] = {"scores": scores, "scored": scored, "ok": ok, "na": na,
                    "total": scored + na, "mean": (statistics.mean(scores) if scores else None),
                    "median": (statistics.median(scores) if scores else None)}
    return out

def load_sub_scores_many(per_file_dirs: Sequence[Path]) -> dict:
    """Aggregate JSON sub-metrics from multiple atomic per-file directories."""
    def complete_complexity_value(metrics: dict, metric_id: str, direct_key: str):
        source = metrics.get(metric_id, {})
        if not isinstance(source, dict) or source.get("status") != "ok":
            return None
        return _nested(metrics, metric_id, "gen", direct_key)

    extractors = {
        "vts_bleu": lambda m: _bleu_component_score(_nested(m, "verus_textual_similarity_spec_only", "components", "bleu")),
        "vts_rouge_l": lambda m: _component_score(_nested(m, "verus_textual_similarity_spec_only", "components", "rouge_l")),
        "vts_key_spec_match": lambda m: _component_score(_nested(m, "verus_textual_similarity_spec_only", "components", "key_spec_match")),
        "trivial_pre_false": lambda m: _first_present(
            _nested(m, "trivial_spec_ratio", "generated", "components", "precondition_always_false"),
            _nested(m, "trivial_spec_ratio", "generated", "components", "precondition_contradiction"),
        ),
        "trivial_post_true": lambda m: _first_present(
            _nested(m, "trivial_spec_ratio", "generated", "components", "postcondition_always_true"),
            _nested(m, "trivial_spec_ratio", "generated", "components", "vacuous_postcondition"),
        ),
        "size_spec_volume": lambda m: complete_complexity_value(m, "spec_size_complexity_spec_only", "node_count"),
        "size_spec_depth": lambda m: complete_complexity_value(m, "spec_size_complexity_spec_only", "logic_depth"),
        "size_spec_quantifier": lambda m: complete_complexity_value(m, "spec_size_complexity_spec_only", "quantifier_count"),
        "size_spec_vocabulary": lambda m: complete_complexity_value(m, "spec_size_complexity_spec_only", "vocabulary_size"),
        "size_proof_volume": lambda m: complete_complexity_value(m, "spec_size_complexity_proof", "node_count"),
        "size_proof_depth": lambda m: complete_complexity_value(m, "spec_size_complexity_proof", "logic_depth"),
        "size_proof_quantifier": lambda m: complete_complexity_value(m, "spec_size_complexity_proof", "quantifier_count"),
        "size_proof_vocabulary": lambda m: complete_complexity_value(m, "spec_size_complexity_proof", "vocabulary_size"),
    }
    buckets = {sid: [] for sid in extractors}
    total_files = 0

    for per_file_dir in per_file_dirs:
        for json_file in sorted(per_file_dir.glob("*.json")):
            total_files += 1
            try:
                import json
                data = json.loads(json_file.read_text(encoding="utf-8"))
                metrics = data.get("metrics", {})
            except Exception:
                metrics = {}
            for sid, fn in extractors.items():
                value = fn(metrics)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    buckets[sid].append(float(value))
                elif sid not in RAW_METRIC_IDS:
                    buckets[sid].append(1.0 if METRIC_LOWER_IS_BETTER[sid] else 0.0)

    out = {}
    for sid, scores in buckets.items():
        out[sid] = {
            "scores": scores,
            "scored": len(scores),
            "ok": len(scores),
            "na": total_files - len(scores),
            "total": total_files,
            "mean": (statistics.mean(scores) if scores else None),
            "median": (statistics.median(scores) if scores else None),
        }
    return out

def _conservative_implication_component(metrics: dict, metric_id: str):
    """Return chart-only score/coverage with unknown implications counted as zero."""

    raw = metrics.get(metric_id, {})
    if not isinstance(raw, dict):
        return None
    body = raw.get("generated", raw)
    if not isinstance(body, dict):
        return None
    passed = body.get("passed")
    total = body.get("total")
    if (
        not isinstance(passed, int)
        or isinstance(passed, bool)
        or not isinstance(total, int)
        or isinstance(total, bool)
        or total < 0
        or passed < 0
        or passed > total
    ):
        return None
    unknown = body.get("unknown")
    determined = body.get("determined")
    if not isinstance(determined, int) or isinstance(determined, bool):
        if isinstance(unknown, int) and not isinstance(unknown, bool):
            determined = total - unknown
        else:
            failed = body.get("failed")
            if not isinstance(failed, int) or isinstance(failed, bool):
                return None
            determined = passed + failed
    determined = max(0, min(total, determined))
    score = passed / total if total else 0.0
    return score, determined, total

def _derive_conservative_rule_scores(source_scores: dict[str, float]) -> dict[str, float]:
    pre_rel = source_scores["precondition_clause_reliability_rate"]
    post_rel = source_scores["postcondition_clause_reliability_rate"]
    pre_comp = source_scores["precondition_clause_completeness_rate"]
    post_comp = source_scores["postcondition_clause_completeness_rate"]

    def strict(value: float) -> float:
        return 1.0 if value == 1.0 else 0.0

    pre_rel_strict = strict(pre_rel)
    post_rel_strict = strict(post_rel)
    pre_comp_strict = strict(pre_comp)
    post_comp_strict = strict(post_comp)
    rule_rel = 1.0 if pre_rel_strict == post_rel_strict == 1.0 else 0.0
    rule_comp = 1.0 if pre_comp_strict == post_comp_strict == 1.0 else 0.0
    approx_rel = (pre_rel + post_rel) / 2.0
    approx_comp = (pre_comp + post_comp) / 2.0
    return {
        "precondition_reliability_rate": pre_rel_strict,
        "postcondition_reliability_rate": post_rel_strict,
        "rule_reliability_rate": rule_rel,
        "approx_rule_reliability_rate": approx_rel,
        "precondition_completeness_rate": pre_comp_strict,
        "postcondition_completeness_rate": post_comp_strict,
        "rule_completeness_rate": rule_comp,
        "approx_rule_completeness_rate": approx_comp,
        "rule_correctness_rate": 1.0 if rule_rel == rule_comp == 1.0 else 0.0,
        "approx_rule_correctness_rate": (approx_rel + approx_comp) / 2.0,
    }

def load_conservative_implication_scores_many(per_file_dirs: Sequence[Path]) -> dict:
    """Aggregate chart-only implication scores without changing stored metrics."""

    buckets = {metric_id: [] for metric_id in CONSERVATIVE_IMPLICATION_METRICS}
    coverage_numerators = {metric_id: 0 for metric_id in CONSERVATIVE_IMPLICATION_METRICS}
    coverage_denominators = {metric_id: 0 for metric_id in CONSERVATIVE_IMPLICATION_METRICS}
    complete_files = {metric_id: 0 for metric_id in CONSERVATIVE_IMPLICATION_METRICS}
    total_files = 0

    derived_sources = {
        "precondition_reliability_rate": ("precondition_clause_reliability_rate",),
        "postcondition_reliability_rate": ("postcondition_clause_reliability_rate",),
        "rule_reliability_rate": (
            "precondition_clause_reliability_rate",
            "postcondition_clause_reliability_rate",
        ),
        "approx_rule_reliability_rate": (
            "precondition_clause_reliability_rate",
            "postcondition_clause_reliability_rate",
        ),
        "precondition_completeness_rate": ("precondition_clause_completeness_rate",),
        "postcondition_completeness_rate": ("postcondition_clause_completeness_rate",),
        "rule_completeness_rate": (
            "precondition_clause_completeness_rate",
            "postcondition_clause_completeness_rate",
        ),
        "approx_rule_completeness_rate": (
            "precondition_clause_completeness_rate",
            "postcondition_clause_completeness_rate",
        ),
        "rule_correctness_rate": RULE_SOURCE_IMPLICATION_METRICS,
        "approx_rule_correctness_rate": RULE_SOURCE_IMPLICATION_METRICS,
    }

    for per_file_dir in per_file_dirs:
        for json_file in sorted(per_file_dir.glob("*.json")):
            total_files += 1
            try:
                import json

                payload = json.loads(json_file.read_text(encoding="utf-8"))
                metrics = payload.get("metrics", {})
            except Exception:
                metrics = {}
            components = {
                metric_id: _conservative_implication_component(metrics, metric_id)
                for metric_id in DIRECT_IMPLICATION_METRICS
            }
            for metric_id, component in components.items():
                if component is None:
                    buckets[metric_id].append(0.0)
                    continue
                score, determined, total = component
                buckets[metric_id].append(score)
                coverage_numerators[metric_id] += determined
                coverage_denominators[metric_id] += total
                if determined == total:
                    complete_files[metric_id] += 1

            if any(components.get(metric_id) is None for metric_id in RULE_SOURCE_IMPLICATION_METRICS):
                for metric_id in DERIVED_IMPLICATION_METRICS:
                    buckets[metric_id].append(0.0)
                continue
            source_scores = {
                metric_id: components[metric_id][0]
                for metric_id in RULE_SOURCE_IMPLICATION_METRICS
            }
            derived_scores = _derive_conservative_rule_scores(source_scores)
            for metric_id, score in derived_scores.items():
                sources = derived_sources[metric_id]
                determined = sum(components[source][1] for source in sources)
                total = sum(components[source][2] for source in sources)
                buckets[metric_id].append(score)
                coverage_numerators[metric_id] += determined
                coverage_denominators[metric_id] += total
                if determined == total:
                    complete_files[metric_id] += 1

    out = {}
    for metric_id, scores in buckets.items():
        denominator = coverage_denominators[metric_id]
        out[metric_id] = {
            "scores": scores,
            "scored": len(scores),
            "ok": complete_files[metric_id],
            "na": total_files - len(scores),
            "total": total_files,
            "mean": statistics.mean(scores) if scores else None,
            "median": statistics.median(scores) if scores else None,
            "semantic_coverage": (
                coverage_numerators[metric_id] / denominator if denominator else None
            ),
            "score_policy": "unknown_as_zero",
        }
    return out

def load_conservative_implication_scores(per_file_dir: Path) -> dict:
    return load_conservative_implication_scores_many((per_file_dir,))

RAW_SPEC_IDS = ("size_spec_volume", "size_spec_depth", "size_spec_quantifier", "size_spec_vocabulary")

RAW_PROOF_IDS = ("size_proof_volume", "size_proof_depth", "size_proof_quantifier", "size_proof_vocabulary")

RAW_COL_LABELS = ("Nodes", "Logical depth", "Quantifiers", "Vocabulary")

RAW_IDS = frozenset(RAW_SPEC_IDS) | frozenset(RAW_PROOF_IDS)

INK = "#17171a"

INK_SECONDARY = "#52514e"

INK_MUTED = "#8a8985"

HAIRLINE = "#e1e0d9"

GRIDLINE = "#e6e6e6"

WARNING_TEXT = "#b3701f"

LOWER_BETTER_EDGE = "#4a4a46"

EMPTY_FILL = "#eeeeee"

RAW_BLOCK_FILL = "#f4f4f1"

ACCENT = DIMENSION_COLOR["spec_correctness"]

_FIG_FORMATS: tuple[str, ...] = ("png",)

def _save_figure(fig, out_dir: Path, stem: str) -> list[Path]:
    """Save figure to configured formats; PDF is vector, PNG is raster preview."""
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    for fmt in _FIG_FORMATS:
        path = out_dir / f"{stem}.{fmt}"
        kwargs: dict = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = 160
        fig.savefig(path, format=fmt, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths

def _tint(hex_color: str, frac: float = 0.5) -> str:
    """Lighten hex_color toward white by frac (0=no change, 1=white)."""
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    r2 = r + (1.0 - r) * frac
    g2 = g + (1.0 - g) * frac
    b2 = b + (1.0 - b) * frac
    return "#{:02x}{:02x}{:02x}".format(
        round(r2 * 255), round(g2 * 255), round(b2 * 255)
    )

def _split_dimensions_into_columns(
    dims_in_order: list[str],
    bar_row_counts: dict[str, int],
    minichart_dim: str | None = "spec_size",
    minichart_weight: float = 4.5,
) -> int:
    """Return split index k so left/right dimension columns are balanced."""
    if len(dims_in_order) <= 1:
        return 1
    weights = []
    for dim in dims_in_order:
        weight = float(bar_row_counts.get(dim, 0))
        if dim == minichart_dim:
            weight += minichart_weight
        weights.append(weight)
    total = sum(weights)
    best_k, best_diff = 1, float("inf")
    for k in range(1, len(dims_in_order)):
        left = sum(weights[:k])
        diff = abs(left - (total - left))
        if diff < best_diff:
            best_k, best_diff = k, diff
    return best_k

def _is_submetric(label: str) -> bool:
    return label.strip().startswith("↳")

def fig_mean_hbar(plt, data, out_dir: Path, dataset_name: str, total_files: int):
    """Horizontal bar chart of mean score per metric, grouped/colored by dimension.

    The 8 "spec size" sub-metrics (node count, logic depth, quantifier count,
    vocabulary size, for spec and proof) are raw counts on a scale that differs
    both from the 0-1 scores and from each other. Render them as a compact
    mini bar chart inside the "Specification complexity" group, with each column normalized within
    its own unit.
    """
    import numpy as np
    from matplotlib.patches import Patch

    dims_in_order = [d for d in DIMENSION_ORDER if any(m[2] == d for m in METRICS)]
    bar_row_counts = {
        dim: sum(1 for m in METRICS if m[2] == dim and m[0] not in RAW_IDS)
        for dim in dims_in_order
    }
    split_idx = _split_dimensions_into_columns(dims_in_order, bar_row_counts)
    column_dims = [dims_in_order[:split_idx], dims_in_order[split_idx:]]

    def build_layout(dims_for_column):
        y, row_kind, row_item = [], [], []
        group_span = {}
        cur = 0.0
        for gi, dim in enumerate(dims_for_column):
            if gi > 0:
                cur += 0.58
            group_start = cur
            dim_items = [(idx, m) for idx, m in enumerate(METRICS)
                         if m[2] == dim and m[0] not in RAW_IDS]
            for idx, m in dim_items:
                y.append(cur)
                row_kind.append("bar")
                row_item.append((idx, m))
                cur += 0.80 if _is_submetric(m[1]) else 1.0
            if dim == "spec_size":
                cur += 0.35
                y.append(cur); row_kind.append("raw_header"); row_item.append(None)
                cur += 0.55
                y.append(cur); row_kind.append("raw_spec"); row_item.append(None)
                cur += 0.72
                y.append(cur); row_kind.append("raw_proof"); row_item.append(None)
                cur += 0.55
                y.append(cur); row_kind.append("raw_caption"); row_item.append(None)
                cur += 0.35
            group_span[dim] = (group_start, y[-1] if y else group_start)
        y_arr = np.array(y, dtype=float)
        return {
            "y": y_arr,
            "row_kind": row_kind,
            "row_item": row_item,
            "group_span": group_span,
            "max_y": float(max(y_arr)) if len(y_arr) else 0.0,
        }

    layouts = [build_layout(dims) for dims in column_dims]
    fig_height = max(10.4, 0.48 * max(layout["max_y"] for layout in layouts) + 2.7)
    fig, axes = plt.subplots(1, 2, figsize=(24, fig_height), sharex=True)

    def draw_raw_size_minichart(ax, layout):
        y = layout["y"]
        row_kind = layout["row_kind"]
        header_y = next((yy for yy, k in zip(y, row_kind) if k == "raw_header"), None)
        spec_y = next((yy for yy, k in zip(y, row_kind) if k == "raw_spec"), None)
        proof_y = next((yy for yy, k in zip(y, row_kind) if k == "raw_proof"), None)
        caption_y = next((yy for yy, k in zip(y, row_kind) if k == "raw_caption"), None)
        if header_y is None or spec_y is None or proof_y is None or caption_y is None:
            return

        ax.axhspan(header_y - 0.38, caption_y + 0.23, color=RAW_BLOCK_FILL, zorder=0)
        col_lefts = (0.04, 0.31, 0.58, 0.84)
        col_w = 0.18
        spec_color = DIMENSION_COLOR["spec_size"]
        proof_color = _tint(spec_color, 0.5)
        for cx, label, spec_mid, proof_mid in zip(
            col_lefts, RAW_COL_LABELS, RAW_SPEC_IDS, RAW_PROOF_IDS
        ):
            ax.text(cx, header_y, label, fontsize=8.1, color=INK_MUTED,
                    ha="left", va="center", fontweight="bold")
            vals = [data[spec_mid]["mean"], data[proof_mid]["mean"]]
            numeric = [v for v in vals if isinstance(v, (int, float))]
            vmax = max(numeric) if numeric else 0.0
            for yy, val, color in ((spec_y, vals[0], spec_color), (proof_y, vals[1], proof_color)):
                if isinstance(val, (int, float)):
                    width = (val / vmax) * col_w if vmax > 0 else 0.0
                    bar = ax.barh(yy, width, left=cx, height=0.32, color=color,
                                  edgecolor=LOWER_BETTER_EDGE, linewidth=0.45,
                                  hatch="//", zorder=3)
                    bar[0].set_alpha(0.92)
                    text_x = min(cx + width + 0.009, cx + col_w + 0.03)
                    ax.text(text_x, yy, f"{val:.1f}", fontsize=8.3, color=INK,
                            va="center", ha="left", fontweight="bold")
                else:
                    ax.text(cx, yy, "N/A", fontsize=8.3, color=INK_MUTED,
                            va="center", ha="left", fontstyle="italic")
        ax.text(0.04, caption_y,
                "Raw counts; each column scaled separately. Dark: specification; light: proof.",
                fontsize=7.3, color=INK_MUTED, fontstyle="italic",
                ha="left", va="center")

    def draw_score_column(ax, layout):
        y = layout["y"]
        row_kind = layout["row_kind"]
        row_item = layout["row_item"]
        group_span = layout["group_span"]

        sub_runs = []
        run = []
        for yy, kind, item in zip(y, row_kind, row_item):
            is_sub = kind == "bar" and _is_submetric(item[1][1])
            if is_sub:
                run.append(float(yy))
            elif run:
                sub_runs.append(run)
                run = []
        if run:
            sub_runs.append(run)

        for yy, kind, item in zip(y, row_kind, row_item):
            if kind != "bar":
                continue
            _idx, metric = item
            mid, label, dim, lower_is_better, _is_raw = metric
            mean = data[mid]["mean"]
            scored = data[mid]["scored"]
            total = data[mid]["total"]
            is_sub = _is_submetric(label)
            if mid == "verification_time":
                label = f"{mean:.3f} s   n={scored}/{total}" if mean is not None else "N/A"
                ax.text(0.012, yy, label, va="center", fontsize=9.2, color=INK)
                continue
            value = mean if mean is not None else 0.0
            color = _tint(DIMENSION_COLOR[dim], 0.5) if is_sub else DIMENSION_COLOR[dim]
            height = 0.32 if is_sub else 0.60
            bars = ax.barh(float(yy), value, color=color, edgecolor="white",
                           linewidth=0.7, height=height, zorder=3)
            bar = bars[0]
            if lower_is_better:
                bar.set_hatch("//")
                bar.set_edgecolor(LOWER_BETTER_EDGE)
                bar.set_linewidth(0.5)
            if mean is not None:
                x = bar.get_width()
                ax.text(x + 0.012, yy, f"{mean:.3f}", va="center", ha="left",
                        fontsize=9.2 if not is_sub else 8.7,
                        fontweight="bold", color=INK)
                n_color = WARNING_TEXT if scored < total else INK_MUTED
                ax.text(x + 0.078, yy, f"n={scored}/{total}", va="center", ha="left",
                        fontsize=8.0 if not is_sub else 7.6, color=n_color)
            else:
                ax.text(0.012, yy, "N/A", va="center", ha="left",
                        fontsize=9.4, color=INK_MUTED, fontstyle="italic")

        draw_raw_size_minichart(ax, layout)

        for run in sub_runs:
            ax.plot([0, 0], [run[0] - 0.34, run[-1] + 0.34],
                    color=HAIRLINE, linewidth=1.0, zorder=2)

        tick_labels = []
        tick_is_sub = []
        for kind, item in zip(row_kind, row_item):
            if kind == "bar":
                label = item[1][1]
                tick_labels.append(label)
                tick_is_sub.append(_is_submetric(label))
            elif kind == "raw_spec":
                tick_labels.append("  ↳ Specification")
                tick_is_sub.append(True)
            elif kind == "raw_proof":
                tick_labels.append("  ↳ Proof")
                tick_is_sub.append(True)
            else:
                tick_labels.append("")
                tick_is_sub.append(False)

        ax.set_yticks(y)
        ax.set_yticklabels(tick_labels, fontsize=9.8, color=INK_SECONDARY)
        for tick, is_sub in zip(ax.get_yticklabels(), tick_is_sub):
            if is_sub:
                tick.set_fontsize(8.8)
                tick.set_color(INK_MUTED)
        ax.invert_yaxis()
        ax.set_ylim(layout["max_y"] + 0.72, -0.92)
        ax.set_xlim(0, 1.14)
        ax.set_xlabel("Mean score (hatched: lower is better; time shown separately in seconds)", fontsize=10.5, color=INK_SECONDARY)

        for dim, (ystart, yend) in group_span.items():
            ax.text(0.0, ystart - 0.42, "●", ha="left", va="bottom", fontsize=9,
                    color=DIMENSION_COLOR[dim], transform=ax.get_yaxis_transform())
            ax.text(0.018, ystart - 0.42, DIMENSION_LABEL[dim], ha="left", va="bottom",
                    fontsize=9.4, fontweight="bold", color=INK,
                    transform=ax.get_yaxis_transform())
            ax.plot([0, 1], [yend + 0.42, yend + 0.42],
                    color=HAIRLINE, linewidth=0.8,
                    transform=ax.get_yaxis_transform(), clip_on=False, zorder=0)

        ax.grid(axis="x", which="major", color=GRIDLINE, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)

    for ax, layout in zip(axes, layouts):
        draw_score_column(ax, layout)

    fig.suptitle(
        f"{dataset_name} · Metric overview ({total_files} programs)",
        x=0.5, y=0.988, ha="center",
        fontsize=15, fontweight="bold", color=INK,
    )
    fig.text(
        0.5, 0.955,
        "Unknown implications count as 0; other missing non-I/O scores use the worst value. Empty I/O and unverified mutation/redundancy remain N/A.",
        ha="center", va="bottom", fontsize=9.4, color=INK_SECONDARY,
    )

    legend_handles = [Patch(facecolor=DIMENSION_COLOR[d], edgecolor="none",
                             label=DIMENSION_LABEL[d]) for d in DIMENSION_ORDER]
    legend_handles.append(Patch(facecolor="white", edgecolor=LOWER_BETTER_EDGE, hatch="//",
                                 label="Lower is better"))
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.94),
               ncol=7, fontsize=8.7, frameon=False, columnspacing=1.3, handlelength=1.4,
               handletextpad=0.5)
    fig.subplots_adjust(wspace=0.20)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return _save_figure(fig, out_dir, "metric_mean_hbar")
