from __future__ import annotations

from metrics_rebuild.share.models import MetricCatalogEntry

METRIC_CATALOG: tuple[MetricCatalogEntry, ...] = (
    MetricCatalogEntry("verus_textual_similarity", "Verus Textual Similarity (full)", "text_quality", "higher_is_better", "native"),
    MetricCatalogEntry("verus_textual_similarity_spec_only", "Verus Textual Similarity (spec only)", "text_quality", "higher_is_better", "native"),
    MetricCatalogEntry("verus_textual_similarity_proof", "Verus Textual Similarity (proof)", "text_quality", "higher_is_better", "native"),
    MetricCatalogEntry("trivial_spec_ratio", "规约平凡性比例", "spec_triviality", "lower_is_better", "native"),
    MetricCatalogEntry("llm_as_judge_intent_consistency", "LLM-as-Judge: Spec 意图一致性", "intent_consistency", "higher_is_better", "native"),
    MetricCatalogEntry("parse_rate", "解析通过率", "formal_validity", "higher_is_better", "native"),
    MetricCatalogEntry("type_check_rate", "类型检查通过率", "formal_validity", "higher_is_better", "native"),
    MetricCatalogEntry("verification_pass_rate", "验证器验证通过率", "formal_validity", "higher_is_better", "native"),
    MetricCatalogEntry("llm_as_judge_spec_code_intent_consistency", "LLM-as-Judge: 代码 Spec 意图一致性", "intent_consistency", "higher_is_better", "native"),
    MetricCatalogEntry("proportion_at_least_gt", "不弱于 GT 比例", "spec_reliability", "higher_is_better", "native"),
    MetricCatalogEntry("postcondition_clause_reliability_rate", "GT 后置条件子句可靠率", "spec_reliability", "higher_is_better", "native"),
    MetricCatalogEntry("precondition_clause_reliability_rate", "GT 前置条件子句可靠率", "spec_reliability", "higher_is_better", "native"),
    MetricCatalogEntry("wrong_io_reject_rate", "错误 I/O 用例拒绝率", "io_testing", "higher_is_better", "native"),
    MetricCatalogEntry("proportion_at_most_gt", "不强于 GT 比例", "spec_completeness", "higher_is_better", "native"),
    MetricCatalogEntry("postcondition_clause_completeness_rate", "GT 后置条件子句完备率", "spec_completeness", "higher_is_better", "native"),
    MetricCatalogEntry("precondition_clause_completeness_rate", "GT 前置条件子句完备率", "spec_completeness", "higher_is_better", "native"),
    MetricCatalogEntry("correct_io_pass_rate", "正确 I/O 用例接收率", "io_testing", "higher_is_better", "native"),
    MetricCatalogEntry("invalid_test_filtering_rate", "无效输入拒绝率", "io_testing", "higher_is_better", "native"),
    MetricCatalogEntry("mutation_kill_rate", "变异击杀率", "robustness", "higher_is_better", "native"),
    MetricCatalogEntry("spec_redundancy_rate", "证明集冗余率", "simplicity", "lower_is_better", "native"),
    # 规约复杂度报告的是原始结构计数，没有归一化分数，也没有好坏方向：
    # 规约过简可能平凡，过繁可能难懂。按源码中规约子句计算。
    MetricCatalogEntry("spec_size_complexity", "规约复杂度", "spec_size", "descriptive", "native"),
    MetricCatalogEntry("spec_size_complexity_spec_only", "规约复杂度 (spec only)", "spec_size", "descriptive", "native"),
    MetricCatalogEntry("spec_size_complexity_proof", "规约复杂度 (proof)", "spec_size", "descriptive", "native"),
    MetricCatalogEntry("verification_time", "验证阶段验证时间", "time_efficiency", "lower_is_better", "native"),
)

DEFAULT_ENABLED_METRIC_IDS = tuple(entry.metric_id for entry in METRIC_CATALOG)
AVAILABLE_METRIC_IDS = DEFAULT_ENABLED_METRIC_IDS
EXTENSION_METRIC_IDS = ("bug_detection_rate",)
