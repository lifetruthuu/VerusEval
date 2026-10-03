import unittest

from scripts.evaluation.evaluate import (
    DERIVED_RULE_METRIC_NAMES,
    add_derived_rule_metrics,
    metric_names_with_derived_rule_metrics,
)


class RuleDerivedMetricsTest(unittest.TestCase):
    def test_adds_rule_metrics_from_clause_scores(self) -> None:
        result = {
            "metrics": {
                "precondition_clause_reliability_rate": {"generated": {"status": "ok", "score": 0.0}},
                "postcondition_clause_reliability_rate": {"generated": {"status": "ok", "score": 1.0}},
                "precondition_clause_completeness_rate": {"generated": {"status": "ok", "score": 1.0}},
                "postcondition_clause_completeness_rate": {"generated": {"status": "ok", "score": 1.0}},
            }
        }

        add_derived_rule_metrics(result)
        metrics = result["metrics"]

        self.assertEqual(metrics["precondition_reliability_rate"]["score"], 0.0)
        self.assertEqual(metrics["postcondition_reliability_rate"]["score"], 1.0)
        self.assertEqual(metrics["rule_reliability_rate"]["score"], 0.0)
        self.assertEqual(metrics["approx_rule_reliability_rate"]["score"], 0.5)
        self.assertEqual(metrics["precondition_completeness_rate"]["score"], 1.0)
        self.assertEqual(metrics["postcondition_completeness_rate"]["score"], 1.0)
        self.assertEqual(metrics["rule_completeness_rate"]["score"], 1.0)
        self.assertEqual(metrics["approx_rule_completeness_rate"]["score"], 1.0)
        self.assertEqual(metrics["rule_correctness_rate"]["score"], 0.0)
        self.assertEqual(metrics["approx_rule_correctness_rate"]["score"], 0.75)
        self.assertEqual(result["rule_metrics"]["approx_rule_correctness_rate"], 0.75)
        self.assertEqual(
            result["metric_scores"]["precondition_clause_reliability_rate"],
            0.0,
        )

    def test_metric_names_append_derived_metrics_once(self) -> None:
        names = metric_names_with_derived_rule_metrics(["parse_rate", "rule_reliability_rate"])

        self.assertEqual(names[0], "parse_rate")
        self.assertEqual(names.count("rule_reliability_rate"), 1)
        for metric_name in DERIVED_RULE_METRIC_NAMES:
            self.assertIn(metric_name, names)


if __name__ == "__main__":
    unittest.main()
