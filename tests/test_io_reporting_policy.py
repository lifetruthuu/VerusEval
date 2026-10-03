import unittest
from scripts.evaluation import evaluate

class TestNumericPartialReportingPolicy(unittest.TestCase):
    @staticmethod
    def _results() -> dict:
        return {
            "ok.rs": {
                "metrics": {
                    "metric_a": {"generated": {"status": "ok", "score": 1.0}},
                    "metric_b": {"generated": {"status": "ok", "score": 2.0}},
                }
            },
            "partial.rs": {
                "metrics": {
                    "metric_a": {"generated": {"status": "partial", "score": 0.0}},
                    "metric_b": {"generated": {"status": "partial", "score": 1.0}},
                }
            },
            "na.rs": {
                "metrics": {
                    "metric_a": {"generated": {"status": "not_available", "score": None}},
                    "metric_b": {"generated": {"status": "partial", "score": 0.0}},
                }
            },
            "partial2.rs": {
                "metrics": {
                    "metric_a": {"generated": {"status": "partial", "score": 0.5}},
                    "metric_b": {"generated": {"status": "partial", "score": 0.5}},
                }
            },
        }

    def test_evaluation_statistics_include_numeric_partial_scores(self) -> None:
        stats = evaluate.build_statistics(self._results(), ["metric_a"])["metric_a"]
        self.assertEqual(stats["ok_count"], 1)
        self.assertEqual(stats["scored_count"], 3)
        self.assertEqual(stats["scores_count"], 3)
        self.assertEqual(stats["mean"], 0.5)

