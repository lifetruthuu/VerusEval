from __future__ import annotations

import unittest

from scripts.evaluation.smoke_lemma_implication import _concise_metric_result


class LemmaSmokeReportTest(unittest.TestCase):
    def test_pair_shaped_clause_metric_uses_generated_result(self) -> None:
        result = _concise_metric_result(
            {
                "generated": {
                    "status": "partial",
                    "score": 1.0,
                    "passed": 1,
                    "failed": 0,
                    "unknown": 1,
                    "determined": 1,
                    "total": 2,
                    "coverage": 0.5,
                },
                "ground": {"status": "ok", "score": 1.0},
                "delta": 0.0,
            },
            0.25,
        )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["unknown"], 1)
        self.assertEqual(result["coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
