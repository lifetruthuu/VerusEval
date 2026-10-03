from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import hashlib

from scripts.plot_gallery import check_group

from scripts.reporting.metric_overview import (
    CONSERVATIVE_IMPLICATION_METRICS,
    load_conservative_implication_scores,
)


def _implication_result(*, passed: int, failed: int, unknown: int) -> dict:
    total = passed + failed + unknown
    determined = passed + failed
    return {
        "generated": {
            "status": "partial" if unknown else "ok",
            "score": passed / determined if determined else None,
            "passed": passed,
            "failed": failed,
            "unknown": unknown,
            "total": total,
            "determined": determined,
            "coverage": determined / total if total else 0.0,
        }
    }


class TestPlotConservativeImplicationScores(unittest.TestCase):
    def test_gallery_rejects_stale_scores_and_changed_source_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'per_file').mkdir()
            record = root / 'per_file/example.json'
            record.write_text(json.dumps({'metrics': {'parse_rate': {'generated': {'score': 1.0}}}}))
            table = root / 'scores.csv'
            table.write_text('filename,parse_rate_score\nexample.rs,1.0\n')
            expected = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (table, record)}
            self.assertEqual(check_group(table, expected, root)[0], 1)
            table.write_text('filename,parse_rate_score\nexample.rs,0.0\n')
            with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
                check_group(table, expected, root)
            expected['scores.csv'] = hashlib.sha256(table.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'CSV/JSON score mismatch'):
                check_group(table, expected, root)

    def test_missing_implication_results_receive_zero_for_every_chart_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            per_file_dir = Path(tmp)
            (per_file_dir / "missing.json").write_text(
                json.dumps({"metrics": {}}),
                encoding="utf-8",
            )

            loaded = load_conservative_implication_scores(per_file_dir)

            for metric_id in CONSERVATIVE_IMPLICATION_METRICS:
                self.assertEqual(loaded[metric_id]["scores"], [0.0])
                self.assertEqual(loaded[metric_id]["scored"], 1)
                self.assertEqual(loaded[metric_id]["na"], 0)

    def test_unknown_is_zero_and_derived_rule_scores_use_conservative_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            per_file_dir = Path(tmp)
            metrics = {
                "precondition_clause_reliability_rate": _implication_result(
                    passed=1, failed=0, unknown=1
                ),
                "postcondition_clause_reliability_rate": _implication_result(
                    passed=2, failed=0, unknown=0
                ),
                "precondition_clause_completeness_rate": _implication_result(
                    passed=0, failed=1, unknown=1
                ),
                "postcondition_clause_completeness_rate": _implication_result(
                    passed=1, failed=0, unknown=1
                ),
                "proportion_at_least_gt": _implication_result(
                    passed=1, failed=0, unknown=1
                ),
                "proportion_at_most_gt": _implication_result(
                    passed=0, failed=1, unknown=1
                ),
            }
            original = {"metrics": metrics, "untouched": {"value": 7}}
            path = per_file_dir / "sample.json"
            path.write_text(json.dumps(original), encoding="utf-8")

            loaded = load_conservative_implication_scores(per_file_dir)

            self.assertEqual(set(loaded), CONSERVATIVE_IMPLICATION_METRICS)
            self.assertEqual(
                loaded["precondition_clause_reliability_rate"]["scores"],
                [0.5],
            )
            self.assertEqual(
                loaded["precondition_clause_reliability_rate"]["semantic_coverage"],
                0.5,
            )
            self.assertEqual(loaded["precondition_reliability_rate"]["scores"], [0.0])
            self.assertEqual(loaded["postcondition_reliability_rate"]["scores"], [1.0])
            self.assertEqual(loaded["rule_reliability_rate"]["scores"], [0.0])
            self.assertEqual(loaded["approx_rule_reliability_rate"]["scores"], [0.75])
            self.assertEqual(loaded["precondition_completeness_rate"]["scores"], [0.0])
            self.assertEqual(loaded["postcondition_completeness_rate"]["scores"], [0.0])
            self.assertEqual(loaded["rule_completeness_rate"]["scores"], [0.0])
            self.assertEqual(loaded["approx_rule_completeness_rate"]["scores"], [0.25])
            self.assertEqual(loaded["rule_correctness_rate"]["scores"], [0.0])
            self.assertEqual(loaded["approx_rule_correctness_rate"]["scores"], [0.5])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), original)


if __name__ == "__main__":
    unittest.main()
