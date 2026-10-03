from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from metrics_rebuild.metrics.llm import intent_consistency as rebuild_intent
from metrics_rebuild.metrics.llm import spec_code_intent_consistency as rebuild_spec_code
from metrics_rebuild.metrics.llm.common import llm_contract_blocks_for_path


def _frontend(success: bool = True, outcome: str = "verified") -> dict:
    return {
        "status": "ok",
        "outcome_status": outcome,
        "success": success,
        "verified": 1 if success else 0,
        "errors": 0 if success else 1,
        "returncode": 0 if success else 1,
        "elapsed_seconds": 0.0,
        "command": ["verus", "--no-verify", "fixture.rs"],
        "stderr": "",
        "stdout": "",
        "score": 1.0 if success else 0.0,
        "method": "verus_no_verify_frontend_check",
        "note": "fixture",
    }


def _write_pair(tmpdir: Path, generated_text: str, ground_text: str) -> tuple[Path, Path]:
    generated = tmpdir / "eval.rs"
    ground = tmpdir / "ref.rs"
    generated.write_text(generated_text, encoding="utf-8")
    ground.write_text(ground_text, encoding="utf-8")
    return generated, ground


VALID_RS = """
verus! {
fn inc(x: int) -> (y: int)
    requires x >= 0,
    ensures y > x,
{
    x + 1
}
}
"""

NO_SPEC_RS = """
verus! {
fn inc(x: int) -> (y: int) {
    x + 1
}
}
"""


class MetricsRebuildLlmTest(unittest.TestCase):
    def test_frontend_failure_short_circuit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generated, ground = _write_pair(Path(tmp), "not rust", VALID_RS)
            with patch.object(rebuild_intent, "verus_frontend_run_to_dict", side_effect=[_frontend(False, "parse_error"), _frontend(True)]):
                result = rebuild_intent.metric_llm_as_judge_intent_consistency(str(generated), str(ground))
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["score"], 0.0)
            self.assertEqual(result["generated"]["verdict"], "invalid_frontend")
            self.assertEqual(result["llm"]["status"], "skipped")

            with patch.object(rebuild_spec_code, "verus_frontend_run_to_dict", side_effect=[_frontend(False, "parse_error"), _frontend(True)]):
                result = rebuild_spec_code.metric_llm_as_judge_spec_code_intent_consistency(str(generated), str(ground))
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["score"], 0.0)
            self.assertEqual(result["generated"]["verdict"], "invalid_frontend")
            self.assertEqual(result["llm"]["status"], "skipped")

    def test_identical_intent_short_circuit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generated, ground = _write_pair(Path(tmp), VALID_RS, VALID_RS)
            with patch.object(rebuild_intent, "verus_frontend_run_to_dict", side_effect=[_frontend(True), _frontend(True)]):
                result = rebuild_intent.metric_llm_as_judge_intent_consistency(str(generated), str(ground))
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["score"], 1.0)
            self.assertEqual(result["generated"]["verdict"], "equivalent")
            self.assertEqual(result["llm"]["reason"], "identical_files")

    def test_missing_llm_config(self) -> None:
        unavailable = {
            "status": "not_available",
            "reason": "missing_llm_config:api_key",
            "llm": {
                "status": "not_available",
                "model_name": "fixture-model",
                "base_url_configured": True,
                "api_key_configured": False,
                "reason": "missing_llm_config:api_key",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            generated, ground = _write_pair(Path(tmp), VALID_RS.replace("y > x", "y >= x"), VALID_RS)
            with patch.object(rebuild_intent, "verus_frontend_run_to_dict", side_effect=[_frontend(True), _frontend(True)]), \
                 patch.object(rebuild_intent, "call_llm_json", return_value=unavailable):
                payload = rebuild_intent.metric_llm_as_judge_intent_consistency(str(generated), str(ground))
            self.assertEqual(payload["status"], "not_available")
            self.assertEqual(payload["reason"], "missing_llm_config:api_key")

            with patch.object(rebuild_spec_code, "verus_frontend_run_to_dict", side_effect=[_frontend(True), _frontend(True)]), \
                 patch.object(rebuild_spec_code, "call_llm_json", return_value=unavailable):
                payload = rebuild_spec_code.metric_llm_as_judge_spec_code_intent_consistency(str(generated), str(ground))
            self.assertEqual(payload["status"], "not_available")
            self.assertEqual(payload["reason"], "missing_llm_config:api_key")

    def test_fake_llm_json(self) -> None:
        intent_result = {
            "status": "ok",
            "json": {
                "score": 0.75,
                "verdict": "too_weak",
                "missing_constraints": ["postcondition weaker"],
                "extra_or_overstrong_constraints": [],
                "vacuity_risks": [],
                "reasoning": "fixture",
            },
            "llm": {"status": "ok", "model_name": "fixture", "base_url_configured": True, "api_key_configured": True, "client": "mock"},
            "cached": False,
            "raw": "{}",
            "attempts": 1,
        }
        spec_code_result = {
            "status": "ok",
            "json": {
                "score": 0.5,
                "verdict": "too_weak",
                "function_judgments": [{"function": "inc", "score": 0.5}],
                "missing_behavior": ["missing strict postcondition"],
                "overstrong_or_wrong_constraints": [],
                "vacuity_risks": [],
                "reasoning": "fixture",
            },
            "llm": {"status": "ok", "model_name": "fixture", "base_url_configured": True, "api_key_configured": True, "client": "mock"},
            "cached": False,
            "raw": "{}",
            "attempts": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            generated, ground = _write_pair(Path(tmp), VALID_RS.replace("y > x", "y >= x"), VALID_RS)
            with patch.object(rebuild_intent, "verus_frontend_run_to_dict", side_effect=[_frontend(True), _frontend(True)]), \
                 patch.object(rebuild_intent, "call_llm_json", return_value=intent_result):
                payload = rebuild_intent.metric_llm_as_judge_intent_consistency(str(generated), str(ground))
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["score"], 0.75)
            self.assertEqual(payload["generated"]["verdict"], "too_weak")

            with patch.object(rebuild_spec_code, "verus_frontend_run_to_dict", side_effect=[_frontend(True), _frontend(True)]), \
                 patch.object(rebuild_spec_code, "call_llm_json", return_value=spec_code_result):
                payload = rebuild_spec_code.metric_llm_as_judge_spec_code_intent_consistency(str(generated), str(ground))
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["score"], 0.5)
            self.assertEqual(payload["generated"]["verdict"], "too_weak")

    def test_no_contract_spec_functions_yields_vacuous_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generated, ground = _write_pair(Path(tmp), NO_SPEC_RS, VALID_RS)
            with patch.object(rebuild_spec_code, "verus_frontend_run_to_dict", side_effect=[_frontend(True), _frontend(True)]):
                payload = rebuild_spec_code.metric_llm_as_judge_spec_code_intent_consistency(str(generated), str(ground))
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["score"], 0.0)
            self.assertEqual(payload["generated"]["verdict"], "vacuous")

    def test_no_executable_spec_functions_yields_zero_with_marker(self) -> None:
        empty_rs = "verus! {\n}\n"
        with tempfile.TemporaryDirectory() as tmp:
            generated, ground = _write_pair(Path(tmp), empty_rs, VALID_RS)
            with patch.object(rebuild_spec_code, "verus_frontend_run_to_dict", side_effect=[_frontend(True), _frontend(True)]):
                payload = rebuild_spec_code.metric_llm_as_judge_spec_code_intent_consistency(str(generated), str(ground))
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["score"], 0.0)
            self.assertEqual(payload["generated"]["verdict"], "no_executable_spec_functions_found")
            self.assertEqual(payload["generated"]["reason"], "no_executable_spec_functions_found")
            self.assertEqual(payload["llm"]["reason"], "no_executable_spec_functions_found")

    def test_boolean_clause_text_does_not_truncate_function_signature(self) -> None:
        source = """
        verus! {
        fn choose(flag: bool, x: int) -> (result: bool)
            requires flag,
            ensures result,
        { result }
        }
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.rs"
            path.write_text(source, encoding="utf-8")
            block = llm_contract_blocks_for_path(str(path))[0]

        self.assertEqual(
            block["signature"],
            "fn choose(flag: bool, x: int) -> (result: bool)",
        )
        self.assertIn("fn choose(flag: bool, x: int)", block["contract"])


if __name__ == "__main__":
    unittest.main()
