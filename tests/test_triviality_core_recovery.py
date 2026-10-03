from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from metrics_rebuild.share import proof_probe
from tests.test_trivial_proof_probes import VERUS_AVAILABLE, _write_verus_file


UNKNOWN = {"holds": None, "status": "unknown", "reason": "verification_unresolved",
           "verification_issue": "trigger_inference"}
REJECTED = {"holds": False, "status": "invalid"}
PROVED = {"holds": True, "status": "valid"}


class TrivialityCoreRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def source(self, body):
        return _write_verus_file(Path(self.directory.name), "input.rs", body)

    def test_repaired_triggers_reach_actual_proof_clauses(self):
        for field, probe in (("requires", proof_probe.probe_precondition_falsity_for_path),
                             ("ensures", proof_probe.probe_postcondition_truth_for_path)):
            with self.subTest(field=field):
                path = self.source(
                    f"fn target(a: &Vec<i32>) {field} "
                    "forall|i: int| #[trigger] (0 <= i < a.len()) ==> a[i] > 0, {}"
                )
                calls = []

                def run(**kwargs):
                    calls.append(kwargs)
                    clauses = kwargs[field + "_clauses"]
                    return dict(REJECTED if "#![trigger a[i]]" in clauses[0] else UNKNOWN)

                with patch.object(proof_probe, "run_probe_lemma", side_effect=run):
                    result = probe(path, function_name="target")
                detail = result["functions"][0]["probe"]
                self.assertEqual(detail["fallback_method"], "explicit_triggers")
                self.assertEqual(len(calls), 3)
                self.assertEqual(detail["initial_probe"]["verification_issue"], "trigger_inference")
                self.assertEqual(len(detail["fallback_attempts"]), 2)
                if field == "ensures":
                    self.assertEqual(calls[-1]["requires_clauses"], [])
                json.dumps(result)

    def test_split_uses_only_selected_conjunct_and_serializes(self):
        path = self.source(
            "fn target(x: i32) -> (r: i32) requires false, "
            "ensures r == 0, forall|c: char| c == '\n', { x }"
        )
        calls = []

        def run(**kwargs):
            calls.append(kwargs)
            if kwargs["ensures_clauses"] == ["r == 0"]:
                self.assertEqual(kwargs["requires_clauses"], [])
                self.assertNotIn("requires", Path(kwargs["host_rs_path"]).read_text())
                return dict(REJECTED)
            return dict(UNKNOWN)

        with patch.object(proof_probe, "run_probe_lemma", side_effect=run):
            result = proof_probe.probe_postcondition_truth_for_path(path, function_name="target")
        entry = result["functions"][0]
        self.assertIs(entry["holds"], False)
        self.assertEqual(entry["probe"]["fallback_method"], "conjunct")
        self.assertEqual(entry["probe"]["fallback_clause_index"], 0)
        self.assertEqual(len(calls), 3)
        json.dumps(result)

    def test_successful_conjuncts_do_not_upgrade_unknown_whole(self):
        path = self.source("fn target(x: i32) ensures x == x, x <= x, {}")

        def run(**kwargs):
            return dict(PROVED if len(kwargs["ensures_clauses"]) == 1 else UNKNOWN)

        with patch.object(proof_probe, "run_probe_lemma", side_effect=run):
            result = proof_probe.probe_postcondition_truth_for_path(path, function_name="target")
        self.assertEqual(result["unresolved"], 1)
        self.assertIsNone(result["functions"][0]["holds"])
        self.assertEqual(len(result["functions"][0]["probe"]["fallback_attempts"]), 3)
        json.dumps(result)

    def test_decided_disabled_and_missing_binary_do_not_retry(self):
        path = self.source("fn target(x: i32) requires x > 0, {}")
        cases = [(PROVED, True), (REJECTED, True), (UNKNOWN, False),
                 ({"holds": None, "status": "unknown", "reason": "verus_binary_not_found"}, True)]
        for initial, enabled in cases:
            with self.subTest(initial=initial, enabled=enabled):
                with patch.object(proof_probe, "run_probe_lemma", return_value=dict(initial)) as run:
                    result = proof_probe.probe_precondition_falsity_for_path(
                        path, function_name="target", recover_unknown=enabled)
                run.assert_called_once()
                self.assertEqual(result["functions"][0]["probe"], initial)

    def test_unresolved_diagnostics_are_retained(self):
        path = self.source("fn target(x: i32) requires x > 0, {}")
        retry = {"holds": None, "status": "unknown", "reason": "verification_unresolved",
                 "verification_issue": "termination_check"}
        with patch.object(proof_probe, "run_probe_lemma", side_effect=[dict(UNKNOWN), retry]):
            result = proof_probe.probe_precondition_falsity_for_path(path, function_name="target")
        detail = result["functions"][0]["probe"]
        self.assertEqual(detail["verification_issue"], "termination_check")
        self.assertEqual(detail["initial_probe"], UNKNOWN)
        self.assertEqual(len(detail["fallback_attempts"]), 1)
        self.assertEqual(len(detail["source_sha256"]), 64)

    def test_changed_logical_clause_is_not_probed(self):
        path = self.source("fn target(x: i32) requires x > 0, {}")
        with patch.object(proof_probe, "run_probe_lemma", return_value=dict(UNKNOWN)) as run:
            with patch.object(proof_probe, "isolated_host", return_value=(
                    "verus! { fn target(x: i32) requires false, {} }", {})):
                result = proof_probe.probe_precondition_falsity_for_path(path, function_name="target")
        run.assert_called_once()
        self.assertEqual(result["functions"][0]["probe"]["reason"], "isolated_contract_mismatch")

    def test_exhausted_recovery_budget_stops_subprocess_attempts(self):
        path = self.source("fn target(x: i32) requires x > 0, {}")
        with patch.object(proof_probe, "run_probe_lemma", return_value=dict(UNKNOWN)) as run:
            with patch.object(proof_probe.time, "monotonic", side_effect=[0, 10]):
                result = proof_probe.probe_precondition_falsity_for_path(
                    path, function_name="target", timeout_seconds=1)
        run.assert_called_once()
        self.assertEqual(result["functions"][0]["probe"]["reason"], "recovery_budget_exhausted")

    @unittest.skipUnless(VERUS_AVAILABLE, "Verus not available")
    def test_generic_mutable_old_context_survives_recovery(self):
        path = self.source(
            "fn target<T>(a: &mut Vec<T>) where T: Copy "
            "ensures old(a).len() == a.len(), {}"
        )
        actual_run = proof_probe.run_probe_lemma
        first = True

        def run(**kwargs):
            nonlocal first
            if first:
                first = False
                return dict(UNKNOWN)
            self.assertEqual(kwargs["parameters"], [{"name": "a", "type": "&mut Vec<T>"}])
            self.assertEqual(kwargs["ensures_clauses"], ["old(a).len() == a.len()"])
            return actual_run(**kwargs)

        with patch.object(proof_probe, "run_probe_lemma", side_effect=run):
            result = proof_probe.probe_postcondition_truth_for_path(path, function_name="target")
        self.assertEqual(result["status"], "ok", result)
        self.assertIs(result["functions"][0]["holds"], False)

    @unittest.skipUnless(VERUS_AVAILABLE, "Verus not available")
    def test_required_nonterminating_spec_remains_unknown(self):
        path = self.source(
            "spec fn recursive(x: int) -> int { recursive(x) } "
            "fn target(x: i32) -> (r: i32) ensures r == recursive(x as int), { x }"
        )
        result = proof_probe.probe_postcondition_truth_for_path(path, function_name="target")
        self.assertEqual(result["unresolved"], 1)
        self.assertIsNone(result["functions"][0]["holds"])
        self.assertIn("initial_probe", result["functions"][0]["probe"])


if __name__ == "__main__":
    unittest.main()
