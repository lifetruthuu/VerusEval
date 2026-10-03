"""Tests for batch poisoning isolation and conservative unsupported handling."""

import unittest
from unittest import mock

from metrics_rebuild.share import contract_eval
from metrics_rebuild.share.contract_eval import contract_evaluation
from metrics_rebuild.share.verus_runner import VerusRun


def _hard_failure_run() -> VerusRun:
    # A non-success run that is neither "unavailable" nor "timeout": the state a
    # sibling compile error produces, which _run_and_parse_batch should isolate.
    return VerusRun(
        status="ok",
        success=False,
        verified=0,
        errors=1,
        returncode=1,
        elapsed_seconds=0.0,
        stdout="",
        stderr="",
        command=(),
    )


class BatchIsolationTest(unittest.TestCase):
    def test_poisoned_batch_recovers_good_cases_via_bisection(self) -> None:
        context = {"ensures": [{"text": "true"}], "spec_preamble": ""}
        cases = [
            {"inputs": {}, "output": {}, "key": "k0"},
            {"inputs": {}, "output": {}, "key": "k1"},
            {"inputs": {}, "output": {}, "key": "k2"},
        ]
        verdicts = {"k0": True, "k1": False, "k2": True}

        def fake_parse(run, items):
            # The full batch is "poisoned" -> all None; sub-batches resolve.
            if len(items) == len(cases):
                return {item.key: None for item in items}
            return {item.key: verdicts[item.key] for item in items}

        with mock.patch.object(contract_eval, "_run_verus_on_text", return_value=_hard_failure_run()), \
                mock.patch.object(contract_eval, "_parse_batch_result", side_effect=fake_parse):
            result = contract_eval.batch_verus_contract_check(context, cases)

        # Without isolation this would be {k0:None, k1:None, k2:None}.
        self.assertEqual(result, verdicts)

    def test_unavailable_batch_is_not_isolated(self) -> None:
        context = {"ensures": [{"text": "true"}], "spec_preamble": ""}
        cases = [{"inputs": {}, "output": {}, "key": "k0"}, {"inputs": {}, "output": {}, "key": "k1"}]
        unavailable = _hard_failure_run().__class__(
            status="unavailable", success=None, verified=None, errors=None,
            returncode=None, elapsed_seconds=0.0, stdout="", stderr="", command=(),
        )
        parse_calls = []

        def fake_parse(run, items):
            parse_calls.append(len(items))
            return {item.key: None for item in items}

        with mock.patch.object(contract_eval, "_run_verus_on_text", return_value=unavailable), \
                mock.patch.object(contract_eval, "_parse_batch_result", side_effect=fake_parse):
            result = contract_eval.batch_verus_contract_check(context, cases)

        # No bisection attempted for unavailable/timeout runs.
        self.assertEqual(result, {"k0": None, "k1": None})
        self.assertEqual(parse_calls, [2])

    def test_contract_failure_is_unknown_until_negation_is_proven(self) -> None:
        context = {"ensures": [{"text": "r == 1"}], "returns": [{"name": "r", "type": "int"}]}
        cases = [{"inputs": {}, "output": {"r": 2}, "key": "k0"}]
        with mock.patch.object(
            contract_eval, "batch_verus_contract_check", return_value={"k0": False},
        ), mock.patch.object(
            contract_eval, "_run_and_parse_batch", return_value={"k0": False},
        ):
            self.assertEqual(
                contract_eval.batch_verus_contract_decide(context, cases),
                {"k0": None},
            )

    def test_contract_false_requires_successful_negation_proof(self) -> None:
        context = {"ensures": [{"text": "r == 1"}], "returns": [{"name": "r", "type": "int"}]}
        cases = [{"inputs": {}, "output": {"r": 2}, "key": "k0"}]
        with mock.patch.object(
            contract_eval, "batch_verus_contract_check", return_value={"k0": False},
        ), mock.patch.object(
            contract_eval, "_run_and_parse_batch", return_value={"k0": True},
        ) as negated:
            self.assertEqual(
                contract_eval.batch_verus_contract_decide(context, cases),
                {"k0": False},
            )
        negation_item = negated.call_args.args[0][0]
        self.assertTrue(any("assert(!" in line for line in negation_item.lines))

    def test_requires_failure_is_unknown_until_negation_is_proven(self) -> None:
        context = {"requires": [{"text": "x > 0"}], "parameters": [{"name": "x", "type": "int"}]}
        cases = [{"inputs": {"x": 0}, "key": "k0"}]
        with mock.patch.object(
            contract_eval, "batch_verus_requires_check", return_value={"k0": False},
        ), mock.patch.object(
            contract_eval, "_run_and_parse_batch", return_value={"k0": False},
        ):
            self.assertEqual(
                contract_eval.batch_verus_requires_decide(context, cases),
                {"k0": None},
            )


class BatchBisectionDepthCapTest(unittest.TestCase):
    def test_depth_cap_bounds_runs_for_fully_poisoned_batch(self) -> None:
        # Every sub-batch is poisoned, so without a cap the recursion would
        # descend to singletons (~2*N runs). The cap must truncate it.
        cap = contract_eval._BATCH_BISECT_MAX_DEPTH
        n = 2 ** (cap + 2)  # large enough that leaves at the cap still have size > 1
        items = [
            contract_eval._BatchItem(key=f"k{i:03d}", fn_name=f"fn{i}", lines=[])
            for i in range(n)
        ]
        run_calls: list[str] = []

        def fake_run(text, filename, timeout_seconds=10):
            run_calls.append(filename)
            return _hard_failure_run()

        with mock.patch.object(contract_eval, "_run_verus_on_text", side_effect=fake_run), \
                mock.patch.object(
                    contract_eval,
                    "_parse_batch_result",
                    side_effect=lambda run, its: {it.key: None for it in its},
                ):
            result = contract_eval._run_and_parse_batch(items, "", "batch_check.rs")

        self.assertEqual(result, {f"k{i:03d}": None for i in range(n)})
        # A complete binary tree truncated at `cap` has 2^(cap+1)-1 nodes.
        self.assertEqual(len(run_calls), 2 ** (cap + 1) - 1)
        # And that is strictly fewer than the uncapped full descent (~2*N-1).
        self.assertLess(len(run_calls), 2 * n - 1)

    def test_depth_cap_surrenders_only_cap_level_leaf_to_none(self) -> None:
        # A single bad case sits below the cap depth. Healthy siblings outside its
        # bisection path are recovered; only the smallest sub-batch the cap allows
        # (which still contains the bad case) is surrendered to None.
        cap = contract_eval._BATCH_BISECT_MAX_DEPTH
        n = 2 ** (cap + 1)
        bad = n // 2 + 1  # forces descent down the right half to the cap depth
        bad_key = f"k{bad:03d}"
        items = [
            contract_eval._BatchItem(key=f"k{i:03d}", fn_name=f"fn{i}", lines=[])
            for i in range(n)
        ]

        def fake_parse(run, its):
            keys = [it.key for it in its]
            if bad_key in keys:
                return {k: None for k in keys}  # poisoned batch
            return {k: True for k in keys}  # healthy batch resolves

        with mock.patch.object(contract_eval, "_run_verus_on_text", return_value=_hard_failure_run()), \
                mock.patch.object(contract_eval, "_parse_batch_result", side_effect=fake_parse):
            result = contract_eval._run_and_parse_batch(items, "", "batch_check.rs")

        none_keys = {k for k, v in result.items() if v is None}
        self.assertIn(bad_key, none_keys)
        # The surrendered leaf has size N / 2^cap.
        self.assertEqual(len(none_keys), n // (2 ** cap))
        # Everything outside that leaf is recovered as a real verdict.
        self.assertTrue(all(v is True for k, v in result.items() if k not in none_keys))


class ConservativeUnsupportedTest(unittest.TestCase):
    # A clause the Python evaluator cannot understand (unsupported syntax) sits
    # next to a clause that evaluates True. Non-strict mode must NOT soften this
    # to accepted=True just because the evaluated count reaches the unsupported
    # count — a False could be hiding in the unsupported clause.
    def _context(self):
        return {
            "requires": [],
            "ensures": [
                {"text": "r == 0"},                      # evaluates (True for r=0)
                {"text": "totally_unknown_builtin(r) && weird"},  # unsupported -> None
            ],
        }

    def test_non_strict_does_not_soften_unsupported_to_true(self) -> None:
        result = contract_evaluation(self._context(), {}, {"r": 0}, strict=False)
        self.assertIsNone(result["accepted"])
        self.assertEqual(result["reason"], "unsupported_expression")

    def test_strict_agrees_with_non_strict_on_unsupported(self) -> None:
        strict = contract_evaluation(self._context(), {}, {"r": 0}, strict=True)
        non_strict = contract_evaluation(self._context(), {}, {"r": 0}, strict=False)
        self.assertEqual(strict["accepted"], non_strict["accepted"])
        self.assertIsNone(strict["accepted"])

    def test_all_supported_clauses_still_accept(self) -> None:
        ctx = {"requires": [], "ensures": [{"text": "r == 0"}]}
        result = contract_evaluation(ctx, {}, {"r": 0}, strict=False)
        self.assertTrue(result["accepted"])


class ContractHarnessPreflightTest(unittest.TestCase):
    def test_bounded_forall_over_concrete_sequence_gets_finite_domain_hint(self) -> None:
        context = {
            "parameters": [
                {"name": "a", "type": "&[i32]"},
                {"name": "offset", "type": "usize"},
            ],
            "returns": [{"name": "result", "type": "Vec<i32>"}],
            "requires": [],
            "ensures": [{
                "text": (
                    "forall|i: int| 0 <= i && i < a.len() ==> "
                    "result@[i] == a[(i + offset as int) % a.len() as int]"
                ),
            }],
        }

        lines = contract_eval._contract_check_proof_fn_lines(
            context,
            {"a": [1, 2], "offset": 1},
            {"result": [2, 1]},
            "check_rotate",
        )
        proof = "\n".join(lines)

        self.assertIn("assert forall|i: int|", proof)
        self.assertIn("implies", proof)
        self.assertIn("assert(i == 0 || i == 1);", proof)

    def test_unbounded_forall_keeps_plain_assertion(self) -> None:
        context = {
            "parameters": [{"name": "a", "type": "&[i32]"}],
            "returns": [{"name": "result", "type": "Vec<i32>"}],
            "requires": [],
            "ensures": [{"text": "forall|i: int| i >= 0 ==> result@[0] == a[0]"}],
        }

        lines = contract_eval._contract_check_proof_fn_lines(
            context, {"a": [1, 2]}, {"result": [1, 2]}, "check_unbounded",
        )
        proof = "\n".join(lines)

        self.assertIn("assert(forall|i: int|", proof)
        self.assertNotIn("assert(i == 0 || i == 1);", proof)

    def test_contract_proof_inherits_generics_and_where_clause(self) -> None:
        context = {
            "parameters": [{"name": "xs", "type": "Vec<T>"}],
            "returns": [{"name": "r", "type": "int"}],
            "requires": [],
            "ensures": [{"text": "xs@.len() == 0 && r == 0"}],
            "generic_parameters": "<T: Copy>",
            "where_clause": "where T: Eq,",
        }
        lines = contract_eval._contract_check_proof_fn_lines(
            context, {"xs": []}, {"r": 0}, "check_generic",
        )
        self.assertEqual(lines[0], "proof fn check_generic<T: Copy>(xs: Vec<T>)")
        self.assertEqual(lines[1], "    where T: Eq,")

    def test_nonempty_generic_literal_is_unknown_without_running_verus(self) -> None:
        context = {
            "parameters": [{"name": "xs", "type": "Vec<T>"}],
            "returns": [{"name": "r", "type": "int"}],
            "requires": [],
            "ensures": [{"text": "xs@[0] == xs@[0] && r == 0"}],
            "generic_parameters": "<T>",
            "where_clause": "",
        }
        cases = [{"key": "k0", "inputs": {"xs": [1]}, "output": {"r": 0}}]
        with mock.patch.object(contract_eval, "_run_verus_on_text") as run:
            result = contract_eval.batch_verus_contract_decide_detailed(context, cases)
        run.assert_not_called()
        self.assertIsNone(result["k0"]["accepted"])
        self.assertEqual(result["k0"]["reason"], "unresolved_generic_instantiation")

    def test_malformed_clause_is_unknown_without_running_verus(self) -> None:
        context = {
            "parameters": [],
            "returns": [{"name": "r", "type": "int"}],
            "requires": [],
            "ensures": [{"text": "(r == 0"}],
        }
        cases = [{"key": "k0", "inputs": {}, "output": {"r": 0}}]
        with mock.patch.object(contract_eval, "_run_verus_on_text") as run:
            result = contract_eval.batch_verus_contract_decide_detailed(context, cases)
        run.assert_not_called()
        self.assertIsNone(result["k0"]["accepted"])
        self.assertEqual(result["k0"]["reason"], "malformed_clause")
        self.assertEqual(result["k0"]["reason_detail"], "unclosed_(")

    def test_detailed_unknown_diagnostics_use_one_representative_case(self) -> None:
        context = {
            "parameters": [],
            "returns": [{"name": "r", "type": "int"}],
            "requires": [],
            "ensures": [{"text": "r >= 0"}],
        }
        cases = [
            {"key": f"k{index}", "inputs": {}, "output": {"r": index}}
            for index in range(5)
        ]
        with mock.patch.object(
            contract_eval,
            "batch_verus_contract_decide",
            return_value={case["key"]: None for case in cases},
        ), mock.patch.object(
            contract_eval,
            "_diagnose_contract_case",
            return_value={"accepted": None, "reason": "compile_error"},
        ) as diagnose:
            result = contract_eval.batch_verus_contract_decide_detailed(context, cases)
        diagnose.assert_called_once()
        self.assertEqual({item["reason"] for item in result.values()}, {"compile_error"})


if __name__ == "__main__":
    unittest.main()
