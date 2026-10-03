"""Tests for the extra facts concrete-pair proof harnesses hand to Verus."""

import unittest
from unittest import mock

from metrics_rebuild.share import contract_eval


def _string_context() -> dict:
    return {
        "parameters": [{"name": "text", "type": "&str"}],
        "returns": [{"name": "result", "type": "bool"}],
        "requires": [],
        "ensures": [{"text": "result == (text@.len() == 2)"}],
    }


class StringLiteralFactTest(unittest.TestCase):
    def test_facts_are_absent_unless_requested(self) -> None:
        lines = contract_eval._contract_check_proof_fn_lines(
            _string_context(), {"text": "ab"}, {"result": True}, "harness",
        )
        text = "\n".join(lines)
        self.assertNotIn("reveal_strlit", text)
        self.assertNotIn("text@[0]", text)

    def test_facts_pin_length_and_every_character(self) -> None:
        lines = contract_eval._contract_check_proof_fn_lines(
            _string_context(), {"text": "ab"}, {"result": True}, "harness",
            string_facts=True,
        )
        text = "\n".join(lines)
        self.assertIn('reveal_strlit("ab");', text)
        self.assertIn("assert(text@.len() == 2);", text)
        self.assertIn("assert(text@[0] == 'a');", text)
        self.assertIn("assert(text@[1] == 'b');", text)
        # The reveal has to precede the binding it talks about.
        self.assertLess(text.index("reveal_strlit"), text.index("let text"))

    def test_long_literals_are_left_alone(self) -> None:
        value = "x" * (contract_eval._STRING_FACT_CHAR_CAP + 1)
        lines = contract_eval._contract_check_proof_fn_lines(
            _string_context(), {"text": value}, {"result": False}, "harness",
            string_facts=True,
        )
        text = "\n".join(lines)
        self.assertNotIn("reveal_strlit", text)
        self.assertNotIn("@[0]", text)

    def test_requires_harness_also_reveals_literals(self) -> None:
        context = {
            "parameters": [{"name": "s", "type": "String"}],
            "requires": [{"text": "s@.len() > 0"}],
        }
        lines = contract_eval._requires_check_proof_fn_lines(
            context, {"s": "hi"}, "harness", string_facts=True,
        )
        text = "\n".join(lines)
        self.assertIn('reveal_strlit("hi");', text)
        self.assertIn("assert(s@[1] == 'i');", text)

    def test_string_literal_detection_ignores_other_types(self) -> None:
        context = {
            "parameters": [{"name": "xs", "type": "Vec<u32>"}],
            "returns": [{"name": "r", "type": "u32"}],
        }
        self.assertFalse(
            contract_eval._case_has_string_literal(
                context, {"inputs": {"xs": [1]}, "output": {"r": 1}},
            )
        )
        self.assertTrue(
            contract_eval._case_has_string_literal(
                _string_context(), {"inputs": {"text": "ab"}, "output": {"result": True}},
            )
        )


class FuelRevealTest(unittest.TestCase):
    PREAMBLE = """
spec fn fibo(n: int) -> nat
    decreases n
{
    if n <= 0 { 0 } else if n == 1 { 1 } else { fibo(n - 2) + fibo(n - 1) }
}

spec fn fibo_fits_i32(n: int) -> bool {
    fibo(n) < 0x8000_0000
}

spec fn unrelated(n: int) -> bool {
    n > 0
}
"""

    def test_fuel_follows_a_non_recursive_wrapper(self) -> None:
        lines = contract_eval._fuel_reveal_lines(
            {"spec_preamble": self.PREAMBLE}, ["fibo_fits_i32(n as int)"],
        )
        self.assertEqual(lines, ["    reveal_with_fuel(fibo, 12);"])

    def test_direct_reference_still_gets_fuel(self) -> None:
        lines = contract_eval._fuel_reveal_lines(
            {"spec_preamble": self.PREAMBLE}, ["r == fibo(3)"],
        )
        self.assertEqual(lines, ["    reveal_with_fuel(fibo, 12);"])

    def test_unrelated_clauses_reveal_nothing(self) -> None:
        lines = contract_eval._fuel_reveal_lines(
            {"spec_preamble": self.PREAMBLE}, ["unrelated(n)"],
        )
        self.assertEqual(lines, [])

    def test_decreases_without_self_call_gets_no_fuel(self) -> None:
        # Verus rejects fuel > 1 for a non-recursive function and the whole
        # harness fails to compile, so a bare `decreases` must not be trusted.
        preamble = """
spec fn looks_recursive(arr: Seq<u32>, index: nat) -> u32
    decreases index
{
    if index == 0 { arr[0 as int] } else { arr[index as int] }
}
"""
        lines = contract_eval._fuel_reveal_lines(
            {"spec_preamble": preamble}, ["r == looks_recursive(arr@, 1)"],
        )
        self.assertEqual(lines, [])

    def test_mutual_recursion_counts_as_recursive(self) -> None:
        preamble = """
spec fn is_even(n: int) -> bool
    decreases n
{
    if n == 0 { true } else { is_odd(n - 1) }
}

spec fn is_odd(n: int) -> bool
    decreases n
{
    if n == 0 { false } else { is_even(n - 1) }
}
"""
        lines = contract_eval._fuel_reveal_lines(
            {"spec_preamble": preamble}, ["r == is_even(4)"],
        )
        self.assertEqual(
            lines,
            ["    reveal_with_fuel(is_even, 12);", "    reveal_with_fuel(is_odd, 12);"],
        )

    def test_bodies_are_extracted_with_balanced_braces(self) -> None:
        bodies = contract_eval._spec_fn_bodies(self.PREAMBLE)
        self.assertEqual(set(bodies), {"fibo", "fibo_fits_i32", "unrelated"})
        self.assertIn("fibo(n - 2)", bodies["fibo"])
        self.assertIn("fibo(n)", bodies["fibo_fits_i32"])


class StringFactRetryTest(unittest.TestCase):
    def test_unresolved_string_case_is_retried_with_facts(self) -> None:
        context = _string_context()
        cases = [{"inputs": {"text": "ab"}, "output": {"result": False}, "key": "k0"}]
        check_calls: list[bool] = []

        def fake_check(_context, _cases, *, string_facts=False, return_params=False):
            check_calls.append(string_facts)
            return {"k0": None}

        def fake_batch(items, *_args, **_kwargs):
            # Rejection is only provable once the literal's view is revealed.
            revealed = any("reveal_strlit" in line for line in items[0].lines)
            return {"k0": revealed}

        with mock.patch.object(contract_eval, "batch_verus_contract_check", side_effect=fake_check), \
                mock.patch.object(contract_eval, "_run_and_parse_batch", side_effect=fake_batch):
            result = contract_eval.batch_verus_contract_decide(context, cases)

        self.assertEqual(result, {"k0": False})
        self.assertEqual(check_calls, [False, True])

    def test_no_retry_without_string_literals(self) -> None:
        context = {
            "parameters": [{"name": "x", "type": "u32"}],
            "returns": [{"name": "r", "type": "u32"}],
            "ensures": [{"text": "r == x"}],
        }
        cases = [{"inputs": {"x": 1}, "output": {"r": 2}, "key": "k0"}]
        check_calls: list[bool] = []

        def fake_check(_context, _cases, *, string_facts=False, return_params=False):
            check_calls.append(string_facts)
            return {"k0": None}

        with mock.patch.object(contract_eval, "batch_verus_contract_check", side_effect=fake_check), \
                mock.patch.object(contract_eval, "_run_and_parse_batch", return_value={"k0": None}):
            contract_eval.batch_verus_contract_decide(context, cases)

        self.assertEqual(check_calls, [False])

    def test_resolved_case_is_not_retried(self) -> None:
        context = _string_context()
        cases = [{"inputs": {"text": "ab"}, "output": {"result": True}, "key": "k0"}]
        check_calls: list[bool] = []

        def fake_check(_context, _cases, *, string_facts=False, return_params=False):
            check_calls.append(string_facts)
            return {"k0": True}

        with mock.patch.object(contract_eval, "batch_verus_contract_check", side_effect=fake_check):
            result = contract_eval.batch_verus_contract_decide(context, cases)

        self.assertEqual(result, {"k0": True})
        self.assertEqual(check_calls, [False])

    def test_wellformedness_errors_mark_ill_formed_contracts(self) -> None:
        from metrics_rebuild.share.verus_runner import VerusRun

        def run(stderr: str) -> VerusRun:
            return VerusRun("failed", False, None, 1, 1, 0.1, "", stderr, ())

        trigger = "error: Could not automatically infer triggers for this quantifer."
        self.assertEqual(contract_eval._unresolved_run_reason(run(trigger)), "contract_ill_formed")
        self.assertEqual(contract_eval._unresolved_run_reason(run("error: could not prove termination")), "contract_ill_formed")
        self.assertEqual(contract_eval._unresolved_run_reason(run("error[E0308]: mismatched types")), "compile_error")

    def test_vec_return_gets_parameter_bound_retry(self) -> None:
        context = {
            "parameters": [{"name": "x", "type": "u32"}],
            "returns": [{"name": "r", "type": "Vec<u32>"}],
            "ensures": [{"text": "r@.len() == 1"}],
        }
        cases = [{"inputs": {"x": 1}, "output": {"r": [1]}, "key": "k0"}]
        check_calls: list[tuple[bool, bool]] = []

        def fake_check(_context, _cases, *, string_facts=False, return_params=False):
            check_calls.append((string_facts, return_params))
            return {"k0": True if return_params else None}

        with mock.patch.object(contract_eval, "batch_verus_contract_check", side_effect=fake_check), \
                mock.patch.object(contract_eval, "_run_and_parse_batch", return_value={"k0": None}):
            result = contract_eval.batch_verus_contract_decide(context, cases)

        self.assertEqual(result, {"k0": True})
        self.assertEqual(check_calls, [(False, False), (False, True)])


if __name__ == "__main__":
    unittest.main()
