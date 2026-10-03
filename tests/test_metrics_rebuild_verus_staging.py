from __future__ import annotations

import unittest
from unittest.mock import patch

from metrics_rebuild.share import verus_runner
from metrics_rebuild.share.verus_runner import VerusRun


def _run(*, success: bool, stderr: str = "", no_verify: bool = False) -> VerusRun:
    return VerusRun(
        status="ok",
        success=success,
        verified=0 if no_verify else 2,
        errors=0 if success else 1,
        returncode=0 if success else 1,
        elapsed_seconds=0.1,
        stdout="{}",
        stderr=stderr,
        command=("verus", "--no-verify", "case.rs") if no_verify else ("verus", "case.rs"),
        encountered_vir_error=False,
    )


class StagedVerusClassificationTest(unittest.TestCase):
    def test_frontend_pass_turns_ambiguous_full_failure_into_verification_failure(self) -> None:
        frontend = _run(success=True, no_verify=True)
        full = _run(
            success=False,
            stderr=(
                '{"$message_type":"diagnostic","message":"invariant not satisfied before loop",'
                '"level":"error","code":null}'
            ),
        )
        with patch.object(verus_runner, "run_verus", side_effect=[frontend, full]) as run:
            result = verus_runner.verus_staged_verification_to_dict("case.rs")

        self.assertEqual(result["outcome_status"], "verification_failed")
        self.assertEqual(result["stage"], "verification")
        self.assertEqual(run.call_count, 2)
        self.assertTrue(run.call_args_list[0].kwargs["no_verify"])
        self.assertFalse(run.call_args_list[1].kwargs["no_verify"])

    def test_frontend_failure_skips_full_verification(self) -> None:
        frontend = _run(
            success=False,
            no_verify=True,
            stderr=(
                '{"$message_type":"diagnostic","message":"mismatched types",'
                '"level":"error","code":{"code":"E0308"}}'
            ),
        )
        with patch.object(verus_runner, "run_verus", return_value=frontend) as run:
            result = verus_runner.verus_staged_verification_to_dict("case.rs")

        self.assertEqual(result["outcome_status"], "frontend_failed")
        self.assertEqual(result["stage"], "frontend")
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
