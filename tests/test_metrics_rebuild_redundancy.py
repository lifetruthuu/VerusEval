from __future__ import annotations

import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics_rebuild.metrics.spec_redundancy_rate import metric_spec_redundancy_rate  # noqa: E402
from metrics_rebuild.share import redundancy  # noqa: E402
from metrics_rebuild.share.config import set_verus_binary  # noqa: E402
from metrics_rebuild.share.redundancy import (  # noqa: E402
    extract_removal_candidates_from_text,
    remove_candidate_text,
    remove_candidates_text,
)
from metrics_rebuild.share.verus_runner import VerusRun  # noqa: E402


# --- VerusRun fixtures whose outcome_status classifies as intended ------------
# Full-run failures are now confirmed with a separate --no-verify frontend run,
# so these fixtures cover verified / verification_failed / parse_error without
# a real Verus binary.

def _verified() -> VerusRun:
    return VerusRun(
        status="ok", success=True, verified=1, errors=0, returncode=0,
        elapsed_seconds=0.0, stdout="", stderr="", command=(), encountered_vir_error=False,
    )


def _verification_failed() -> VerusRun:
    # encountered_vir_error=True forces verus_run_to_dict -> "verification_failed"
    # (otherwise an empty-stderr failure is treated as a parse error).
    return VerusRun(
        status="ok", success=False, verified=0, errors=1, returncode=1,
        elapsed_seconds=0.0, stdout="", stderr="", command=(), encountered_vir_error=True,
    )


def _parse_error() -> VerusRun:
    # success=False with no VIR error and no error diagnostics -> "parse_error".
    return VerusRun(
        status="ok", success=False, verified=None, errors=None, returncode=1,
        elapsed_seconds=0.0, stdout="", stderr="", command=(), encountered_vir_error=False,
    )


def _timeout() -> VerusRun:
    return VerusRun(
        status="timeout", success=None, verified=None, errors=None, returncode=None,
        elapsed_seconds=12.0, stdout="", stderr="", command=(), encountered_vir_error=None,
    )


THREE_ASSERTS = textwrap.dedent(
    """\
    use vstd::prelude::*;
    verus! {
    proof fn f() {
        assert(a);
        assert(b);
        assert(c);
    }
    } // verus!
    """
)


class RedundancyControlFlowTest(unittest.TestCase):
    """Mock-driven tests of the optimistic + greedy control flow (no Verus)."""

    def setUp(self) -> None:
        redundancy._HOUDINI_REDUNDANCY_CACHE.clear()
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir, ignore_errors=True)
        self.src = Path(self._tmpdir) / "t.rs"
        self.src.write_text(THREE_ASSERTS, encoding="utf-8")

    def _run(self, side_effect, frontend_side_effect=None, *, runtime=None, **kwargs):
        if frontend_side_effect is None:
            frontend_side_effect = lambda text, tmpdir, filename, timeout_seconds: _verified()
        if runtime is None:
            runtime = {"configured_path": "verus-a", "version": "1"}
        with mock.patch.object(redundancy, "run_verus", return_value=_verified()), \
                mock.patch.object(redundancy, "run_verus_on_text", side_effect=side_effect) as on_text, \
                mock.patch.object(redundancy, "verus_runtime_fingerprint", return_value=runtime), \
                mock.patch.object(
                    redundancy,
                    "run_verus_frontend_on_text",
                    side_effect=frontend_side_effect,
                ) as frontend:
            result = redundancy.houdini_redundancy_for_path(str(self.src), **kwargs)
        return result, on_text, frontend

    def test_optimistic_removes_all_when_jointly_redundant(self) -> None:
        # Every clause independently removable AND jointly removable -> one
        # optimistic reverify strips all three; greedy then finds nothing.
        def se(text, tmpdir, filename, timeout_seconds):
            return _verified()

        result, on_text, frontend = self._run(se)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["strategy"], "optimistic")
        self.assertEqual(result["removable_clause_candidates"], 3)
        self.assertEqual(result["clauses_total"], 3)
        self.assertEqual(result["unknown_clause_candidates"], 0)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["coverage"], 1.0)
        # 3 independent delta runs + 1 optimistic; greedy scan finds no clauses.
        self.assertEqual(on_text.call_count, 4)
        self.assertEqual(frontend.call_count, 0)

    def test_greedy_fallback_when_optimistic_fails(self) -> None:
        # All three independently removable, but joint removal fails (they
        # interact); greedy commits one and the rest turn necessary.
        def se(text, tmpdir, filename, timeout_seconds):
            if "optimistic_" in filename:
                return _verification_failed()
            if "greedy_0_" in filename:
                return _verified()
            if "greedy_" in filename:
                return _verification_failed()
            return _verified()  # delta_* independent pass

        result, on_text, frontend = self._run(se)
        self.assertEqual(result["strategy"], "greedy")
        self.assertEqual(result["removable_clause_candidates"], 1)
        # resolved_total = 3 (no unknown); necessary = resolved - removed = 2.
        self.assertEqual(result["necessary_clause_candidates"], 2)
        self.assertEqual(result["unknown_clause_candidates"], 0)
        self.assertAlmostEqual(result["score"], 1 / 3)
        # independent pass still saw all three as individually removable.
        self.assertEqual(result["independent_delta_pass"]["removable"], 3)
        self.assertEqual(frontend.call_count, 0)

    def test_parse_error_deletion_is_unknown_not_necessary(self) -> None:
        # The middle clause's deletion breaks syntax (parse_error). It must be
        # classified unknown and excluded from the denominator, NOT counted as a
        # necessary clause (the core bug this refactor fixes).
        def se(text, tmpdir, filename, timeout_seconds):
            if "delta_1_" in filename or "greedy_" in filename:
                return _parse_error()
            return _verified()

        def frontend_se(text, tmpdir, filename, timeout_seconds):
            return _parse_error()

        result, on_text, frontend = self._run(se, frontend_se)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["unknown_clause_candidates"], 1)
        self.assertEqual(result["necessary_clause_candidates"], 0)
        # Unknown candidates remain in the conservative denominator.
        self.assertAlmostEqual(result["score"], 2 / 3)
        self.assertAlmostEqual(result["coverage"], 2 / 3)
        idp = result["independent_delta_pass"]
        self.assertEqual(idp["unknown"], 1)
        self.assertEqual(idp["necessary"], 0)
        self.assertEqual(idp["attempted"], 3)
        self.assertAlmostEqual(idp["score"], 2 / 3)
        self.assertEqual(frontend.call_count, 1)

    def test_full_failure_with_frontend_pass_is_necessary(self) -> None:
        # Even when the legacy diagnostic heuristic calls a full-run proof
        # failure a parse_error, a successful --no-verify run proves that the
        # mutated file passed frontend checks and the deletion is necessary.
        def full_se(text, tmpdir, filename, timeout_seconds):
            return _parse_error()

        def frontend_se(text, tmpdir, filename, timeout_seconds):
            return _verified()

        result, on_text, frontend = self._run(full_se, frontend_se)
        self.assertEqual(result["removable_clause_candidates"], 0)
        self.assertEqual(result["necessary_clause_candidates"], 3)
        self.assertEqual(result["unknown_clause_candidates"], 0)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["independent_delta_pass"]["necessary"], 3)
        self.assertEqual(frontend.call_count, 3)
        self.assertEqual(result["frontend_checks"], 3)

    def test_duplicate_occurrences_keep_independent_ids(self) -> None:
        self.src.write_text(
            "use vstd::prelude::*; verus! { proof fn f() { assert(a); assert(a); } }",
            encoding="utf-8",
        )

        def full_run(text, tmpdir, filename, timeout_seconds):
            if "delta_0_" in filename:
                return _parse_error()
            if "delta_1_" in filename or "optimistic_" in filename:
                return _verified()
            return _verification_failed()

        result, _on_text, _frontend = self._run(
            full_run,
            lambda text, tmpdir, filename, timeout_seconds: _parse_error(),
        )
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["removable_clause_candidates"], 1)
        self.assertEqual(result["unknown_clause_candidates"], 1)
        self.assertEqual(result["necessary_clause_candidates"], 0)
        self.assertEqual(result["removed_clauses"][0]["id"], 1)
        self.assertEqual(result["unknown_clauses"][0]["id"], 0)
        self.assertEqual(
            result["removable_clause_candidates"] + result["necessary_clause_candidates"],
            result["clauses_total"] - result["unknown_clause_candidates"],
        )

    def test_cache_key_tracks_runtime_timeout_and_actual_budget(self) -> None:
        all_verified = lambda text, tmpdir, filename, timeout_seconds: _verified()
        _first, first_runs, _ = self._run(all_verified)
        _cached, cached_runs, _ = self._run(all_verified)
        _runtime_changed, runtime_runs, _ = self._run(
            all_verified,
            runtime={"configured_path": "verus-b", "version": "2"},
        )
        _timeout_changed, timeout_runs, _ = self._run(all_verified, timeout_seconds=13)
        _budget_changed, budget_runs, _ = self._run(all_verified, max_attempts=100)

        self.assertGreater(first_runs.call_count, 0)
        self.assertEqual(cached_runs.call_count, 0)
        self.assertGreater(runtime_runs.call_count, 0)
        self.assertGreater(timeout_runs.call_count, 0)
        self.assertGreater(budget_runs.call_count, 0)
        self.assertEqual(len(redundancy._HOUDINI_REDUNDANCY_CACHE), 4)

    def test_timeout_results_are_retried_instead_of_cached(self) -> None:
        result, first_runs, _ = self._run(lambda *args: _timeout())
        repeated, repeated_runs, _ = self._run(lambda *args: _timeout())

        self.assertEqual(result["status"], "partial")
        self.assertEqual(repeated["status"], "partial")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(repeated["score"], 0.0)
        self.assertEqual(first_runs.call_count, 3)
        self.assertEqual(repeated_runs.call_count, 3)
        self.assertEqual(redundancy._HOUDINI_REDUNDANCY_CACHE, {})


class RemoveCandidatesTextTest(unittest.TestCase):
    """Unit tests for the multi-clause splice helper."""

    def _assert_candidates(self, src: str):
        clean, cands = extract_removal_candidates_from_text(src)
        return clean, [c for c in cands if c.kind == "assert"]

    def test_removing_all_distinct_clauses(self) -> None:
        clean, asserts = self._assert_candidates(THREE_ASSERTS)
        self.assertEqual(len(asserts), 3)
        out = remove_candidates_text(clean, asserts)
        for name in ("assert(a)", "assert(b)", "assert(c)"):
            self.assertNotIn(name, out)

    def test_multi_removal_matches_sequential_descending(self) -> None:
        clean, asserts = self._assert_candidates(THREE_ASSERTS)
        selected = [asserts[0], asserts[2]]  # first and last, keep the middle
        multi = remove_candidates_text(clean, selected)
        sequential = clean
        for cand in sorted(selected, key=lambda c: c.removal_start, reverse=True):
            sequential = remove_candidate_text(sequential, cand)
        self.assertEqual(multi, sequential)
        self.assertIn("assert(b)", multi)

    def test_overlapping_sibling_parts_merge_without_corruption(self) -> None:
        src = textwrap.dedent(
            """\
            use vstd::prelude::*;
            verus! {
            proof fn f() {
                while cond
                    invariant a, b, c,
                {
                }
            }
            } // verus!
            """
        )
        clean, cands = extract_removal_candidates_from_text(src)
        inv = [c for c in cands if c.kind == "invariant"]
        self.assertEqual(len(inv), 3)
        # Remove first and last invariant parts (adjacent/overlapping spans); the
        # middle part must survive intact.
        out = remove_candidates_text(clean, [inv[0], inv[2]])
        invariant_region = out.split("invariant", 1)[1].split("{", 1)[0]
        self.assertIn("b", invariant_region)
        self.assertNotIn("a", invariant_region)
        self.assertNotIn("c", invariant_region)

    def test_empty_selection_is_noop(self) -> None:
        clean, _ = self._assert_candidates(THREE_ASSERTS)
        self.assertEqual(remove_candidates_text(clean, []), clean)


def _load_verus_path() -> str | None:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        return None
    for line in config_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("verus_path:"):
            return stripped.split(":", 1)[1].strip().strip('"').strip("'") or None
    return None


class RedundancyBehaviorWithVerusTest(unittest.TestCase):
    """End-to-end behavioral checks that need a real Verus binary."""

    @classmethod
    def setUpClass(cls) -> None:
        verus_path = _load_verus_path()
        if not verus_path or not Path(verus_path).is_file():
            verus_path = shutil.which("verus")
        if not verus_path:
            raise unittest.SkipTest("verus binary not available")
        set_verus_binary(verus_path)

    def _case_with_proof_clauses(self):
        from metrics_rebuild.share.redundancy import extract_removal_candidates_from_text
        from metrics_rebuild.share.redundancy import REDUNDANCY_DELETE_AND_REVERIFY_KINDS
        from metrics_rebuild.share.text import read_text

        cases_dir = PROJECT_ROOT / "cases"
        if not cases_dir.is_dir():
            self.skipTest("no cases/ directory")
        for case_dir in sorted(cases_dir.iterdir()):
            generated = case_dir / "eval.rs"
            ground = case_dir / "ref.rs"
            if not (generated.exists() and ground.exists()):
                continue
            _, cands = extract_removal_candidates_from_text(read_text(str(generated)))
            if sum(c.kind in REDUNDANCY_DELETE_AND_REVERIFY_KINDS for c in cands) >= 2:
                return generated, ground
        self.skipTest("no case with >=2 testable proof clauses found")

    def test_metric_shape_and_bounds(self) -> None:
        generated, ground = self._case_with_proof_clauses()
        result = metric_spec_redundancy_rate(str(generated), str(ground))
        inner = result["generated"]
        self.assertIn(inner["status"], {"ok", "partial", "skipped", "no_testable_clauses"})
        score = inner.get("score")
        if score is not None:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        # Full-path result shape + accounting identity only apply when we actually
        # ran the delete-and-reverify passes (not the skipped / no-clauses branches).
        if inner["status"] in {"ok", "partial"}:
            for key in (
                "clauses_total", "removable_clause_candidates", "necessary_clause_candidates",
                "unknown_clause_candidates", "minimal_size", "removed_clauses",
                "remaining_clauses", "independent_delta_pass", "coverage", "strategy",
            ):
                self.assertIn(key, inner, f"missing key: {key}")
            # removed + necessary == resolved == clauses_total - unknown
            resolved = inner["clauses_total"] - inner["unknown_clause_candidates"]
            self.assertEqual(
                inner["removable_clause_candidates"] + inner["necessary_clause_candidates"],
                resolved,
            )


if __name__ == "__main__":
    unittest.main()
