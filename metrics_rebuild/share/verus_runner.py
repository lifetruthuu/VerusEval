from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_VERUS_TIMEOUT_SECONDS = 30

# In-memory stderr cap for VerusRun. Batch harness parsers attribute JSON
# diagnostics back to individual proof functions, so silently dropping leading
# diagnostic lines turns proof failures into false successes. Large failing
# batches emit tens of kilobytes of diagnostics; keep a generous cap and only
# guard against pathological output sizes.
_STDERR_MEMORY_CAP_CHARS = 4_000_000
# Persisted excerpt size used by verus_run_to_dict; matches the historical
# 32KB tail so result JSON files do not grow.
_STDERR_PERSISTED_TAIL_CHARS = 32_000


def _tail_on_line_boundary(text: str, limit: int) -> str:
    """Return the trailing ``limit`` characters aligned to a full line."""
    if len(text) <= limit:
        return text
    tail = text[-limit:]
    newline = tail.find("\n")
    if newline == -1:
        return tail
    return tail[newline + 1 :]


_verus_binary: Optional[str] = None
_VERUS_FINGERPRINT_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


@dataclass(frozen=True)
class VerusRun:
    status: str
    success: Optional[bool]
    verified: Optional[int]
    errors: Optional[int]
    returncode: Optional[int]
    elapsed_seconds: float
    stdout: str
    stderr: str
    command: tuple[str, ...]
    encountered_vir_error: Optional[bool] = None


def set_verus_binary(path: Optional[str]) -> None:
    global _verus_binary
    _verus_binary = path
    _VERUS_FINGERPRINT_CACHE.clear()


def get_verus_binary() -> str:
    return _verus_binary or "verus"


def verus_runtime_fingerprint(binary: Optional[str] = None) -> dict[str, Any]:
    """Identify the exact Verus runtime used by cached verifier results."""
    configured = str(binary or get_verus_binary())
    configured_path = Path(configured).expanduser()
    discovered = str(configured_path) if configured_path.is_file() else shutil.which(configured)
    if discovered is None:
        return {
            "configured_path": configured,
            "resolved_path": None,
            "file_identity": None,
            "version": None,
        }

    resolved = Path(discovered).resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return {
            "configured_path": configured,
            "resolved_path": str(resolved),
            "file_identity": None,
            "version": None,
        }
    identity = {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    cache_key = (configured, str(resolved), *identity.values())
    cached = _VERUS_FINGERPRINT_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    try:
        version_run = subprocess.run(
            [str(resolved), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        version_text = (version_run.stdout or version_run.stderr or "").strip()
        version = version_text.splitlines()[0] if version_run.returncode == 0 and version_text else None
    except (OSError, subprocess.TimeoutExpired):
        version = None

    result = {
        "configured_path": configured,
        "resolved_path": str(resolved),
        "file_identity": identity,
        "version": version,
    }
    if version is not None:
        _VERUS_FINGERPRINT_CACHE[cache_key] = dict(result)
    return result


def run_verus(
    path: str,
    *,
    no_verify: bool = False,
    timeout_seconds: int = DEFAULT_VERUS_TIMEOUT_SECONDS,
) -> VerusRun:
    verus_bin = get_verus_binary()
    command = [verus_bin, "--output-json", "--crate-type=lib", "--error-format=json"]
    if no_verify:
        command.append("--no-verify")
    command.append(str(path))

    started = time.monotonic()
    if shutil.which(verus_bin) is None and not Path(verus_bin).is_file():
        return VerusRun(
            status="unavailable",
            success=None,
            verified=None,
            errors=None,
            returncode=None,
            elapsed_seconds=0.0,
            stdout="",
            stderr=f"verus binary not found: {verus_bin}",
            command=tuple(command),
        )

    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return VerusRun(
            status="timeout",
            success=None,
            verified=None,
            errors=None,
            returncode=None,
            elapsed_seconds=time.monotonic() - started,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
            command=tuple(command),
        )
    except OSError as exc:
        return VerusRun(
            status="unavailable",
            success=None,
            verified=None,
            errors=None,
            returncode=None,
            elapsed_seconds=time.monotonic() - started,
            stdout="",
            stderr=str(exc),
            command=tuple(command),
        )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    success: Optional[bool] = None
    verified: Optional[int] = None
    errors: Optional[int] = None
    encountered_vir_error: Optional[bool] = None
    status = "ok"

    try:
        parsed = json.loads(stdout)
        results = parsed.get("verification-results", {})
        success = results.get("success")
        verified = results.get("verified")
        errors = results.get("errors")
        encountered_vir_error = results.get("encountered-vir-error")
    except json.JSONDecodeError:
        match = re.search(r"verification results::\s*(\d+) verified,\s*(\d+) errors", stdout)
        if match:
            verified = int(match.group(1))
            errors = int(match.group(2))
            success = errors == 0 and proc.returncode == 0
        else:
            status = "parse_error"

    return VerusRun(
        status=status,
        success=success,
        verified=verified,
        errors=errors,
        returncode=proc.returncode,
        elapsed_seconds=time.monotonic() - started,
        stdout=stdout[-4000:],
        stderr=_tail_on_line_boundary(stderr, _STDERR_MEMORY_CAP_CHARS),
        command=tuple(command),
        encountered_vir_error=encountered_vir_error,
    )


def _is_parse_error_by_diagnostics(stderr: str, encountered_vir_error: Optional[bool]) -> bool:
    """Determine if a failure is a parse error using structured JSON diagnostics.

    Uses an exclusion approach: if there is evidence the error is NOT a parse error
    (E-codes from later compiler phases, VIR errors, or Verus-specific messages),
    return False.  Otherwise the error is from the parsing phase.
    """
    if encountered_vir_error:
        return False

    for line in stderr.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            diag = json.loads(line)
        except json.JSONDecodeError:
            continue
        if diag.get("$message_type") != "diagnostic":
            continue
        if diag.get("level") != "error":
            continue
        msg = diag.get("message", "")
        if "aborting due to" in msg:
            continue
        code = diag.get("code")
        if isinstance(code, dict) and code.get("code"):
            return False
        if "verus" in msg.lower():
            return False

    return True


def verus_run_to_dict(run: VerusRun) -> dict:
    combined_output = f"{run.stdout}\n{run.stderr}".lower()
    if run.status == "timeout":
        outcome_status = "timeout"
    elif run.status == "unavailable":
        outcome_status = "tool_error"
    elif run.status == "parse_error":
        outcome_status = "parse_error"
    elif run.success is True:
        outcome_status = "verified"
    elif run.success is False:
        if any(marker in combined_output for marker in ("out of memory", "oom", "memory exhausted", "resource exhausted")):
            outcome_status = "resource_exhausted"
        elif run.returncode is not None and run.returncode < 0:
            outcome_status = "tool_error"
        elif _is_parse_error_by_diagnostics(run.stderr, run.encountered_vir_error):
            outcome_status = "parse_error"
        else:
            outcome_status = "verification_failed"
    else:
        outcome_status = "tool_error"
    return {
        "status": run.status,
        "outcome_status": outcome_status,
        "success": run.success,
        "verified": run.verified,
        "errors": run.errors,
        "returncode": run.returncode,
        "elapsed_seconds": run.elapsed_seconds,
        "command": list(run.command),
        "stderr": _tail_on_line_boundary(run.stderr, _STDERR_PERSISTED_TAIL_CHARS),
        "stdout": run.stdout,
        "score": 1.0 if run.success is True else 0.0 if run.success is False else None,
    }


def looks_like_parse_error_output(output: str) -> bool:
    haystack = str(output or "").lower()
    markers = (
        "syntax error",
        "expected one of",
        "unexpected token",
        "mismatched closing delimiter",
        "unclosed delimiter",
        "unexpected closing delimiter",
        "this file contains an unclosed delimiter",
        "expected identifier",
        "expected expression",
        "expected type",
        "expected item",
    )
    if any(marker in haystack for marker in markers):
        return True
    return bool(re.search(r"\bparse(?:d|r)?\b", haystack) or re.search(r"expected .{0,80}, found", haystack))


def verus_frontend_run_to_dict(path: str) -> dict:
    result = verus_run_to_dict(run_verus(path, no_verify=True))
    result["method"] = "verus_no_verify_frontend_check"
    result["note"] = "Runs Verus with --no-verify, so SMT proof obligations are not discharged."
    return result


def verus_staged_verification_to_dict(path: str) -> dict:
    """Run frontend checks first, then classify failures from full verification.

    A proof diagnostic such as ``invariant not satisfied`` has no Rust error code.
    Looking at that diagnostic in isolation can therefore resemble a parse error.
    A successful ``--no-verify`` run proves that parsing, name resolution, type
    checking, and Verus mode checking already succeeded; any subsequent ordinary
    full-run failure is consequently a verification failure.
    """
    frontend = verus_frontend_run_to_dict(path)
    if frontend.get("success") is not True:
        result = dict(frontend)
        if result.get("outcome_status") not in {"parse_error", "tool_error", "timeout"}:
            result["outcome_status"] = "frontend_failed"
        result["stage"] = "frontend"
        result["frontend"] = frontend
        result["method"] = "verus_staged_frontend_then_verify"
        result["note"] = "Full SMT verification was not run because --no-verify frontend checks failed."
        return result

    result = verus_run_to_dict(run_verus(path, no_verify=False))
    if result.get("success") is False and result.get("outcome_status") == "parse_error":
        result["outcome_status"] = "verification_failed"
    result["stage"] = "verification"
    result["frontend"] = frontend
    result["method"] = "verus_staged_frontend_then_verify"
    result["note"] = (
        "The source passed --no-verify frontend checks before full SMT verification; "
        "a subsequent ordinary failure is classified as verification_failed."
    )
    return result


def verus_verification_success(run: dict) -> dict:
    return {
        "status": run.get("status"),
        "outcome_status": run.get("outcome_status"),
        "success": run.get("success"),
        "verified": run.get("verified"),
        "errors": run.get("errors"),
        "returncode": run.get("returncode"),
        "elapsed_seconds": run.get("elapsed_seconds"),
        "command": list(run.get("command", ())),
        "stderr": run.get("stderr", ""),
        "stdout": run.get("stdout", ""),
        "score": 1.0 if run.get("success") is True else 0.0 if run.get("success") is False else None,
    }


__all__ = [
    "DEFAULT_VERUS_TIMEOUT_SECONDS",
    "VerusRun",
    "get_verus_binary",
    "looks_like_parse_error_output",
    "run_verus",
    "set_verus_binary",
    "verus_frontend_run_to_dict",
    "verus_run_to_dict",
    "verus_runtime_fingerprint",
    "verus_staged_verification_to_dict",
    "verus_verification_success",
]
