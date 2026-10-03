"""Shared dataset and prompt helpers for the two spec-generation baselines."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SPEC_SYSTEM_PROMPT = (
    "You are an expert in Verus, a verification-aware programming language. "
    "Write formally verified Rust code with meaningful specifications, including "
    "preconditions, postconditions, loop invariants, decreases clauses, and proof blocks. "
    "Preserve the executable behavior of the input program. Do not use assume, admit, "
    "external, or external_body to bypass verification."
)

SPEC_USER_TEMPLATE = """Consider the following Verus code which lacks formal specifications:
```rust
{program}
```

Add meaningful preconditions (`requires`), postconditions (`ensures`), specification
functions when needed, loop invariants, decreases clauses, assertions, and proof blocks so
that Verus can verify the program. Preserve the original executable statements and function
behavior. Output the complete Verus program, not an explanation. Do not use `assume`,
`admit`, `#[verifier::external]`, or `#[verifier::external_body]`.
"""


@dataclass(frozen=True)
class DatasetTask:
    task_id: str
    subset: str
    input_path: Path
    reference_path: Path
    shot_ids: tuple[str, ...]


def _file_map(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*.rs")):
        if path.stem in result:
            raise ValueError(f"duplicate task id {path.stem}: {result[path.stem]} and {path}")
        result[path.stem] = path
    return result


def load_dataset(dataset_root: Path) -> tuple[list[DatasetTask], dict[str, Path], dict[str, Path]]:
    """Load the canonical X/Y files and ordered KNN mapping."""
    x_root = dataset_root / "X_code"
    y_root = dataset_root / "Y"
    x_map = _file_map(x_root)
    y_map = _file_map(y_root)
    if len(x_map) != 762 or set(x_map) != set(y_map):
        raise ValueError(
            f"expected 762 aligned X/Y tasks, got X={len(x_map)}, Y={len(y_map)}"
        )

    knn = json.loads((dataset_root / "knn_similar.json").read_text(encoding="utf-8"))
    if set(knn) != set(x_map):
        raise ValueError("knn_similar.json task IDs do not match X_code")

    tasks: list[DatasetTask] = []
    for task_id in sorted(x_map):
        shots = tuple(knn[task_id])
        if len(shots) != 5 or len(set(shots)) != 5 or task_id in shots:
            raise ValueError(f"invalid five-shot mapping for {task_id}: {shots}")
        missing = [shot for shot in shots if shot not in x_map or shot not in y_map]
        if missing:
            raise ValueError(f"missing shots for {task_id}: {missing}")
        relative = x_map[task_id].relative_to(x_root)
        reference = y_root / relative
        if reference != y_map[task_id]:
            raise ValueError(f"X/Y relative path mismatch for {task_id}")
        tasks.append(
            DatasetTask(
                task_id=task_id,
                subset=relative.parent.as_posix(),
                input_path=x_map[task_id],
                reference_path=reference,
                shot_ids=shots,
            )
        )
    return tasks, x_map, y_map


def spec_messages(
    program: str,
    exemplars: Iterable[tuple[str, str]] = (),
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SPEC_SYSTEM_PROMPT}]
    for example_input, example_output in exemplars:
        messages.append(
            {"role": "user", "content": SPEC_USER_TEMPLATE.format(program=example_input)}
        )
        messages.append(
            {"role": "assistant", "content": f"```rust\n{example_output}\n```"}
        )
    messages.append({"role": "user", "content": SPEC_USER_TEMPLATE.format(program=program)})
    return messages


def extract_rust_code(text: str) -> str:
    matches = re.findall(r"```(?:rust)?\s*\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return max(matches, key=len).strip()
    return text.strip()


def unsafe_reason(code: str) -> str | None:
    compact = re.sub(r"\s+", "", code).lower()
    checks = {
        "assume(": "assume is forbidden",
        "admit(": "admit is forbidden",
        "#[verifier::external]": "external verifier is forbidden",
        "#[verifier::external_body]": "external body is forbidden",
        "ensurestrue": "ensures true is forbidden",
    }
    for needle, reason in checks.items():
        if needle in compact:
            return reason
    if "ensures" not in code:
        return "generated program has no ensures clause"
    if "verus!" not in code:
        return "generated output is not a Verus program"
    return None


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

