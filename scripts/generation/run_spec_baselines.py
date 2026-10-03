#!/usr/bin/env python
"""Run the spec-generating AlphaVerus or AutoVerus pipeline over 762 tasks."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generation.spec_baseline_common import (  # noqa: E402
    DatasetTask,
    atomic_write_json,
    load_dataset,
)

DATASET_ROOT = ROOT / "data" / "generation"
ALPHA_DIR = ROOT / "baselines" / "alphaverus" / "inference"
AUTO_ROOT = ROOT / "baselines" / "verus-proof-synthesis"
AUTO_DIR = AUTO_ROOT / "autoverus"


def load_api_environment(path: Path) -> None:
    """Load local baseline credentials without overriding explicit shell values."""
    load_dotenv(path, override=False)
    shared_key = os.getenv("BASELINE_API_KEY")
    if shared_key:
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = shared_key
        if not os.getenv("ALPHAVERUS_API_KEY"):
            os.environ["ALPHAVERUS_API_KEY"] = shared_key


def verus_score(path: Path, verus_path: str, timeout: int = 120) -> tuple[int, int, str]:
    try:
        result = subprocess.run(
            [verus_path, str(path), "--crate-name", "generated_program", "--crate-type=lib",
             "--multiple-errors", "100"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 0, 100, "Verus timed out"
    output = result.stdout + "\n" + result.stderr
    matches = re.findall(r"(\d+) verified, (\d+) errors", output)
    if matches:
        verified, errors = map(int, matches[-1])
    else:
        verified = 0
        errors = max(1, len(re.findall(r"^error", result.stderr, re.MULTILINE)))
    return verified, errors, output


def verification_succeeded(verified: int, errors: int) -> bool:
    """Follow Verus success semantics; zero verified obligations may still pass."""
    return errors == 0


def write_auto_config(path: Path, verus_path: str, model: str, base_url: str) -> None:
    config = {
        "use_openai": True,
        "aoai_api_base": [base_url],
        "aoai_api_version": "",
        "aoai_api_key": [],
        "aoai_max_retries": 5,
        "max_token": 32000,
        "aoai_generation_model": model,
        "aoai_debug_model": model,
        "verus_path": verus_path,
        "example_path": str(AUTO_DIR / "examples"),
        "lemma_path": str(AUTO_DIR / "lemmas"),
        "util_path": str(AUTO_ROOT / "utils"),
    }
    atomic_write_json(path, config)


def run_logged(command: list[str], cwd: Path, log_path: Path, timeout: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}; see {log_path}")


def alpha_candidate(
    task_dir: Path,
    task_id: str,
    verus_path: str,
) -> tuple[Path, int, int]:
    """Select the best candidate using fresh Verus results, not filename metadata."""
    candidates = list((task_dir / "generation" / "dumps").glob(f"verified_prog={task_id}_*.rs"))
    if not candidates:
        raise RuntimeError("AlphaVerus produced no candidate file")
    scored = []
    for candidate in candidates:
        verified, errors, _ = verus_score(candidate, verus_path)
        correct = verification_succeeded(verified, errors)
        total = verified + errors
        ratio = verified / total if total else 0.0
        scored.append((correct, ratio, verified, -errors, candidate.name, candidate))
    _, _, verified, neg_errors, _, selected = max(scored)
    return selected, verified, -neg_errors


def run_alpha(
    task: DatasetTask,
    task_dir: Path,
    final_path: Path,
    args: argparse.Namespace,
    x_map: dict[str, Path] | None = None,
    y_map: dict[str, Path] | None = None,
) -> dict:
    shot_ids = task.shot_ids if getattr(args, "shot", "zero-shot") == "few-shot" else ()
    # AlphaVerus writes scratch files in its working directory. Give each task
    # its own source copy so concurrent runs keep all outputs under output-root.
    workspace = task_dir / "workflow"
    workspace.mkdir(parents=True, exist_ok=True)
    for source in ALPHA_DIR.rglob("*.py"):
        destination = workspace / source.relative_to(ALPHA_DIR)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    record_path = task_dir / "input.jsonl"
    record_path.write_text(
        json.dumps(
            {
                "task_id": task.task_id,
                "x": task.input_path.read_text(encoding="utf-8"),
                "y": "",
                "spec_exemplars": [
                    {"input": x_map[key].read_text(encoding="utf-8"),
                     "output": y_map[key].read_text(encoding="utf-8")}
                    for key in shot_ids
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    generation_dir = task_dir / "generation"
    command = [
        sys.executable,
        "solve.py",
        "--SAVE_DIR",
        str(generation_dir),
        "--PROGRAMS_FILE",
        str(record_path),
        "--model",
        args.alpha_model,
        "--base_url",
        args.alpha_base_url,
        "--temperature",
        str(args.temperature),
        "--batch_size",
        "1",
        "--zero_shot",
        "--generate_specs",
    ]
    run_logged(command, workspace, task_dir / "generation.log", args.task_timeout)
    selected, verified, errors = alpha_candidate(task_dir, task.task_id, args.verus_path)
    selected_stage = "inference"

    if not verification_succeeded(verified, errors) and not args.skip_alpha_treefinement:
        started = time.time()
        command = [
            sys.executable,
            "rebase.py",
            "0",
            str(selected),
            "0",
            str(args.alpha_tree_width),
            str(args.alpha_repair_rounds),
        ]
        run_logged(command, workspace, task_dir / "treefinement.log", args.task_timeout)
        possible = []
        for path in workspace.rglob("correct_code.rs"):
            if path.stat().st_mtime >= started and selected.name in path.parent.name:
                possible.append(path)
        if possible:
            selected = max(possible, key=lambda path: path.stat().st_mtime)
            selected_stage = "treefinement"

    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected, final_path)
    verified, errors, _ = verus_score(final_path, args.verus_path)
    return {
        "selected_stage": selected_stage,
        "verified": verified,
        "errors": errors,
        "correct": verification_succeeded(verified, errors),
        "num_shots": len(shot_ids),
        "shot_ids": list(shot_ids),
        "tree_width": args.alpha_tree_width,
        "repair_rounds": args.alpha_repair_rounds,
    }


def run_auto(
    task: DatasetTask,
    task_dir: Path,
    final_path: Path,
    args: argparse.Namespace,
    x_map: dict[str, Path],
    y_map: dict[str, Path],
    config_path: Path,
) -> dict:
    exemplars = [
        {
            "task_id": shot_id,
            "input": x_map[shot_id].read_text(encoding="utf-8"),
            "output": y_map[shot_id].read_text(encoding="utf-8"),
        }
        for shot_id in (task.shot_ids if args.shot == "few-shot" else ())
    ]
    exemplar_path = task_dir / "spec_exemplars.json"
    atomic_write_json(exemplar_path, exemplars)
    command = [
        sys.executable,
        "main.py",
        "--config",
        str(config_path),
        "--input",
        str(task.input_path),
        "--output",
        str(final_path),
        "--repair",
        str(args.auto_repair_rounds),
        "--merge",
        str(args.auto_merge_candidates),
        "--temp",
        str(args.temperature),
        "--generate-specs",
        "--spec-exemplars-json",
        str(exemplar_path),
    ]
    run_logged(command, AUTO_DIR, task_dir / "autoverus.log", args.task_timeout)
    verified, errors, _ = verus_score(final_path, args.verus_path)
    return {
        "selected_stage": "autoverus_native_pipeline",
        "verified": verified,
        "errors": errors,
        "correct": verification_succeeded(verified, errors),
        "num_shots": len(exemplars),
        "shot_ids": [item["task_id"] for item in exemplars],
        "repair_rounds": args.auto_repair_rounds,
        "merge_candidates": args.auto_merge_candidates,
    }


def process_task(
    task: DatasetTask,
    args: argparse.Namespace,
    x_map: dict[str, Path],
    y_map: dict[str, Path],
    config_path: Path | None,
) -> dict:
    task_dir = args.output_root / "tasks" / task.subset / task.task_id
    state_path = task_dir / "state.json"
    final_path = args.output_root / "results" / task.subset / f"{task.task_id}.rs"
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "complete" and final_path.exists():
            return state

    task_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "task_id": task.task_id,
        "subset": task.subset,
        "pipeline": args.pipeline,
        "status": "running",
        "input": str(task.input_path),
        "output": str(final_path),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_write_json(state_path, state)
    started = time.monotonic()
    try:
        if args.pipeline == "alphaverus":
            result = run_alpha(task, task_dir, final_path, args, x_map, y_map)
        else:
            assert config_path is not None
            result = run_auto(task, task_dir, final_path, args, x_map, y_map, config_path)
        state.update(result)
        state["status"] = "complete"
        category = "verified" if result["correct"] else "unverified"
        classified = args.output_root / category / task.subset / final_path.name
        classified.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(final_path, classified)
        state["classified_output"] = str(classified)
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["traceback"] = traceback.format_exc()
    state["elapsed_seconds"] = round(time.monotonic() - started, 3)
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_write_json(state_path, state)
    return state


def main() -> int:
    load_api_environment(ROOT / "baseline_api.env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("alphaverus", "autoverus"), required=True)
    parser.add_argument("--shot", choices=("zero-shot", "few-shot"))
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent task pipelines (default: AlphaVerus 2, AutoVerus 4)",
    )
    parser.add_argument("--task-timeout", type=int, default=7200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--verus-path", default=None)
    parser.add_argument(
        "--alpha-model",
        default=os.getenv("ALPHAVERUS_MODEL", "llama-3.3-70b"),
    )
    parser.add_argument(
        "--alpha-base-url",
        default=os.getenv("ALPHAVERUS_API_BASE", "https://api.openai.com/v1"),
    )
    parser.add_argument("--skip-alpha-treefinement", action="store_true")
    parser.add_argument("--alpha-tree-width", type=int, default=3)
    parser.add_argument("--alpha-repair-rounds", type=int, default=3)
    parser.add_argument("--auto-model", default="gpt-4o")
    parser.add_argument(
        "--auto-base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--auto-repair-rounds", type=int, default=5)
    parser.add_argument("--auto-merge-candidates", type=int, default=5)
    args = parser.parse_args()
    args.shot = args.shot or ("zero-shot" if args.pipeline == "alphaverus" else "few-shot")

    config_path = ROOT / "config.yaml"
    root_config = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}) if config_path.is_file() else {}
    args.verus_path = args.verus_path or root_config.get("verus_path") or shutil.which("verus")
    if not args.verus_path and not args.dry_run:
        parser.error("Set --verus-path or verus_path in config.yaml to the pinned Verus binary.")
    default_name = f"{args.pipeline}_{args.shot}"
    args.output_root = (args.output_root or ROOT / "runs" / "generation" / default_name).resolve()
    args.workers = args.workers or (2 if args.pipeline == "alphaverus" else 4)

    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.alpha_tree_width < 1 or args.alpha_repair_rounds < 1:
        parser.error("AlphaVerus tree width and repair rounds must be positive")
    if (
        args.pipeline == "alphaverus"
        and not args.dry_run
        and (not args.alpha_model or not args.alpha_base_url)
    ):
        parser.error("AlphaVerus requires ALPHAVERUS_MODEL and ALPHAVERUS_API_BASE")
    if (
        args.pipeline == "alphaverus"
        and not args.dry_run
        and not os.getenv("ALPHAVERUS_API_KEY")
    ):
        parser.error("AlphaVerus requires ALPHAVERUS_API_KEY")
    if args.pipeline == "autoverus" and not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        parser.error("AutoVerus requires OPENAI_API_KEY")
    if args.pipeline == "alphaverus" and not args.dry_run:
        os.environ["ALPHAVERUS_MODEL"] = args.alpha_model
        os.environ["ALPHAVERUS_API_BASE"] = args.alpha_base_url
        os.environ["VERUS_PATH"] = args.verus_path

    tasks, x_map, y_map = load_dataset(args.dataset_root.resolve())
    if args.task_id:
        selected = set(args.task_id)
        tasks = [task for task in tasks if task.task_id in selected]
        missing = selected - {task.task_id for task in tasks}
        if missing:
            parser.error(f"unknown task IDs: {sorted(missing)}")
    if args.limit is not None:
        tasks = tasks[: args.limit]

    config_path = None
    if args.pipeline == "autoverus":
        config_path = args.output_root / "autoverus.runtime.json"
        write_auto_config(config_path, args.verus_path, args.auto_model, args.auto_base_url)

    manifest = {
        "pipeline": args.pipeline,
        "shot": args.shot,
        "temperature": args.temperature,
        "auto_merge_candidates": args.auto_merge_candidates if args.pipeline == "autoverus" else None,
        "dataset_root": str(args.dataset_root.resolve()),
        "num_tasks": len(tasks),
        "workers": args.workers,
        "verus_path": args.verus_path,
        "auto_repair_rounds": args.auto_repair_rounds if args.pipeline == "autoverus" else None,
        "auto_model": args.auto_model if args.pipeline == "autoverus" else None,
        "auto_base_url": args.auto_base_url if args.pipeline == "autoverus" else None,
        "alpha_tree_width": args.alpha_tree_width if args.pipeline == "alphaverus" else None,
        "alpha_repair_rounds": args.alpha_repair_rounds if args.pipeline == "alphaverus" else None,
        "alpha_model": args.alpha_model if args.pipeline == "alphaverus" else None,
        "alpha_base_url": args.alpha_base_url if args.pipeline == "alphaverus" else None,
        "tasks": [task.task_id for task in tasks],
    }
    atomic_write_json(args.output_root / "manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    counts = {"complete": 0, "failed": 0, "verified": 0, "unverified": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_task, task, args, x_map, y_map, config_path): task
            for task in tasks
        }
        for index, future in enumerate(as_completed(futures), 1):
            state = future.result()
            counts[state["status"]] += 1
            if state["status"] == "complete":
                counts["verified" if state.get("correct") else "unverified"] += 1
            print(
                f"[{index}/{len(tasks)}] {state['task_id']}: {state['status']} "
                f"verified={state.get('verified', '-')} errors={state.get('errors', '-')}"
            )
    atomic_write_json(args.output_root / "summary.json", counts)
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
