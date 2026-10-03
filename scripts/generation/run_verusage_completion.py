#!/usr/bin/env python
"""Generate VeruSAGE contracts and proofs with up to five repair rounds."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generation.spec_baseline_common import atomic_write_json, load_dataset  # noqa: E402

VERUSAGE_DIR = ROOT / "baselines" / "verus-proof-synthesis" / "verusage"
DATASET_ROOT = ROOT / "data" / "generation"
OUTPUT_ROOT = ROOT / "runs" / "generation" / "verusage"
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_MODELS = {
    "qwen-coder": "qwen3-coder-480b-a35b-instruct",
    "deepseek-chat": "deepseek-v4-flash",
}
MODEL_API_KEY_ENVS = {
    "qwen-coder": "VERUSAGE_QWEN_CODER_API_KEY",
    "deepseek-chat": "VERUSAGE_DEEPSEEK_API_KEY",
}
MANIFEST_FIELDS = (
    "source_path",
    "target_path",
    "filename",
    "model",
    "shot",
    "status",
    "task_name",
    "generation_candidate_id",
    "source_record_id",
    "final_state_id",
    "final_round_id",
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    api_model: str


@dataclass(frozen=True)
class CompletionTask:
    shot: str
    model: ModelSpec
    dataset: str
    task_name: str
    raw_path: Path


def parse_model(value: str) -> ModelSpec:
    label, separator, api_model = value.partition("=")
    if not label:
        raise argparse.ArgumentTypeError("model label cannot be empty")
    if "/" in label or label in {".", ".."}:
        raise argparse.ArgumentTypeError("model label must be a safe directory name")
    return ModelSpec(
        label=label,
        api_model=api_model if separator else DEFAULT_API_MODELS.get(label, label),
    )


def resolve_api_keys(models: list[ModelSpec], explicit_key: str | None) -> dict[str, str]:
    shared_key = (
        explicit_key
        or os.getenv("VERUSAGE_API_KEY")
        or os.getenv("BASELINE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    resolved: dict[str, str] = {}
    for model in models:
        env_name = MODEL_API_KEY_ENVS.get(model.label)
        model_key = os.getenv(env_name) if env_name else None
        resolved[model.label] = model_key or shared_key or ""
    return resolved


def verus_score(path: Path, verus_path: str, timeout: int = 120) -> tuple[int, int, str]:
    # Match metrics_rebuild/share/verus_runner.py: these benchmarks are libraries,
    # so default binary mode falsely fails with `main` not found (E0601).
    try:
        result = subprocess.run(
            [
                verus_path,
                "--crate-type=lib",
                "--multiple-errors",
                "100",
                str(path),
            ],
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


def load_existing_keys(output_root: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for path in (output_root / "manifest.csv", output_root / "completion_manifest.csv"):
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                keys.add((row["shot"], row["model"], row["task_name"]))
    return keys


def discover_tasks(
    dataset_root: Path,
    output_root: Path,
    shots: list[str],
    models: list[ModelSpec],
) -> list[CompletionTask]:
    existing = load_existing_keys(output_root)
    tasks: list[CompletionTask] = []
    raw_paths = sorted((dataset_root / "X_code").glob("*/*.rs"))
    if len(raw_paths) != 762:
        raise RuntimeError(f"expected 762 canonical inputs, found {len(raw_paths)}")
    for shot in shots:
        for raw_path in raw_paths:
            dataset = raw_path.parent.name
            task_name = raw_path.stem
            for model in models:
                key = (shot, model.label, task_name)
                verified_path = result_path(output_root, shot, model.label, task_name, "verified")
                unverified_path = result_path(
                    output_root, shot, model.label, task_name, "unverified"
                )
                if key in existing or verified_path.exists() or unverified_path.exists():
                    continue
                tasks.append(
                    CompletionTask(
                        shot=shot,
                        model=model,
                        dataset=dataset,
                        task_name=task_name,
                        raw_path=raw_path,
                    )
                )
    return tasks


def result_path(
    output_root: Path,
    shot: str,
    model: str,
    task_name: str,
    status: str,
) -> Path:
    filename = f"verusage_{task_name}_{model}_{shot}_{status}.rs"
    return output_root / shot / model / status / "verusage" / filename


def write_runtime_config(
    path: Path,
    model: ModelSpec,
    base_url: str,
    verus_path: str,
) -> None:
    atomic_write_json(
        path,
        {
            "use_openai": True,
            "aoai_api_base": [base_url],
            "aoai_api_version": "",
            "aoai_api_key": [],
            "aoai_max_retries": 5,
            "aoai_generation_model": model.api_model,
            "aoai_debug_model": model.api_model,
            "verus_path": verus_path,
            "debug_max_attempt": 3,
            "debug_temp": 1.0,
            "max_token": 20000,
        },
    )


def write_exemplars(
    path: Path,
    task_name: str,
    shot: str,
    x_map: dict[str, Path],
    y_map: dict[str, Path],
    shot_map: dict[str, tuple[str, ...]],
) -> Path | None:
    if shot == "zero-shot":
        return None
    exemplars = [
        {
            "task_id": example_id,
            "input": x_map[example_id].read_text(encoding="utf-8"),
            "output": y_map[example_id].read_text(encoding="utf-8"),
        }
        for example_id in shot_map[task_name]
    ]
    atomic_write_json(path, exemplars)
    return path


def run_task(
    task: CompletionTask,
    args: argparse.Namespace,
    config_path: Path,
    x_map: dict[str, Path],
    y_map: dict[str, Path],
    shot_map: dict[str, tuple[str, ...]],
) -> dict:
    work_dir = args.work_root / task.shot / task.model.label / task.task_name
    state_path = work_dir / "state.json"
    work_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "shot": task.shot,
        "model": task.model.label,
        "api_model": task.model.api_model,
        "task_name": task.task_name,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_write_json(state_path, state)
    started = time.monotonic()
    try:
        candidate_id = ""
        source = task.raw_path
        verified, errors = 0, 1

        repair_ran = False
        repair_rounds_run = 0
        staged_output = work_dir / "output.rs"
        if verification_succeeded(verified, errors):
            shutil.copyfile(source, staged_output)
        else:
            repair_ran = True
            command = [
                sys.executable,
                "main.py",
                "--config",
                str(config_path),
                "--mode",
                "agent",
                "--input",
                str(source),
                "--output",
                str(staged_output),
                "--outdir",
                str(work_dir),
                "--repair",
                str(args.repair_rounds),
                "--spec-repair",
                "--temp",
                str(args.temperature),
            ]
            command.append("--generate-specs")
            exemplar_path = write_exemplars(
                work_dir / "spec_exemplars.json",
                task.task_name,
                task.shot,
                x_map,
                y_map,
                shot_map,
            )
            if exemplar_path is not None:
                command.extend(["--spec-exemplars-json", str(exemplar_path)])
            environment = os.environ.copy()
            environment["OPENAI_API_KEY"] = args.api_keys[task.model.label]
            runner_log = work_dir / "runner.log"
            with runner_log.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=VERUSAGE_DIR,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=args.task_timeout,
                    check=False,
                )
            if result.returncode not in (0, 233):
                raise RuntimeError(f"VeruSAGE exited with {result.returncode}")
            if not staged_output.exists():
                raise RuntimeError("VeruSAGE produced no output file")
            attempts = re.findall(
                r"Repair attempt (\d+)/\d+",
                runner_log.read_text(encoding="utf-8", errors="replace"),
            )
            repair_rounds_run = max(map(int, attempts), default=0)

        verified, errors, _ = verus_score(staged_output, args.verus_path)
        status = "verified" if verification_succeeded(verified, errors) else "unverified"
        target = result_path(
            args.output_root, task.shot, task.model.label, task.task_name, status
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged_output, target)
        state.update(
            {
                "status": "complete",
                "classification": status,
                "verified": verified,
                "errors": errors,
                "repair_ran": repair_ran,
                "repair_rounds": args.repair_rounds,
                "repair_rounds_run": repair_rounds_run,
                "generation_candidate_id": candidate_id,
                "source": str(source),
                "output": str(target),
            }
        )
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["traceback"] = traceback.format_exc()
    state["elapsed_seconds"] = round(time.monotonic() - started, 3)
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_write_json(state_path, state)
    return state


def run_task_claimed(
    task: CompletionTask,
    args: argparse.Namespace,
    config_path: Path,
    x_map: dict[str, Path],
    y_map: dict[str, Path],
    shot_map: dict[str, tuple[str, ...]],
) -> dict:
    """Serialize the same shot/model/task across independently launched processes."""
    work_dir = args.work_root / task.shot / task.model.label / task.task_name
    work_dir.mkdir(parents=True, exist_ok=True)
    with (work_dir / ".task.lock").open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        for status in ("verified", "unverified"):
            target = result_path(
                args.output_root,
                task.shot,
                task.model.label,
                task.task_name,
                status,
            )
            if target.exists():
                return {
                    "status": "skipped",
                    "shot": task.shot,
                    "model": task.model.label,
                    "task_name": task.task_name,
                    "output": str(target),
                }
        return run_task(task, args, config_path, x_map, y_map, shot_map)


def append_manifest(path: Path, state: dict, lock: Lock) -> None:
    target = Path(state["output"])
    row = {
        "source_path": state["source"],
        "target_path": str(
            Path(path.parent.name) / target.relative_to(path.parent)
        ),
        "filename": target.name,
        "model": state["model"],
        "shot": state["shot"],
        "status": state["classification"],
        "task_name": state["task_name"],
        "generation_candidate_id": state["generation_candidate_id"],
        "source_record_id": f"verusage-completion-{state['task_name']}-{state['model']}-{state['shot']}",
        "final_state_id": "",
        "final_round_id": state["repair_rounds_run"] if state["repair_ran"] else "",
    }
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with (path.parent / ".completion_manifest.lock").open(
            "a+", encoding="utf-8"
        ) as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            write_header = not path.exists() or path.stat().st_size == 0
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())


def main() -> int:
    load_dotenv(ROOT / "baseline_api.env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument(
        "--shot",
        action="append",
        choices=("zero-shot", "few-shot"),
        default=None,
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("VERUSAGE_API_BASE", DEFAULT_API_BASE),
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--verus-path", default=None)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repair-rounds", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--task-timeout", type=int, default=7200)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.shot = args.shot or ["zero-shot", "few-shot"]
    args.dataset_root = args.dataset_root.resolve()
    args.output_root = args.output_root.resolve()
    args.work_root = (args.work_root or args.output_root / ".completion_work").resolve()
    args.api_keys = resolve_api_keys(args.model, args.api_key)
    config_path = ROOT / "config.yaml"
    root_config = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}) if config_path.is_file() else {}
    args.verus_path = args.verus_path or root_config.get("verus_path") or shutil.which("verus")
    if not args.verus_path and not args.dry_run:
        parser.error("Set --verus-path or verus_path in config.yaml to the pinned Verus binary.")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.repair_rounds != 5:
        parser.error("VeruSAGE completion must use exactly 5 repair rounds")
    if not args.api_base and not args.dry_run:
        parser.error("--api-base or VERUSAGE_API_BASE is required")
    missing_api_keys = [model.label for model in args.model if not args.api_keys[model.label]]
    if missing_api_keys and not args.dry_run:
        parser.error(
            "missing API key for models: "
            f"{missing_api_keys}; configure model-specific VERUSAGE_*_API_KEY values"
        )

    tasks = discover_tasks(args.dataset_root, args.output_root, args.shot, args.model)
    if args.task_id:
        selected = set(args.task_id)
        tasks = [task for task in tasks if task.task_name in selected]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    counts_by_combo: dict[str, int] = {}
    for task in tasks:
        key = f"{task.shot}/{task.model.label}"
        counts_by_combo[key] = counts_by_combo.get(key, 0) + 1
    summary = {"missing": len(tasks), "by_combo": counts_by_combo}
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    dataset_tasks, x_map, y_map = load_dataset(args.dataset_root)
    shot_map = {task.task_id: task.shot_ids for task in dataset_tasks}
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    run_dir = args.work_root / "runs" / run_id
    config_paths: dict[str, Path] = {}
    for model in args.model:
        config_path = run_dir / f"runtime-{model.label}.json"
        write_runtime_config(config_path, model, args.api_base, args.verus_path)
        config_paths[model.label] = config_path

    manifest_path = args.output_root / "completion_manifest.csv"
    manifest_lock = Lock()
    result_counts = {
        "complete": 0,
        "failed": 0,
        "skipped": 0,
        "verified": 0,
        "unverified": 0,
    }
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_task_claimed,
                task,
                args,
                config_paths[task.model.label],
                x_map,
                y_map,
                shot_map,
            ): task
            for task in tasks
        }
        for index, future in enumerate(as_completed(futures), 1):
            state = future.result()
            result_counts[state["status"]] += 1
            if state["status"] == "complete":
                result_counts[state["classification"]] += 1
                append_manifest(manifest_path, state, manifest_lock)
            print(
                f"[{index}/{len(tasks)}] {state['shot']}/{state['model']}/"
                f"{state['task_name']}: {state['status']}"
            )
    atomic_write_json(run_dir / "summary.json", result_counts)
    print(json.dumps(result_counts, ensure_ascii=False, indent=2))
    return 1 if result_counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
