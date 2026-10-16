"""Evaluate program sources and write fresh per-file records independently of RQ reproduction."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IDENTITY = ("workflow", "model", "shot", "task_id")
INDEX_FIELDS = ("sample_id", *IDENTITY, "generated_path", "reference_path", "result_path",
                "evaluation_status")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def input_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def make_job(identity, generated, reference):
    for key in IDENTITY:
        value = identity.get(key)
        if not isinstance(value, str) or not value.strip() or "|" in value:
            raise ValueError(f"Invalid {key}: {value!r}")
    sample_id = "|".join(identity[key] for key in IDENTITY)
    filename = quote(sample_id, safe="") + ".json"
    if len(filename.encode()) > 251:  # Leave room for the atomic-write .tmp suffix.
        raise ValueError(f"Sample ID is too long for a result filename: {sample_id}")
    for path in (generated, reference):
        if not path.is_file():
            raise ValueError(f"Missing input file: {path}")
    return {"sample_id": sample_id, **{key: identity[key] for key in IDENTITY},
            "generated_path": str(generated), "reference_path": str(reference),
            "result_path": "per_file/" + filename, "evaluation_status": "pending"}


def discover_jobs(args):
    jobs = []
    if args.mode == "dataset":
        catalog = read_csv(ROOT / "data/evaluation/target_functions.csv")
        references = {row["task_id"]: row for row in catalog}
        if len(references) != len(catalog):
            raise ValueError("Duplicate task IDs in target_functions.csv")
        for row in read_csv(args.manifest):
            if any(getattr(args, key) and row[key] != getattr(args, key) for key in IDENTITY):
                continue
            if row["task_id"] not in references:
                raise ValueError(f"Unknown task: {row['task_id']}")
            job = make_job(row, input_path(row["generated_path"]),
                           input_path(references[row["task_id"]]["reference_path"]))
            if row.get("sample_id") and row["sample_id"] != job["sample_id"]:
                raise ValueError(f"Manifest sample ID disagrees with its identity: {row['sample_id']}")
            jobs.append(job)
    else:
        generated, reference = Path(args.generated).resolve(), Path(args.reference).resolve()
        identity = {key: getattr(args, key) for key in IDENTITY[:3]}
        if args.mode == "file":
            jobs.append(make_job({**identity, "task_id": args.task_id or reference.stem}, generated, reference))
        else:
            if not generated.is_dir() or not reference.is_dir():
                raise ValueError("Directory mode requires two existing directories")
            gen = {p.name: p for p in generated.glob("*.rs") if p.is_file()}
            ref = {p.name: p for p in reference.glob("*.rs") if p.is_file()}
            if gen.keys() != ref.keys():
                raise ValueError(f"Unmatched filenames: generated-only={sorted(gen.keys() - ref.keys())[:10]}, "
                                 f"reference-only={sorted(ref.keys() - gen.keys())[:10]}")
            for name in sorted(gen):
                jobs.append(make_job({**identity, "task_id": ref[name].stem}, gen[name], ref[name]))
    if not jobs:
        raise ValueError("No samples selected")
    if len({job["sample_id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate sample IDs in selected inputs")
    return sorted(jobs, key=lambda job: job["sample_id"])


def run_worker(args):
    from metrics_rebuild.share.io_cases import set_io_suite_dir
    from scripts.evaluation.evaluate import evaluate_single
    from scripts.evaluation.target_summary import resolve_summary_target, summarize_target

    job = json.loads(args.job)
    set_io_suite_dir(args.io_root)
    output = Path(args.result)
    target = resolve_summary_target(job["generated_path"], job["reference_path"])

    def save(result, status="running"):
        write_json(output, {**result, "sample": {key: job[key] for key in ("sample_id", *IDENTITY)},
                            "target_evaluation": summarize_target(result, target, job["sample_id"]),
                            "evaluation_status": status})

    result = evaluate_single(job["generated_path"], job["reference_path"], args.verus_path, progress=save)
    save(result, "completed_with_errors" if result["errors"] else "completed")


def stop_worker(process):
    """Stop the worker and any verifier/compiler children in its process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def finish_failed(job, path, status, error):
    """Retain metrics already saved by the worker's last atomic checkpoint."""
    from scripts.evaluation.target_summary import summarize_target

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    if payload.get("sample", {}).get("sample_id") != job["sample_id"]:
        payload = {"generated_path": job["generated_path"], "reference_path": job["reference_path"],
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "sample": {key: job[key] for key in ("sample_id", *IDENTITY)},
                   "metrics": {}, "errors": []}
    payload["evaluation_status"] = status
    payload["errors"].append({"metric": "__task__", "error": error})
    if "target_evaluation" not in payload:
        payload["target_evaluation"] = summarize_target(
            payload, {"status": "unavailable", "reason": "evaluation_not_started"}, job["sample_id"])
    write_json(path, payload)
    return payload


def evaluate_jobs(args, jobs, output, verus):
    (output / "per_file").mkdir()
    (output / "worker_logs").mkdir()
    summary = {"status": "running", "mode": args.mode, "samples": len(jobs),
               "tasks": len({job["task_id"] for job in jobs}),
               "configurations": len({tuple(job[k] for k in IDENTITY[:3]) for job in jobs}),
               "verus_path": verus, "io_root": args.io_root, "workers": args.workers,
               "worker_timeout_seconds": args.worker_timeout_seconds}
    started = time.monotonic()

    def save():
        temporary = output / "artifact_index.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS)
            writer.writeheader()
            writer.writerows(jobs)
        temporary.replace(output / "artifact_index.csv")
        summary["counts"] = dict(Counter(job["evaluation_status"] for job in jobs))
        summary["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write_json(output / "summary.json", summary)

    if args.dry_run:
        summary["status"] = "checked"
        save()
        print(json.dumps({key: summary[key] for key in ("status", "samples", "tasks", "configurations")}))
        return 0
    save()
    pending, running = deque(jobs), {}
    completed, last_saved = 0, started
    try:
        while pending or running:
            while pending and len(running) < args.workers:
                job = pending.popleft()
                result_path = output / job["result_path"]
                log_path = output / "worker_logs" / (result_path.stem + ".log")
                log = log_path.open("w", encoding="utf-8")
                command = [sys.executable, str(Path(__file__).resolve()), "_worker",
                           "--job", json.dumps(job), "--result", str(result_path), "--verus-path", verus]
                if args.io_root:
                    command.extend(["--io-root", args.io_root])
                try:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                except OSError:
                    log.close()
                    raise
                job["evaluation_status"] = "running"
                running[process] = (job, log, time.monotonic())
            for process, (job, log, start) in list(running.items()):
                timed_out = (process.poll() is None and args.worker_timeout_seconds > 0
                             and time.monotonic() - start >= args.worker_timeout_seconds)
                if timed_out:
                    stop_worker(process)
                if process.poll() is None:
                    continue
                log.close()
                del running[process]
                path = output / job["result_path"]
                try:
                    if timed_out:
                        raise TimeoutError(f"Worker exceeded {args.worker_timeout_seconds}s")
                    if process.returncode:
                        raise RuntimeError(f"Worker exited with code {process.returncode}; see worker_logs/{path.stem}.log")
                    result = json.loads(path.read_text(encoding="utf-8"))
                    if (result.get("sample", {}).get("sample_id") != job["sample_id"]
                            or result.get("evaluation_status") not in {"completed", "completed_with_errors"}):
                        raise ValueError("Worker did not produce a completed sample record")
                except (OSError, ValueError, RuntimeError, TimeoutError) as error:
                    result = finish_failed(job, path, "timeout" if timed_out else "failed", str(error))
                job["evaluation_status"] = result["evaluation_status"]
                completed += 1
                print(f"[{completed}/{len(jobs)}] {job['sample_id']}: {job['evaluation_status']}", flush=True)
            if time.monotonic() - last_saved >= 10:
                save()
                last_saved = time.monotonic()
            if running:
                time.sleep(0.1)
    except BaseException:
        summary["status"] = "interrupted"
        for process, (job, log, _) in running.items():
            stop_worker(process)
            log.close()
            finish_failed(job, output / job["result_path"], "interrupted", "Parent run interrupted")
            job["evaluation_status"] = "interrupted"
        save()
        raise
    summary["status"] = "completed" if all(job["evaluation_status"] == "completed" for job in jobs) else "completed_with_errors"
    save()
    print(f"Saved {completed} per-file records to {output / 'per_file'}", flush=True)
    return 0 if summary["status"] == "completed" else 1


def main():
    if sys.argv[1:2] == ["_worker"]:
        worker = argparse.ArgumentParser()
        worker.add_argument("--job", required=True)
        worker.add_argument("--result", required=True)
        worker.add_argument("--verus-path", required=True)
        worker.add_argument("--io-root")
        run_worker(worker.parse_args(sys.argv[2:]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("file", "dir", "dataset"):
        sub = modes.add_parser(mode)
        if mode == "dataset":
            sub.add_argument("--manifest", type=Path, default=ROOT / "data/generated/manifest.csv")
        else:
            sub.add_argument("-g", "--generated", required=True)
            sub.add_argument("-r", "--reference", required=True)
        for name in IDENTITY[:3]:
            sub.add_argument("--" + name, default=None if mode == "dataset" else "local" if name == "workflow" else "unspecified")
        if mode != "dir":
            sub.add_argument("--task-id")
        sub.add_argument("-o", "--output-dir", required=True, type=Path)
        sub.add_argument("-w", "--workers", type=int, default=4)
        sub.add_argument("--worker-timeout-seconds", type=float, default=1800,
                         help="Wall time per sample, including metric calls; 0 disables the limit.")
        sub.add_argument("--verus-path")
        sub.add_argument("--io-root", help="Offline I/O suite root; dataset mode defaults to data/io.")
        sub.add_argument("--dry-run", action="store_true", help="Write the pairing index without running metrics.")
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.worker_timeout_seconds < float("inf"):
        parser.error("workers must be positive and the timeout must be finite and nonnegative")
    try:
        jobs = discover_jobs(args)
        output = args.output_dir.resolve()
        if output == ROOT or output.is_relative_to(ROOT / "data"):
            raise ValueError("Choose an output directory outside the released data")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("Output directory must be new or empty")
        if args.io_root is None and args.mode == "dataset":
            args.io_root = str(ROOT / "data/io")
        if args.io_root:
            args.io_root = str(Path(args.io_root).resolve())
            if not Path(args.io_root).is_dir():
                raise ValueError(f"I/O suite directory does not exist: {args.io_root}")
        from scripts.evaluation.evaluate import _load_verus_path_from_config
        verus = args.verus_path or os.environ.get("VERUS_PATH") or _load_verus_path_from_config() or "verus"
        if not args.dry_run:
            resolved = shutil.which(verus)
            if not resolved:
                raise ValueError(f"Verus executable not found: {verus}")
            verus = str(Path(resolved).resolve())
        output.mkdir(parents=True, exist_ok=True)
        return evaluate_jobs(args, jobs, output, verus)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Re-evaluation failed: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Re-evaluation interrupted; saved records and index are in the output directory.\n")


if __name__ == "__main__":
    sys.exit(main())
