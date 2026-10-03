"""Generate contracts and proofs for the released baseline configuration matrix."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generation.spec_baseline_common import atomic_write_json, load_dataset

VERUS_VERSION = "0.2025.09.25.04e8687"
BASELINES = ("alphaverus", "autoverus", "verusage", "starverus")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_runs(config, baseline, model, shot):
    runs = [r for r in config["runs"]
            if (baseline == "all" or r["baseline"] == baseline)
            and (model is None or r["model"] == model)
            and (shot == "all" or r["shot"] == shot)]
    if not runs:
        raise ValueError("No configuration matches the baseline/model/shot selection")
    names = []
    for run in runs:
        if run["baseline"] not in BASELINES or run["shot"] not in ("zero-shot", "few-shot"):
            raise ValueError("Invalid baseline or shot in generation config")
        label = run["model"]
        if not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in label):
            raise ValueError("Invalid model label in generation config")
        names.append((run["baseline"], label, run["shot"]))
    if len(set(names)) != len(names):
        raise ValueError("Duplicate generation configuration")
    return runs


def command_for(run, settings, args, output, model_id, base_url, tasks):
    baseline = run["baseline"]
    common = ["--dataset-root", str(args.dataset_root), "--output-root", str(output),
              "--shot", run["shot"], "--workers", str(args.workers),
              "--verus-path", args.verus_path]
    if baseline in ("alphaverus", "autoverus"):
        prefix = "alpha" if baseline == "alphaverus" else "auto"
        command = [sys.executable, str(ROOT / "scripts/generation/run_spec_baselines.py"),
                   "--pipeline", baseline, f"--{prefix}-model", model_id,
                   f"--{prefix}-base-url", base_url]
    elif baseline == "verusage":
        command = [sys.executable, str(ROOT / "scripts/generation/run_verusage_completion.py"),
                   "--model", f"{run['model']}={model_id}", "--api-base", base_url]
    else:
        command = [sys.executable, str(ROOT / "scripts/generation/run_starverus.py"),
                   "--model", model_id, "--base-url", base_url, "--benchmark", args.benchmark]
        if run["model"] == "deepseek-reasoner":
            command.append("--thinking")
    command += common
    for key, value in settings["arguments"].items():
        command.extend(["--" + key, str(value)])
    if baseline != "starverus" and args.benchmark != "all":
        for task in tasks:
            command.extend(["--task-id", task.task_id])
    if args.dry_run:
        command.append("--dry-run")
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "baselines/generation.json")
    parser.add_argument("--baseline", choices=("all", *BASELINES), default="all")
    parser.add_argument("--model", help="Corpus model label, e.g. gpt-4o or qwen-coder.")
    parser.add_argument("--api-model", help="Override provider model identifier; requires --model.")
    parser.add_argument("--base-url", help="Override the endpoint for all selected runs.")
    parser.add_argument("--shot", choices=("all", "zero-shot", "few-shot"), default="all")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/generation")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/generation/dataset")
    parser.add_argument("--benchmark", default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verus-path", default=os.environ.get("VERUS_PATH", "verus"))
    parser.add_argument("--dry-run", action="store_true", help="Check all selected launchers without model calls.")
    args = parser.parse_args(argv)
    try:
        if args.workers < 1 or (args.api_model and not args.model):
            raise ValueError("workers must be positive; --api-model requires --model")
        args.dataset_root = args.dataset_root.resolve()
        args.output_root = args.output_root.resolve()
        if (args.output_root.is_relative_to(args.dataset_root)
                or args.dataset_root.is_relative_to(args.output_root)):
            raise ValueError("output-root must be separate from the generation inputs")
        if args.output_root.exists() and any(args.output_root.iterdir()):
            raise ValueError("Choose a new, empty output-root")
        config = json.loads(args.config.read_text())
        runs = select_runs(config, args.baseline, args.model, args.shot)
        tasks, x_map, y_map = load_dataset(args.dataset_root)
        counts = Counter(t.subset for t in tasks)
        if args.benchmark != "all":
            if args.benchmark not in counts:
                raise ValueError(f"Unknown benchmark: {args.benchmark}")
            tasks = [t for t in tasks if t.subset == args.benchmark]
        executable = shutil.which(args.verus_path)
        if not executable:
            raise ValueError(f"Verus executable not found: {args.verus_path}")
        version = subprocess.run([executable, "--version"], check=True, capture_output=True, text=True)
        if VERUS_VERSION not in version.stdout + version.stderr:
            raise ValueError(f"Requires Verus {VERUS_VERSION}")
        args.verus_path = executable
        jobs = []
        for run in runs:
            provider = config["models"][run["model"]]
            settings = config["workflows"][run["baseline"]]
            model_id = args.api_model or settings.get("model_overrides", {}).get(
                run["model"], provider["api_model"])
            base_url = (args.base_url or os.getenv(provider["base_url_env"])
                        or os.getenv("OPENAI_BASE_URL")
                        or settings.get("endpoint_overrides", {}).get(run["model"])
                        or provider["base_url"])
            key = (os.getenv(provider["api_key_env"]) or os.getenv("BASELINE_API_KEY")
                   or os.getenv("OPENAI_API_KEY"))
            if not args.dry_run and not key:
                raise ValueError(f"Set {provider['api_key_env']} or OPENAI_API_KEY for {run['model']}")
            if not args.dry_run and not base_url:
                raise ValueError(f"Set {provider['base_url_env']} or OPENAI_BASE_URL for {run['model']}")
            name = f"{run['baseline']}/{run['model']}/{run['shot']}"
            command = command_for(run, settings, args, args.output_root / name,
                                  model_id, base_url or "https://example.invalid/v1", tasks)
            env = os.environ.copy()
            env["VERUS_PATH"] = executable
            if key:
                for variable in ("OPENAI_API_KEY", "BASELINE_API_KEY", "ALPHAVERUS_API_KEY",
                                 "VERUSAGE_QWEN_CODER_API_KEY", "VERUSAGE_DEEPSEEK_API_KEY"):
                    env[variable] = key
            jobs.append(({**run, "name": name, "api_model": model_id, "base_url": base_url,
                          "tasks": len(tasks), "command": command}, env))
        # Hash inputs and examples once so a new run can be tied to its exact inputs.
        inputs = sorted([*x_map.values(), *y_map.values(), args.dataset_root / "knn_similar.json"])
        atomic_write_json(args.output_root / "inputs.json", {
            str(p.relative_to(args.dataset_root)): file_hash(p) for p in inputs})
        manifest = {"status": "running", "dry_run": args.dry_run,
                    "config_sha256": file_hash(args.config), "configuration": config,
                    "verus_version": VERUS_VERSION, "benchmark": args.benchmark,
                    "runs": [job for job, _ in jobs]}
        manifest_path = args.output_root / "generation.json"
        atomic_write_json(manifest_path, manifest)
        failed = False
        for index, (job, env) in enumerate(jobs, 1):
            log = args.output_root / "logs" / (job["name"].replace("/", "__") + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as stream:
                result = subprocess.run(job["command"], cwd=ROOT, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT)
            job["returncode"] = result.returncode
            job["log"] = str(log.relative_to(args.output_root))
            job["status"] = "passed" if result.returncode == 0 else "failed"
            failed |= result.returncode != 0
            atomic_write_json(manifest_path, manifest)
            print(f"[{index}/{len(jobs)}] {job['name']}: {job['status']}", flush=True)
        manifest["status"] = "failed" if failed else ("checked" if args.dry_run else "completed")
        atomic_write_json(manifest_path, manifest)
        if failed:
            raise ValueError(f"A baseline failed; see {args.output_root / 'logs'}")
        print(f"{len(jobs)} configurations, {len(tasks)} tasks each; manifest: {manifest_path}")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Generation failed: {error}\n")


if __name__ == "__main__":
    main()
