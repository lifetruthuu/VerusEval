"""Run the bundled StarVerus benchmark workflow on the released inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generation.spec_baseline_common import load_dataset

UPSTREAM = ROOT / "baselines/starverus/workflows/benchmark/run.py"
VERUS_VERSION = "0.2025.09.25.04e8687"


def runtime_config(verus_path, model, base_url, api_key, thinking=False):
    return {
        "root_dir": str(ROOT),
        "verus": {"verus_path": verus_path, "timeout_duration": 120},
        "models": {"configured-model": {
            "api_key": api_key, "base_url": base_url, "model_name": model,
            "max_n": 1, "is_thinking": thinking,
        }},
        "initial_generator": {"temperature": 1.0, "num_candidates": 5, "few_shot_examples": 5},
        "contract_alignment": {"max_iterations": 3},
        "proof_repair": {"temperature": 0.3, "max_epochs": 3},
        "actor": {"num_candidates": 3},
        "rewriter": {"max_invocations": 1, "additional_epochs": 3},
    }


def command_for(args, config, subset):
    return [
        sys.executable, str(UPSTREAM), "--config", str(config),
        "--mode", args.shot, "--model_name", "configured-model",
        "--source_bench_dir", str(args.dataset_root / "X_code" / subset),
        "--x_dir", str(args.dataset_root / "X_code"),
        "--y_dir", str(args.dataset_root / "Y"),
        "--knn_json_path", str(args.dataset_root / "knn_similar.json"),
        "--result_save_path", str(args.output_root / "generation"),
        "--repair_save_path", str(args.output_root / "repair"),
        "--num_workers", str(args.workers),
        "--num_generation_candidates", str(args.candidates),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/generation")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/generation/starverus")
    parser.add_argument("--benchmark", default="all", help="Benchmark directory name, or all.")
    parser.add_argument("--shot", choices=["zero-shot", "few-shot"], default="few-shot")
    parser.add_argument("--model", required=True, help="Provider model identifier.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--verus-path", default=os.getenv("VERUS_PATH", "verus"))
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alignment-rounds", type=int, default=3)
    parser.add_argument("--repair-temperature", type=float, default=0.3)
    parser.add_argument("--repair-rounds", type=int, default=3)
    parser.add_argument("--actor-candidates", type=int, default=3)
    parser.add_argument("--rewriter-invocations", type=int, default=1)
    parser.add_argument("--rewriter-rounds", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Check inputs, imports and pinned Verus without model calls.")
    args = parser.parse_args(argv)
    try:
        if min(args.workers, args.candidates, args.actor_candidates) < 1:
            raise ValueError("workers and candidate counts must be positive")
        if min(args.temperature, args.repair_temperature, args.alignment_rounds,
               args.repair_rounds, args.rewriter_invocations, args.rewriter_rounds) < 0:
            raise ValueError("temperatures and round limits must be non-negative")
        args.dataset_root = args.dataset_root.resolve()
        args.output_root = args.output_root.resolve()
        if args.output_root.is_relative_to(args.dataset_root):
            raise ValueError("output-root must be outside the generation inputs")
        tasks, _, _ = load_dataset(args.dataset_root)
        counts = Counter(t.subset for t in tasks)
        if args.benchmark != "all" and args.benchmark not in counts:
            raise ValueError(f"Unknown benchmark: {args.benchmark}; choose from {sorted(counts)}")
        selected = sorted(counts) if args.benchmark == "all" else [args.benchmark]
        executable = shutil.which(args.verus_path)
        if not executable:
            raise ValueError(f"Verus executable not found: {args.verus_path}")
        version = subprocess.run([executable, "--version"], check=True, capture_output=True, text=True)
        if VERUS_VERSION not in version.stdout + version.stderr:
            raise ValueError(f"StarVerus requires Verus {VERUS_VERSION}")
        key = os.getenv("OPENAI_API_KEY") or os.getenv("BASELINE_API_KEY")
        if not args.dry_run and not key:
            raise ValueError("Set OPENAI_API_KEY or BASELINE_API_KEY before generation")
        config = runtime_config(str(Path(executable).resolve()), args.model, args.base_url,
                                "offline-check" if args.dry_run else key, args.thinking)
        config["initial_generator"].update(temperature=args.temperature, num_candidates=args.candidates)
        config["contract_alignment"]["max_iterations"] = args.alignment_rounds
        config["proof_repair"].update(temperature=args.repair_temperature, max_epochs=args.repair_rounds)
        config["actor"]["num_candidates"] = args.actor_candidates
        config["rewriter"].update(max_invocations=args.rewriter_invocations,
                                  additional_epochs=args.rewriter_rounds)
        # Upstream loads its YAML before imports. Keep credentials in a private,
        # temporary directory and remove the config when the subprocess exits.
        with tempfile.TemporaryDirectory(prefix="veruseval-starverus-") as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            path.chmod(0o600)
            env = {k: v for k, v in os.environ.items()
                   if k not in {"STARVERUS_CONFIG", "STARVERUS_RUN_CONFIG"}}
            if args.dry_run:
                subprocess.run([sys.executable, str(UPSTREAM), "--config", str(path), "--help"],
                               check=True, env=env, capture_output=True, text=True)
            else:
                if args.output_root.exists() and any(args.output_root.iterdir()):
                    raise ValueError("Choose a new, empty output-root for this generation run")
                for subset in selected:
                    subprocess.run(command_for(args, path, subset), check=True, env=env)
        print(json.dumps({"status": "checked" if args.dry_run else "completed",
                          "shot": args.shot, "model": args.model,
                          "tasks": sum(counts[s] for s in selected),
                          "benchmarks": {s: counts[s] for s in selected},
                          "output_root": str(args.output_root)}, indent=2))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"StarVerus failed: {error}\n")


if __name__ == "__main__":
    main()
