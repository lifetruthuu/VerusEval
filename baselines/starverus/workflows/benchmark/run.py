#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional


WORKFLOW_DIR = Path(__file__).resolve().parent
REPO_DIR = WORKFLOW_DIR.parents[1]
DEFAULT_CONFIG = REPO_DIR / "configs" / "benchmark.yaml"


def _bootstrap_runtime() -> Path:
    """Select the config before importing modules that load it globally."""
    inherited_config = os.environ.get("STARVERUS_RUN_CONFIG")
    if inherited_config:
        config_path = Path(inherited_config)
    else:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--config",
            default=os.environ.get("STARVERUS_CONFIG", str(DEFAULT_CONFIG)),
        )
        known, _ = parser.parse_known_args()
        config_path = Path(known.config).expanduser().resolve()

    os.environ["STARVERUS_CONFIG"] = str(config_path)
    os.environ["STARVERUS_RUN_CONFIG"] = str(config_path)
    os.chdir(WORKFLOW_DIR)
    return config_path


CONFIG_PATH = _bootstrap_runtime()

from pipeline.agents import (  # noqa: E402
    BestCandidateSelector,
    InitialGenerator,
    Verifier,
)
from pipeline import StarVerusPipeline  # noqa: E402
from utils.file_util import (  # noqa: E402
    check_rs_files,
    get_file_name_by_path,
    get_path_by_file_name,
    load_content,
    load_config,
    load_json,
    save_content,
)


CONFIG = load_config()
ROOT_DIR = Path(CONFIG["root_dir"]).expanduser().resolve()

PIPELINE_SECTION_NAMES = (
    "initial_generator",
    "contract_alignment",
    "proof_repair",
    "actor",
    "rewriter",
)


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT_DIR / path).resolve()


def _format_duration(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"


def _get_few_shot_examples(
    file_path: Path,
    knn_similar: Dict[str, Any],
    x_dir: str,
    y_dir: str,
) -> list[list[str]]:
    task_name = get_file_name_by_path(file_path)
    if task_name not in knn_similar:
        raise KeyError(f"KNN examples are missing for task: {task_name}")

    examples = []
    for neighbor in knn_similar[task_name]:
        file_name = f"{neighbor}.rs"
        x_path = get_path_by_file_name(file_name, x_dir)
        y_path = get_path_by_file_name(file_name, y_dir)
        if x_path is None or y_path is None:
            raise FileNotFoundError(
                f"Could not resolve few-shot pair for neighbor: {neighbor}"
            )
        examples.append([load_content(x_path), load_content(y_path)])
    return examples


def _generate_one(
    source_file: str,
    mode: str,
    model_name: str,
    num_candidates: int,
    generation_temperature: float,
    few_shot_examples: int,
    generation_dir: str,
    knn_similar: Optional[Dict[str, Any]],
    x_dir: str,
    y_dir: str,
) -> tuple[str, int]:
    source_path = Path(source_file)
    examples = None
    if mode == "few-shot":
        if knn_similar is None:
            raise ValueError("Few-shot mode requires KNN examples")
        examples = _get_few_shot_examples(
            source_path,
            knn_similar,
            x_dir,
            y_dir,
        )[:few_shot_examples]

    generator = InitialGenerator(model_name, generation_temperature)
    results = generator.generate(source_path, num_candidates, examples)
    if not results:
        raise RuntimeError("InitialGenerator returned no candidates")

    task_dir = Path(generation_dir) / get_file_name_by_path(source_path)
    model_dir = task_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    raw_path = task_dir / "raw.rs"
    if not raw_path.exists():
        save_content(source_path.read_text(encoding="utf-8"), str(raw_path))

    verifier = Verifier()
    evaluated = []
    for index, code in enumerate(results):
        candidate_path = model_dir / f"result_{index}.rs"
        save_content(code, str(candidate_path))
        evaluated.append(verifier.verify(candidate_path))

    best = BestCandidateSelector().select(evaluated)
    return str(best.path), len(evaluated)


def _repair_one(
    candidate_file: str,
    generation_dir: str,
    repair_dir: str,
    backbone_model: str,
    pipeline_config: Dict[str, Any],
) -> tuple[str, bool]:
    candidate_path = Path(candidate_file)
    relative_path = candidate_path.relative_to(generation_dir)
    task_name = relative_path.parts[0]
    save_dir = (
        Path(repair_dir)
        / task_name
        / candidate_path.stem
        / backbone_model
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    pipeline = StarVerusPipeline(
        source_file=candidate_path,
        save_dir=save_dir,
        config=pipeline_config,
        backbone_model=backbone_model,
    )
    result = pipeline.run()
    return candidate_file, result.verified


def _process_one(
    source_file: str,
    mode: str,
    model_name: str,
    num_candidates: int,
    generation_temperature: float,
    few_shot_examples: int,
    generation_dir: str,
    repair_dir: str,
    pipeline_config: Dict[str, Any],
    knn_similar: Optional[Dict[str, Any]],
    x_dir: str,
    y_dir: str,
) -> tuple[str, int, bool]:
    candidate_file, candidate_count = _generate_one(
        source_file=source_file,
        mode=mode,
        model_name=model_name,
        num_candidates=num_candidates,
        generation_temperature=generation_temperature,
        few_shot_examples=few_shot_examples,
        generation_dir=generation_dir,
        knn_similar=knn_similar,
        x_dir=x_dir,
        y_dir=y_dir,
    )
    _, verified = _repair_one(
        candidate_file=candidate_file,
        generation_dir=generation_dir,
        repair_dir=repair_dir,
        backbone_model=model_name,
        pipeline_config=pipeline_config,
    )
    return candidate_file, candidate_count, verified


def _build_parser() -> argparse.ArgumentParser:
    generation = CONFIG.get("initial_generator", {})
    contract_alignment = CONFIG.get("contract_alignment", {})
    proof_repair = CONFIG.get("proof_repair", {})
    actor = CONFIG.get("actor", {})
    rewriter = CONFIG.get("rewriter", {})

    parser = argparse.ArgumentParser(
        description="Run the StarVerus benchmark pipeline."
    )
    parser.add_argument(
        "--mode",
        choices=("zero-shot", "few-shot"),
        required=True,
        help="InitialGenerator prompting mode.",
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help="Path to the benchmark YAML config.",
    )
    parser.add_argument(
        "--source_bench_dir",
        default="dataset/benchmark/unverified/DAFNY2VERUS-COLLECTION",
    )
    parser.add_argument(
        "--result_save_path",
        default="outputs/benchmark/generation",
        help="Base directory for InitialGenerator candidates.",
    )
    parser.add_argument(
        "--repair_save_path",
        default="outputs/benchmark/repair",
        help="Base directory for alignment and proof-repair results.",
    )
    parser.add_argument("--model_name", default="deepseek-reasoner")
    parser.add_argument(
        "--num_generation_candidates",
        "--n",
        dest="num_generation_candidates",
        type=int,
        default=generation.get("num_candidates", 5),
    )
    parser.add_argument(
        "--generation_temperature",
        type=float,
        default=generation.get("temperature", 1.0),
    )
    parser.add_argument(
        "--few_shot_examples",
        type=int,
        default=generation.get("few_shot_examples", 5),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of source-file tasks processed in parallel.",
    )

    parser.add_argument("--x_dir", default="dataset/benchmark/unverified")
    parser.add_argument("--y_dir", default="dataset/benchmark/verified")
    parser.add_argument(
        "--knn_json_path",
        default="dataset/benchmark/knn_similar.json",
    )

    parser.add_argument(
        "--max_alignment_iterations",
        type=int,
        default=contract_alignment.get("max_iterations", 3),
    )
    parser.add_argument(
        "--repair_temperature",
        type=float,
        default=proof_repair.get("temperature", 0.3),
    )
    parser.add_argument(
        "--max_repair_epochs",
        "--max_epoch",
        dest="max_repair_epochs",
        type=int,
        default=proof_repair.get("max_epochs", 3),
    )
    parser.add_argument(
        "--num_actor_candidates",
        "--width",
        dest="num_actor_candidates",
        type=int,
        default=actor.get("num_candidates", 3),
    )
    parser.add_argument(
        "--max_rewriter_invocations",
        type=int,
        default=rewriter.get("max_invocations", 1),
    )
    parser.add_argument(
        "--additional_repair_epochs",
        type=int,
        default=rewriter.get("additional_epochs", 3),
    )
    return parser


def _validate_config(parser: argparse.ArgumentParser) -> None:
    required_sections = (*PIPELINE_SECTION_NAMES, "models")
    missing = [name for name in required_sections if name not in CONFIG]
    if missing:
        parser.error("config is missing sections: " + ", ".join(missing))


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive_values = {
        "num_generation_candidates": args.num_generation_candidates,
        "few_shot_examples": args.few_shot_examples,
        "num_workers": args.num_workers,
        "max_alignment_iterations": args.max_alignment_iterations,
        "max_repair_epochs": args.max_repair_epochs,
        "num_actor_candidates": args.num_actor_candidates,
        "additional_repair_epochs": args.additional_repair_epochs,
    }
    for name, value in positive_values.items():
        if value <= 0:
            parser.error(f"--{name} must be greater than zero")
    if args.generation_temperature < 0:
        parser.error("--generation_temperature must be non-negative")
    if args.repair_temperature < 0:
        parser.error("--repair_temperature must be non-negative")
    if args.max_rewriter_invocations < 0:
        parser.error("--max_rewriter_invocations must be non-negative")


def _build_pipeline_config(args: argparse.Namespace) -> Dict[str, Any]:
    pipeline_config = {
        name: copy.deepcopy(CONFIG[name]) for name in PIPELINE_SECTION_NAMES
    }
    pipeline_config["models"] = {}
    pipeline_config["initial_generator"]["num_candidates"] = (
        args.num_generation_candidates
    )
    pipeline_config["initial_generator"]["temperature"] = (
        args.generation_temperature
    )
    pipeline_config["initial_generator"]["few_shot_examples"] = (
        args.few_shot_examples
    )
    pipeline_config["contract_alignment"]["max_iterations"] = (
        args.max_alignment_iterations
    )
    pipeline_config["proof_repair"]["temperature"] = args.repair_temperature
    pipeline_config["proof_repair"]["max_epochs"] = args.max_repair_epochs
    pipeline_config["actor"]["num_candidates"] = args.num_actor_candidates
    pipeline_config["rewriter"]["max_invocations"] = (
        args.max_rewriter_invocations
    )
    pipeline_config["rewriter"]["additional_epochs"] = (
        args.additional_repair_epochs
    )
    return pipeline_config


def main() -> int:
    parser = _build_parser()
    _validate_config(parser)
    args = parser.parse_args()
    _validate_args(args, parser)

    if args.model_name not in CONFIG.get("models", {}):
        parser.error(f"model is not configured: {args.model_name}")

    pipeline_config = _build_pipeline_config(args)
    for role, configured_model in pipeline_config.get("models", {}).items():
        model_name = configured_model or args.model_name
        if model_name not in CONFIG.get("models", {}):
            parser.error(f"model for {role} is not configured: {model_name}")

    source_dir = _resolve_path(args.source_bench_dir)
    if not source_dir.is_dir():
        parser.error(f"source benchmark directory does not exist: {source_dir}")

    x_dir = _resolve_path(args.x_dir)
    y_dir = _resolve_path(args.y_dir)
    knn_path = _resolve_path(args.knn_json_path)
    if args.mode == "few-shot":
        for label, path in (("x_dir", x_dir), ("y_dir", y_dir)):
            if not path.is_dir():
                parser.error(f"{label} does not exist: {path}")
        if not knn_path.is_file():
            parser.error(f"knn_json_path does not exist: {knn_path}")

    bench_name = source_dir.name
    generation_dir = (
        _resolve_path(args.result_save_path)
        / args.mode
        / f"{bench_name}_{args.mode}"
    )
    repair_dir = _resolve_path(args.repair_save_path) / args.mode / bench_name
    generation_dir.mkdir(parents=True, exist_ok=True)
    repair_dir.mkdir(parents=True, exist_ok=True)

    log_path = generation_dir / f"{bench_name}_{args.model_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("benchmark_pipeline")
    logger.info("Arguments: %s", args)
    logger.info("Generation output: %s", generation_dir)
    logger.info("Repair output: %s", repair_dir)

    source_files = sorted(check_rs_files(source_dir))
    if not source_files:
        logger.error("No .rs files found in %s", source_dir)
        return 1

    knn_similar = load_json(str(knn_path)) if args.mode == "few-shot" else None
    completed_count = 0
    task_failures = 0
    verified_count = 0
    pipeline_start = time.time()

    logger.info(
        "Starting %d source-file tasks with %d workers",
        len(source_files),
        args.num_workers,
    )
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                _process_one,
                str(source_file),
                args.mode,
                args.model_name,
                args.num_generation_candidates,
                args.generation_temperature,
                args.few_shot_examples,
                str(generation_dir),
                str(repair_dir),
                pipeline_config,
                knn_similar,
                str(x_dir),
                str(y_dir),
            ): source_file
            for source_file in source_files
        }
        for future in as_completed(futures):
            source_file = futures[future]
            try:
                best_candidate, candidate_count, verified = future.result()
                completed_count += 1
                verified_count += int(verified)
                logger.info(
                    "[TASK %s] %s: generated=%d, selected=%s",
                    "VERIFIED" if verified else "DONE",
                    source_file,
                    candidate_count,
                    best_candidate,
                )
            except Exception as exc:
                task_failures += 1
                logger.exception("[TASK FAIL] %s: %s", source_file, exc)

    logger.info(
        "Pipeline finished in %s: completed=%d, verified=%d, failures=%d",
        _format_duration(time.time() - pipeline_start),
        completed_count,
        verified_count,
        task_failures,
    )
    return 1 if task_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
