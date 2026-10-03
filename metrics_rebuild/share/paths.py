from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LLM_CACHE_PATH = PROJECT_ROOT / ".cache" / "llm_metrics_cache.json"
STARVERUS_BENCHMARKS_DIR = PROJECT_ROOT / "data/references"
STARVERUS_IO_DIR = PROJECT_ROOT / "data/io"
