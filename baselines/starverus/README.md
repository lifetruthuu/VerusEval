# StarVerus benchmark workflow

Source: https://github.com/Je5s1e/KDD26-ADS-StarVerus

Pinned upstream commit: `0c0ce03c7f68027085bdb70c82075510cf0a8f57`.
The complete `workflows/benchmark/` subtree is included unchanged, with its
generation, candidate selection, contract alignment, proof repair and prompts.
The upstream MIT license is retained in [LICENSE](LICENSE).

Run from the VerusEval root using the pinned Docker environment:

```bash
docker compose run --rm veruseval python scripts/generation/run_starverus.py \
  --model gpt-4o --shot few-shot --dry-run
```

For generation, export `OPENAI_API_KEY` and `OPENAI_BASE_URL` on the host, then:

```bash
docker compose run --rm -e OPENAI_API_KEY -e OPENAI_BASE_URL veruseval \
  python scripts/generation/run_starverus.py --model gpt-4o --shot few-shot \
  --output-root runs/generation/starverus-gpt4o-few
```

Use `--shot zero-shot` for zero-shot generation, `--benchmark HumanEval-Verus`
to select a benchmark, and `--workers` or `--candidates` to control concurrency
and initial candidate count. The default covers all 762 tasks. Few-shot runs use
the released `data/generation/X_code`, `Y`, and ordered `knn_similar.json` mapping.
The adapter creates a temporary upstream configuration with your endpoint and
the pinned Verus executable. Its default verifier timeout is 120 seconds.
Credentials are read from environment variables and the temporary configuration
is removed when the run exits. Select a fresh output directory for each run.

The benchmark workflow needs only Python, PyYAML and the OpenAI client already
provided by the artifact Docker image; Verus is fixed to `0.2025.09.25.04e8687`.
`--dry-run` checks all dataset mappings, the Verus version and upstream imports
without making model calls. The upstream multi-file FVT workflow and its separate
datasets are outside this artifact's single-file benchmark scope.

This pinned public implementation is provided for new generation runs. Paper
result reproduction uses the released StarVerus outputs and evaluation records;
new model calls can produce different programs.
