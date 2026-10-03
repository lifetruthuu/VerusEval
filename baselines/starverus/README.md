# StarVerus

Source: https://github.com/Je5s1e/KDD26-ADS-StarVerus

Commit: `0c0ce03c7f68027085bdb70c82075510cf0a8f57`. The single-file benchmark
source and prompts are included unchanged under the [MIT license](LICENSE).

After downloading the data and building Docker, check the setup:

```bash
docker compose run --rm veruseval python scripts/generation/run_starverus.py \
  --model gpt-4o --shot few-shot --dry-run
```

To generate programs, export `OPENAI_API_KEY` and `OPENAI_BASE_URL`, then run:

```bash
docker compose run --rm -e OPENAI_API_KEY -e OPENAI_BASE_URL veruseval \
  python scripts/generation/run_starverus.py --model gpt-4o --shot few-shot \
  --output-root runs/generation/starverus-gpt4o-few
```

The default covers all 762 tasks using `data/generation/`. Use `--shot zero-shot`
to change prompting, `--benchmark HumanEval-Verus` to select a subset, and
`--help` for other options. Choose a fresh output directory for each run.
Docker already contains the required dependencies and pinned Verus version.
Paper reproduction uses the frozen outputs; model generation produces new runs.
