# Evaluate programs

After [installation](installation.md), run commands from the project root using
either Docker or your local Python environment.

## Docker

```bash
# One generated program and its reference
docker compose run --rm veruseval python scripts/evaluation/evaluate.py file \
  -g generated.rs -r reference.rs -o runs/evaluation --verus-path /opt/verus/verus

# Directories containing matching filenames
docker compose run --rm veruseval python scripts/evaluation/evaluate.py dir \
  -g generated/ -r references/ -o runs/directory-evaluation \
  --workers 4 --verus-path /opt/verus/verus
```

## Local Python

Complete the [local environment setup](installation.md#local-environment), then
activate that environment in your terminal. Set `VERUS_PATH` to the absolute path
of your installed Verus executable, as shown in the setup instructions.

```bash
conda activate veruseval

# One generated program and its reference
python scripts/evaluation/evaluate.py file \
  -g generated.rs -r reference.rs -o runs/evaluation --verus-path "$VERUS_PATH"

# Directories containing matching filenames
python scripts/evaluation/evaluate.py dir \
  -g generated/ -r references/ -o runs/directory-evaluation \
  --workers 4 --verus-path "$VERUS_PATH"
```

## Inputs, outputs, and optional configuration

Directory evaluation matches top-level `.rs` files by name. For baseline outputs,
use the [pairing instructions](../baselines/README.md#evaluate-newly-generated-programs)
to collect final programs and their evaluation references.

Each run writes `per_file/*.json`, `scores.csv`, `statistics.json`, `summary.json`,
and plots. Check metric states alongside scores: unavailable metrics and empty
I/O categories do not count as successful checks.

For offline I/O suites, copy `config.example.yaml` to `config.yaml` and set
`io_suite_dir: data/io`. Set `verus_path` to `/opt/verus/verus` for Docker or to
your local executable's absolute path for local Python. The `--verus-path`
argument overrides this configuration value.

For LLM judgments, export `VERUSEVAL_LLM_API_KEY`, `VERUSEVAL_LLM_BASE_URL`, and
`VERUSEVAL_LLM_MODEL_NAME` in your terminal. Local Python reads them directly;
with Docker, pass each with `-e VARIABLE` before `veruseval` in the command.
Without a configured judge, LLM metrics are unavailable.

To reproduce the paper's released results, follow [Part 1](reproduction.md).
