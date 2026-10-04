# Evaluate programs

After [installation](installation.md), run these commands from the project root:

```bash
# One generated program and its reference
docker compose run --rm veruseval python scripts/evaluation/evaluate.py file \
  -g generated.rs -r reference.rs -o runs/evaluation --verus-path /opt/verus/verus

# Directories containing matching filenames
docker compose run --rm veruseval python scripts/evaluation/evaluate.py dir \
  -g generated/ -r references/ -o runs/directory-evaluation \
  --workers 4 --verus-path /opt/verus/verus
```

Directory evaluation matches top-level `.rs` files by name. For baseline outputs,
use the [pairing instructions](../baselines/README.md#evaluate-newly-generated-programs)
to collect final programs and their evaluation references.

Each run writes `per_file/*.json`, `scores.csv`, `statistics.json`, `summary.json`,
and plots. Check metric states alongside scores: unavailable metrics and empty
I/O categories do not count as successful checks.

For offline I/O suites, copy `config.example.yaml` to `config.yaml` and set
`verus_path: /opt/verus/verus` and `io_suite_dir: data/io`. For LLM judgments,
export `VERUSEVAL_LLM_API_KEY`, `VERUSEVAL_LLM_BASE_URL`, and
`VERUSEVAL_LLM_MODEL_NAME` on the host, then pass each with `-e VARIABLE` before
`veruseval` in the Docker command. Without a configured judge, LLM metrics
are unavailable.

To reproduce the paper's released results, follow [Part 1](reproduction.md).
