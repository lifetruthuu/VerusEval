# Part 2: Re-evaluate programs and generate per-file records

After [installation](installation.md), run `scripts/reevaluate.py` from the
project root to evaluate generated programs against their references. It writes
fresh `per_file/*.json` records, a sample index, and a run summary.

New values may differ from the paper's saved records, which include subsequent
rechecks and reviews; tool versions, timeouts, and model judgments can also
affect results. [Part 1](reproduction.md) uses the saved paper records to
reproduce RQ1–RQ4. New evaluation outputs are kept in their own run directory.

## Docker

Start with a pairing check or one task from the released programs:

```bash
# Check all 13,716 program/reference pairs without running metrics
docker compose run --rm veruseval python scripts/reevaluate.py dataset \
  --dry-run -o runs/pairing-check

# Re-evaluate one released sample
docker compose run --rm veruseval python scripts/reevaluate.py dataset \
  --workflow AutoVerus --model gpt-4o --shot zero-shot \
  --task-id VeriCoding_VV0178_vericoded \
  -o runs/evaluation-one --verus-path /opt/verus/verus

# Re-evaluate the full released corpus
docker compose run --rm veruseval python scripts/reevaluate.py dataset \
  -o runs/evaluation-full --workers 4 --verus-path /opt/verus/verus
```

For your own program pairs:

```bash
docker compose run --rm veruseval python scripts/reevaluate.py file \
  -g generated.rs -r reference.rs -o runs/evaluation-file \
  --verus-path /opt/verus/verus

docker compose run --rm veruseval python scripts/reevaluate.py dir \
  -g generated/ -r references/ -o runs/evaluation-directory \
  --workers 4 --verus-path /opt/verus/verus
```

## Local Python

Activate the [local environment](installation.md#local-environment) and set
`VERUS_PATH` to your installed Verus executable. The same commands work without
the Docker prefix:

```bash
conda activate veruseval

python scripts/reevaluate.py dataset --dry-run -o runs/local-pairing-check

python scripts/reevaluate.py dataset \
  --workflow AutoVerus --model gpt-4o --shot zero-shot \
  --task-id VeriCoding_VV0178_vericoded \
  -o runs/local-evaluation-one --verus-path "$VERUS_PATH"

python scripts/reevaluate.py dataset \
  -o runs/local-evaluation-full --workers 4 --verus-path "$VERUS_PATH"

python scripts/reevaluate.py file \
  -g generated.rs -r reference.rs -o runs/local-evaluation-file \
  --verus-path "$VERUS_PATH"

python scripts/reevaluate.py dir \
  -g generated/ -r references/ -o runs/local-evaluation-directory \
  --workers 4 --verus-path "$VERUS_PATH"
```

## Inputs and configuration

Dataset mode reads `data/generated/manifest.csv` and matches task IDs to
`data/evaluation/target_functions.csv`. Its optional `--workflow`, `--model`,
`--shot`, and `--task-id` filters select exact values; omit them to evaluate all
released samples. It automatically uses the offline I/O suites in `data/io`.

Use `--manifest path/to/programs.csv` for newly generated benchmark programs.
The required columns are `workflow,model,shot,task_id,generated_path`; an optional
`sample_id` must equal `workflow|model|shot|task_id`. Relative source paths are
resolved from the project root. References come from the target catalog. The
[baseline guide](../baselines/README.md#evaluate-newly-generated-programs) shows
how to prepare this manifest from each baseline's final outputs.

Directory mode matches top-level `.rs` filenames and reports unmatched files
as an input error. File and directory modes accept `--workflow`, `--model`, and
`--shot` labels, defaulting to `local`, `unspecified`, and `unspecified`. Task IDs
default to reference filename stems; file mode also accepts `--task-id`.

Use `--io-root data/io` to select the released offline suites explicitly.
File and directory modes otherwise use `io_suite_dir` from `config.yaml`, if
configured. Keep benchmark references at their catalog paths so their target
metadata and I/O suites can be found. For custom references without offline
suites, the existing metric evaluator constructs available checks and records
unavailable categories.

Verus selection follows `--verus-path`, then `VERUS_PATH`, then `verus_path`
in `config.yaml`, then `verus` on `PATH`. Use the version from the installation
guide. `--worker-timeout-seconds` limits the total time for each sample; the
default is 1800 seconds, and 0 disables this limit. Individual metrics also
have their own verifier limits.

For LLM judgments, export `VERUSEVAL_LLM_API_KEY`, `VERUSEVAL_LLM_BASE_URL`, and
`VERUSEVAL_LLM_MODEL_NAME`. Local Python reads them directly; with Docker, pass
each using `-e VARIABLE` before `veruseval`. You can also use the `llm` section
of `config.yaml`, following `config.example.yaml`. Judgments requiring a model
are unavailable without judge configuration. Configured calls use your model service.

## Outputs and completion

Choose a new or empty output directory for each invocation, including dry runs.
The output contains:

- `per_file/*.json`: the evaluator's metric details, sample identity, and errors.
- `artifact_index.csv`: one row per selected sample, with input paths, the result
  path relative to this run directory, and evaluation status.
  JSON filenames encode the full sample ID.
- `summary.json`: sample, task, configuration, and completion counts, plus run settings.
- `worker_logs/`: individual worker logs.

The `metrics` object retains the existing evaluator's outputs, including scores,
statuses, per-function or per-clause details, I/O case results, and diagnostics
where provided by each metric. Select the fields relevant to your analysis.
The downloaded paper records additionally contain merged review results and
`target_evaluation`; this entry produces fresh evaluator records.

A run exits with code 0 and summary status `completed` when all samples finish
without execution exceptions. Program rejection and unavailable metrics are
ordinary evaluation outcomes. Worker failures, timeouts, or metric exceptions
produce `completed_with_errors` and exit code 1; inspect the affected JSON and
log. Each metric completion saves a checkpoint, so a timed-out sample retains
its completed metrics. An interrupted run keeps its records and an index that
distinguishes pending and interrupted samples. A dry run reports `checked` and
writes the planned index without evaluating programs.

Checks cover files, unique sample IDs, pairing, and record completeness.
The evaluation entry does not scan release-wide hashes or compare new values
against paper tables. Existing metric checks still match offline suites to
their reference programs. Unavailable judgments and empty I/O categories do
not count as successful checks.

Continue with [Part 3: generate programs with baselines](../baselines/README.md).
