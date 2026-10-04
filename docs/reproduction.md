# Part 1: Reproduce the paper's tables and results

[Download and extract](../downloads/README.md) the matching code and data
archives into one directory. With [Docker and Docker Compose](installation.md)
installed, run from that directory:

```bash
docker compose run --build --rm veruseval
docker compose run --rm veruseval python scripts/reproduce.py \
  --rq all --output-dir runs/full
```

This rebuilds tables, statistics, and figures from the released evaluation
records without model API calls. Use a new or empty output directory for each run.

For statistics and LaTeX tables only, or a single RQ:

```bash
# Statistics and tables
docker compose run --rm veruseval python scripts/reproduce.py \
  --rq all --statistics-only --output-dir runs/statistics

# One result group: --rq accepts 1, 2, 3, or 4
docker compose run --rm veruseval python scripts/reproduce.py \
  --rq 2 --output-dir runs/rq2
```

RQ2 and RQ3 build their required RQ1 intermediate results automatically.
The [RQ script index](../RQs/README.md) maps each analysis to its code and inputs.

## Outputs and validation

A successful run exits with code 0 and sets `"status": "passed"` in
`runs/full/reproduction.json`. This file lists stage logs and the numerical
tables compared against `data/evidence/paper_results/`, with absolute tolerance
`1e-12`.

Main result paths below are relative to `runs/full/work/RQs/`:

| Group | Result files |
| --- | --- |
| RQ1 | `RQ1/results/rq1_acceptance_association.csv` |
| RQ2 | `RQ2/results/rq2_configuration_profiles.csv`, `RQ2/results/rq2_prompt_deltas.csv` |
| RQ3 | `RQ3/results/natural/conditional_rates.csv`, `RQ3/results/variants/detection_matrix.csv` |
| RQ4 | `RQ4/results/directions.csv`, `RQ4/results/reference_screen/screen_signals.csv`, `RQ4/results/reference_repair.csv` |

LaTeX tables are under `RQ1/tables/` and `RQ3/tables/`; rendered figures are under
each RQ's `figures/` directory. Recheck a completed run with:

```bash
docker compose run --rm veruseval python scripts/validate.py --results runs/full/work
```

For a single-RQ run, use its output path and matching selector, for example
`--results runs/rq2/work --rq 2`. Keep the extracted data available:
`work/data` links to it.

For missing files or hash mismatches, extract matching archives into a new
directory. For a failed stage, open its log listed in `reproduction.json`.
Numerical comparison errors identify the CSV, row, and field.

To generate and evaluate new programs, continue with
[Part 2: run and evaluate baselines](../baselines/README.md).
The optional [metric gallery](../gallery/README.md) has its own regeneration command.
