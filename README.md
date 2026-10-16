# VerusEval

VerusEval evaluates generated Verus specifications. This artifact includes the
released results, datasets, analysis scripts, and four baseline implementations.
The dataset contains 762 tasks and 13,716 generated programs across 18 configurations.

## 1. Reproduce the paper's tables and results

Start with [the result reproduction guide](docs/reproduction.md). Download the
matching code and data archives, extract both into one empty directory, and run:

```bash
docker compose run --build --rm veruseval
docker compose run --rm veruseval python scripts/reproduce.py --rq all --output-dir runs/full
```

This rebuilds tables, statistics, and figures from the released evaluation
records and checks numerical results against the released expected tables.
It needs no model API credentials. The guide lists the output files, how to
check success, and commands for statistics only or an individual RQ.

## 2. Re-evaluate programs and generate per-file records

Use [the evaluation guide](docs/usage.md) to run VerusEval on the released
programs or your own outputs. `scripts/reevaluate.py` writes fresh per-file JSON
records with the evaluator's metric details and a `target_evaluation` summary,
plus a sample index and a run summary.
The guide provides Docker and local Python commands, including a pairing check
and a single-sample run. New evaluation values may differ from the saved paper records.

## 3. Generate programs with the baselines

Continue with [the baseline guide](baselines/README.md) to run AutoVerus,
VeruSAGE, StarVerus, and AlphaVerus on the dataset. It provides model and API
configuration, commands for each tool, links to the generation and repair code,
and instructions for evaluating newly generated programs with VerusEval.
These runs use your model service and API credentials.

Additional references: [downloads](downloads/README.md),
[installation](docs/installation.md), [evaluation options](docs/usage.md),
[data layout](docs/data.md), and [metric image gallery](gallery/README.md).

Code: [MIT](LICENSE). Data and third-party terms:
[data license](DATA_LICENSE.md), [third-party notices](THIRD_PARTY.md).
