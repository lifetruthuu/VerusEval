# RQ1–RQ4 reproduction

From the repository root in the [configured environment](../docs/installation.md):

```bash
python scripts/reproduce.py --rq all --output-dir runs/full
```

Use `--rq 1`, `2`, `3`, or `4` to select one RQ, and `--statistics-only` to skip
figures. Each run needs a new or empty output directory. RQ2 and RQ3 automatically
build their required RQ1 results. No model API or Verus execution is needed.

| RQ | Analysis | Main scripts | Released inputs |
| --- | --- | --- | --- |
| RQ1 | Metric association with verifier acceptance; checks on accepted programs | [refresh_rq1_paper.py](RQ1/scripts/refresh_rq1_paper.py), [refresh_rq1_screening.py](RQ1/scripts/refresh_rq1_screening.py) | `data/evaluation/` |
| RQ2 | Metric profiles by generation configuration and paired comparisons | [analyze_rq2_configurations.py](RQ2/scripts/analyze_rq2_configurations.py) | `data/evaluation/` and RQ1 results |
| RQ3 | Conditional quality of accepted programs; I/O detection of controlled contract variants | [analyze_rq3_natural.py](RQ3/scripts/analyze_rq3_natural.py), [analyze_rq3_variants.py](RQ3/scripts/analyze_rq3_variants.py) | `data/evaluation/`, `data/evidence/contract_variants/`, and RQ1 results |
| RQ4 | Directional flags, clause cases, reference screening, and a reference repair case | [build_rq4.py](RQ4/scripts/build_rq4.py) | `data/evaluation/`, `data/evidence/{clause_cases,reference_review,reference_repair}/` |

Results, tables, and figures are written under `runs/full/work/RQs/RQ*/`.
`runs/full/reproduction.json` records stage logs and comparisons with the expected
tables in `data/evidence/paper_results/`. Each RQ's `validate_*.py` script checks
its records during reproduction. `shared/` provides record readers and contract
parsing helpers.

See [Part 1](../docs/reproduction.md) for Docker commands, output filenames, and
validation. The data archive's `data/evidence/README.md` describes the supporting
inputs. For new generation and evaluation, use [Part 2](../baselines/README.md).
