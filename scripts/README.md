# Entry points

- `reproduce.py`: reproduces RQ1–RQ4. It first runs `rebuild_rq1_inputs.py` to rebuild the RQ1 input CSVs from per-file JSON records.
- `validate.py`: checks data and results.
- `package.py`: builds archives from the release whitelist.
- `reevaluate.py`: evaluates program sources and writes per-file JSON, a sample index, and a run summary. It reuses `evaluation/evaluate.py`, so metric results are unchanged.
- `evaluation/target_summary.py`: adds `target_evaluation` from existing results, without running more metrics.
- `plot_gallery.py`: regenerates all metric overview images.
- `generation/`: baseline launchers.
- `io/`: generates and validates I/O suites.

Where to start:

- Reproduce paper results: [Part 1](../docs/reproduction.md)
- Re-evaluate programs: [Part 2](../docs/usage.md)
- Generate programs with baselines: [Part 3](../baselines/README.md)
