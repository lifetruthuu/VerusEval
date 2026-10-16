# Entry points

`reproduce.py` reproduces RQ1–RQ4. It first runs `rebuild_rq1_inputs.py` to rebuild
the two RQ1 input CSV files from per-file JSON records. `validate.py` verifies
data and results, and `package.py` builds archives from the release whitelist. `plot_gallery.py`
regenerates all metric overview images. `evaluation/evaluate.py`
evaluates file or directory pairs. `generation/` contains the baseline launchers;
`io/` generates and validates I/O suites. Start with
[Part 1: reproduce paper results](../docs/reproduction.md) or
[Part 2: run and evaluate baselines](../baselines/README.md).
