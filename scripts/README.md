# Entry points

`reproduce.py` reproduces RQ1–RQ4. It first runs `rebuild_rq1_inputs.py` to rebuild
the two RQ1 input CSV files from per-file JSON records. `validate.py` verifies
data and results, and `package.py` builds archives from the release whitelist.
`reevaluate.py` evaluates program sources and writes fresh per-file JSON, a
sample index, and a run summary. It reuses the single-sample evaluator in
`evaluation/evaluate.py`, preserving its metric results. The helper
`evaluation/target_summary.py` adds `target_evaluation` from those results
without running additional metrics.
`plot_gallery.py` regenerates all metric overview images. `generation/` contains the baseline launchers;
`io/` generates and validates I/O suites. Start with
[Part 1: reproduce paper results](../docs/reproduction.md) or
[Part 2: re-evaluate programs](../docs/usage.md), or
[Part 3: generate programs with baselines](../baselines/README.md).
