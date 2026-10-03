# Evaluating programs and generating specifications

Configure the pinned Verus binary as described in [installation](installation.md).
Inside Docker it is already available at `/opt/verus/verus`. Prefix the commands
below with `docker compose run --rm veruseval` when running from the host.
The evaluation entry supports a single program/reference pair and directories:

```bash
python scripts/evaluation/evaluate.py file -g generated.rs -r reference.rs \
  -o runs/evaluation --verus-path /opt/verus/verus
python scripts/evaluation/evaluate.py dir -g generated/ -r references/ \
  -o runs/directory-evaluation --workers 4 --verus-path /opt/verus/verus
```

Each run writes per-file JSON, scores, summary statistics and plots. The JSON
contains metric states as well as values. An unavailable metric or empty I/O
category is not a successful check. Set LLM endpoint/model/key configuration
when requesting intent judgments; retained offline results need no API calls.

For an inexpensive identical-file smoke check:

```bash
verus examples/identity.rs
python -m metrics_rebuild.cli.main examples/identity.rs examples/identity.rs \
  --no-mutation --compact
```

The identity example should pass parsing, type checking and verification. Its
I/O metric can be unavailable because the example has no offline suite.

`scripts/generation/run_spec_baselines.py` launches the adapted AlphaVerus or
AutoVerus implementation. `scripts/generation/run_verusage_completion.py`
launches VeruSAGE. Use `--help` for workflow controls, `--dataset-root
data/generation`, an explicit model/endpoint, and a fresh `--output-root`.
The `--dry-run` option checks dataset selection and launcher configuration
without paid model calls. Defaults write new generation under `runs/generation/`.
Set `VERUS_PATH` to the pinned binary for AlphaVerus's internal checks.
Optional error exemplars can be supplied with `ALPHAVERUS_ERROR_EXEMPLARS`.

Generation inputs comprise 762 unannotated `X_code/` programs, their `Y/`
examples and the ordered five-example mapping `knn_similar.json`. Generation
examples can differ from the evaluation references; see [data](data.md).
The StarVerus output corpus is released, but this repository does not contain
a complete StarVerus generation launcher.

`scripts/io/generate_io_tests_llm.py` generates candidate suites into a required
output directory. `scripts/io/revalidate_io_tests.py` checks strict I/O validity
and defaults to `runs/io/revalidated`; `scripts/io/rebuild_io_suite_summary.py`
rebuilds summary metadata from a selected suite directory.

Run the retained tests with `python -m unittest discover -s tests`. The standalone
scripts `tests/test_trivial_enhancements.py` and `tests/test_z3_parser_enhancements.py`
cover additional parser and triviality cases. Some proof tests require the pinned
Verus on `PATH`; offline baseline tests do not call model services.
