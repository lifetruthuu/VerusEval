# Data

The data archive contains six directories under `data/`:

- `evaluation/`: per-file metric records and indexes.
- `generated/`: the 13,716 evaluated Rust programs across 18 configurations.
- `references/`: one evaluation reference for each of the 762 tasks.
- `io/`: offline I/O suites and their validation records.
- `generation/`: baseline inputs, annotated examples, and retrieval mappings.
- `evidence/`: proof evidence, controlled variants, review and repair cases, and expected results. Its `README.md` maps these inputs to the RQs.

`evaluation/artifact_index.csv` maps sample IDs to result files and hashes.
`generated/manifest.csv` maps samples to generated programs.
`evaluation/target_functions.csv` maps task IDs to reference files and target
functions. Quality analyses use 13,659 eligible programs; the 57 exclusions have
`analysis_eligible=False` in `evaluation/artifact_index.csv`.

Use `references/` for evaluation. The examples under `generation/Y/` differ from
evaluation references for some tasks. An offline I/O suite is used only when its
stored reference hash matches the selected reference.

Per-file JSON records contain metric values and states. Failed, unresolved,
unavailable, and missing-target results remain distinct; empty I/O categories
have null scores.

`evidence/reference_review/candidate_reviews.csv` contains the 61 RQ4 candidates,
both human reviewers' judgments, final labels, and the expert's decisions on
the four reviewer disagreements. The protocol is in
`evidence/reference_review/protocol.md` in the data archive.

Check all released files from the project root:

```bash
docker compose run --rm veruseval python scripts/validate.py --all-files
```

See [Part 1](reproduction.md) for result reproduction and
[Part 2](../baselines/README.md) for generating and evaluating programs.
