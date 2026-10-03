# Data

The data archive contains six directories. `evaluation/` holds final per-file
records and indexes; `generated/` holds the 13,716 evaluated Rust sources;
`references/` holds 762 reference files; `io/` holds validated
I/O suites; `generation/` holds baseline inputs, examples and retrieval mappings;
`evidence/` holds proofs, annotation records, controlled variants, repair cases
and frozen expected results.

`evaluation/artifact_index.csv` maps each `sample_id` to its relative
`result_path` and `result_sha256`, with `analysis_eligible`. `artifact_outcomes.csv`
and `artifact_labels.csv` cover the full population. The `rq1_artifact_*` views
cover 13,659 eligible artifacts; `evaluation/provenance/excluded_missing_target.csv`
records the 57 exclusions. Sample IDs encode workflow, model, prompting mode
and task. `evaluation/target_functions.csv` defines 762 tasks and target functions.

`generated/manifest.csv` maps each sample to exactly one source, source hash,
evaluation record, record hash and eligibility state. The corpus comprises
1,524 AlphaVerus, 1,524 AutoVerus, 7,620 StarVerus and 3,048 VeruSAGE artifacts,
covering 18 configurations. The alias inventory records removed exact duplicate
source copies. These aliases do not add samples.

Per-file JSON is authoritative for metric values. `target_evaluation` describes
the target function, eligibility, verifier stage, four comparison directions
and triviality labels. Whole-target comparisons and function/clause aggregates
have different scopes. The four directions are `pre_ref_to_gen` (L1),
`post_gen_to_ref` (L2), `pre_gen_to_ref` (L3), and `post_ref_to_gen` (L4).
Postcondition implication is checked without adding preconditions.

Passed, failed, unresolved, unavailable and missing-target states remain distinct.
A rejected proof attempt is not a verified counterexample. I/O scores include
all available validated cases in the denominator; unresolved cases do not pass.
Empty categories have null scores. Strict wrong-output checks require both input
admission and rejection of that output. Triviality scoring retains the experiment's
definition, including non-detection when a tautology proof is rejected.

Public commands read the final records in `data/evaluation/`.
RQ3 evidence contains 395 bases and 943 controlled variants, separate from the
13,716-program population. RQ4 evidence covers 61 screened references,
36 candidate defects, four witness cases and 11 repair comparisons. Supporting
files include annotations, proof harnesses and verification results.

Run `python scripts/validate.py --all-files` to check the released files against
`artifact_manifest.json`, including program-to-record pairs and reference hashes.

There are 762 reference files, one per task. The target catalog provides their
SHA-256 hashes; `references/availability.json` records availability.
Generation examples differ from evaluation references for
`HumanEval-Verus_task_36`, `MBPP-verified_task_47`,
`VerusBench_MBPP_task_id_476` and `VerusBench_MBPP_task_id_588`; examples are not
silently substituted. The I/O supplement retains two filtered suites and six
quarantined positive cases. The usable suite totals 11,641 cases;
86 tasks have at least one category below the generation target. Recorded
unresolved cases and missing categories remain represented in the released data.
New evaluations use an offline I/O suite only when its stored source hash matches
the selected reference; a mismatch makes that suite unavailable.
