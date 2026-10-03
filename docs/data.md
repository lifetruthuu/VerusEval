# Data and provenance

The data archive contains six directories. `evaluation/` holds final per-file
records and indexes; `generated/` holds the 13,716 evaluated Rust sources;
`references/` holds recovered exact evaluation references; `io/` holds validated
I/O suites; `generation/` holds baseline inputs, examples and retrieval mappings;
`evidence/` holds proofs, annotation records, controlled variants, repair cases
and frozen expected results.

`evaluation/artifact_index.csv` maps each `sample_id` to its relative
`result_path`, released `result_sha256`, historical `source_path` and original
`source_sha256`, with `analysis_eligible`. Historical input locators identify
pre-merge records that are not execution dependencies. `artifact_outcomes.csv`
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

The final records derive from the internal merged revision called v5, combining
the four original evaluation trees with registered I/O, target-binding,
triviality and formal-comparison corrections. The predecessor v4 supplied the
original records and repair evidence. These version names describe provenance;
public commands read `data/evaluation/`. The JSON `migration_v5` field is retained
as a schema field for compatibility. Original timestamps and unresolved outcomes
are preserved; no historical overlay should be applied again.

`evidence/contract_variants/` is the separate RQ3 experiment with 395 bases,
943 retained variants and its checked candidate/evidence records. It is not part
of the 13,716-program population. `reference_review/` contains prior labels,
annotations and the subsequent review. `reference_repair/original/` contains the
min_array correction and its original verification; `reference_repair/comparisons/`
contains the current 11-artifact comparisons. Evidence records identify source
files, hashes, proof harnesses and the pinned Verus release.

`artifact_manifest.json` records released file hashes and original pre-export
hashes. Path normalization and linked-hash/cache-key updates do not change measured
values or logical judgments. Historical code snapshots are evidence under
`evidence/source-code/`; active reproduction uses the public analysis modules.
The hash update follows dependencies, including structured RQ3 cache
keys. Package checks validate each generated source and per-file record plus all
files when `--all-files` is selected.

Exact references were recovered only when matching the target catalog's hash.
`references/availability.json` records 761 available references and the unavailable
original `VeriCoding_VT0545_vericoded`. Its frozen measurements support statistical
reproduction, but reevaluation against that exact original requires obtaining it.
Generation examples differ from evaluation references for
`HumanEval-Verus_task_36`, `MBPP-verified_task_47`, `VeriCoding_VT0545_vericoded`,
`VerusBench_MBPP_task_id_476` and `VerusBench_MBPP_task_id_588`; examples are not
silently substituted. The I/O supplement retains two filtered suites and six
quarantined historical positive cases. The usable suite totals 11,641 cases;
86 tasks have at least one category below the generation target. Recorded
unresolved cases and missing categories remain represented in the released data.
