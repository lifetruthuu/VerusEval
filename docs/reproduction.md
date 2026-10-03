# Reproducing the paper

Run `python scripts/reproduce.py --rq all --output-dir runs/full` after installing
the figure dependencies. `--statistics-only` generates statistics and LaTeX
tables without rendering figures. `--rscript` selects the R executable.
In Docker, prefix this command with `docker compose run --rm veruseval`; all
figure dependencies and Rscript are already installed.
`--rq` accepts exactly `all`, `1`, `2`, `3`, and `4`; each selection is runnable
from the [matching code and data archives](../downloads/README.md) extracted
into the same directory.

The entry validates every released file before starting, copies analysis code
to the output workspace and reads the released data. The workspace links its
`data/` directory to the extracted data, so retain that data while inspecting
the workspace. The entry starts with no generated RQ outputs. RQ2 and RQ3 first
build the RQ1 encodings they need. Statistical scripts use final merged records
directly and do not reapply historical corrections.

RQ1 produces acceptance/quality associations, task-bootstrap intervals and
accepted-artifact screening profiles. RQ2 compares the 18 generation configurations
and renders the current workflow and model/prompt figures. RQ3 computes conditional
quality among accepted artifacts and checks 395 bases and 943 retained variants,
including witnesses, proof-source identities, case outcomes and fixed denominators.

RQ4 includes all contract-difference, reference-screening, review and repair
results. It checks 3,765 flagged accepted artifacts, 61 screened references,
36 candidate reference defects, four witness cases and 11 repair comparisons.
Ten of the 11 contracts become equivalent to the corrected reference; the remaining
unconditional postcondition rejection has a separate recorded explanation.
Candidate annotations, supporting evidence and the review scope are recorded
in the released data.

CSV results live under `work/RQs/RQ*/results/`, LaTeX tables under `tables/`, and
figures under `figures/`. RQ4 reference-screen statistics are under
`RQ4/results/reference_screen/`. `reproduction.json` records each command, return
code and log. Numerical comparisons use the frozen expected tables in
`data/evidence/paper_results/`, with absolute tolerance `1e-12`. PDF byte identity
is not expected across font and renderer versions.

RQ4 also writes `reference_repair.csv`, `reference_repair_summary.json` and
`case_summary.json`, so the repair comparisons and four witnesses are available
beside the direction and screening results.

Use `python scripts/validate.py --all-files` for corpus integrity and
`python scripts/validate.py --results runs/full/work` to compare a completed run.
Missing files, altered hashes and unsupported RQ selections fail explicitly.
The published whitelist `release-files.json` governs packaging;
`python scripts/package.py --output-dir /new/archive/directory` rebuilds the two
archives without including unlisted local files.

Reproduction checks frozen outcomes and archived proof evidence. It does not
rerun every verifier call, regenerate I/O cases or contact the original models.
New evaluations and generation use the separate workflows in [usage](usage.md).

The [metric image gallery](../gallery/README.md) contains 36 complete PNG
overviews regenerated from the released records. Run
`python scripts/plot_gallery.py --output-dir runs/gallery` to rebuild it.
Its manifest records input and image hashes, sample counts and plotted summaries.
Gallery means use the documented chart policy and all 13,716 generation outputs;
the paper's quality analyses use 13,659 eligible outputs. Original generation
partitions named verified/unverified are distinct from final acceptance labels.
