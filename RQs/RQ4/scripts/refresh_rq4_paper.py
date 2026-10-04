"""Generate the RQ4 direction figure and the reference-screening figure from checked summaries.

The REFERENCE_SCREEN defect examples are built by RQs/RQ4/scripts/build_reference_defects_figure.py.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory, offset_copy

from build_rq4 import DIRECTIONS, ROOT, RQ4, REFERENCE_SCREEN, read_csv, source, write_json

PAPER = ROOT / "RQs/RQ4"


def plot():
    rows = read_csv(RQ4 / "combinations.csv")
    directions = read_csv(RQ4 / "directions.csv")
    total = sum(int(r["artifacts"]) for r in rows)
    # The tight bounding box of this canvas is about the text width, so the paper prints it near full size.
    fig, (dots, bars) = plt.subplots(1, 2, figsize=(7.15, 1.7), sharey=True,
                                    width_ratios=(1.8, 2.15), gridspec_kw={"wspace": .035})
    xs = [0, 1.75, 3.85, 5.6]
    colors = ["#2a78d6", "#eb6834", "#eb6834", "#2a78d6"]
    ys = list(range(len(rows) - 1, -1, -1))
    for ax in (dots, bars):
        for y in ys[1::2]:
            ax.axhspan(y - .5, y + .5, color="#F4F4F1", linewidth=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    for y, row in zip(ys, rows):
        present = [x for x, d in zip(xs, directions) if row[d["key"]] == "True"]
        if len(present) > 1:
            dots.plot([min(present), max(present)], [y, y], color="#5f5f5c", linewidth=.8)
        for x, d, color in zip(xs, directions, colors):
            dots.scatter(x, y, s=13, color=color if row[d["key"]] == "True" else "#DEDED9", linewidths=0, zorder=3)
    dots.set_xlim(-.9, 6.5)
    dots.set_ylim(-.5, len(rows) - .5)
    labels = [f"{d['obligation']}: {'Pre' if d['key'].startswith('pre') else 'Post'}\n{d['diagnostic'].split()[1]}\n{float(d['share'])*100:.1f}%" for d in directions]
    dots.set_xticks(xs, labels, fontsize=6.5, linespacing=1.05)
    dots.xaxis.tick_top()
    dots.set_yticks([])
    header = offset_copy(blended_transform_factory(dots.transData, dots.transAxes), fig=fig, y=25, units="points")
    for label, left, right in (("Soundness", xs[0], xs[1]), ("Completeness", xs[2], xs[3])):
        dots.text((left + right) / 2, 1, label, transform=header, ha="center", va="bottom", fontsize=7.2, weight="bold")
        dots.plot([left - .7, right + .7], [1, 1], transform=header, color="#5f5f5c", linewidth=.6, clip_on=False)
    counts = [int(r["artifacts"]) for r in rows]
    bars.barh(ys, counts, height=.68, color="#5f5f5c")
    for y, count in zip(ys, counts):
        bars.text(count + 10, y, f"{count:,} ({100*count/total:.1f}%)", va="center", fontsize=6.3)
    bars.set_xlim(0, max(counts) * 1.36)
    bars.set_xticks([])
    bars.set_title(f"Flagged artifacts per combination (n = {total:,})", loc="left", fontsize=7.2, weight="bold")
    fig.savefig(PAPER / "figures/rq4_contract_relations.pdf", bbox_inches="tight", pad_inches=.02)
    destination = ROOT / "RQs/RQ4/figures"
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "rq4_contract_relations.png", dpi=200, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


# Candidate-defect kinds in the order used by the text and both reference-review figures.
KINDS = (("reference_under_specified", "Missing guarantee"), ("reference_vacuous", "Vacuous contract"),
         ("reference_encoding_bug", "Incorrect constraint"), ("encoding_limitation", "Inexpressible property"))
CANDIDATE, NO_CANDIDATE, INK, MUTED = "#2a78d6", "#8a8984", "#222222", "#6b6b6b"


def kind_counts():
    tasks = read_csv(REFERENCE_SCREEN / "screen_tasks.csv")
    union = next(r for r in read_csv(REFERENCE_SCREEN / "screen_signals.csv") if r["signal"] == "selected")
    candidates = [t for t in tasks if t["selected"] == "True" and t["final_reference_defect"] == "yes"]
    counts = Counter(t["category"] for t in candidates)
    assert set(counts) == {k for k, _ in KINDS} and len(candidates) == int(union["candidate"])
    return [(label, counts[key]) for key, label in KINDS]


def screen_figure():
    """Selected reference contracts by signal and label status, and candidate defects by kind."""
    rows = read_csv(REFERENCE_SCREEN / "screen_signals.csv")
    kinds = kind_counts()
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.3, 1.35), width_ratios=(1.55, 1), gridspec_kw={"wspace": .62})
    labels = [("Any signal" if r["signal"] == "selected" else r["signal"]) for r in rows]
    ys = list(range(len(rows)))[::-1]
    ys = [y - (.35 if r["signal"] == "selected" else 0) for y, r in zip(ys, rows)]
    segments = (("candidate", "Candidate defect", dict(color=CANDIDATE)),
                ("no_candidate", "No candidate", dict(color=NO_CANDIDATE)),
                ("not_reviewed", "Unlabeled", dict(facecolor="white", edgecolor=NO_CANDIDATE, hatch="////", linewidth=.6)))
    for y, r in zip(ys, rows):
        start = 0
        for key, name, style in segments:
            width = int(r[key])
            if width:
                # A white edge leaves a thin surface gap between touching segments.
                left.barh(y, width - .25, left=start, height=.62, label=name if y == ys[-1] or r["signal"] == "selected" else None, **style)
            start += width
        left.text(start + .8, y, f"{r['candidate']}/{r['tasks']}", va="center", fontsize=6.8, color=INK)
    handles, names = left.get_legend_handles_labels()
    unique = dict(zip(names, handles))
    left.legend(unique.values(), unique.keys(), loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=6.5, handlelength=1.2, handleheight=.8)
    left.axhline((ys[-2] + ys[-1]) / 2, color="#c8c8c3", linewidth=.6)
    left.set_yticks(ys, labels, fontsize=6.8)
    left.set_xlim(0, max(int(r["tasks"]) for r in rows) * 1.12)
    left.set_title("(a) Selected reference contracts by signal", loc="left", fontsize=7.2, weight="bold")
    ky = list(range(len(kinds)))[::-1]
    right.barh(ky, [n for _, n in kinds], height=.62, color=CANDIDATE)
    for y, (_, n) in zip(ky, kinds):
        right.text(n + .4, y, str(n), va="center", fontsize=6.8, color=INK)
    right.set_yticks(ky, [label for label, _ in kinds], fontsize=6.8)
    right.set_xlim(0, max(n for _, n in kinds) * 1.18)
    right.set_title(f"(b) Candidate defects by kind (n = {sum(n for _, n in kinds)})", loc="left", fontsize=7.2, weight="bold")
    for ax in (left, right):
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#c8c8c3")
        ax.tick_params(axis="y", length=0, colors=INK)
        ax.tick_params(axis="x", labelsize=6.3, colors=MUTED, color="#c8c8c3")
    fig.savefig(PAPER / "figures/rq4_reference_screen.pdf", bbox_inches="tight", pad_inches=.02)
    destination = ROOT / "RQs/RQ4/figures"
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "rq4_reference_screen.png", dpi=200, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def main(statistics_only=False):
    plt.rcParams.update({"font.family": "Liberation Sans", "pdf.fonttype": 42})
    if not statistics_only:
        plot()
        screen_figure()
    inputs = [RQ4 / name for name in ("combinations.csv", "directions.csv", "clause_summary.csv")]
    inputs += [REFERENCE_SCREEN / "screen_signals.csv", REFERENCE_SCREEN / "screen_tasks.csv"]
    outputs = [PAPER / name for name in ("figures/rq4_contract_relations.pdf", "figures/rq4_reference_screen.pdf")]
    write_json(RQ4 / "paper_manifest.json", {"script": source(Path(__file__)), "inputs": [source(p) for p in inputs], "outputs": [] if statistics_only else [source(p) for p in outputs]})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statistics-only', action='store_true')
    main(parser.parse_args().statistics_only)
