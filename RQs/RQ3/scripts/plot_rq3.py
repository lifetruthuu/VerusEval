"""Plot eight-operator behavioral coverage without treating unknowns as misses."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rq3_io_common import ROOT, RESULTS, read_csv
from rq3_mutations import OPERATORS


def plot(results, figures):
    cells = read_csv(Path(results) / "variants/detection_matrix.csv")
    operators = list(OPERATORS)
    categories = ("positive", "negative", "invalid", "union")
    lookup = {(r["operator"], r["category"]): r for r in cells}
    values = np.array([[float(lookup[op, cat]["behavior_detection_rate"] or "nan") for cat in categories] for op in operators])
    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    matrix = ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), ("Correct\npairs", "Wrong\noutputs", "Invalid\ninputs", "Any kind"))
    ax.set_yticks(range(8), [op.replace("_", " ").title() for op in operators])
    for i, op in enumerate(operators):
        for j, cat in enumerate(categories):
            r = lookup[op, cat]
            text = f'{r["behavior_detected"]}/{r["n"]}' if int(r["n"]) else "N/A"
            ax.text(j, i, text, ha="center", va="center", color="white" if values[i, j] > .65 else "black", fontsize=10)
    fig.colorbar(matrix, ax=ax, label="Behavioral counterexample detection rate", shrink=.8)
    ax.set_title("Fixed I/O suite: confirmed nondegenerate contract changes", fontsize=11)
    fig.tight_layout()
    figures = Path(figures)
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(figures / ("rq3_detection_matrix." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    left = np.zeros(8)
    styles = (("behavior_detected", "Behavioral counterexample", "#326A9C"),
              ("diagnostic_only", "Diagnostic only", "#D39645"),
              ("unresolved", "Unresolved", "#B6BEC8"), ("missed", "All cases pass", "#E5EBF1"))
    for key, label, color in styles:
        widths = np.array([int(lookup[op, "union"][key]) / int(lookup[op, "union"]["n"]) if int(lookup[op, "union"]["n"]) else 0 for op in operators])
        ax.barh(range(8), widths, left=left, color=color, label=label)
        left += widths
    ax.set_yticks(range(8), [f'{op.replace("_", " ").title()} (n={lookup[op, "union"]["n"]})' for op in operators])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of retained variants")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=2, frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figures / ("rq3_detection_states." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--figures", type=Path, default=ROOT / "RQs/RQ3/figures")
    args = parser.parse_args()
    plot(args.results, args.figures)
