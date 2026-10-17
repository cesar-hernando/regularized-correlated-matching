"""
Figures of the low-weight fault analysis, from `fault_counts.csv` and `failure_classes.csv`
(written by `enumerate_faults.py`) and the grid-search logical error rates.

    failing_sets_vs_p.pdf           failing sets of MWPM, CM and RCM against p, one row per
                                    distance, weights (d - 1) / 2 and (d + 1) / 2;
    failing_probability_vs_p.pdf    the same for the failing probability;
    low_weight_vs_measured_ler.pdf  the measured logical error rates next to the part the two
                                    weights account for;
    failure_classes.pdf             the classes of the weight-(d - 1) / 2 sets CM and RCM fail
                                    on while MWPM corrects them, against p.

Example:
    python effective_distance_loss/plot_fault_counts.py
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "figures"))

from lcd_figures import COLUMN_WIDTH, PAGE_WIDTH, USE_LATEX, paper_style, save  # noqa: E402

DECODERS = {
    # name: colour; every decoder is drawn with filled dots
    "MWPM": "#555555",
    "CM": "#dc4405",
    "RCM": "#006f62",
}
DOTS = {"marker": "o", "markersize": 2.5}
CLASS_COLORS = ["#006f62", "#dc4405", "#3ccbda", "#cf6f7f", "#8a6d3b", "#6a51a3", "#999999"]
TOP_CLASSES = 6
SHORT_NAMES = {"two-qubit gate": "CZ", "idle during meas./reset": "meas. idle", "single-qubit gate/idle": "1q/idle",
               "measurement flip": "meas. flip", "reset flip": "reset flip"}
"""Shorter source names for the failure-class legends."""


def short_class(name):
    for long, short in SHORT_NAMES.items():
        name = name.replace(long, short)
    return name


def read(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def by_decoder(rows):
    """{name: rows} for MWPM, CM and RCM, each sorted by p."""
    return {name: sorted((r for r in rows if r["decoder"] == name), key=lambda r: float(r["p"]))
            for name in DECODERS}


def linear_axis(ax, values, headroom=1.1):
    """
    Linear y axis with room above the largest value (a factor `headroom`), starting a little
    below zero so points at zero stay visible.
    """
    top = max(values, default=0)
    top = top * headroom if top > 0 else 1
    ax.set_ylim(-0.05 * top, top)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)


def grid(distances, n_columns, height=1.9):
    figure, axes = plt.subplots(len(distances), n_columns, squeeze=False, sharex=True, layout="constrained",
                                figsize=(PAGE_WIDTH if n_columns > 1 else COLUMN_WIDTH, height * len(distances)))
    return figure, axes


def plot_vs_p(rows, quantity, label, path):
    distances = sorted({int(r["distance"]) for r in rows})
    figure, axes = grid(distances, 2)
    for row, d in enumerate(distances):
        for column, weight in enumerate(((d - 1) // 2, (d + 1) // 2)):
            ax = axes[row, column]
            subset = [r for r in rows if int(r["distance"]) == d and int(r["weight"]) == weight]
            values = []
            for name, points in by_decoder(subset).items():
                p = np.array([float(r["p"]) for r in points])
                y = np.array([float(r[quantity]) for r in points])
                values += list(y)
                if quantity == "failing_sets":
                    ax.plot(p, y, color=DECODERS[name], lw=0.8, label=name, **DOTS)
                else:
                    # a log axis cannot show zero: leave those points out so the line just ends
                    error = np.array([float(r[quantity + "_std"]) for r in points])
                    y = np.where(y > 0, y, np.nan)
                    ax.errorbar(p, y, error, color=DECODERS[name], lw=0.8, label=name,
                                elinewidth=0.6, capsize=1.5, **DOTS)
            ax.set_xscale("log")
            if quantity == "failing_sets":
                linear_axis(ax, values)
            else:
                ax.set_yscale("log")  # zero probabilities are left out
            ax.grid(visible=True, linewidth=0.4, alpha=0.5)
            ax.set_title(f"$d={d}$, weight ${weight}$ $= (d{'-' if column == 0 else '+'}1)/2$", pad=3)
            if column == 0:
                ax.set_ylabel(label)
            if row == len(distances) - 1:
                ax.set_xlabel("Physical error rate, $p$")
    axes[0, 0].legend(loc="best", handlelength=1.0, borderaxespad=0.4, labelspacing=0.25)
    save(figure, path)


def plot_vs_measured(rows, alpha_star_path, path):
    measured = read(alpha_star_path)
    columns = {"MWPM": "ler_mwpm", "CM": "ler_cm_hard", "RCM": "ler_rcm_alpha_star"}
    # only distances whose logical error rates were measured
    distances = sorted({int(r["distance"]) for r in rows} & {int(r["distance"]) for r in measured})
    figure, axes = plt.subplots(1, len(distances), squeeze=False, layout="constrained",
                                figsize=(PAGE_WIDTH, 2.4))
    for ax, d in zip(axes[0], distances):
        subset = [r for r in rows if int(r["distance"]) == d]
        low_weight = collections.defaultdict(lambda: collections.defaultdict(float))
        for name, points in by_decoder(subset).items():
            for r in points:
                low_weight[name][float(r["p"])] += float(r["failing_probability"])
        for name, color in DECODERS.items():
            ps = sorted(low_weight[name])
            predicted = [low_weight[name][p] if low_weight[name][p] > 0 else np.nan for p in ps]  # log axis
            ax.plot(ps, predicted, color=color, lw=0.9, label=f"{name}, weights $\\le (d+1)/2$")
            points = sorted((r for r in measured if int(r["distance"]) == d), key=lambda r: float(r["p"]))
            ax.errorbar([float(r["p"]) for r in points], [float(r[columns[name]]) for r in points],
                        [float(r[columns[name] + "_std"]) for r in points], color=color, linestyle="None",
                        elinewidth=0.6, label=f"{name}, measured", **DOTS)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(visible=True, linewidth=0.4, alpha=0.5)
        ax.set_title(f"$d={d}$", pad=3)
        ax.set_xlabel("Physical error rate, $p$")
    axes[0, 0].set_ylabel("Logical error probability")
    axes[0, -1].legend(loc="upper left", fontsize=5.5, handlelength=1.5, labelspacing=0.2)
    save(figure, path)


def plot_classes(classes, counts, path):
    rows = [r for r in classes if r["subset"] == "MWPM correct" and r["kind"] == "source, shape"]
    distances = sorted({int(r["distance"]) for r in rows})
    if not distances:
        return
    decoded = {(int(r["distance"]), r["decoder"]) for r in counts}
    figure, axes = grid(distances, 2, height=2.3)
    for row, d in enumerate(distances):
        subset = [r for r in rows if int(r["distance"]) == d]
        totals = collections.Counter()
        for r in subset:
            totals[r["fault_class"]] += float(r["failing_sets"])
        top = [name for name, _ in totals.most_common(TOP_CLASSES)]
        for column, decoder in enumerate(("CM", "RCM")):
            ax = axes[row, column]
            series = collections.defaultdict(lambda: collections.defaultdict(float))
            ps = sorted({float(r["p"]) for r in subset})
            for r in subset:
                if r["decoder"] == decoder:
                    name = r["fault_class"] if r["fault_class"] in top else "other"
                    series[name][float(r["p"])] += float(r["failing_sets"])
            for color, name in zip(CLASS_COLORS, top + ["other"]):
                if name in series:
                    ax.plot(ps, [series[name][p] for p in ps], color=color, marker="o", markersize=2.5,
                            lw=0.8, label=short_class(name))
            if not series:
                message = "no failures" if (d, decoder) in decoded else f"{decoder} not run at $d={d}$"
                ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")
            ax.set_xscale("log")
            linear_axis(ax, [v for s in series.values() for v in s.values()], headroom=1.8)  # room for the legend
            ax.grid(visible=True, linewidth=0.4, alpha=0.5)
            ax.set_title(f"{decoder}, $d={d}$, weight ${(d - 1) // 2}$, MWPM correct", pad=3)
            if column == 0:
                ax.set_ylabel("Failing sets")
            if row == len(distances) - 1:
                ax.set_xlabel("Physical error rate, $p$")
        if axes[row, 0].get_legend_handles_labels()[0]:
            axes[row, 0].legend(loc="upper left", fontsize=4.5, labelspacing=0.15, handlelength=1.2)
    save(figure, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=HERE, help="folder with the CSVs; figures go there too")
    parser.add_argument("--alpha-star", default=ROOT / "data/mwpm/rcm_grid_search_si1000/analysis/alpha_star_vs_p.csv")
    args = parser.parse_args()
    folder = Path(args.dir)
    rows = read(folder / "fault_counts.csv")

    with paper_style(use_latex=USE_LATEX):
        plot_vs_p(rows, "failing_sets", "Failing sets", folder / "failing_sets_vs_p.pdf")
        plot_vs_p(rows, "failing_probability", "Failing probability", folder / "failing_probability_vs_p.pdf")
        plot_vs_measured(rows, args.alpha_star, folder / "low_weight_vs_measured_ler.pdf")
        if (folder / "failure_classes.csv").exists():
            plot_classes(read(folder / "failure_classes.csv"), rows, folder / "failure_classes.pdf")


if __name__ == "__main__":
    main()
