"""
One 2x3 figure, a column per code distance (`d = 3, 5, 7`):

    row 1  posterior edge weights against `p` (from `posterior_weights.csv`);
    row 2  both enumerated weights together (from `fault_counts.csv`), filled squares for
           `(d - 1) / 2` and filled triangles for `(d + 1) / 2`.

The two sources are the ones behind `posterior_weights_vs_p.pdf` and
`failing_probability_vs_p.pdf`; this draws them together so a column reads as one distance.
Example:
    python effective_distance_loss/combined_figure.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "figures"))

from lcd_figures import PAGE_WIDTH, USE_LATEX, paper_style, save  # noqa: E402
from plot_fault_counts import DECODERS, DOTS, by_decoder, read  # noqa: E402
from posterior_weights import SERIES  # noqa: E402

DISTANCES = [3, 5, 7]
YLABEL = r"$p_L$ contribution"
# the key names the series without their reweighting strength: the caption carries
# `alpha = 1` for CM and `alpha*` for RCM, and the short names keep the box small
POSTERIOR_LABELS = {"prior": "Prior", "cm": "CM posterior", "rcm": "RCM posterior"}

RCM_COLOR = "#0b6fa4"
"""RCM in blue: against the orange of CM, the pair that survives being printed in grey or read by a
colourblind eye."""

COLORS = {**DECODERS, "RCM": RCM_COLOR}
"""Decoder colours for this figure: the module's, with RCM's swapped."""


def _series_color(prefix, color):
    """The prior and CM keep their colours; RCM takes the blue."""
    return RCM_COLOR if prefix == "rcm" else color

# (marker, face, linestyle) for the lower and the upper weight, where one panel holds both
WEIGHT_STYLES = [("s", True, "-"), ("^", True, "-")]


def _decades_only(ax):
    """Label the decades and nothing else: a panel spanning ~1 decade otherwise collides."""
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def _posterior_row(axes_row, weights):
    """The posterior edge weights, one panel per distance."""
    for ax, d in zip(axes_row, DISTANCES, strict=True):
        points = sorted((r for r in weights if int(r["distance"]) == d),
                        key=lambda r: float(r["p"]))
        p = [float(r["p"]) for r in points]
        drawn = []
        for prefix, (label, color) in SERIES.items():
            # the mean alone: the median tracks it closely and doubles the key
            y = [float(r[f"{prefix}_mean"]) for r in points]
            drawn += y
            ax.plot(p, y, color=_series_color(prefix, color), lw=0.8,
                    label=POSTERIOR_LABELS.get(prefix, label), **DOTS)
        # these weights never approach zero, so starting there would waste half the panel:
        # a margin of the drawn range instead, which keeps CM's flat line off the frame
        low, high = min(drawn), max(drawn)
        margin = 0.08 * (high - low)
        ax.set_ylim(low - margin, high + margin)
        ax.set_title(f"$d={d}$", pad=4)
    axes_row[0].set_ylabel("Mean edge weight")


def _draw_series(ax, subset, marker="o", filled=True, linestyle="-", label=True):
    """One weight's three decoders on `ax`."""
    for name, points in by_decoder(subset).items():
        p = np.array([float(r["p"]) for r in points])
        y = np.array([float(r["failing_probability"]) for r in points])
        style = {"color": COLORS[name], "lw": 0.8, "marker": marker,
                 "markersize": DOTS["markersize"], "linestyle": linestyle,
                 "markerfacecolor": COLORS[name] if filled else "white",
                 "markeredgewidth": 0.7}
        if label:
            style["label"] = name
        # a log axis cannot show zero: leave those points out so the line just ends
        error = np.array([float(r["failing_probability_std"]) for r in points])
        ax.errorbar(p, np.where(y > 0, y, np.nan), error, elinewidth=0.6, capsize=1.5,
                    **style)


def _finish(axes):
    """Log x, decade labels, grid, and the x label under the bottom row."""
    for ax in axes.flat:
        ax.set_xscale("log")
        _decades_only(ax)
        ax.grid(visible=True, linewidth=0.4, alpha=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("Physical error rate, $p$ (CZ gates)")


def _boxed(ax, handles=None, **kwargs):
    """A key with the marker close against its label, on white to lift it off the grid."""
    options = {"loc": "best", "fontsize": 6, "labelspacing": 0.25, "borderaxespad": 0.4,
               "handlelength": 0.8, "handletextpad": 0.3, "borderpad": 0.35,
               "frameon": True, "framealpha": 1.0, "facecolor": "white",
               "edgecolor": "0.8", **kwargs}
    # the handles carry their own labels; without them the axes supply both
    key = ax.legend(handles=handles, **options) if handles else ax.legend(**options)
    key.get_frame().set_linewidth(0.5)
    return key


def _dot(color, marker="o", label=""):
    """A key entry that is only its marker: the lines and the error bars go without saying."""
    return plt.Line2D([], [], color=color, marker=marker, linestyle="none",
                      markersize=DOTS["markersize"], markerfacecolor=color,
                      markeredgewidth=0.7, label=label)


def _series_key(ax):
    """The edge-weight key, inside the d = 3 panel of the top row."""
    _boxed(ax, [_dot(_series_color(prefix, color), label=POSTERIOR_LABELS.get(prefix, label))
                for prefix, (label, color) in SERIES.items()])


def _contribution_row(axes_row, counts):
    """Both enumerated weights together, one panel per distance, with their keys in the d = 3 panel."""
    for ax, d in zip(axes_row, DISTANCES, strict=True):
        for weight, (marker, filled, linestyle) in zip(
                ((d - 1) // 2, (d + 1) // 2), WEIGHT_STYLES, strict=True):
            subset = [r for r in counts if int(r["distance"]) == d
                      and int(r["weight"]) == weight]
            _draw_series(ax, subset, marker=marker, filled=filled,
                         linestyle=linestyle, label=False)
        ax.set_yscale("log")
    axes_row[0].set_ylabel(YLABEL)

    # every combination named outright: colour alone or marker alone invites a misread.
    # One box per weight, in the corner its series runs through: the triangles of
    # (d + 1) / 2 across the top, the squares of (d - 1) / 2 along the bottom.
    for index, (marker, _, _) in reversed(list(enumerate(WEIGHT_STYLES))):
        entries = [_dot(COLORS[name], marker, f"{name}-weight $(d{'-+'[index]}1)/2$")
                   for name in COLORS]
        corner = "upper left" if index else "lower right"
        key = _boxed(axes_row[0], entries, loc=corner)
        # a second legend replaces the first unless the first is kept as an artist
        axes_row[0].add_artist(key)


def plot_merged(counts, weights, path):
    """2x3: the posterior weights above, both enumerated weights together below."""
    figure, axes = plt.subplots(2, len(DISTANCES), squeeze=False, layout="constrained",
                                sharex="col", figsize=(PAGE_WIDTH, 4.4))
    _posterior_row(axes[0], weights)
    _contribution_row(axes[1], counts)
    _finish(axes)
    _series_key(axes[0, 0])
    save(figure, path)


def plot_split(counts, weights, path_weights, path_contributions):
    """The two rows of `plot_merged` as figures of their own, each 1x3 with the distances titled."""
    figure, axes = plt.subplots(1, len(DISTANCES), squeeze=False, layout="constrained",
                                figsize=(PAGE_WIDTH, 2.3))
    _posterior_row(axes[0], weights)
    _finish(axes)
    _series_key(axes[0, 0])
    save(figure, path_weights)

    figure, axes = plt.subplots(1, len(DISTANCES), squeeze=False, layout="constrained",
                                figsize=(PAGE_WIDTH, 2.4))
    _contribution_row(axes[0], counts)
    for ax, d in zip(axes[0], DISTANCES, strict=True):
        ax.set_title(f"$d={d}$", pad=4)
    _finish(axes)
    save(figure, path_contributions)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=HERE, help="folder with the CSVs; the figure goes there too")
    parser.add_argument("--out", help="output name (default: named after the figure)")
    parser.add_argument("--counts", help="fault-count CSV (default: fault_counts.csv in --dir)")
    parser.add_argument("--split", action="store_true",
                        help="the two rows as separate figures, <out>_weights.pdf and <out>_contributions.pdf")
    args = parser.parse_args()

    folder = Path(args.dir)
    counts = read(Path(args.counts) if args.counts else folder / "fault_counts.csv")
    weights = read(folder / "posterior_weights.csv")
    name = args.out or "posterior_and_failing_probability_merged.pdf"
    with paper_style(use_latex=USE_LATEX):
        if args.split:
            stem = Path(name).stem
            top, bottom = folder / f"{stem}_weights.pdf", folder / f"{stem}_contributions.pdf"
            plot_split(counts, weights, top, bottom)
            print(f"wrote {top} and {bottom}")
        else:
            plot_merged(counts, weights, folder / name)
            print(f"wrote {folder / name}")


if __name__ == "__main__":
    main()
