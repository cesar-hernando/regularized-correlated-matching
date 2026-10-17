"""
The two fit figures of the SI1000 grid search, one panel per distance, from the analysis
tables of `scripts/analyze_rcm_grid_search.py`:

    best_alpha_vs_p_fit.pdf        the swept optimal damping alpha* against p, with the fit
                                   alpha_0 / (1 + (p_0/p)^k);
    ler_ratio_fits_bydistance.pdf  the logical error rate of RCM (at alpha*) and of CM over
                                   MWPM's, with the fits 1 + a + b p and
                                   1 + a + b p + c ln(1/p) / p.

Points come from `alpha_star_vs_p.csv` and the fitted coefficients from
`ratio_fits_coefficients.csv`, so the curves are the ones the analysis fitted, not a refit.

Example:
    uv run figures/si1000_fit_figures.py data/mwpm/rcm_grid_search_si1000/analysis figures/mwpm/si1000
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lcd_figures import (  # noqa: E402
    COLUMN_WIDTH, DECODER_TITLES, DISTANCE_COLORS, PAGE_WIDTH, PARITY_STYLE, POINT_STYLE,
    USE_LATEX, XLABEL, inside_key, paper_style, save, series_handle, two_panel_axes,
)
from RegularizedCM.models import cm_ratio, optimal_alpha, rcm_ratio  # noqa: E402

ALPHA_COLOR = "#006f62"
SERIES = {
    # key: (colour, marker, legend label)
    "rcm": ("#006f62", "o", r"$\alpha^*(p)$-RCM"),
    "cm": ("#dc4405", "s", "CM"),
}
PANEL = "abcdefgh"


def read(path):
    with open(path, newline="") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def by_distance(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["distance"])].append(row)
    return {d: sorted(points, key=lambda r: r["p"]) for d, points in sorted(grouped.items())}


def panels(distances, height=4.2):
    """A 2 x 2 grid (or as many rows as the distances need), one panel per distance."""
    columns = 2
    rows = -(-len(distances) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(PAGE_WIDTH, height), squeeze=False,
                                sharex=True, layout="constrained")
    return figure, [axes[i // columns][i % columns] for i in range(rows * columns)]


def smooth(points):
    """A dense p grid for drawing a fitted curve over the points it was fitted to."""
    p = [row["p"] for row in points]
    return fit_grid(min(p), max(p))


def fit_grid(low, high):
    """
    The p grid a fitted curve is drawn on: the range it was fitted over and no further. Below
    the lowest sampled p nothing is measured, and an extrapolated curve would read as a result
    (a large-distance alpha=1 fit, run down to 1e-5, crosses parity where no point was taken).
    """
    return np.geomspace(low, high, 300)


FONTSIZE = 7
SYMBOL_HEIGHT = 4.0     # the box every symbol is drawn in, in points
MARKER_WIDTH = 3.4
LINE_WIDTH = 14.0
LABEL_GAP = 4.5         # symbol to its own label
ENTRY_GAP = 7.0         # one entry to the next within a group
GROUP_GAP = 16.0        # one group to the next


def _text(label, bold=False):
    return TextArea(label, textprops={"fontsize": FONTSIZE,
                                      "fontweight": "bold" if bold else "normal"})


def _marker_box(color, filled):
    """A circle in a box just wide enough to hold it, so the gap either side is the packer's."""
    area = DrawingArea(MARKER_WIDTH, SYMBOL_HEIGHT, 0, 0)
    area.add_artist(plt.Line2D([MARKER_WIDTH / 2], [SYMBOL_HEIGHT / 2], marker="o", markersize=2.5,
                               color=color, markerfacecolor=color if filled else "white",
                               markeredgewidth=0.7, linestyle="None"))
    return area


def _line_box(linestyle):
    area = DrawingArea(LINE_WIDTH, SYMBOL_HEIGHT, 0, 0)
    line = plt.Line2D([0, LINE_WIDTH], [SYMBOL_HEIGHT / 2] * 2, color="black", lw=0.9,
                      linestyle=linestyle)
    if linestyle == "dashed":
        line.set_dashes((3, 1.5))
    area.add_artist(line)
    return area


def _entry(symbol, label):
    """A symbol and its label, centred on each other rather than sharing a text baseline."""
    return HPacker(children=[symbol, _text(label)], align="center", pad=0, sep=LABEL_GAP)


def _row(children, sep):
    return HPacker(children=children, align="center", pad=0, sep=sep)


def strength_legend(figure, distances, point_labels, fit_labels):
    """
    The key above the panels, in two rows: the points (circles, filled for alpha* and hollow
    for alpha = 1) with the fitted models (lines) on the first, the distances on the second.

    Built from offset boxes rather than a legend: each symbol sits in a box tight around its own
    ink, so the symbols are spaced by their own ink rather than by a fixed handle width, and
    everything is centred on one another vertically.
    """
    points = [_entry(_marker_box("black", filled), label) for label, filled in point_labels]
    fits = [_entry(_line_box(linestyle), label) for label, linestyle in fit_labels]
    colours = [_entry(_marker_box(DISTANCE_COLORS.get(d, "#006f62"), True), f"$d={d}$")
               for d in distances]
    rows = [_row([*points, *fits], GROUP_GAP), _row(colours, GROUP_GAP)]
    # each row is anchored on its own, so a shorter row sits centred instead of lining up with
    # the row above it
    boxes = []
    for row in rows:
        box = AnchoredOffsetbox(loc="upper center", child=row, pad=0.0, borderpad=0.0,
                                frameon=False, bbox_to_anchor=(0.5, 1.0),
                                bbox_transform=figure.transFigure)
        figure.add_artist(box)
        boxes.append(box)
    # the rows' heights are known only once they are laid out; stack them on what they measure
    figure.canvas.draw()
    heights = [box.get_window_extent().height / figure.bbox.height for box in boxes]
    gap, top = 0.028, 0.014
    for index, box in enumerate(boxes):
        box.set_bbox_to_anchor((0.5, 1.0 - top - sum(heights[:index]) - index * gap),
                               transform=figure.transFigure)
    # the key clear of the top edge, and a little air between it and the panel titles below
    figure.get_layout_engine().set(
        rect=(0, 0, 1, 1 - top - sum(heights) - len(boxes) * gap - 0.015))


def _alpha_panel(ax, rows, fit, markersize=3, color=ALPHA_COLOR):
    """The swept alpha* of one distance, its fit and the alpha_0 the fit settles on."""
    ax.plot([r["p"] for r in rows], [r["alpha_star"] for r in rows], linestyle="None",
            marker="o", markersize=markersize, color=color, label=r"swept $\alpha^*$")
    grid = smooth(rows)
    ax.plot(grid, optimal_alpha(grid, fit["alpha_star_fit_alpha0"], fit["alpha_star_fit_p0"],
                                fit["alpha_star_fit_k"]),
            color=color, lw=1.0, label=r"fit $\alpha_0/(1+(p_0/p)^k)$")
    ax.axhline(fit["alpha_star_fit_alpha0"], color="0.6", lw=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.grid(visible=True, linewidth=0.4, alpha=0.5)


def plot_alpha_star(points, fits, path):
    """The swept alpha* and its fit, one panel per distance."""
    figure, axes = panels(sorted(points))
    for index, (ax, d) in enumerate(zip(axes, sorted(points), strict=False)):
        _alpha_panel(ax, points[d], fits[d])
        ax.set_title(f"({PANEL[index]}) $d={d}$", pad=7)
    for ax in axes[len(points):]:
        ax.set_visible(False)
    for i, ax in enumerate(axes):
        if i % 2 == 0:
            ax.set_ylabel(r"Optimal reweighting strength, $\alpha^*$")
        if i >= len(axes) - 2:
            ax.set_xlabel(XLABEL)
    axes[0].legend(loc="lower right", handlelength=1.4, labelspacing=0.25, borderaxespad=0.4)
    save(figure, path)


def plot_alpha_star_single(points, fits, distance, path):
    """One distance on its own, in a single column."""
    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.6), layout="constrained")
    _alpha_panel(ax, points[distance], fits[distance], markersize=2.5,
                 color=DISTANCE_COLORS.get(distance, ALPHA_COLOR))
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(r"Optimal reweighting strength, $\alpha^*$")
    ax.legend(loc="upper left", handlelength=1.4, labelspacing=0.25, borderaxespad=0.4)
    save(figure, path)


def plot_ratio_fits(points, fits, path):
    """RCM and CM over MWPM with their fits, every distance in one panel."""
    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH * 1.6, 3.0), layout="constrained")
    for d in sorted(points):
        rows, fit = points[d], fits[d]
        color = DISTANCE_COLORS.get(d, "#006f62")
        p = np.array([r["p"] for r in rows])
        grid = smooth(rows)
        for key, (filled, linestyle) in (("rcm", (True, "solid")), ("cm", (False, "dashed"))):
            ratio = np.array([r[f"{key}_over_mwpm"] for r in rows])
            # the ratio's error, from the two logical error rates behind it
            top = "ler_rcm_alpha_star" if key == "rcm" else "ler_cm_hard"
            error = ratio * np.hypot(
                np.array([r[f"{top}_std"] / r[top] for r in rows]),
                np.array([r["ler_mwpm_std"] / r["ler_mwpm"] for r in rows]),
            )
            ax.errorbar(p, ratio, error, linestyle="None", marker="o", markersize=2.5, color=color,
                        markerfacecolor=color if filled else "white", markeredgewidth=0.7,
                        elinewidth=0.6, capsize=1.5, capthick=0.6)
            model = (rcm_ratio(grid, fit["rcm_mwpm_fit_1plusA"] - 1, fit["rcm_mwpm_fit_b"]) if key == "rcm"
                     else cm_ratio(grid, fit["cm_mwpm_fit_1plusA"] - 1, fit["cm_mwpm_fit_b"],
                                   fit["cm_mwpm_fit_c"]))
            ax.plot(grid, model, color=color, lw=0.9, linestyle=linestyle)
    ax.axhline(1.0, color="0.6", lw=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1.0, 2.0, 3.0, 5.0)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(visible=True, linewidth=0.4, alpha=0.5)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(r"$P_L \,/\, P_L^{\mathrm{MWPM}}$")

    strength_legend(
        figure, sorted(points),
        [(r"$\alpha^*(p)$-RCM", True), (r"CM ($\alpha=1$)", False)],
        [(r"$1+a+bp$", "solid"), (r"$1+a+bp+c\ln(1/p)/p$", "dashed")],
    )
    save(figure, path)


def read_curves(path):
    """Ratio curves grouped as {(decoder, distance, condition): rows sorted by p}."""
    grouped = defaultdict(list)
    with open(path, newline="") as handle:
        for raw in csv.DictReader(handle):
            grouped[(raw["decoder"], int(float(raw["distance"])), raw["condition"])].append({
                "p": float(raw["p"]),
                "ratio": float(raw["ratio"]),
                "sigma": (float(raw["ratio_high"]) - float(raw["ratio_low"])) / 2,
                "low": float(raw["ratio_low"]),
                "high": float(raw["ratio_high"]),
            })
    return {key: sorted(rows, key=lambda r: r["p"]) for key, rows in grouped.items()}


def shared_curves(curves):
    """
    The curves restricted to the (distance, p) points every decoder has, so both panels cover
    the same grid. Same rule as `lcd_figures.shared_points`, on the grouped curves.
    """
    seen = {
        decoder: {(d, round(row["p"], 12))
                  for (dec, d, _), rows in curves.items() if dec == decoder
                  for row in rows}
        for decoder, _, _ in curves
    }
    shared = set.intersection(*seen.values())
    kept = {key: [row for row in rows if (key[1], round(row["p"], 12)) in shared]
            for key, rows in curves.items()}
    return {key: rows for key, rows in kept.items() if rows}


def fitted(model, rows, p0):
    """Fit `model` to the ratios, as the analysis does; None when the fit fails."""
    from scipy.optimize import curve_fit

    p = np.array([r["p"] for r in rows])
    ratio = np.array([r["ratio"] for r in rows])
    sigma = np.array([r["sigma"] or np.inf for r in rows])
    try:
        parameters, _ = curve_fit(model, p, ratio, p0=p0, sigma=sigma, absolute_sigma=True,
                                  maxfev=40000)
    except (RuntimeError, ValueError, TypeError):
        return None
    return parameters


def cm_ratio_inverse(p, a, b, c):
    """The alpha = 1 ratio without the logarithm: 1 + a + b p + c / p (a p-independent t-fault coefficient)."""
    return 1 + a + b * p + c / p


ALPHA1_MODELS = {
    # name: (model, initial parameters, key label)
    "log": (cm_ratio, [-0.4, 30.0, 5e-6], r"$1+a+bp+c\ln(1/p)/p$"),
    "inverse": (cm_ratio_inverse, [-0.4, 30.0, 3e-5], r"$1+a+bp+c/p$"),
}


GLOBAL_MODELS = {
    # condition: (model of (p, d) with distance-independent parameters, initial parameters, key label)
    "alpha_1": (lambda x, A, r, B, C: r ** x[1] * (A + B * (x[1] - 3) * x[0] + C / x[0]),
                [2.0, 0.77, 160.0, 8e-5], r"$r^d[A+B(d-3)p+C/p]$"),
    "alpha_star": (lambda x, A, r, B1, B0: r ** x[1] * (A + (B1 * x[1] + B0) * x[0]),
                   [1.8, 0.78, 125.0, -350.0], r"$r^d[A+(B_1d+B_0)p]$"),
}


def global_fit(curves, decoder, condition):
    """The parameters of GLOBAL_MODELS[condition] fitted to every distance of `decoder` at once, or None."""
    from scipy.optimize import curve_fit

    model, start, _ = GLOBAL_MODELS[condition]
    rows = [(d, r) for (dec, d, cond), points in curves.items() if dec == decoder and cond == condition
            for r in points]
    if not rows:
        return None
    p = np.array([r["p"] for _, r in rows])
    d = np.array([float(d) for d, _ in rows])
    try:
        parameters, _ = curve_fit(model, (p, d), np.array([r["ratio"] for _, r in rows]), p0=start,
                                  sigma=np.array([r["sigma"] or np.inf for _, r in rows]),
                                  absolute_sigma=True, maxfev=200000)
    except (RuntimeError, ValueError, TypeError):
        return None
    return parameters


def plot_two_panel(curves, path, asymmetric=False, alpha1_model="log", global_models=False):
    """
    Both decoders side by side, every distance in one panel each: the two-pass logical error
    rate over the one-pass reference's, at alpha = 1 and at alpha*, with their fits. Each fit
    covers only its own points. The
    key names the two strengths in place and sits inside the first panel. `asymmetric` draws
    each interval as it is (ratio_low to ratio_high) instead of its mean half-width. `alpha1_model`
    names the fit of the alpha = 1 points (ALPHA1_MODELS). `global_models` instead fits each
    decoder and strength once over every distance (GLOBAL_MODELS), drawn over each distance's points.
    """
    model1, start1, label1 = ALPHA1_MODELS[alpha1_model]
    labels = ({c: GLOBAL_MODELS[c][2] for c in GLOBAL_MODELS} if global_models
              else {"alpha_star": r"$1+a+bp$", "alpha_1": label1})
    # matching first, then LCD, as in the other two-panel figures
    order = {"CM": 0, "LCD": 1}
    decoders = sorted({decoder for decoder, _, _ in curves}, key=lambda name: (order.get(name, 2), name))
    axes = two_panel_axes(len(decoders))
    for index, (ax, decoder) in enumerate(zip(axes, decoders, strict=True)):
        distances = sorted({d for dec, d, _ in curves if dec == decoder})
        shared = ({c: global_fit(curves, decoder, c) for c in GLOBAL_MODELS} if global_models else {})
        for d in distances:
            color = DISTANCE_COLORS.get(d, "#006f62")
            for condition, model, p0, filled, linestyle in (
                ("alpha_star", rcm_ratio, [-0.4, 30.0], True, "solid"),
                ("alpha_1", model1, start1, False, "dashed"),
            ):
                rows = curves.get((decoder, d, condition))
                if not rows:
                    continue
                p = np.array([r["p"] for r in rows])
                error = ([[r["ratio"] - r["low"] for r in rows], [r["high"] - r["ratio"] for r in rows]]
                         if asymmetric else [r["sigma"] for r in rows])
                ax.errorbar(p, [r["ratio"] for r in rows], error,
                            linestyle="None", marker="o", color=color,
                            markerfacecolor=color if filled else "white", **POINT_STYLE)
                grid = fit_grid(p.min(), p.max())
                if global_models:
                    if shared[condition] is not None:
                        curve = GLOBAL_MODELS[condition][0]((grid, float(d)), *shared[condition])
                        ax.plot(grid, curve, color=color, lw=0.9, linestyle=linestyle)
                    continue
                parameters = fitted(model, rows, p0)
                if parameters is not None:
                    ax.plot(grid, model(grid, *parameters), color=color, lw=0.9, linestyle=linestyle)
        ax.axhline(1.0, **PARITY_STYLE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1.0, 2.0, 3.0, 5.0)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.grid(visible=True, linewidth=0.4, alpha=0.5)
        ax.set_xlabel(XLABEL)
        ax.set_title(f"({PANEL[index]}) {DECODER_TITLES.get(decoder, decoder)}", pad=3)
    axes[0].set_ylabel(r"$P_L$ ratio ($d$ rounds)" "\n" "two-pass / one-pass")

    # the first panel carries the key, on white so the grid does not read through it
    inside_key(
        axes[0],
        [series_handle(r"Regularized $\alpha^*(p)$", filled=True),
         series_handle(r"Standard $\alpha=1$", filled=False),
         series_handle(labels["alpha_star"], linestyle="solid"),
         series_handle(labels["alpha_1"], linestyle="dashed")],
        sorted({d for _, d, _ in curves}),
    )
    save(axes[0].figure, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analysis", help="the analysis folder of a grid search")
    parser.add_argument("out", help="folder for the two figures")
    parser.add_argument("--lcd", help="Correlated LCD ratio curves; adds the two-decoder figure")
    parser.add_argument("--lcd-out", default="figures/cm_clcd_ratio_fits.pdf",
                        help="path of the two-decoder figure (default: %(default)s)")
    parser.add_argument("--cm-curves", default="ratio_vs_p_curves.csv",
                        help="CM ratio curves, in the analysis folder (default: %(default)s; "
                             "ratio_vs_p_curves_binomial.csv from binomial_ratio_intervals.py has the paper's intervals)")
    parser.add_argument("--asymmetric", action="store_true",
                        help="draw each interval as it is rather than its mean half-width")
    parser.add_argument("--suffix", default="", help="appended to the two-decoder figure's name, e.g. _v2")
    parser.add_argument("--global-fit", action="store_true",
                        help="one fit per decoder and strength over every distance (GLOBAL_MODELS)")
    parser.add_argument("--alpha1-model", choices=sorted(ALPHA1_MODELS), default="log",
                        help="fit of the alpha = 1 points: 1 + a + bp + c ln(1/p)/p (log) or + c/p (inverse)")
    parser.add_argument("--two-panel-only", action="store_true",
                        help="draw only the two-decoder figure (needs --lcd)")
    args = parser.parse_args()

    analysis, out = Path(args.analysis), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    points = by_distance(read(analysis / "alpha_star_vs_p.csv"))
    fits = {int(row["distance"]): row for row in read(analysis / "ratio_fits_coefficients.csv")}

    with paper_style(use_latex=USE_LATEX):
        if not args.two_panel_only:
            plot_alpha_star(points, fits, out / "best_alpha_vs_p_fit.pdf")
            if 5 in points:
                plot_alpha_star_single(points, fits, 5, out / "best_alpha_vs_p_fit_d5.pdf")
            plot_ratio_fits(points, fits, out / "ler_ratio_fits_bydistance.pdf")
        if args.lcd:
            curves = read_curves(analysis / args.cm_curves) | read_curves(args.lcd)
            lcd_out = Path(args.lcd_out)
            # only the points both decoders share, so the panels cover the same grid
            tail = "_common_named_inside" + args.suffix
            plot_two_panel(shared_curves(curves),
                           lcd_out.with_name(lcd_out.stem + tail + lcd_out.suffix), asymmetric=args.asymmetric,
                           alpha1_model=args.alpha1_model, global_models=args.global_fit)
    written = [] if args.two_panel_only else [str(out / "best_alpha_vs_p_fit.pdf"),
                                              str(out / "ler_ratio_fits_bydistance.pdf")]
    if args.lcd:
        written.append(f"the _common_named_inside{args.suffix} variant of {args.lcd_out}")
    print("wrote " + " and ".join(written))


if __name__ == "__main__":
    main()
