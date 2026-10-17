"""
The circuit-level depolarizing LER-ratio figure in the style of the main-text ratio figure
(`figures/cm_clcd_ratio_fits_common_named_inside_v4.pdf`): every distance in one panel, coloured by distance,
standard CM (alpha = 1) hollow and dashed, RCM (alpha*) filled and solid, each with its own fit
(1 + a + bp + c/p and 1 + a + bp), and the key inside the panel in three centred rows (the two updates, the
distances, the two fits).

Every ratio's interval is sinter's likelihood-ratio interval (maximum likelihood factor 10, 96.8%) on each
decoder's rate, combined in log space under an independence approximation, as in Fig. 1
(`figures/binomial_ratio_intervals.py`); the covariance from decoding the same shots is neglected. Points with a
result file in `data/mwpm/rcm_grid_search_depolarizing` (written by `scripts/run_depolarizing.py`) take their shots
and failures from that file. The other points use the rates in `data/mwpm/alpha_star_vs_p.csv` and their binomial
standard errors, from which the shots and failures are recovered as N = P(1 - P) / sigma^2 and E = P N.
`source_report` lists which points come from where.

Writes:
    figures/mwpm/circuit_level_depolarizing/ler_ratio_fits_v4_ratio.pdf

Example:
    python scripts/depolarizing_ler_ratio_v4.py
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import sinter
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))

from lcd_figures import (COLUMN_WIDTH, DISTANCE_COLORS, LAYOUT_PAD, PARITY_STYLE, POINT_STYLE, USE_LATEX,  # noqa: E402
                         paper_style, save, series_handle)
from si1000_fit_figures import by_distance, fit_grid, read  # noqa: E402
from binomial_ratio_intervals import ratio_interval  # noqa: E402

DATA = ROOT / "data/mwpm"
RESULTS = DATA / "rcm_grid_search_depolarizing"
EXCLUDED = {(7, 1e-4), (9, 4e-4)}  # the lowest p of d = 7 and 9: too few failures for a useful ratio (left out)
OUT = ROOT / "figures/mwpm/circuit_level_depolarizing"
Z = math.sqrt(2 * math.log(10))
XLABEL = "Physical error rate, $p$"
SIZE = (1.3 * COLUMN_WIDTH, 3.1)   # a little wider than a column, so the key fits inside the panel
RATIO_TOP = 2.6                    # the y axis opened up above the points (none reaches past ~1.4) for the key

CONDITIONS = (
    # rate column, filled, linestyle, model, initial parameters
    ("ler_rcm_alpha_star", True, "solid", lambda p, a, b: 1 + a + b * p, [-0.4, 30.0]),
    ("ler_cm_hard", False, "dashed", lambda p, a, b, c: 1 + a + b * p + c / p, [-0.4, 30.0, 3e-5]),
)

KEY_FONTSIZE = 6.5
SYMBOL = (13.0, 5.0)    # width and height, in points, of the box each key symbol is drawn in
LABEL_GAP, ENTRY_GAP, ROW_GAP = 3.0, 9.0, 2.5


def counts(rate, std):
    """(shots, failures) behind a binomial rate and its standard error."""
    shots = rate * (1 - rate) / std ** 2
    return shots, rate * shots


def interval(rate, std, reference, reference_std):
    """rho and its 96.8% interval, both rates' likelihood-ratio intervals combined in log space."""
    fits = []
    for r, s in ((rate, std), (reference, reference_std)):
        shots, errors = counts(r, s)
        f = sinter.fit_binomial(num_shots=round(shots), num_hits=round(errors), max_likelihood_factor=10)
        fits.append((f.low, f.best, f.high))
    (lo2, b2, hi2), (lo1, b1, hi1) = fits
    rho = rate / reference
    down = math.hypot(math.log(b2 / lo2), math.log(hi1 / b1))
    up = math.hypot(math.log(hi2 / b2), math.log(b1 / lo1))
    return rho, rho * math.exp(-down), rho * math.exp(up)


def counts_interval(d, p, alpha):
    """rho and its interval from the result file for (d, p), or None if there is none."""
    path = RESULTS / f"rotated_surface_code_circuit-level-depolarizing_z_d{d}_r{d}_p{p:.6e}.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    one = next(r for r in rows if r["decoder"] == "mwpm")
    two = next((r for r in rows if r["decoder"] == "cm" and abs(float(r["alpha"]) - alpha) < 1e-9), None)
    if two is None or int(two["errors"]) == 0 or int(one["errors"]) == 0:
        return None
    e2, e1 = int(two["errors"]), int(one["errors"])
    rho = e2 / e1
    low, high = ratio_interval(rho, int(one["shots"]), e2, e1)
    return rho, low, high


def ratio(d, r, column):
    """rho, low, high for one point and condition: from its result file if there is one, else from its rates."""
    alpha = 1.0 if column == "ler_cm_hard" else float(r["alpha_star"])
    return counts_interval(d, r["p"], alpha) or interval(r[column], r[f"{column}_std"], r["ler_mwpm"], r["ler_mwpm_std"])


def source_report(points):
    """Which (d, p) points take their counts from a result file and which from the rates in alpha_star_vs_p.csv."""
    from_files = [(d, r["p"]) for d, rows in sorted(points.items()) for r in rows
                  if counts_interval(d, r["p"], float(r["alpha_star"])) is not None]
    from_rates = [(d, r["p"]) for d, rows in sorted(points.items()) for r in rows if (d, r["p"]) not in from_files]
    return from_files, from_rates


def draw_ratio(ax, points):
    """The ratio panel: points, their fits and the parity line."""
    for d, rows in sorted(points.items()):
        color = DISTANCE_COLORS.get(d, "#006f62")
        p = np.array([r["p"] for r in rows])
        for column, filled, linestyle, model, start in CONDITIONS:
            rho, low, high = np.array([ratio(d, r, column) for r in rows]).T
            ax.errorbar(p, rho, [rho - low, high - rho], linestyle="None", marker="o", color=color,
                        markerfacecolor=color if filled else "white", **POINT_STYLE)
            try:
                parameters, _ = curve_fit(model, p, rho, p0=start, sigma=(high - low) / 2 / Z,
                                          absolute_sigma=True, maxfev=40000)
            except RuntimeError:
                continue
            grid = fit_grid(p.min(), p.max())
            ax.plot(grid, model(grid, *parameters), color=color, lw=0.9, linestyle=linestyle)
    ax.axhline(1.0, **PARITY_STYLE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(top=RATIO_TOP)
    ax.yaxis.set_major_locator(mticker.FixedLocator([0.2, 0.3, 0.5, 0.7, 1, 1.5]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(visible=True, linewidth=0.4, alpha=0.5)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(r"$P_L$ ratio ($d$ rounds)" "\n" "two-pass / one-pass")


def key_box(rows):
    """
    The key as independent rows (each centred, its entries a fixed gap apart), stacked: unlike a legend, whose
    columns are as wide as their widest entry, this spaces the entries of every row evenly. Every symbol is
    redrawn from its handle in a box of the same size.
    """
    from matplotlib.offsetbox import DrawingArea, HPacker, TextArea, VPacker

    def symbol(handle):
        width, height = SYMBOL
        area = DrawingArea(width, height, 0, 0)
        if handle.get_linestyle() not in ("None", "none", "", " "):
            line = plt.Line2D([0, width], [height / 2] * 2)
            line.update_from(handle)
            line.set_marker("None")
            area.add_artist(line)
        if handle.get_marker() not in ("None", "none", "", " ", None):
            dot = plt.Line2D([width / 2], [height / 2])
            dot.update_from(handle)
            dot.set_linestyle("None")
            area.add_artist(dot)
        return area

    def entry(handle):
        label = TextArea(handle.get_label(), textprops={"fontsize": KEY_FONTSIZE})
        return HPacker(children=[symbol(handle), label], align="center", pad=0, sep=LABEL_GAP)

    lines = [HPacker(children=[entry(h) for h in row], align="center", pad=0, sep=ENTRY_GAP) for row in rows]
    return VPacker(children=lines, align="center", pad=0, sep=ROW_GAP)


def key_rows_inside(ax, rows, loc):
    """A boxed key inside `ax`, laid out in `rows`."""
    from matplotlib.offsetbox import AnchoredOffsetbox

    box = AnchoredOffsetbox(loc=loc, child=key_box(rows), pad=0.35, borderpad=0.4, frameon=True)
    box.patch.set(facecolor="white", edgecolor="0.8", linewidth=0.5, alpha=0.95)
    box.set_zorder(10)
    ax.add_artist(box)


def plot(points, path):
    figure, ax = plt.subplots(figsize=SIZE, layout="constrained")
    figure.get_layout_engine().set(w_pad=LAYOUT_PAD, h_pad=LAYOUT_PAD)
    draw_ratio(ax, points)
    updates = [series_handle(r"Standard $\alpha=1$", filled=False),
               series_handle(r"Regularized $\alpha^*(p)$", filled=True)]
    distances = [series_handle(f"$d={d}$", filled=True, color=DISTANCE_COLORS.get(d, "#006f62")) for d in sorted(points)]
    fits = [series_handle(r"$1+a+bp+c/p$", linestyle="dashed"), series_handle(r"$1+a+bp$", linestyle="solid")]
    key_rows_inside(ax, [updates, distances, fits], "upper center")
    save(figure, path)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--out-dir", default=OUT)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    points = by_distance([r for r in read(Path(args.data) / "alpha_star_vs_p.csv")
                          if not any(r["distance"] == d and math.isclose(r["p"], q) for d, q in EXCLUDED)])
    path = out / "ler_ratio_fits_v4_ratio.pdf"
    with paper_style(use_latex=USE_LATEX):
        plot(points, path)
    from_files, from_rates = source_report(points)
    print(f"wrote {path}: {len(from_files)} points from result files, {len(from_rates)} from the rates in "
          "alpha_star_vs_p.csv:", ", ".join(f"d={d} p={p:g}" for d, p in from_rates))


if __name__ == "__main__":
    main()
