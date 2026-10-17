"""
The paper's figures, drawn from CSV. Standalone: matplotlib, numpy, stdlib.

Example run:
    uv run figures/lcd_figures.py ratio  figures/lcd/lcd_ratio.pdf data/lcd/ratio_vs_p_curves.csv
    uv run figures/lcd_figures.py landscape figures/lcd/lcd_landscape.pdf data/lcd/alpha_landscape_overlay.csv data/lcd/alpha_landscape_surface.csv
    uv run figures/lcd_figures.py ler figures/rcm_rclcd_ler.pdf data/lcd/ratio_vs_p_curves.csv data/mwpm/rcm_grid_search_si1000/analysis/ratio_vs_p_curves.csv
    uv run figures/lcd_figures.py scaling figures/cm_lcd_scaling.pdf data/mwpm/rcm_grid_search_si1000/analysis/lep_scaling.csv data/lcd/lep_scaling.csv

All figures take long-format CSVs keyed on a `decoder` column.

The LER figure:
---------------

Absolute logical error probability `P_L` against p at alpha*, i.e. of the regularized
two-pass decoders, one line per (decoder, distance), from the same CSVs as the ratio figure
(columns `P_L` and `shots` of the `alpha_star` rows).

The ratio figure:
-----------------

Takes in a CSV comparing two-pass/one-pass P_L against p, at alpha=1
and at alpha*. One row per (decoder, distance, p, condition):

    decoder          panel identity, e.g. "LCD" or "MWPM"
    distance         code distance
    rounds           rounds per shot, so P_L is unambiguous
    basis            memory basis
    noise_model      e.g. "si1000"
    p                physical error rate
    condition        "alpha_1" or "alpha_star"
    alpha            the regularisation actually sampled
    ratio            P_L(two-pass at alpha) / P_L(one-pass reference)
    ratio_low        lower end of the interval on `ratio`
    ratio_high       upper end
    mcnemar_p        p-value against parity, where the two decoders shared shots
    significant      1/0, or blank if the source cannot say
    P_L              absolute logical error probability at that alpha
    P_L_reference    the one-pass decoder's, over the same shots
    shots            shots behind that alpha
    errors           its failures
    errors_reference the reference's failures over the same shots

The scaling figure
------------------

Absolute P_L against p for the one-pass reference, the standard two-pass rule
and the regularised one, against dotted guides of the nominal slope p^((d+1)/2).
Laid out as the ratio figure: one panel per decoder, every distance in each.
One row per (decoder, distance, p, condition):

    decoder, distance, rounds, basis, noise_model, p
    condition        "one_pass", "alpha_1" or "alpha_star"
    alpha            the regularisation actually sampled; blank for one-pass
    P_L              absolute logical error probability per shot
    P_L_low/high     interval on it
    shots, errors    what it was measured from
    fit_exponent     n in P_L ~ p^n, fitted over the lowest-p window
    fit_exponent_err its error, propagated from the P_L intervals
    fit_prefactor    the matching A in P_L = A p^n, so the line can be drawn
    fit_p_min/max    the window it was fitted over
    fit_n_points     how many points that was
    ideal_exponent   (d+1)/2, the distance-preserving value

The five `fit_*` columns repeat on every row of a (decoder, distance,
condition), so one file draws the whole figure.

The landscape figure
--------------------
The cost of mis-tuning alpha across (p, alpha), one panel per distance. It
takes two CSV files.

The *surface*, one row per sampled (decoder, distance, p, alpha) cell:

    decoder, distance, rounds, basis, noise_model, p, alpha
    ratio            P_L at that alpha / P_L of the one-pass reference
    ratio_low/high   interval on it
    excess           ratio divided by the best ratio in the same p column
    P_L, shots, errors

And the *overlay*, one row per (decoder, distance, p):

    decoder, distance, rounds, basis, noise_model, p
    alpha_star, alpha_star_low, alpha_star_high
    grid_argmin          the sampled alpha that won, before interpolation
    censored, unresolved flags: not a measurement, drawn but not fitted
    fit_alpha_0, fit_p_0, fit_k    alpha_0/(1+(p_0/p)^k) for that distance

The fit coefficients repeat on every row of a distance. Either file may be
given alone: the surface draws without the alpha* overlay, and the overlay
without a surface draws the points on bare axes.
"""

import argparse
import contextlib
import csv
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# --- knobs -----------------------------------------------------------------

USE_LATEX = True
"""Typeset through a real TeX toolchain. Falls back to matplotlib's mathtext
with a warning when `latex`/`dvipng` are not installed, so this is safe to
leave on."""

LOG_Y = True
"""Log y-axis on the ratio figure."""

YLIM = None
"""`(low, high)` to pin the ratio figure's y-axis, or None to fit the data."""

MAX_EXCESS = 2.0
"""Top of the landscape's colour scale, in P_L(alpha)/P_L(alpha*)."""

LANDSCAPE_CMAP = "Spectral_r"
"""The landscape's colour scale, cool at alpha* and warm at the top of it."""

LANDSCAPE_BAR_HEIGHT = 0.855
"""How much of the panels' height the colour bar takes, standing on their floor. The rest
is the gap under the stacked ratio, which is pinned to their ceiling."""

LANDSCAPE_SINGLE_WIDTH = 1.24
"""A lone panel's canvas, in columns: square panel plus room for the bar beside it."""

LANDSCAPE_BAR_RECT = 0.85
LANDSCAPE_BAR_RECT_SINGLE = 0.77
LANDSCAPE_RECT_LEFT_SINGLE = 0.05
"""A lone panel also needs a left margin: its y label is long, and the constrained layout
measures the panels only, not the text hanging off them."""
"""The fraction of the figure width the panels may use, leaving a margin for the bar, its
tick labels and the ratio above it. A lone panel needs the wider margin: its bar stands
further off the frame."""

LANDSCAPE_BAR_GRID_HEIGHT = 0.5
"""On several panels the bar is centred on their box at this fraction of its height, with
the ratio just above it, rather than standing on the floor with the ratio on the ceiling."""

LANDSCAPE_BAR_WIDTH = 0.022
LANDSCAPE_BAR_PAD = 0.032
LANDSCAPE_BAR_PAD_SINGLE = 0.045
LANDSCAPE_BAR_LABEL_GAP = 0.012
LANDSCAPE_BAR_DROP = 0.0
"""How far a lone panel's bar hangs below the panel floor: none, so the two
feet line up."""

LANDSCAPE_GRID_TOP = 0.93
LANDSCAPE_GRID_KEY_Y = 0.975
"""The top of the panels' area on a grid, and where its key sits above them."""

LANDSCAPE_COLUMN_ASPECT = 0.55
LANDSCAPE_COLUMN_HEIGHT = 2.45
LANDSCAPE_COLUMN_TOP = 0.965
LANDSCAPE_COLUMN_KEY_Y = 0.993
LANDSCAPE_COLUMN_RECT_BOTTOM = 0.085
LANDSCAPE_COLUMN_XLABEL_GAP = 0.004
LANDSCAPE_BAR_COLUMN_Y = 0.036
LANDSCAPE_BAR_COLUMN_HEIGHT = 0.013
LANDSCAPE_BAR_COLUMN_LABEL_GAP = 0.02
"""A single column stacks the panels and lays the bar flat beneath them, where one above
the other they are too narrow to give it a side: each panel's height over its width; the
canvas height in columns; the top of the panels' area and where the key sits above it; the
strip kept clear under them for the x label and the bar, and the gap between the bottom
panel's tick labels and that label; then where in the strip the bar sits, how thick it is,
and the gap between its right end and the quantity it carries. The bar starts at the
panels' left edge and runs as far as that quantity leaves room for."""
"""The bar's width, its distance from the panels, and the gap under the ratio where the
ratio rides above the bar, all in figure coordinates."""
COLUMN_WIDTH = 246 / 72
PAGE_WIDTH = 510 / 72
"""Target widths, in inches, from REVTeX's APS class: \\columnwidth is 246 pt
(86.8 mm) and \\textwidth 510 pt (179.9 mm)."""

RATIO_HEIGHT = 2.8
LANDSCAPE_HEIGHT = 2.9
"""Figure heights, in inches. Height sets how much of PRL's 3750-word budget a
figure spends, since the count scales with the area a figure occupies."""

TWO_PANEL_SIZE = (PAGE_WIDTH, 3.2)
"""Width and height, in inches, of the two-decoder figures (ratio, scaling and the
headline fit figure of `si1000_fit_figures.py`). The key sits inside the first panel, so
the axes take the whole height."""

LAYOUT_PAD = 1 / 72
"""Constrained-layout padding around and between the panels, in inches: 1 pt, against
matplotlib's 3 pt, so the axes reach the edges of the figure."""

XLABEL = "Physical error rate, $p$ (CZ gates)"
"""x-axis label of every p sweep. p is SI1000's two-qubit gate error rate: every other
channel scales with it."""

POINT_STYLE = {
    "markersize": 2.5, "markeredgewidth": 0.7,
    "elinewidth": 0.6, "capsize": 1.5, "capthick": 0.6,
}
"""Marker and error-bar sizes of the two-decoder figures."""

PARITY_STYLE = {"color": "0.6", "linewidth": 0.8, "linestyle": "dashed", "zorder": 1}
"""The reference line at ratio 1."""

DECODER_TITLES = {"CM": "Correlated Matching", "LCD": "Correlated LCD"}
"""Panel titles of the ratio figure, keyed on the `decoder` column; other names are shown as is."""

LER_DECODERS = {
    # decoder: (marker size, filled marker, legend label)
    "CM": (2.5, True, "CM"),
    "LCD": (6.0, False, "CLCD"),
}
"""Marker style and legend label of each decoder in the LER figure: a small filled marker
inside a large hollow one, so both stay visible where the points coincide. Labels get an
"R" (regularized) prefix when the figure is drawn at alpha*."""


def _ler_marker_style(decoder, color):
    size, filled, _ = LER_DECODERS.get(decoder, (3.0, True, decoder))
    return {
        "markersize": size, "color": color, "linestyle": "None",
        "markerfacecolor": color if filled else "none", "markeredgewidth": 0.5 if filled else 0.8,
        "zorder": 3 if filled else 2,
    }

DISTANCE_COLORS = {3: "#006f62", 5: "#dc4405", 7: "#3ccbda", 9: "#cf6f7f"}
DISTANCE_MARKERS = {3: "o", 5: "o", 7: "o", 9: "o"}
"""Circles at every distance; the colour carries the distance."""
FALLBACK_COLOR = "#006f62"
UNSAMPLED_COLOR = "#d9d9d9"
FALLBACK_MARKER = "o"

CONDITIONS = (
    # (condition, linestyle, filled, legend label)
    ("alpha_1", "dashed", False, r"Standard $\alpha=1$"),
    ("alpha_star", "solid", True, r"Regularized $\alpha^*(p)$"),
)

PAPER_RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth": 1.1,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
}

LATEX_RC = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
}


def latex_available() -> bool:
    """Whether a real TeX toolchain is present for `text.usetex`."""
    return all(shutil.which(tool) for tool in ("latex", "dvipng"))


@contextlib.contextmanager
def paper_style(use_latex=True):
    """
    Typeset the enclosed figures for print, with real LaTeX where possible. Falls
    back to mathtext with a warning when no TeX is installed.
    """
    settings = dict(PAPER_RC)
    if use_latex:
        if latex_available():
            settings.update(LATEX_RC)
        else:
            print(
                "warning: text.usetex requested but latex/dvipng were not found; "
                "falling back to mathtext",
                file=sys.stderr,
            )
    with mpl.rc_context(settings):
        yield


def save(figure, path):
    """
    Write a figure at exactly its `figsize`.
    """
    figure.savefig(path)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _maybe_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_curves(*paths) -> list[dict]:
    """
    Read one or more curve CSVs into typed rows, concatenated.
    """
    rows = []
    for path in paths:
        with Path(path).open(newline="") as handle:
            for raw in csv.DictReader(handle):
                rows.append({
                    "decoder": (raw.get("decoder") or "").strip() or "decoder",
                    "distance": int(float(raw["distance"])),
                    "rounds": _maybe_float(raw.get("rounds")),
                    "p": float(raw["p"]),
                    "condition": (raw.get("condition") or "").strip(),
                    "ratio": float(raw["ratio"]),
                    "ratio_low": _maybe_float(raw.get("ratio_low")),
                    "ratio_high": _maybe_float(raw.get("ratio_high")),
                    "significant": _maybe_float(raw.get("significant")),
                    "P_L": _maybe_float(raw.get("P_L")),
                    "shots": _maybe_float(raw.get("shots")),
                })
    return rows


def shared_points(rows) -> list[dict]:
    """
    The rows at the (distance, p) points every decoder has, so panels drawn from several
    decoders cover the same grid.
    """
    decoders = {row["decoder"] for row in rows}
    seen = {
        decoder: {(row["distance"], round(row["p"], 12)) for row in rows if row["decoder"] == decoder}
        for decoder in decoders
    }
    shared = set.intersection(*seen.values())
    return [row for row in rows if (row["distance"], round(row["p"], 12)) in shared]


def _select(rows, **constraints):
    return [
        row for row in rows
        if all(row.get(key) == value for key, value in constraints.items())
    ]


def _per_shot(rows) -> str:
    """What one shot covers, taken from the data: "$d$ rounds" where every row agrees."""
    per_shot = {
        "$d$ rounds" if row["rounds"] == row["distance"]
        else f"{row['rounds']:g} rounds"
        for row in rows if row.get("rounds")
    }
    return per_shot.pop() if len(per_shot) == 1 else "one shot"


def _ylabel(rows) -> str:
    """
    Axis label `P_L`, with what one shot covers taken from the data.
    """
    return f"$P_L$ ratio ({_per_shot(rows)})\ntwo-pass / one-pass"


def read_scaling(*paths) -> list[dict]:
    """Read P_L-scaling CSVs into typed rows, concatenated."""
    rows = []
    for path in paths:
        with Path(path).open(newline="") as handle:
            for raw in csv.DictReader(handle):
                rows.append({
                    "decoder": (raw.get("decoder") or "").strip() or "decoder",
                    "distance": int(float(raw["distance"])),
                    "rounds": _maybe_float(raw.get("rounds")),
                    "p": float(raw["p"]),
                    "condition": (raw.get("condition") or "").strip(),
                    "P_L": float(raw["P_L"]),
                    "P_L_low": _maybe_float(raw.get("P_L_low")),
                    "P_L_high": _maybe_float(raw.get("P_L_high")),
                    "fit_exponent": _maybe_float(raw.get("fit_exponent")),
                    "fit_exponent_err": _maybe_float(raw.get("fit_exponent_err")),
                    "fit_prefactor": _maybe_float(raw.get("fit_prefactor")),
                    "fit_p_min": _maybe_float(raw.get("fit_p_min")),
                    "fit_p_max": _maybe_float(raw.get("fit_p_max")),
                    "ideal_exponent": _maybe_float(raw.get("ideal_exponent")),
                })
    return rows


def read_surface(*paths) -> list[dict]:
    """Read landscape surface CSVs into typed rows, concatenated."""
    rows = []
    for path in paths:
        with Path(path).open(newline="") as handle:
            for raw in csv.DictReader(handle):
                rows.append({
                    "decoder": (raw.get("decoder") or "").strip() or "decoder",
                    "distance": int(float(raw["distance"])),
                    "p": float(raw["p"]),
                    "alpha": float(raw["alpha"]),
                    "excess": _maybe_float(raw.get("excess")),
                    "ratio": _maybe_float(raw.get("ratio")),
                })
    return rows


def read_overlay(*paths) -> list[dict]:
    """Read landscape overlay CSVs into typed rows, concatenated."""
    rows = []
    for path in paths:
        with Path(path).open(newline="") as handle:
            for raw in csv.DictReader(handle):
                rows.append({
                    "decoder": (raw.get("decoder") or "").strip() or "decoder",
                    "distance": int(float(raw["distance"])),
                    "p": float(raw["p"]),
                    "alpha_star": float(raw["alpha_star"]),
                    "grid_argmin": _maybe_float(raw.get("grid_argmin")),
                    "alpha_star_low": _maybe_float(raw.get("alpha_star_low")),
                    "alpha_star_high": _maybe_float(raw.get("alpha_star_high")),
                    "fit_alpha_0": _maybe_float(raw.get("fit_alpha_0")),
                    "fit_p_0": _maybe_float(raw.get("fit_p_0")),
                    "fit_k": _maybe_float(raw.get("fit_k")),
                })
    return rows


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def _draw_series(ax, rows, color, marker, linestyle, filled, mark_insignificant=True):
    """
    One ratio curve, with hollow markers where it cannot be told from parity.
    On a filled series, "insignificant" is drawn as a white face with a coloured
    edge. With `mark_insignificant` False every point is drawn like the rest of
    its series instead.
    """
    rows = sorted(rows, key=lambda row: row["p"])
    if not rows:
        return
    p_values = [row["p"] for row in rows]
    ratios = [row["ratio"] for row in rows]

    has_interval = all(
        row["ratio_low"] is not None and row["ratio_high"] is not None for row in rows
    )
    errors = (
        [[row["ratio"] - row["ratio_low"] for row in rows],
         [row["ratio_high"] - row["ratio"] for row in rows]]
        if has_interval else None
    )
    ax.errorbar(
        p_values, ratios, errors,
        marker=marker, color=color, linestyle=linestyle, linewidth=0.9,
        markerfacecolor=color if filled else "white", ecolor=color, **POINT_STYLE,
    )

    insignificant = [
        (row["p"], row["ratio"]) for row in rows if row["significant"] == 0
    ]
    if not insignificant or not mark_insignificant:
        return
    if filled:
        ax.scatter(
            *zip(*insignificant, strict=True),
            facecolors="white", edgecolors=color, zorder=5, s=14, linewidths=0.8,
        )
    else:
        ax.scatter(
            *zip(*insignificant, strict=True),
            facecolors=color, edgecolors="none", zorder=5, s=6,
        )


def plot_panel(ax, rows, distances=None, log_y=True, mark_insignificant=True):
    """Draw one decoder's curves onto an existing axis."""
    distances = distances if distances is not None else sorted(
        {row["distance"] for row in rows},
    )
    for d in distances:
        color = DISTANCE_COLORS.get(d, FALLBACK_COLOR)
        marker = DISTANCE_MARKERS.get(d, FALLBACK_MARKER)
        for condition, linestyle, filled, _ in CONDITIONS:
            _draw_series(
                ax, _select(rows, distance=d, condition=condition),
                color, marker, linestyle, filled, mark_insignificant,
            )

    ax.axhline(1.0, **PARITY_STYLE)
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
        # Ticks fixed so that the scale is easier to grasp
        ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1.0, 2.0, 3.0, 5.0)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(visible=True, linewidth=0.4, alpha=0.5)
    ax.set_xlabel(XLABEL)
    return distances


def series_handle(label, filled=None, linestyle="None", color="black"):
    """A key entry: a marker (filled, open, or none when `filled` is None) and/or a line."""
    marker = {"marker": "o", "markersize": 3.6, "markeredgewidth": 0.8,
              "markerfacecolor": color if filled else "white"} if filled is not None else {}
    return plt.Line2D([], [], color=color, linestyle=linestyle, linewidth=0.9, label=label,
                      **marker)


def inside_key(ax, series_handles, distances, loc="best"):
    """
    The key of the two-decoder figures, inside one panel on a white box: what each kind of
    series is in the left column, the distances as coloured markers in the right.
    """
    distance_handles = [
        series_handle(f"$d={d}$", filled=True, color=DISTANCE_COLORS.get(d, FALLBACK_COLOR))
        for d in distances
    ]
    # a legend fills its columns top to bottom, so pad the shorter one to keep the groups apart
    blank = plt.Line2D([], [], linestyle="None", label=" ")
    rows = max(len(series_handles), len(distance_handles))
    handles = (list(series_handles) + [blank] * (rows - len(series_handles))
               + distance_handles + [blank] * (rows - len(distance_handles)))
    key = ax.legend(handles=handles, loc=loc, ncol=2, fontsize=7.5, handlelength=1.8,
                    handletextpad=0.5, labelspacing=0.3, columnspacing=1.2,
                    borderpad=0.45, borderaxespad=0.5, frameon=True,
                    framealpha=0.9, facecolor="white", edgecolor="0.8")
    key.get_frame().set_linewidth(0.5)
    return key


def two_panel_axes(count):
    """
    `count` panels side by side at TWO_PANEL_SIZE, sharing both axes, tightly padded. A lone
    panel is one column wide instead, at the same height.
    """
    width, height = TWO_PANEL_SIZE
    figure, axes = plt.subplots(1, count, figsize=(width if count > 1 else COLUMN_WIDTH, height),
                                sharex=True, sharey=True, squeeze=False, layout="constrained")
    figure.get_layout_engine().set(w_pad=LAYOUT_PAD, h_pad=LAYOUT_PAD)
    return axes[0]


def plot_ratio_vs_p(rows, decoders=None, axes=None, log_y=True, ylim=None, mark_insignificant=True):
    """
    Two-pass/one-pass LEP ratio against p, at alpha=1 and alpha*.
    """
    decoders = decoders if decoders is not None else sorted(
        {row["decoder"] for row in rows},
    )
    if axes is None:
        axes = two_panel_axes(len(decoders))

    for index, (ax, decoder) in enumerate(zip(axes, decoders, strict=True)):
        plot_panel(ax, _select(rows, decoder=decoder), log_y=log_y,
                   mark_insignificant=mark_insignificant)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if len(decoders) > 1:
            title = DECODER_TITLES.get(decoder, decoder)
            ax.set_title(f"({'abcdefg'[index]}) {title}", pad=3)

    axes[0].set_ylabel(_ylabel(rows))
    inside_key(
        axes[0],
        [series_handle(label, filled, linestyle) for _, linestyle, filled, label in reversed(CONDITIONS)],
        sorted({row["distance"] for row in rows}),
    )
    return axes[0].figure


# ---------------------------------------------------------------------------
# absolute P_L, and the exponent it decays with
# ---------------------------------------------------------------------------

ONE_PASS_COLOR = "#8c8c8c"
"""The one-pass reference, the same grey at every distance: it is the baseline
both re-weighting rules are read against, not a series of its own."""

GUIDE_LINESTYLE = (0, (1, 1.5))
"""The nominal-slope guides, dotted so they cannot be mistaken for the dashed
alpha=1 series."""

COMPENSATE_SCALING = False
"""Divide P_L by the distance-preserving power p^((d+1)/2) before plotting.

Off, the figure is the raw log-log: absolute rates, directly comparable with
the literature, and the dotted guides carry the nominal slope. The cost is that
over seven decades of P_L a slope difference of 0.67 is a few millimetres of
divergence at the very left edge.

On, every series is divided by p^((d+1)/2) and normalised to its own one-pass
fit, so a decoder that preserves the nominal distance is a horizontal line at 1
and one that does not climbs away from it as p falls - at the full 1/p rate the
theory predicts. The same data, with the eye doing the subtraction instead of
having to measure a slope."""


def _scaling_fit(rows):
    """The (exponent, error, prefactor, window) carried by a set of rows."""
    for row in rows:
        if row["fit_exponent"] is not None and np.isfinite(row["fit_exponent"]):
            return row
    return None


def _scaling_errors(series, leps, divisor):
    """
    Asymmetric (lower, upper) bars on P_L, or None when the source omits them.

    These are the likelihood-ratio intervals on an *absolute* rate - one
    decoder against its own shots. The ratio figures combine two such
    intervals, one per decoder, in log space under an independence
    approximation (binomial_ratio_intervals.py).

    On the raw log-log they are mostly smaller than the markers: at the ~6000
    reference failures this sweep targets, a 96.8% interval is about +-3%, which
    is a fraction of a point across seven decades. They matter at the few
    cap-limited points, and they matter everywhere once `compensate` expands the
    y-axis onto a linear-ish band.
    """
    if any(row["P_L_low"] is None or row["P_L_high"] is None for row in series):
        return None
    return [
        leps - np.array([row["P_L_low"] for row in series]) / divisor,
        np.array([row["P_L_high"] for row in series]) / divisor - leps,
    ]


def _ideal_guide(series, ideal):
    """
    Fowler's guide: the nominal power law through the lowest data point.

    Anchored on the datum rather than on a fit, so the line is a statement about
    the code's nominal scaling and not about anything measured here. Where the
    one-pass series follows its own guide, the decoder keeps the distance.
    """
    if not series or not ideal:
        return None, None
    anchor = min(series, key=lambda row: row["p"])
    span = np.geomspace(anchor["p"], max(row["p"] for row in series), 2)
    return span, anchor["P_L"] * (span / anchor["p"]) ** ideal


def plot_scaling_panel(ax, rows, distances=None, show_ideal=True, compensate=False):
    """
    One decoder's absolute P_L against p, every distance together.

    The point of the panel is the *slope*. A decoder preserving the nominal
    distance decays as p^((d+1)/2); the dotted guide for each distance has
    exactly that slope and passes through that distance's lowest one-pass point,
    so a series running parallel to its guide keeps the distance and one that
    flattens against it does not.

    Grey open markers are the one-pass reference. The two-pass series are drawn
    as in the ratio figure: dashed and open at alpha=1, solid and filled at
    alpha*. `compensate` divides out p^((d+1)/2) and normalises to the one-pass
    fit, turning every guide into the line y=1 - see `COMPENSATE_SCALING`.
    """
    distances = distances if distances is not None else sorted(
        {row["distance"] for row in rows},
    )
    for d in distances:
        at_d = _select(rows, distance=d)
        one_pass = sorted(_select(at_d, condition="one_pass"), key=lambda row: row["p"])
        color = DISTANCE_COLORS.get(d, FALLBACK_COLOR)
        marker = DISTANCE_MARKERS.get(d, FALLBACK_MARKER)
        reference = _scaling_fit(one_pass)
        ideal = reference["ideal_exponent"] if reference else None

        def compensator(p_values, ideal=ideal, reference=reference):
            """P_L divided by the nominal power, scaled to the one-pass fit."""
            p_values = np.asarray(p_values, dtype=float)
            if not compensate or reference is None or not ideal:
                return np.ones(len(p_values))
            anchor = reference["fit_p_max"]
            scale = (
                reference["fit_prefactor"] * anchor ** reference["fit_exponent"]
                / anchor ** ideal
            )
            return scale * p_values ** ideal

        if show_ideal and ideal and not compensate:
            span, guide = _ideal_guide(one_pass, ideal)
            if span is not None:
                ax.plot(span, guide, color=color, linewidth=0.8,
                        linestyle=GUIDE_LINESTYLE, zorder=2)

        if one_pass:
            p_values = np.array([row["p"] for row in one_pass])
            divisor = compensator(p_values)
            leps = np.array([row["P_L"] for row in one_pass]) / divisor
            ax.errorbar(
                p_values, leps, _scaling_errors(one_pass, leps, divisor),
                marker=marker, linestyle="None", color=ONE_PASS_COLOR,
                markerfacecolor="white", ecolor=ONE_PASS_COLOR, zorder=3, **POINT_STYLE,
            )

        for condition, linestyle, filled, _ in CONDITIONS:
            series = sorted(_select(at_d, condition=condition), key=lambda row: row["p"])
            if not series:
                continue
            p_values = np.array([row["p"] for row in series])
            divisor = compensator(p_values)
            leps = np.array([row["P_L"] for row in series]) / divisor
            ax.errorbar(
                p_values, leps, _scaling_errors(series, leps, divisor),
                marker=marker, color=color, linestyle=linestyle, linewidth=0.9,
                markerfacecolor=color if filled else "white", ecolor=color,
                zorder=5 if filled else 4, **POINT_STYLE,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(visible=True, linewidth=0.4, alpha=0.5)
    ax.set_xlabel(XLABEL)
    if compensate:
        # The guides all collapse onto this one line once the nominal power
        # is divided out, so it is drawn once rather than per distance.
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle=GUIDE_LINESTYLE, zorder=1)
        ax.set_ylim(0.12, 8.0)
    return distances


def plot_scaling(rows, decoders=None, axes=None, show_ideal=True, compensate=None):
    """
    Absolute P_L against p, one panel per decoder, laid out as the ratio figure:
    same size, panel titles and key inside the first panel.
    """
    compensate = COMPENSATE_SCALING if compensate is None else compensate
    decoders = decoders if decoders is not None else sorted(
        {row["decoder"] for row in rows},
    )
    if axes is None:
        axes = two_panel_axes(len(decoders))

    for index, (ax, decoder) in enumerate(zip(axes, decoders, strict=True)):
        plot_scaling_panel(ax, _select(rows, decoder=decoder), show_ideal=show_ideal,
                           compensate=compensate)
        if len(decoders) > 1:
            title = DECODER_TITLES.get(decoder, decoder)
            ax.set_title(f"({'abcdefg'[index]}) {title}", pad=3)

    axes[0].set_ylabel(
        f"$P_L \\,/\\, p^{{(d+1)/2}}$, normalised\n({_per_shot(rows)})" if compensate
        else f"Logical error probability\n$P_L$ ({_per_shot(rows)})",
    )
    series_handles = [
        series_handle(label, filled, linestyle) for _, linestyle, filled, label in reversed(CONDITIONS)
    ]
    series_handles.append(series_handle("One-pass", filled=False, color=ONE_PASS_COLOR))
    if show_ideal and not compensate:
        series_handles.append(series_handle(r"Nominal $p^{(d+1)/2}$", linestyle=GUIDE_LINESTYLE))
    # the data climbs from lower left to upper right, so the upper left is clear
    inside_key(axes[0], series_handles, sorted({row["distance"] for row in rows}),
               loc="upper left")
    return axes[0].figure


# ---------------------------------------------------------------------------
# absolute logical error probability
# ---------------------------------------------------------------------------

def plot_ler_vs_p(rows, condition="alpha_star"):
    """
    Logical error probability against p at `condition` (alpha* by default), unconnected
    points: colour and marker shape by distance, marker size and fill by decoder.
    """
    rows = [row for row in rows if row["condition"] == condition and row["P_L"]]
    decoders = sorted({row["decoder"] for row in rows})
    distances = sorted({row["distance"] for row in rows})
    prefix = "R" if condition == "alpha_star" else ""
    names = {decoder: prefix + LER_DECODERS.get(decoder, (0, 0, decoder))[2] for decoder in decoders}
    figure, (ax, ratio_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(COLUMN_WIDTH, 1.5 * RATIO_HEIGHT),
        height_ratios=(2.2, 1), layout="constrained",
    )

    for decoder in decoders:
        for d in distances:
            points = sorted(_select(rows, decoder=decoder, distance=d), key=lambda r: r["p"])
            if not points:
                continue
            color = DISTANCE_COLORS.get(d, FALLBACK_COLOR)
            errors = [
                np.sqrt(r["P_L"] * (1 - r["P_L"]) / r["shots"]) if r["shots"] else 0.0
                for r in points
            ]
            ax.errorbar(
                [r["p"] for r in points], [r["P_L"] for r in points], errors,
                marker=DISTANCE_MARKERS.get(d, FALLBACK_MARKER), **_ler_marker_style(decoder, color),
                capthick=0.6, capsize=1.5, elinewidth=0.6, ecolor=color,
            )

    # lower panel: LER of the second decoder over the first at the same (d, p); the two
    # decoders see independent shots, so relative binomial errors add in quadrature
    if len(decoders) == 2:
        reference, other = ("CM", "LCD") if set(decoders) == {"CM", "LCD"} else decoders
        for d in distances:
            by_p = {round(r["p"], 12): r for r in _select(rows, decoder=reference, distance=d)}
            pairs = sorted(
                (r["p"], by_p[round(r["p"], 12)], r)
                for r in _select(rows, decoder=other, distance=d) if round(r["p"], 12) in by_p
            )
            if not pairs:
                continue
            color = DISTANCE_COLORS.get(d, FALLBACK_COLOR)
            ratios = [o["P_L"] / ref["P_L"] for _, ref, o in pairs]
            errors = [
                ratio * np.sqrt(1 / (ref["P_L"] * ref["shots"]) + 1 / (o["P_L"] * o["shots"]))
                for ratio, (_, ref, o) in zip(ratios, pairs)
            ]
            ratio_ax.errorbar(
                [p for p, _, _ in pairs], ratios, errors,
                marker=DISTANCE_MARKERS.get(d, FALLBACK_MARKER), markersize=3, color=color,
                linestyle="None", capthick=0.7, capsize=1.8, elinewidth=0.7, ecolor=color,
            )
        ratio_ax.axhline(1.0, color="black", linewidth=0.7, linestyle=":")
        ratio_ax.set_ylabel(f"{names[other]} / {names[reference]}")
    else:
        ratio_ax.set_visible(False)

    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax, ratio_ax):
        axis.grid(visible=True, linewidth=0.4, alpha=0.5)
    ratio_ax.set_xlabel("Physical error rate, $p$")
    rounds = {"$d$ rounds" if r["rounds"] == r["distance"] else f"{r['rounds']:g} rounds"
              for r in rows if r["rounds"]}
    per_shot = rounds.pop() if len(rounds) == 1 else "one shot"
    ax.set_ylabel(f"Logical error probability, $P_L$ ({per_shot})")

    distance_handles = [
        plt.Line2D([], [], color=DISTANCE_COLORS.get(d, FALLBACK_COLOR),
                   marker=DISTANCE_MARKERS.get(d, FALLBACK_MARKER), markersize=3,
                   linestyle="None", label=f"$d={d}$")
        for d in distances
    ]
    decoder_handles = [
        plt.Line2D([], [], marker="o", label=names[decoder],
                   **_ler_marker_style(decoder, "black"))
        for decoder in decoders
    ]
    distance_legend = ax.legend(handles=distance_handles, loc="upper left", handlelength=1.0,
                                borderaxespad=0.4, labelspacing=0.25)
    ax.add_artist(distance_legend)
    ax.legend(handles=decoder_handles, loc="lower right", handlelength=1.0,
              borderaxespad=0.4, labelspacing=0.25)
    return figure


# ---------------------------------------------------------------------------
# the (p, alpha) cost landscape
# ---------------------------------------------------------------------------

def _log_edges(values):
    """Cell boundaries at the geometric midpoints of unevenly log-spaced values."""
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.array([values[0] * 0.9, values[0] * 1.1])
    inner = np.sqrt(values[:-1] * values[1:])
    first = values[0] ** 2 / inner[0]
    last = values[-1] ** 2 / inner[-1]
    return np.concatenate([[first], inner, [last]])


def _grid(rows):
    """(p values, alphas, excess[i, j]) from long-format surface rows."""
    p_values = sorted({row["p"] for row in rows})
    alphas = sorted({row["alpha"] for row in rows})
    p_index = {p: i for i, p in enumerate(p_values)}
    alpha_index = {a: j for j, a in enumerate(alphas)}
    excess = np.full((len(p_values), len(alphas)), np.nan)
    for row in rows:
        if row["excess"] is not None:
            excess[p_index[row["p"]], alpha_index[row["alpha"]]] = row["excess"]
    return np.array(p_values), np.array(alphas), excess


def alpha_star_model(p, alpha_0, p_0, k):
    """alpha*(p) = alpha_0 / (1 + (p_0/p)**k), the form the theory predicts."""
    return alpha_0 / (1 + (p_0 / np.asarray(p, dtype=float)) ** k)


def plot_alpha_landscape(surface, overlay=(), decoder=None, distances=None,
                         figure=None, max_excess=2.0, error_bars=True, grid_argmin=False,
                         fit_k1=False, panel_grid=False, panel_column=False):
    """
    Cost of mis-tuning alpha, across (p, alpha), one panel per distance.
    """
    decoders = sorted({row["decoder"] for row in surface})
    decoder = decoder if decoder is not None else (decoders[0] if decoders else None)
    surface = [row for row in surface if row["decoder"] == decoder]
    overlay = [row for row in overlay if row["decoder"] == decoder]

    distances = distances if distances is not None else sorted(
        {row["distance"] for row in surface} or {row["distance"] for row in overlay},
    )
    if figure is None:
        if panel_column and len(distances) > 1:
            # One column of panels down a page column, each on its own p range, with the
            # colour bar laid flat under them and one x label under the stack. The panels
            # are wider than they are tall: stacked square at this width they would run
            # well past a page.
            figure, grid = plt.subplots(
                len(distances), 1,
                figsize=(COLUMN_WIDTH, COLUMN_WIDTH * LANDSCAPE_COLUMN_HEIGHT),
                sharex=False, sharey=True, squeeze=False, layout="constrained",
            )
            axes = [row[0] for row in grid]
        elif panel_grid and len(distances) > 1:
            # Two columns of square panels, each on its own p range. Sharing the axis puts
            # every distance on the widest of them and leaves the unsampled corners grey.
            columns = 2
            rows = -(-len(distances) // columns)
            width = PAGE_WIDTH * 0.72
            figure, grid = plt.subplots(
                rows, columns, figsize=(width, width * rows / columns * 1.02),
                sharex=False, sharey=True, squeeze=False, layout="constrained",
            )
            axes = [grid[i // columns][i % columns] for i in range(rows * columns)]
            for spare in axes[len(distances):]:
                spare.set_visible(False)
            axes = axes[:len(distances)]
        else:
            # Sized for a two-column page: 180 mm wide; one distance fits a column, and is
            # squared off, where a landscape of the page's width would be a letterbox.
            single = len(distances) == 1
            # wider than it is tall, so the square panel keeps its size and the extra
            # width goes to the bar, its tick labels and the ratio above them
            size = ((COLUMN_WIDTH * LANDSCAPE_SINGLE_WIDTH, COLUMN_WIDTH) if single
                    else (PAGE_WIDTH, LANDSCAPE_HEIGHT))
            figure, axes = plt.subplots(
                1, len(distances), figsize=size,
                sharex=True, sharey=True, squeeze=False, layout="constrained",
            )
            axes = axes[0]
    else:
        axes = figure.axes

    mesh = None
    norm = mpl.colors.Normalize(vmin=1.0, vmax=max_excess)
    for index, (ax, d) in enumerate(zip(axes, distances, strict=True)):
        ax.set_facecolor(UNSAMPLED_COLOR)
        at_d = _select(surface, distance=d)
        alphas = np.array([])
        if at_d:
            p_array, alphas, excess = _grid(at_d)
            mesh = ax.pcolormesh(
                _log_edges(p_array), _log_edges(alphas),
                np.ma.masked_invalid(excess).T,
                cmap=LANDSCAPE_CMAP, norm=norm,
                shading="flat", rasterized=True,
            )

        points = sorted(_select(overlay, distance=d), key=lambda row: row["p"])
        if points:
            _draw_alpha_star(ax, points, error_bars=error_bars, grid_argmin=grid_argmin,
                             fit_k1=fit_k1)

        ax.set_xscale("log")
        ax.set_yscale("log")
        # a panel spanning about a decade would label its minor ticks too, and those
        # labels run into the decades
        ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        # alpha runs up to 1, which reads better as itself than as a power of ten
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
            lambda value, _: "1" if np.isclose(value, 1.0)
            else f"$10^{{{round(np.log10(value))}}}$",
        ))
        ax.set_title(f"$d={d}$", fontsize=9, pad=3)
        # the full label needs a column to itself; beside three others it would collide
        # in a column the label goes under the whole stack, below; nothing per panel
        if not panel_column:
            ax.set_xlabel(XLABEL if len(distances) == 1 or panel_grid else "$p$",
                          fontsize=9)
        ax.tick_params(labelsize=7.5)
        if len(alphas):
            ax.set_ylim(alphas.min(), alphas.max())
        if panel_column:
            ax.set_box_aspect(LANDSCAPE_COLUMN_ASPECT)
        elif len(distances) == 1 or panel_grid:
            ax.set_box_aspect(1)
        if len(distances) > 1:
            # a panel letter only where there are panels to tell apart, above the frame so
            # it covers no data and needs no patch behind it
            ax.annotate(
                f"({'abcdefg'[index]})", xy=(0.0, 1.02), xycoords="axes fraction",
                va="bottom", ha="left", fontsize=8,
            )

    if panel_grid:
        for ax in axes:
            ax.set_ylabel(r"Reweighting strength, $\alpha$", fontsize=9)
    elif panel_column:
        # one label for the stack: four of them down the side would repeat the same words
        figure.supylabel(r"Reweighting strength, $\alpha$", fontsize=9)
    else:
        axes[0].set_ylabel(r"Reweighting strength, $\alpha$", fontsize=9)

    column_xlabel = None
    if panel_column:
        # one label for the stack, parked here and moved under the panels once the layout
        # has settled; left in the layout it would be pushed into the bar's strip
        column_xlabel = figure.supxlabel("Physical error rate, $p$ (CZ gates)", fontsize=9)
        column_xlabel.set_in_layout(False)

    if len(distances) == 1 or panel_grid or panel_column:
        # the overlay's key: the swept points on top, the curve fitted through them below.
        # On one panel the distance shares the row, to its left; on a grid every panel
        # already carries its own d, so the key sits above the top right one alone.
        if len(distances) == 1:
            # matplotlib keeps a left, centre and right title; the centred one was already
            # set in the loop above, so it has to be cleared or both are drawn
            axes[0].set_title("", loc="center")
            axes[0].set_title(rf"$d={distances[0]}$", fontsize=9, loc="left", pad=12,
                              x=0.08)
        handles = [
            plt.Line2D([], [], color="black", marker="o", markersize=2.5, linestyle="None",
                       label=r"grid-searched $\alpha^*(p)$"),
            plt.Line2D([], [], color="black", lw=1.0,
                       label=(r"fit $\alpha^*(p)=\alpha_0/(1+p_0/p)$" if fit_k1
                              else r"fit $\alpha^*(p)=\alpha_0/[1+(p_0/p)^k]$")),
        ]
        if len(distances) == 1:
            key = axes[0].legend(
                handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=1,
                fontsize=7, handlelength=1.6, handletextpad=0.5, labelspacing=0.25,
                borderaxespad=0.0, frameon=False,
            )
            key._legend_box.align = "left"
        else:
            # one key for the grid, centred above it, both entries on a single row. It is
            # anchored in figure coordinates rather than placed "outside", so it can sit
            # closer to the panels; the layout rect below keeps that strip clear.
            key = figure.legend(
                handles=handles, loc="upper center",
                bbox_to_anchor=(0.5, LANDSCAPE_COLUMN_KEY_Y if panel_column
                                else LANDSCAPE_GRID_KEY_Y), ncol=2, fontsize=7,
                handlelength=1.6, handletextpad=0.5, labelspacing=0.25,
                columnspacing=2.0, borderaxespad=0.0, frameon=False,
            )

    # No key on either figure: the caption names the curve and the points, and the
    # landscape is worth more than the space a key would cover.

    if mesh is not None:
        # `aspect` ties a colorbar's length to its width, so its ends land wherever the
        # figure's proportions put them. Placing it by hand instead: reserve a margin,
        # let the constrained layout settle the panels, then measure their box and stand
        # the bar on its floor, LANDSCAPE_BAR_HEIGHT of the way up, leaving the rest for
        # the quantity pinned to the ceiling.
        single_panel = len(distances) == 1
        rect = 1.0 if panel_column else (
            LANDSCAPE_BAR_RECT_SINGLE if single_panel else LANDSCAPE_BAR_RECT)
        top_rect = (LANDSCAPE_COLUMN_TOP if panel_column
                    else 1.0 if single_panel else LANDSCAPE_GRID_TOP)
        left_rect = LANDSCAPE_RECT_LEFT_SINGLE if single_panel else 0.0
        bottom_rect = LANDSCAPE_COLUMN_RECT_BOTTOM if panel_column else 0.0
        figure.get_layout_engine().set(
            rect=(left_rect, bottom_rect, rect - left_rect, top_rect - bottom_rect))
        figure.canvas.draw()
        boxes = [ax.get_position() for ax in axes]
        left = min(box.x0 for box in boxes)
        right = max(box.x1 for box in boxes)
        bottom = min(box.y0 for box in boxes)
        top = max(box.y1 for box in boxes)
        figure.set_layout_engine("none")

        pad = LANDSCAPE_BAR_PAD
        if panel_column:
            # the x label first, just under the bottom panel's tick labels, then the bar
            # flat below it, standing on the panels' left edge and running the width of the
            # stack but for the quantity beside its right end. The quantity is laid out and
            # measured first, since how long the bar can be depends on how wide it is.
            inverse = figure.transFigure.inverted()
            ink = min(ax.get_tightbbox().transformed(inverse).y0 for ax in axes)
            column_xlabel.set_position(((left + right) / 2,
                                        ink - LANDSCAPE_COLUMN_XLABEL_GAP))
            column_xlabel.set_verticalalignment("top")

            middle = LANDSCAPE_BAR_COLUMN_Y + LANDSCAPE_BAR_COLUMN_HEIGHT / 2
            quantity = figure.text(0.0, middle, r"$\dfrac{P_L(\alpha)}{P_L(\alpha^*)}$",
                                   ha="left", va="center", fontsize=8.5)
            figure.canvas.draw()
            label_width = quantity.get_window_extent().width / figure.bbox.width
            length = (right - left) - LANDSCAPE_BAR_COLUMN_LABEL_GAP - label_width
            quantity.set_position((left + length + LANDSCAPE_BAR_COLUMN_LABEL_GAP, middle))
            cax = figure.add_axes([left, LANDSCAPE_BAR_COLUMN_Y,
                                   length, LANDSCAPE_BAR_COLUMN_HEIGHT])
            cax.set_in_layout(False)
            bar = figure.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
        elif len(distances) == 1:
            # one panel: the bar stands on its floor, the ratio sits on its ceiling, and
            # the pair stands further off the frame than on a grid, where the panels are
            # narrower and the margin tighter
            pad = LANDSCAPE_BAR_PAD_SINGLE
            height = (top - bottom) * LANDSCAPE_BAR_HEIGHT
            bar_bottom = bottom - LANDSCAPE_BAR_DROP
            label_y, label_va = top, "top"
        else:
            # several: the bar is centred on their box, the ratio rides just above it
            height = (top - bottom) * LANDSCAPE_BAR_GRID_HEIGHT
            bar_bottom = bottom + (top - bottom - height) / 2
            label_y, label_va = bar_bottom + height + LANDSCAPE_BAR_LABEL_GAP, "bottom"
        if not panel_column:
            cax = figure.add_axes([right + pad, bar_bottom,
                                   LANDSCAPE_BAR_WIDTH, height])
            cax.set_in_layout(False)
            bar = figure.colorbar(mesh, cax=cax, orientation="vertical", extend="max")
            # The bar sits on the panel's floor, which leaves its head clear for the
            # quantity, the right way up. Stacked as a fraction it is about as wide as the
            # bar and its tick labels together, where written on one line it would overrun
            # the landscape. As the bar's title the fraction would ride on the bar's head,
            # wherever that falls. Pinned instead to the panel's own top edge, with x past
            # the middle of the bar (the fraction is wider than it, and the room is on the
            # right, where the tick labels leave a margin), it stays level with the frame
            # however short the bar is.
            figure.text(right + pad + LANDSCAPE_BAR_WIDTH * 1.6, label_y,
                        r"$\dfrac{P_L(\alpha)}{P_L(\alpha^*)}$",
                        ha="center", va=label_va, fontsize=8.5)
        bar.set_ticks(mpl.ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        bar_axis = bar.ax.xaxis if panel_column else bar.ax.yaxis
        bar_axis.set_major_formatter(mpl.ticker.FuncFormatter(lambda t, _: f"{t:g}"))
        bar.ax.tick_params(labelsize=7.5)
    return figure


def _draw_alpha_star(ax, points, error_bars=True, grid_argmin=False, fit_k1=False):
    """
    alpha*(p) and its fit, in black over the valley floor. Only some overlays carry an
    interval for alpha*; `error_bars` drops it even where it is there, so landscapes from
    different sources can be shown side by side without one of them looking better resolved.

    `grid_argmin` plots the best sampled alpha instead of the reported alpha*. Where a table
    refines its minimum between grid points, the refined value falls between the heatmap's
    cells; the raw argmin sits on them, as the grid search's own alpha* does. The interval
    belongs to the refined value, so it is dropped along with it.
    """
    if grid_argmin and all(row.get("grid_argmin") is not None for row in points):
        points = [{**row, "alpha_star": row["grid_argmin"]} for row in points]
        error_bars = False
    has_interval = error_bars and all(
        row["alpha_star_low"] is not None and row["alpha_star_high"] is not None
        for row in points
    )
    errors = (
        [[row["alpha_star"] - row["alpha_star_low"] for row in points],
         [row["alpha_star_high"] - row["alpha_star"] for row in points]]
        if has_interval else None
    )
    ax.errorbar(
        [row["p"] for row in points], [row["alpha_star"] for row in points], errors,
        marker="o", markersize=2.5, linestyle="None", color="black",
        ecolor="black", elinewidth=0.7, capsize=0, zorder=4,
        label=r"swept $\alpha^*$",
    )
    p_values = [row["p"] for row in points]
    smooth = np.geomspace(min(p_values), max(p_values), 200)

    if fit_k1:
        # alpha_0/(1 + p_0/p): the k = 1 case, fitted here since the tables store only the
        # three-parameter coefficients
        from scipy.optimize import curve_fit

        x = np.array(p_values, dtype=float)
        y = np.array([row["alpha_star"] for row in points], dtype=float)
        try:
            (alpha_0, p_0), _ = curve_fit(
                lambda p, a0, p0: a0 / (1.0 + p0 / p), x, y, p0=[0.9, 4e-4], maxfev=40000)
        except Exception:
            return
        ax.plot(
            smooth, alpha_0 / (1.0 + p_0 / smooth),
            color="black", linewidth=1.2, zorder=5,
            label=r"fit $\alpha_0/(1+p_0/p)$",
        )
        return

    fit = next(
        (row for row in points
         if row["fit_alpha_0"] is not None and row["fit_p_0"] is not None
         and row["fit_k"] is not None),
        None,
    )
    if fit is None:
        return
    ax.plot(
        smooth,
        alpha_star_model(smooth, fit["fit_alpha_0"], fit["fit_p_0"], fit["fit_k"]),
        color="black", linewidth=1.2, zorder=5,
        label=r"fit $\alpha_0/(1+(p_0/p)^k)$",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def csv_kind(path) -> str:
    """
    Which of the three schemas a file holds, from its header.

    So the command line takes a plain list of CSVs and works out what each one
    is, rather than asking for `--surface` and `--overlay` separately. One
    column is unique to each schema: `fit_exponent` to the scaling data,
    `alpha_star` to the landscape overlay, `excess` to the landscape surface,
    `condition` to the ratio curves. The scaling file is tested first because
    it carries `condition` too.
    """
    with Path(path).open(newline="") as handle:
        header = set(next(csv.reader(handle), []))
    for marker, kind in (
        ("fit_exponent", "scaling"), ("alpha_star", "overlay"),
        ("excess", "surface"), ("condition", "curves"),
    ):
        if marker in header:
            return kind
    msg = (
        f"{path}: cannot tell what this is. Expected one of the columns "
        f"'condition' (ratio curves), 'fit_exponent' (P_L scaling), "
        f"'alpha_star' (landscape overlay) or 'excess' (landscape surface)."
    )
    raise ValueError(msg)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Draw one of the paper's figures from its data CSVs. "
                    "Everything else is a constant at the top of this file.",
    )
    parser.add_argument("figure", choices=("ratio", "landscape", "ler", "scaling"))
    parser.add_argument("out", help="output figure path")
    parser.add_argument("csv", nargs="+", help="source CSVs, in any order")
    parser.add_argument(
        "--condition", choices=("alpha_star", "alpha_1"), default="alpha_star",
        help="ler figure only: the regularized decoders at alpha* (default) or the "
             "unregularized ones at alpha = 1",
    )
    parser.add_argument("--panel-grid", action="store_true",
                        help="landscape figure only: lay the panels out two to a row, each "
                             "square and on its own p range, so no unsampled corners remain")
    parser.add_argument("--panel-column", action="store_true",
                        help="landscape figure only: stack the panels in one column, each "
                             "on its own p range, with the colour bar beneath them")
    parser.add_argument("--fit-k1", action="store_true",
                        help="landscape figure only: draw alpha_0/(1+p_0/p), the k = 1 case, "
                             "refitted here, instead of the stored three-parameter fit")
    parser.add_argument("--grid-argmin", action="store_true",
                        help="landscape figure only: mark the best sampled alpha rather than "
                             "a refined alpha*, so the points sit on the cells")
    parser.add_argument("--no-error-bars", action="store_true",
                        help="landscape figure only: draw alpha* as bare points, without "
                             "the interval, for overlays that carry one")
    parser.add_argument("--distance", type=int, action="append",
                        help="landscape figure only: draw only this code distance; repeat the "
                             "flag for several")
    parser.add_argument("--shared-points", action="store_true",
                        help="ratio figure only: keep the (distance, p) points every decoder "
                             "has, so the panels cover the same grid")
    parser.add_argument("--fill-insignificant", action="store_true",
                        help="ratio figure only: draw the points that cannot be told from "
                             "parity like the rest of their series, instead of white-faced")
    args = parser.parse_args(argv)

    by_kind = {}
    for path in args.csv:
        try:
            by_kind.setdefault(csv_kind(path), []).append(path)
        except (OSError, ValueError) as error:
            parser.error(str(error))

    with paper_style(use_latex=USE_LATEX):
        if args.figure == "ratio":
            rows = read_curves(*by_kind.get("curves", ()))
            if not rows:
                parser.error("no ratio curve rows in the files given")
            if args.shared_points:
                rows = shared_points(rows)
                if not rows:
                    parser.error("the decoders share no (distance, p) point")
            figure = plot_ratio_vs_p(rows, log_y=LOG_Y, ylim=YLIM,
                                     mark_insignificant=not args.fill_insignificant)
            count = len(rows)
        elif args.figure == "ler":
            rows = read_curves(*by_kind.get("curves", ()))
            if not rows:
                parser.error("no ratio curve rows in the files given")
            figure = plot_ler_vs_p(rows, condition=args.condition)
            count = len(rows)
        elif args.figure == "scaling":
            rows = read_scaling(*by_kind.get("scaling", ()))
            if not rows:
                parser.error("no P_L scaling rows in the files given")
            figure = plot_scaling(rows)
            count = len(rows)
        else:
            surface = read_surface(*by_kind.get("surface", ()))
            overlay = read_overlay(*by_kind.get("overlay", ()))
            if not surface and not overlay:
                parser.error("no landscape rows in the files given")
            decoders = sorted({row["decoder"] for row in surface or overlay})
            if len(decoders) > 1:
                print(
                    f"note: {len(decoders)} decoders present {decoders}; "
                    f"drawing {decoders[0]}. Split the files to draw another.",
                    file=sys.stderr,
                )
            available = sorted({row["distance"] for row in surface or overlay})
            missing = sorted(set(args.distance or ()) - set(available))
            if missing:
                parser.error(f"no landscape rows at distance(s) {missing}; have {available}")
            figure = plot_alpha_landscape(surface, overlay, distances=args.distance,
                                          max_excess=MAX_EXCESS,
                                          error_bars=not args.no_error_bars,
                                          grid_argmin=args.grid_argmin,
                                          fit_k1=args.fit_k1,
                                          panel_grid=args.panel_grid,
                                          panel_column=args.panel_column)
            count = len(surface) + len(overlay)
        save(figure, args.out)
    print(f"{count} rows -> {args.out}")


if __name__ == "__main__":
    main()
