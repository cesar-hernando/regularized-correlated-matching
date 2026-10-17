"""
Fig. S1: the ratio of the d-round logical error probability of correlated matching (alpha = 1) to that of MWPM
at d = 5 under SI1000 noise, for Z-memory and X-memory circuits.

Both come from the grid-search result files (`data/mwpm/rcm_grid_search_si1000/` for Z memory,
`data/mwpm/rcm_grid_search_si1000_xmem_d5/` for X memory). As in Fig. 1, the ratio's error bar combines each
decoder's binomial likelihood interval (sinter, maximum likelihood factor 10) in log space under an independence
approximation (`figures/binomial_ratio_intervals.py`), neglecting the covariance from decoding the same shots.
Z memory keeps the p values of Fig. 1, those correlated LCD also has.

    python figures/xz_memory_ratio.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from lcd_figures import COLUMN_WIDTH, LAYOUT_PAD, PARITY_STYLE, POINT_STYLE, XLABEL, paper_style  # noqa: E402
from binomial_ratio_intervals import ratio_interval  # noqa: E402

GRID = ROOT / "data" / "mwpm"
OUT = ROOT / "figures" / "fig_mc_si1000_xz_ratio.pdf"
COLOUR = {"Z": "#1f77b4", "X": "#d62728"}   # Z memory blue, X memory red, as Z- and X-type in Fig. S4
D = 5


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def points(folder):
    """{(basis, d): [(p, ratio, low, high)]} of CM (alpha = 1) to MWPM, from the common shots."""
    out = {}
    for f in sorted((GRID / folder).glob("*.csv")):
        rows = read(f)
        m = next(r for r in rows if r["decoder"] == "mwpm")
        c = next(r for r in rows if r["decoder"] == "cm" and float(r["alpha"]) == 1.0)
        em, ec = int(m["errors"]), int(c["errors"])
        if em == 0:
            continue
        ratio = ec / em
        low, high = ratio_interval(ratio, int(m["shots"]), ec, em)
        out.setdefault((m["basis"].upper(), int(m["distance"])), []).append(
            (float(m["p"]), ratio, low, high)
        )
    return {k: sorted(v) for k, v in out.items()}


def main():
    shared = {round(float(r["p"]), 12) for r in read(ROOT / "data" / "lcd" / "ratio_vs_p_curves.csv")
              if int(float(r["distance"])) == D}
    pts = {**points("rcm_grid_search_si1000"), **points("rcm_grid_search_si1000_xmem_d5")}
    series = {"Z": [r for r in pts[("Z", D)] if round(r[0], 12) in shared], "X": pts[("X", D)]}
    with paper_style(use_latex=True):
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.6), layout="constrained")
        fig.get_layout_engine().set(w_pad=LAYOUT_PAD, h_pad=LAYOUT_PAD)
        for basis in ("X", "Z"):
            p, ratio, low, high = zip(*series[basis])
            yerr = [[r - l for r, l in zip(ratio, low)],
                    [h - r for r, h in zip(ratio, high)]]
            ax.errorbar(p, ratio, yerr, linestyle="dashed", lw=0.9, marker="o", color=COLOUR[basis],
                        markerfacecolor="white", label=f"{basis} memory", **POINT_STYLE)
        ax.axhline(1.0, **PARITY_STYLE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.FixedLocator([0.6, 0.8, 1, 1.5, 2, 3]))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.grid(visible=True, linewidth=0.4, alpha=0.5)
        ax.set_xlabel(XLABEL)
        ax.set_ylabel(r"$p_L^{\mathrm{CM}} / p_L^{\mathrm{MWPM}}$")
        key = ax.legend(loc="upper right", fontsize=7.5, handlelength=1.8, handletextpad=0.5, labelspacing=0.3,
                        borderpad=0.45, borderaxespad=0.5, frameon=True, framealpha=0.9, facecolor="white",
                        edgecolor="0.8")
        key.get_frame().set_linewidth(0.5)
        fig.savefig(OUT)
        plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
