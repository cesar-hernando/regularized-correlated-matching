"""
Table S III: the optimal reweighting strength model alpha*(p) = alpha_0 / [1 + (p_0/p)^k] fitted to the
grid-searched alpha* of regularized correlated matching (MWPM grid search) and regularized correlated LCD, per
code distance.

The fit is a least-squares fit to the `alpha_star` column of each `alpha_landscape_overlay.csv`, bounded to
0.3 <= alpha_0 <= 1, 1e-5 <= p_0 <= 0.1 and 0.2 <= k <= 4 (as in scripts/analyze_rcm_grid_search.py); the
uncertainties are standard errors from the fit's covariance. The rows are printed as LaTeX, each error rounded to
one significant figure and its value to match, with p_0 in units of 1e-4.

    python figures/alpha_star_fit_table.py
"""
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (("CM", ROOT / "data/mwpm/rcm_grid_search_si1000/analysis/alpha_landscape_overlay.csv"),
           ("C-LCD", ROOT / "data/lcd/alpha_landscape_overlay.csv"))
DISTANCES = (3, 5, 7, 9)
BOUNDS = ([0.3, 1e-5, 0.2], [1.0, 1e-1, 4.0])


def model(p, alpha_0, p_0, k):
    return alpha_0 / (1 + (p_0 / p) ** k)


def pm(value, error):
    """value +- error, the error to one significant figure and the value to the same decimal place."""
    error = float(f"{error:.1g}")
    places = max(0, -math.floor(math.log10(error)))
    return f"{value:.{places}f} \\pm {error:.{places}f}"


def main():
    print(r"decoder & $d$ & $\alpha_0$ & $p_0$ & $k$ & $R^2$ \\")
    for name, path in SOURCES:
        rows = list(csv.DictReader(open(path)))
        for d in DISTANCES:
            points = [r for r in rows if int(float(r["distance"])) == d and r["alpha_star"]]
            p = np.array([float(r["p"]) for r in points])
            y = np.array([float(r["alpha_star"]) for r in points])
            par, cov = curve_fit(model, p, y, p0=[0.9, 7e-4, 1.0], bounds=BOUNDS, maxfev=40000)
            err = np.sqrt(np.diag(cov))
            r2 = 1 - np.sum((y - model(p, *par)) ** 2) / np.sum((y - y.mean()) ** 2)
            print(f"{name} & {d} & ${pm(par[0], err[0])}$ & $({pm(par[1] * 1e4, err[1] * 1e4)})\\times10^{{-4}}$ & "
                  f"${pm(par[2], err[2])}$ & {r2:.3f} \\\\")
        print(r"\addlinespace")


if __name__ == "__main__":
    main()
