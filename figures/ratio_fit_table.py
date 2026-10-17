"""
Fit coefficients of the two-decoder ratio figure, as LaTeX table rows, from the same curves and the same rule
`si1000_fit_figures.plot_two_panel` fits (only the points both decoders share): alpha = 1 points with
1 + a + bp + c/p (or + c ln(1/p)/p), alpha* points with 1 + a + bp. Each point is weighted by its interval's
half-width converted to one standard deviation (divided by z = 2.146), so the quoted errors are 1 sigma.

    python figures/ratio_fit_table.py data/mwpm/rcm_grid_search_si1000/analysis/ratio_vs_p_curves_binomial.csv \
        data/lcd/ratio_vs_p_curves_binomial.csv [--alpha1-model inverse]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from si1000_fit_figures import ALPHA1_MODELS, read_curves, shared_curves  # noqa: E402
from RegularizedCM.models import rcm_ratio  # noqa: E402

Z = math.sqrt(2 * math.log(10))
NAMES = {("CM", "alpha_1"): "CM", ("LCD", "alpha_1"): "C-LCD", ("CM", "alpha_star"): "RCM",
         ("LCD", "alpha_star"): "RC-LCD"}


def fit(model, rows, p0):
    p = np.array([r["p"] for r in rows])
    y = np.array([r["ratio"] for r in rows])
    s = np.array([r["sigma"] / Z for r in rows])
    par, cov = curve_fit(model, p, y, p0=p0, sigma=s, absolute_sigma=True, maxfev=40000)
    residual = y - model(p, *par)
    r2 = 1 - np.sum(residual ** 2) / np.sum((y - y.mean()) ** 2)
    chi2 = float(np.sum((residual / s) ** 2)) / (len(p) - len(par))
    return par, np.sqrt(np.diag(cov)), r2, chi2


def latex(value, error):
    """value +- error with the error to two significant figures and the value to match."""
    if error == 0 or not math.isfinite(error):
        return f"{value:.3g}"
    exponent = math.floor(math.log10(abs(value))) if value else 0
    if abs(exponent) >= 3:
        scale = 10.0 ** exponent
        digits = max(0, -math.floor(math.log10(error / scale)) + 1)
        return rf"$({value / scale:.{digits}f} \pm {error / scale:.{digits}f}) \times 10^{{{exponent}}}$"
    digits = max(0, -math.floor(math.log10(error)) + 1)
    return rf"${value:.{digits}f} \pm {error:.{digits}f}$"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cm_curves")
    parser.add_argument("lcd_curves")
    parser.add_argument("--alpha1-model", choices=sorted(ALPHA1_MODELS), default="inverse")
    args = parser.parse_args()
    curves = shared_curves(read_curves(args.cm_curves) | read_curves(args.lcd_curves))
    model1, start1, label1 = ALPHA1_MODELS[args.alpha1_model]
    for condition, model, p0, title in (("alpha_1", model1, start1, f"alpha = 1: {label1}"),
                                        ("alpha_star", rcm_ratio, [-0.4, 30.0], "alpha*: 1 + a + bp")):
        print(f"% {title}  (decoder & d & a & b{' & c' if condition == 'alpha_1' else ''} & R^2 & chi^2/dof)")
        for decoder in ("CM", "LCD"):
            for d in (3, 5, 7, 9):
                rows = curves.get((decoder, d, condition))
                if not rows:
                    continue
                par, err, r2, chi2 = fit(model, rows, p0)
                cells = [NAMES[(decoder, condition)], str(d)] + [latex(v, e) for v, e in zip(par, err)]
                print(" & ".join(cells) + rf" & ${r2:.3f}$ \\  % chi2/dof = {chi2:.2f}, {len(rows)} points")
            print(r"\midrule" if decoder == "CM" else "")


if __name__ == "__main__":
    main()
