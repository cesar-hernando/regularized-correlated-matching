"""
Analyse an RCM grid search.

Follows the analysis of the circuit-level depolarizing noise RCM results in `data/mwpm/alpha_star_vs_p.csv`:

  * LER = errors / shots, with its binomial standard error;
  * alpha* = the searched alpha < 1 with the lowest LER;
  * ratios to MWPM, with standard errors propagated from both LERs;
  * weighted least-squares fits per distance of the models in `RegularizedCM.models`:
        RCM(alpha*)/MWPM = 1 + a + b p
        CM(alpha=1)/MWPM = 1 + a + b p + c ln(1/p) / p
        alpha*(p)        = alpha0 / (1 + (p0/p)^k)

Writes to `--out-dir`:
  alpha_star_vs_p.csv, ratio_fits_coefficients.csv
      same columns as the files in data/mwpm/
  ratio_vs_p_curves.csv, alpha_landscape_surface.csv, alpha_landscape_overlay.csv
      the format read by figures/lcd_figures.py, with decoder=CM

The results directory must hold a single code, noise model and memory basis.

Example (from the repository root):
    uv run scripts/analyze_rcm_grid_search.py --results-dir data/mwpm/rcm_grid_search_si1000 \\
        --tasks tasks/rcm_grid_search_tasks_si1000.txt
"""

import argparse
import csv
import inspect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

import tasks_and_results as files
from RegularizedCM import (
    cm_ratio,
    error_rate_ratio,
    logical_error_rate,
    mcnemar_p_value,
    optimal_alpha,
    rcm_ratio,
)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

@dataclass
class Point:
    """Counts at one (code, noise model, basis, distance, rounds, p). Arrays follow `alphas`."""

    code: str
    noise_model: str
    basis: str
    distance: int
    rounds: int
    p: float
    shots: int
    mwpm_errors: int
    alphas: np.ndarray
    cm_errors: np.ndarray
    only_cm_fails: np.ndarray
    only_mwpm_fails: np.ndarray

    def ler(self, errors):
        """LER and its binomial standard error."""
        return logical_error_rate(errors, self.shots)

    def ratio(self, i):
        """LER of correlated matching at alphas[i] over the MWPM LER, with its standard error."""
        return error_rate_ratio(self.cm_errors[i], self.mwpm_errors, self.shots)

    def mcnemar_p(self, i):
        """McNemar p-value of correlated matching at alphas[i] against MWPM."""
        return mcnemar_p_value(self.only_cm_fails[i], self.only_mwpm_fails[i])

    @property
    def hard_index(self):
        """Index of alpha = 1, standard correlated matching."""
        return int(np.flatnonzero(np.isclose(self.alphas, 1.0))[0])

    @property
    def best_index(self):
        """Index of alpha*, the alpha < 1 with the fewest errors."""
        damped = np.flatnonzero(self.alphas < 1.0)
        return int(damped[np.argmin(self.cm_errors[damped])])


def load_points(results_dir) -> list[Point]:
    points = []
    for path in sorted(Path(results_dir).glob("*.csv")):
        mwpm, *cm = files.read_results(path)
        points.append(Point(
            code=mwpm["code"],
            noise_model=mwpm["noise_model"],
            basis=mwpm["basis"],
            distance=mwpm["distance"],
            rounds=mwpm["rounds"],
            p=mwpm["p"],
            shots=mwpm["shots"],
            mwpm_errors=mwpm["errors"],
            alphas=np.array([row["alpha"] for row in cm]),
            cm_errors=np.array([row["errors"] for row in cm]),
            only_cm_fails=np.array([row["only_this_fails"] for row in cm]),
            only_mwpm_fails=np.array([row["only_mwpm_fails"] for row in cm]),
        ))
    return sorted(points, key=lambda point: (point.distance, point.p))


def warn_incomplete(points, tasks_file):
    """Warn about points with fewer shots than their task asks for."""
    target = {
        (task.code, task.noise_model, task.basis, task.distance, task.rounds, task.p): task.shots
        for task in files.read_tasks(tasks_file)
    }
    for point in points:
        key = (point.code, point.noise_model, point.basis, point.distance, point.rounds, point.p)
        shots = target.get(key)
        if shots is not None and point.shots < shots:
            print(f"warning: d={point.distance} p={point.p:.6e} has {point.shots:,} "
                  f"of {shots:,} shots")


# ---------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------

def fit(model, x, y, **kwargs):
    """curve_fit returning (parameters, R^2), or NaNs when the fit fails."""
    n_parameters = len(inspect.signature(model).parameters) - 1
    try:
        parameters, _ = curve_fit(model, x, y, **kwargs)
    except (RuntimeError, ValueError, TypeError):
        return np.full(n_parameters, np.nan), np.nan
    residual = np.sum((y - model(x, *parameters)) ** 2)
    total = np.sum((y - np.mean(y)) ** 2)
    return parameters, 1.0 - residual / total if total > 0 else np.nan


def fit_distance(points):
    """Fit coefficients for one distance, as in data/mwpm/ratio_fits_coefficients.csv."""
    p = np.array([point.p for point in points])
    rcm_ratios, rcm_se = np.array([point.ratio(point.best_index) for point in points]).T
    cm_ratios, cm_se = np.array([point.ratio(point.hard_index) for point in points]).T
    alpha_stars = np.array([point.alphas[point.best_index] for point in points])

    (a_rcm, b_rcm), r2_rcm = fit(
        rcm_ratio, p, rcm_ratios, sigma=rcm_se, absolute_sigma=True, p0=[-0.4, 30.0],
    )
    (a_cm, b_cm, c_cm), r2_cm = fit(
        cm_ratio, p, cm_ratios, sigma=cm_se, absolute_sigma=True,
        p0=[-0.4, 30.0, 5e-6], maxfev=40000,
    )
    (alpha0, p0, k), r2_alpha = fit(
        optimal_alpha, p, alpha_stars, p0=[0.9, 7e-4, 1.0],
        bounds=([0.3, 1e-5, 0.2], [1.0, 1e-1, 4.0]), maxfev=40000,
    )
    return {
        "distance": points[0].distance,
        "rcm_mwpm_fit_1plusA": 1 + a_rcm,
        "rcm_mwpm_fit_b": b_rcm,
        "rcm_mwpm_fit_R2": r2_rcm,
        "cm_mwpm_fit_1plusA": 1 + a_cm,
        "cm_mwpm_fit_b": b_cm,
        "cm_mwpm_fit_c": c_cm,
        "cm_mwpm_fit_R2": r2_cm,
        "alpha_star_fit_alpha0": alpha0,
        "alpha_star_fit_p0": p0,
        "alpha_star_fit_k": k,
        "alpha_star_fit_R2": r2_alpha,
    }


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def alpha_star_row(point):
    """One row of alpha_star_vs_p.csv."""
    mwpm, mwpm_std = point.ler(point.mwpm_errors)
    hard, hard_std = point.ler(point.cm_errors[point.hard_index])
    best, best_std = point.ler(point.cm_errors[point.best_index])
    return {
        "distance": point.distance,
        "p": point.p,
        "alpha_star": point.alphas[point.best_index],
        "ler_mwpm": mwpm,
        "ler_mwpm_std": mwpm_std,
        "ler_cm_hard": hard,
        "ler_cm_hard_std": hard_std,
        "ler_rcm_alpha_star": best,
        "ler_rcm_alpha_star_std": best_std,
        "rcm_over_mwpm": best / mwpm,
        "cm_over_mwpm": hard / mwpm,
    }


def lcd_keys(point):
    """The identifying columns of the lcd_figures.py tables."""
    return {
        "decoder": "CM",
        "distance": point.distance,
        "rounds": point.rounds,
        "basis": point.basis,
        "noise_model": point.noise_model,
        "p": point.p,
    }


def curve_rows(point):
    """Rows of ratio_vs_p_curves.csv, at alpha = 1 and at alpha*."""
    mwpm, _ = point.ler(point.mwpm_errors)
    for condition, i in (("alpha_1", point.hard_index), ("alpha_star", point.best_index)):
        ratio, se = point.ratio(i)
        mcnemar_p = point.mcnemar_p(i)
        yield {
            **lcd_keys(point),
            "condition": condition,
            "alpha": point.alphas[i],
            "ratio": ratio,
            "ratio_low": ratio - se,
            "ratio_high": ratio + se,
            "mcnemar_p": mcnemar_p,
            "significant": int(mcnemar_p < 0.05),
            "P_L": point.ler(point.cm_errors[i])[0],
            "P_L_reference": mwpm,
            "shots": point.shots,
            "errors": point.cm_errors[i],
            "errors_reference": point.mwpm_errors,
        }


def surface_rows(point):
    """Rows of alpha_landscape_surface.csv, one per alpha."""
    ratios = [point.ratio(i) for i in range(len(point.alphas))]
    best_ratio = min(ratio for ratio, _ in ratios)
    for i, (ratio, se) in enumerate(ratios):
        yield {
            **lcd_keys(point),
            "alpha": point.alphas[i],
            "ratio": ratio,
            "ratio_low": ratio - se,
            "ratio_high": ratio + se,
            "excess": ratio / best_ratio,
            "P_L": point.ler(point.cm_errors[i])[0],
            "shots": point.shots,
            "errors": point.cm_errors[i],
        }


def overlay_row(point, coefficients):
    """One row of alpha_landscape_overlay.csv. The interval and flags are not computed."""
    alpha_star = point.alphas[point.best_index]
    return {
        **lcd_keys(point),
        "alpha_star": alpha_star,
        "alpha_star_low": "",
        "alpha_star_high": "",
        "grid_argmin": alpha_star,
        "censored": "",
        "unresolved": "",
        "fit_alpha_0": coefficients["alpha_star_fit_alpha0"],
        "fit_p_0": coefficients["alpha_star_fit_p0"],
        "fit_k": coefficients["alpha_star_fit_k"],
    }


def write_csv(path, rows):
    rows = list(rows)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-dir", default=None, help="default: <results-dir>/analysis")
    parser.add_argument("--tasks", default=None,
                        help="task list, to warn about points with missing shots")
    args = parser.parse_args()

    points = load_points(args.results_dir)
    if not points:
        raise SystemExit(f"no result files in {args.results_dir}")
    for field in ("code", "noise_model", "basis"):
        values = {getattr(point, field) for point in points}
        if len(values) > 1:
            raise SystemExit(f"{args.results_dir} mixes {field} values {sorted(values)}")
    if args.tasks:
        warn_incomplete(points, args.tasks)

    distances = sorted({point.distance for point in points})
    coefficients = {
        d: fit_distance([point for point in points if point.distance == d])
        for d in distances
    }

    out_dir = Path(args.out_dir or Path(args.results_dir) / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "alpha_star_vs_p.csv", map(alpha_star_row, points))
    write_csv(out_dir / "ratio_fits_coefficients.csv", coefficients.values())
    write_csv(out_dir / "ratio_vs_p_curves.csv",
              (row for point in points for row in curve_rows(point)))
    write_csv(out_dir / "alpha_landscape_surface.csv",
              (row for point in points for row in surface_rows(point)))
    write_csv(out_dir / "alpha_landscape_overlay.csv",
              (overlay_row(point, coefficients[point.distance]) for point in points))


if __name__ == "__main__":
    main()
