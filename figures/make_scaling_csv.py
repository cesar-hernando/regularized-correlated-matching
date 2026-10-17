"""
Build `lep_scaling.csv` from raw alpha-scan output.

Standalone: sinter, numpy and the standard library. Nothing here imports from
this study, so it runs anywhere an alpha-scan sampler has been run - give it
the raw CSVs and it produces the interchange file that
`figures/lcd_figures.py scaling` draws.

    uv run figures/make_scaling_csv.py \\
        data/mwpm/rcm_grid_search_si1000/*.csv \\
        -o data/mwpm/rcm_grid_search_si1000/analysis/lep_scaling.csv

    uv run figures/lcd_figures.py scaling figures/cm_lcd_scaling.pdf \\
        data/mwpm/rcm_grid_search_si1000/analysis/lep_scaling.csv \\
        data/lcd/lep_scaling.csv

Two input formats are understood, told apart per file by its header (see
`fold`): the CSVs `sinter.collect` writes, and the per-point tables of the RCM
grid search in `data/mwpm/`. Several input CSVs may be given and are summed;
sinter's resume format writes one line per batch, so a point split across
shards folds correctly by addition.

One trap it handles for you. The sampler's decoder name does not change when
the alpha grid does, so rows taken under an earlier grid fold into the same
points and their alphas reappear carrying only the shots of that run. Left
alone they sit on the curve 100x noisier than their neighbours and win the
argmin for alpha* at random. Alphas seeing far fewer shots than the rest of
their point are therefore dropped, and the count is reported on stderr - see
`alpha_grid`. If that note appears, it is worth knowing why.

What it reads
-------------
Sinter rows written by an alpha-scan sampler, such as the LCD one, whose
`custom_counts` carry, with `tag = f"{alpha:.3f}"`:

    err_ref            the one-pass reference failed
    err_a<tag>         two-pass at this alpha failed
    both_a<tag>        both failed
    only_a<tag>        only this alpha failed
    only_ref_a<tag>    only the reference failed
    shots_a<tag>       this alpha's own denominator

and whose `json_metadata` carries `p`, `d`, `r`, `basis` and `noise_model`. The
alpha grid is recovered from the count keys, so nothing has to be told to this
script twice; one consequence is that the `alpha` column comes back rounded to
the three decimals the keys use, which is immaterial here because this file is
about P_L and not about locating alpha* precisely.

Or the RCM grid search's tables (`data/mwpm/rcm_grid_search_*/*.csv`), one row
per (point, decoder, alpha) with columns `basis`, `distance`, `rounds`, `p`,
`noise_model`, `decoder`, `alpha`, `shots`, `errors`, `only_this_fails` and
`only_mwpm_fails`. The row with `decoder = mwpm` and no alpha is the one-pass
reference; every other row is the two-pass decoder at its alpha, decoded on the
same shots, with the paired counts against that reference. These are folded
into the same counts as above, so everything downstream is shared.

What it writes
--------------
One row per (decoder, distance, p, condition), where `condition` is

    one_pass     the uncorrelated reference
    alpha_1      the standard re-weighting rule
    alpha_star   the best regularisation sampled at that point

`alpha_star` is the grid value actually run, chosen exactly as the alpha-scan
analysis chooses it: the grid minimum of the paired ratio, interpolated by a
parabola in log(alpha) through that minimum and its two neighbours, then
snapped back to the nearest sampled alpha - reporting the interpolated vertex
would be quoting a decoder nobody ran.

Each row also carries a fit of P_L ~ p^n over the lowest `--fit-points` sampled
p at that distance. A decoder preserving the nominal code distance has
n = (d+1)/2. The window is a count rather than a p range because the asymptotic
regime starts at a different p for each distance, and it is written into every
row so a caption can state it.

Intervals on P_L are likelihood-ratio intervals at maximum likelihood factor
`--max-likelihood-factor` (default 10), whose Gaussian half-width is
sqrt(2 ln 10) = 2.146 standard deviations, i.e. 96.8% coverage. Keep this the
same across decoders being plotted together.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import sinter   # type: ignore


FIELDS = [
    "decoder", "distance", "rounds", "basis", "noise_model", "p",
    "condition", "alpha",
    "P_L", "P_L_low", "P_L_high", "shots", "errors",
    "fit_exponent", "fit_exponent_err", "fit_prefactor",
    "fit_p_min", "fit_p_max", "fit_n_points", "ideal_exponent",
]

DEFAULT_FIT_POINTS = 6
DEFAULT_MAX_LIKELIHOOD_FACTOR = 10.0
DEFAULT_MIN_SHOT_FRACTION = 0.5
"""Keep an alpha only if it saw at least this share of the best-sampled
alpha's shots at that point. See `alpha_grid` for what it is protecting
against; 0.5 is loose enough for the few per cent of extra shots alpha=1 picks
up when it is a member of two grids, and tight enough for the factor ~150 a
stale grid leaves behind."""
BASELINE = 1.0
"""The unregularised rule. Must be on the sampled grid."""


# ---------------------------------------------------------------------------
# folding raw rows into points
# ---------------------------------------------------------------------------

GRID_SEARCH_COLUMNS = {"only_this_fails", "only_mwpm_fails", "alpha"}
"""Columns that mark a file as an RCM grid-search table rather than sinter's."""
GRID_SEARCH_REFERENCE = "mwpm"
"""`decoder` value of the grid search's one-pass reference row."""
GRID_SEARCH_DECODER = "CM"
"""Decoder name given to grid-search points, as its analysis CSVs name it."""


def fold(*paths) -> dict:
    """
    {(decoder, p, d, basis, rounds, noise_model): {"shots", "errors", "counts"}}.

    Everything is summed. sinter writes one line per batch and splits a point
    across shards, so addition is the whole of the folding; the only thing to
    be careful of is not reading the same file twice.

    Each file is routed by its header: a grid-search table goes through
    `_fold_grid_search`, anything else is handed to sinter.
    """
    points: dict = {}
    sinter_paths = []
    for path in paths:
        with Path(path).open(newline="") as handle:
            header = set(next(csv.reader(handle), []))
        if GRID_SEARCH_COLUMNS <= header:
            _fold_grid_search(points, path)
        else:
            sinter_paths.append(path)
    if sinter_paths:
        _fold_sinter(points, sinter_paths)
    if not points:
        msg = "no rows with p and d in their json_metadata"
        raise ValueError(msg)
    return points


def _add(points, key, shots, errors, counts):
    entry = points.setdefault(key, {"shots": 0, "errors": 0, "counts": {}})
    entry["shots"] += shots
    entry["errors"] += errors
    for name, value in counts.items():
        entry["counts"][name] = entry["counts"].get(name, 0) + int(value)


def _fold_grid_search(points, path):
    """
    Fold one grid-search table into `points`, as the sinter sampler's counts.

    The reference row supplies the point's shots and `err_ref`. Each two-pass
    row at alpha supplies `err_a`, `only_a` (its `only_this_fails`),
    `only_ref_a` (`only_mwpm_fails`) and `both_a`, which is its failures less
    the ones only it had. Every row of a point shares one set of shots, so that
    is also each alpha's `shots_a`.
    """
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                GRID_SEARCH_DECODER, float(row["p"]), int(row["distance"]),
                row["basis"], int(row["rounds"]), row["noise_model"],
            )
            shots, errors = int(row["shots"]), int(row["errors"])
            if row["decoder"] == GRID_SEARCH_REFERENCE and not row["alpha"]:
                _add(points, key, shots, errors, {"err_ref": errors})
                continue
            suffix = tag(float(row["alpha"]))
            only_alpha = int(row["only_this_fails"])
            _add(points, key, 0, 0, {
                f"err_a{suffix}": errors,
                f"both_a{suffix}": errors - only_alpha,
                f"only_a{suffix}": only_alpha,
                f"only_ref_a{suffix}": int(row["only_mwpm_fails"]),
                f"shots_a{suffix}": shots,
            })


def _fold_sinter(points, paths):
    for stat in sinter.read_stats_from_csv_files(*paths):
        meta = stat.json_metadata or {}
        if "p" not in meta or "d" not in meta:
            continue
        key = (
            stat.decoder, float(meta["p"]), int(meta["d"]),
            str(meta.get("basis", "")), int(meta.get("r", meta["d"])),
            str(meta.get("noise_model", "")),
        )
        _add(points, key, stat.shots, stat.errors, stat.custom_counts or {})


def tag(alpha: float) -> str:
    """
    Key suffix for one alpha, as the sampler writes it.

    Three decimals, unless that would round the alpha: the grid search samples
    values such as 0.0365, which three decimals would report as a neighbour.
    """
    short = f"{alpha:.3f}"
    return short if float(short) == alpha else repr(float(alpha))


def alpha_grid(counts, min_shot_fraction=None) -> list[float]:
    """
    The sampled alphas, recovered from the count keys.

    Alphas that saw far fewer shots than the rest are dropped, and that is not
    a nicety. The sampler's decoder name does not change when the alpha grid
    does, so rows taken under an earlier grid fold into the same point and
    their alphas reappear here carrying only the shots of whatever run
    produced them. In this study's own data a scout taken on a 15-point grid
    leaves 17 stale alphas at d=3, p=1e-5, each with 9.4e7 shots against the
    production grid's 1.5e10 - a factor 156. Keeping them puts points on the
    curve that are 150 times noisier than their neighbours, and since alpha* is
    an argmin over that curve, a stale alpha wins it roughly at random.

    Comparing shot counts catches this without having to be told what the grid
    is: every alpha in one run sees the same batches, so the current grid is
    the plateau and anything well below it is from another.
    """
    fraction = (
        DEFAULT_MIN_SHOT_FRACTION if min_shot_fraction is None else min_shot_fraction
    )
    found = {}
    for name, value in counts.items():
        for prefix in ("shots_a", "err_a"):
            if name.startswith(prefix):
                try:
                    alpha = float(name[len(prefix):])
                except ValueError:
                    continue
                if prefix == "shots_a":
                    found[alpha] = int(value)
                else:
                    found.setdefault(alpha, None)
    if not found:
        return []
    shots = [n for n in found.values() if n is not None]
    if not shots:
        # No per-alpha denominators at all: an older sampler, where every alpha
        # on the grid did see every shot. Nothing to compare, so keep them all.
        return sorted(found)
    threshold = fraction * max(shots)
    return sorted(
        alpha for alpha, count in found.items()
        if count is None or count >= threshold
    )


def alpha_shots(entry, alpha) -> int:
    """
    Shots this alpha saw.

    Falls back to the point's own total when no per-alpha denominator was
    recorded, which is correct for such rows: every alpha then on the grid saw
    every shot.
    """
    return int(entry["counts"].get(f"shots_a{tag(alpha)}", entry["shots"]))


def paired_ratio(entry, alpha):
    """
    P_L(two-pass at alpha) / P_L(one-pass), both on the same shots.

    A ratio of failure counts, because the shot count is shared and cancels.
    None where either decoder recorded no failures at all.
    """
    counts = entry["counts"]
    suffix = tag(alpha)
    both = counts.get(f"both_a{suffix}")
    only_alpha = counts.get(f"only_a{suffix}")
    only_reference = counts.get(f"only_ref_a{suffix}")
    if None in (both, only_alpha, only_reference):
        return None
    numerator, denominator = both + only_alpha, both + only_reference
    if denominator <= 0 or numerator <= 0:
        return None
    return numerator / denominator


# ---------------------------------------------------------------------------
# which alpha won
# ---------------------------------------------------------------------------

def _vertex(log_alphas, log_ratios):
    """Vertex of a parabola through three points, or None if it has no minimum."""
    coefficients = np.polyfit(log_alphas, log_ratios, deg=2)
    if not np.isfinite(coefficients[0]) or coefficients[0] <= 0:
        return None
    return -coefficients[1] / (2 * coefficients[0])


def best_alpha(entry, alphas):
    """
    The sampled alpha to report as alpha*, or None.

    Ties are handled explicitly because they are neither rare nor random: as
    alpha falls the two-pass decoder converges on the reference, so the
    low-alpha end of the curve is flat by construction, and at a point with few
    errors it is flat *exactly* - several alphas decode every shot identically.
    A plain argmin breaks that toward the grid floor every time. A tied run is
    reported at its midpoint instead, unless alpha=1 is among the tied, which
    is the reading that matters where it holds.
    """
    curve = {a: paired_ratio(entry, a) for a in alphas}
    usable = [a for a in alphas if curve.get(a) and curve[a] > 0]
    if len(usable) < 3:
        return None

    ratios = [curve[a] for a in usable]
    lowest = min(ratios)
    tied = [i for i, ratio in enumerate(ratios) if ratio == lowest]
    index = tied[-1] if tied[-1] == len(usable) - 1 else tied[len(tied) // 2]

    # At either end of the grid there is no bracket to interpolate inside, so
    # the grid value is the answer such as it is.
    if index in (0, len(usable) - 1):
        return usable[index]

    bracket = usable[index - 1:index + 2]
    log_alphas = np.log(bracket)
    vertex = _vertex(log_alphas, np.log([curve[a] for a in bracket]))
    if vertex is None:
        return usable[index]
    estimate = float(np.clip(np.exp(vertex), bracket[0], bracket[-1]))
    # The vertex is an interpolation; report the decoder that was actually run.
    return min(usable, key=lambda a: abs(np.log(a / estimate)))


# ---------------------------------------------------------------------------
# the exponent
# ---------------------------------------------------------------------------

def fit_exponent(p_values, leps, lows, highs):
    """
    (exponent, error, prefactor) of a weighted log-log straight line.

    Weights come from the per-point intervals as a standard deviation on
    log P_L; `cov="unscaled"` so the quoted error is propagated from those
    rather than rescaled by the residual. NaNs rather than an exception if the
    fit cannot be made, so one thin distance does not take the file down.
    """
    p_values, leps = np.asarray(p_values, float), np.asarray(leps, float)
    usable = (p_values > 0) & (leps > 0) & np.isfinite(leps)
    if usable.sum() < 3:
        return (np.nan, np.nan, np.nan)
    x, y = np.log(p_values[usable]), np.log(leps[usable])

    weights = None
    lows, highs = np.asarray(lows, float)[usable], np.asarray(highs, float)[usable]
    sigma = np.where((lows > 0) & (highs > 0), (np.log(highs) - np.log(lows)) / 2, np.nan)
    if np.all(np.isfinite(sigma)) and np.all(sigma > 0):
        weights = 1 / sigma

    try:
        params, covariance = np.polyfit(x, y, deg=1, w=weights, cov="unscaled")
    except (np.linalg.LinAlgError, ValueError):
        return (np.nan, np.nan, np.nan)
    return (
        float(params[0]), float(np.sqrt(np.diag(covariance))[0]),
        float(np.exp(params[1])),
    )


# ---------------------------------------------------------------------------
# assembling the file
# ---------------------------------------------------------------------------

def scaling_rows(points, decoder_label=None, fit_points=DEFAULT_FIT_POINTS,
                 max_likelihood_factor=DEFAULT_MAX_LIKELIHOOD_FACTOR,
                 min_shot_fraction=DEFAULT_MIN_SHOT_FRACTION) -> list[dict]:
    """Every row of the interchange file."""
    def interval(shots, errors):
        if shots <= 0:
            return None
        fit = sinter.fit_binomial(
            num_shots=int(shots), num_hits=int(errors),
            max_likelihood_factor=max_likelihood_factor,
        )
        return (fit.low, fit.best, fit.high)

    # One series per experiment, not per distance: an X- and a Z-basis memory
    # at the same d are separate curves, and folding them would fit one
    # exponent through both.
    by_group: dict = {}
    for (decoder, p, d, basis, rounds, noise), entry in points.items():
        label = decoder_label or decoder
        by_group.setdefault((label, d, basis, rounds, noise), []).append((p, entry))

    rows, dropped = [], 0
    for (label, d, basis, rounds, noise), entries in sorted(by_group.items()):
        entries.sort(key=lambda item: item[0])
        series: dict = {"one_pass": [], "alpha_1": [], "alpha_star": []}
        for p, entry in entries:
            alphas = alpha_grid(entry["counts"], min_shot_fraction)
            dropped += len(alpha_grid(entry["counts"], 0.0)) - len(alphas)
            if not alphas:
                continue
            wanted = {"one_pass": None}
            if any(np.isclose(a, BASELINE) for a in alphas):
                wanted["alpha_1"] = min(alphas, key=lambda a: abs(a - BASELINE))
            starred = best_alpha(entry, alphas)
            if starred is not None:
                wanted["alpha_star"] = starred

            for condition, alpha in wanted.items():
                if alpha is None and condition != "one_pass":
                    continue
                if condition == "one_pass":
                    shots = entry["shots"]
                    errors = entry["counts"].get("err_ref", entry["errors"])
                else:
                    shots = alpha_shots(entry, alpha)
                    errors = entry["counts"].get(f"err_a{tag(alpha)}")
                if errors is None or shots <= 0:
                    continue
                lep = interval(shots, errors)
                if lep is None or lep[1] <= 0:
                    continue
                series[condition].append((p, alpha, lep, shots, errors))

        for condition, found in series.items():
            if not found:
                continue
            window = found[:fit_points]
            exponent, error, prefactor = fit_exponent(
                [item[0] for item in window],
                [item[2][1] for item in window],
                [item[2][0] for item in window],
                [item[2][2] for item in window],
            )
            for p, alpha, lep, shots, errors in found:
                rows.append({
                    "decoder": label,
                    "distance": d,
                    "rounds": rounds,
                    "basis": basis,
                    "noise_model": noise,
                    "p": f"{p:.6e}",
                    "condition": condition,
                    "alpha": "" if alpha is None else f"{alpha:.4f}",
                    "P_L": f"{lep[1]:.6e}",
                    "P_L_low": f"{lep[0]:.6e}",
                    "P_L_high": f"{lep[2]:.6e}",
                    "shots": int(shots),
                    "errors": int(errors),
                    "fit_exponent": f"{exponent:.4f}",
                    "fit_exponent_err": f"{error:.4f}",
                    "fit_prefactor": f"{prefactor:.6e}",
                    "fit_p_min": f"{window[0][0]:.6e}",
                    "fit_p_max": f"{window[-1][0]:.6e}",
                    "fit_n_points": len(window),
                    "ideal_exponent": (d + 1) // 2,
                })
    if dropped:
        print(
            f"note: dropped {dropped} under-sampled alpha entries across all "
            f"points - almost certainly an earlier alpha grid folded into the "
            f"same decoder name. See `alpha_grid`.",
            file=sys.stderr,
        )
    rows.sort(key=lambda row: (row["decoder"], row["distance"], row["basis"],
                               row["rounds"], row["noise_model"], row["condition"],
                               float(row["p"])))
    return rows


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def summarise(rows, handle=sys.stdout):
    """The fitted exponents, which are the thing worth reading before plotting."""
    seen: dict = {}
    for row in rows:
        seen.setdefault(
            (row["decoder"], row["distance"], row["basis"], row["condition"]), row,
        )
    header = (
        f"{'decoder':>8} {'d':>3} {'basis':>5} {'window':>21} {'n':>2} "
        f"{'one-pass':>13} {'alpha=1':>13} {'alpha*':>13} {'ideal':>5}"
    )
    print(header, file=handle)
    print("-" * len(header), file=handle)
    for decoder, d, basis in sorted(
        {(row["decoder"], row["distance"], row["basis"]) for row in rows},
    ):
        reference = seen.get((decoder, d, basis, "one_pass"))
        if reference is None:
            continue
        cells = []
        for condition in ("one_pass", "alpha_1", "alpha_star"):
            row = seen.get((decoder, d, basis, condition))
            cells.append(
                f"{float(row['fit_exponent']):.2f}+-{float(row['fit_exponent_err']):.2f}"
                if row else "-",
            )
        print(
            f"{decoder:>8} {d:>3} {basis:>5} {float(reference['fit_p_min']):.2e}-"
            f"{float(reference['fit_p_max']):.2e} {reference['fit_n_points']:>2} "
            f"{cells[0]:>13} {cells[1]:>13} {cells[2]:>13} "
            f"{reference['ideal_exponent']:>5}",
            file=handle,
        )
    print(
        "\nExponent of P_L ~ p^n over the lowest-p window. A decoder preserving "
        "the nominal\ndistance has n = (d+1)/2. The standard rule tends to "
        "n = (d-1)/2 as p -> 0, so a\nvalue below the ideal - and falling as the "
        "window moves down - is the signature.",
        file=handle,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Turn raw alpha-scan CSVs (sinter or RCM grid search) into "
                    "lep_scaling.csv.",
    )
    parser.add_argument("csv", nargs="+",
                        help="sinter output files or RCM grid-search tables")
    parser.add_argument("-o", "--out", type=Path, default=Path("lep_scaling.csv"))
    parser.add_argument(
        "--decoder", default=None,
        help="label for the `decoder` column, e.g. MWPM. Defaults to sinter's "
             "own decoder name, which is usually too long for a legend, or to "
             f"{GRID_SEARCH_DECODER!r} for grid-search tables.",
    )
    parser.add_argument("--fit-points", type=int, default=DEFAULT_FIT_POINTS)
    parser.add_argument(
        "--min-shot-fraction", type=float, default=DEFAULT_MIN_SHOT_FRACTION,
        help="drop an alpha seeing less than this share of the best-sampled "
             "alpha's shots at a point; 0 keeps everything",
    )
    parser.add_argument(
        "--max-likelihood-factor", type=float,
        default=DEFAULT_MAX_LIKELIHOOD_FACTOR,
        help="coverage of the P_L intervals; keep it the same across decoders "
             "plotted together",
    )
    args = parser.parse_args(argv)

    try:
        points = fold(*args.csv)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    rows = scaling_rows(
        points, args.decoder, args.fit_points, args.max_likelihood_factor,
        args.min_shot_fraction,
    )
    if not rows:
        parser.error("no usable points; are these alpha-scan rows?")
    summarise(rows)
    print(f"\n{len(rows)} rows -> {write(args.out, rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
