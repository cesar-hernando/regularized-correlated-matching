"""
Does the correlated decoder keep the code's distance? Fit the exponent and say.

Standalone: numpy and the standard library. It reads the `lep_scaling.csv`
interchange schema that `make_scaling_csv.py` writes, so it runs on any decoder
family that can produce one:

    uv run figures/exponents.py data/lcd/lep_scaling.csv \\
        data/mwpm/rcm_grid_search_si1000/analysis/lep_scaling.csv -o exponents.csv

The output is a plain CSV, one row per (decoder, basis, distance), with the
columns of the exponent table:

    decoder, basis, distance
    n_points, p_min, p_max         the window: the lowest n sampled p
    one_pass, one_pass_err         exponent n of P_L ~ p^n, and its error
    alpha_1, alpha_1_err
    alpha_star, alpha_star_err
    ideal_exponent                 (d+1)/2, the distance-preserving value

It goes to stdout unless `-o` is given. The two diagnostics below are opt-in
and are written to stderr, so they never mix into the CSV.

A decoder preserving the nominal distance of a code has

    P_L ~ p^((d+1)/2)                                        as p -> 0,

because every configuration of at most (d-1)/2 faults is corrected. A decoder
that fails on some (d-1)/2-fault configurations opens a p^((d-1)/2) term and
tends to that lower exponent instead. The exponent is therefore a direct,
model-free test of effective distance: no fault enumeration, no importance
sampling, nothing to take on trust but a straight line through log P_L against
log p.

The table is the same one `make_scaling_csv.py` reports when it builds the
file. What is here and not there are the two diagnostics
that decide whether that table means anything:

**Window sensitivity** (`--windows N ...`). The fit covers the lowest n points at
each distance, and n is a choice. Re-running the fit over a range of n settles
whether the result is a property of the data or of the choice. The expected
signature, if the mechanism is real, is asymmetric: the one-pass and alpha*
exponents should barely move, because both decoders are already in their
asymptotic regime and there is nothing left for the window to change; the
alpha=1 exponent should move monotonically, because it is mid-crossover
between p^((d+1)/2) and p^((d-1)/2) and a window reaching higher in p captures
more of the former. A flat alpha=1 column would mean the deficit is an artefact.

**Local slope** (`--local CONDITION ...`). The same question asked the other way: fit a short
sliding window and watch the exponent as the window walks down in p. A
finite-range artefact has no reason to trend; a genuine crossover must fall
monotonically toward (d-1)/2. This is the stronger of the two, and the cheaper
to explain in a caption.

Nothing here re-reads raw shots: every P_L and interval comes from the CSV, so
the numbers are the same ones the figures are drawn from.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


DEFAULT_FIT_POINTS = 6
"""Points per distance, counted up from the lowest p, for the headline table.
Matches `make_scaling_csv.py`, so the two agree by construction."""

DEFAULT_WINDOWS = (4, 5, 6, 7, 8, 10)
"""Window sizes for the sensitivity scan."""

DEFAULT_LOCAL_WIDTH = 4
"""Points in each sliding window. Four is the fewest that still has two degrees
of freedom after a straight line, so the trend is not drawn through noise."""

CONDITIONS = ("one_pass", "alpha_1", "alpha_star")
LABELS = {
    "one_pass": "one-pass",
    "alpha_1": "alpha=1",
    "alpha_star": "alpha*",
}


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_scaling(*paths) -> list[dict]:
    """Read `lep_scaling.csv` files into typed rows, concatenated."""
    rows = []
    for path in paths:
        with Path(path).open(newline="") as handle:
            for raw in csv.DictReader(handle):
                if not raw.get("P_L"):
                    continue
                rows.append({
                    "decoder": (raw.get("decoder") or "").strip() or "decoder",
                    "basis": (raw.get("basis") or "").strip(),
                    "distance": int(float(raw["distance"])),
                    "p": float(raw["p"]),
                    "condition": (raw.get("condition") or "").strip(),
                    "P_L": float(raw["P_L"]),
                    "P_L_low": _float(raw.get("P_L_low")),
                    "P_L_high": _float(raw.get("P_L_high")),
                    "ideal_exponent": _float(raw.get("ideal_exponent")),
                })
    if not rows:
        msg = "no usable rows; expected the lep_scaling.csv schema"
        raise ValueError(msg)
    return rows


def experiments(rows) -> list[tuple]:
    """Every (decoder, basis) in the rows: each is one table of distances."""
    return sorted({(row["decoder"], row["basis"]) for row in rows})


def distances(rows, decoder, basis) -> list[int]:
    return sorted({row["distance"] for row in rows
                   if row["decoder"] == decoder and row["basis"] == basis})


def series(rows, decoder, basis, distance, condition) -> list[dict]:
    """One decoder's P_L against p at one basis and distance, ascending in p."""
    return sorted(
        (row for row in rows
         if row["decoder"] == decoder and row["basis"] == basis
         and row["distance"] == distance and row["condition"] == condition),
        key=lambda row: row["p"],
    )


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------

def fit_exponent(points):
    """
    (exponent, error, prefactor) of a weighted log-log straight line.

    Weights are the per-point confidence intervals as a standard deviation on
    log P_L, and the covariance is taken unscaled, so the quoted error is
    propagated from those intervals rather than rescaled by the residual - a
    fit through four points has too few degrees of freedom for the residual to
    be worth trusting. NaNs rather than an exception when the fit cannot be
    made, so one thin distance does not take the table down.
    """
    p_values = np.array([row["p"] for row in points], dtype=float)
    leps = np.array([row["P_L"] for row in points], dtype=float)
    usable = (p_values > 0) & (leps > 0) & np.isfinite(leps)
    if usable.sum() < 3:
        return (np.nan, np.nan, np.nan)
    x, y = np.log(p_values[usable]), np.log(leps[usable])

    weights = None
    lows = np.array([row["P_L_low"] or np.nan for row in points])[usable]
    highs = np.array([row["P_L_high"] or np.nan for row in points])[usable]
    sigma = np.where(
        (lows > 0) & (highs > 0), (np.log(highs) - np.log(lows)) / 2, np.nan,
    )
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


def exponent_table(rows, decoder, basis, n=DEFAULT_FIT_POINTS) -> list[dict]:
    """One entry per distance: the three exponents over the lowest n points."""
    out = []
    for d in distances(rows, decoder, basis):
        reference = series(rows, decoder, basis, d, "one_pass")[:n]
        if len(reference) < 3:
            continue
        entry = {
            "decoder": decoder, "basis": basis, "distance": d, "n": len(reference),
            "p_min": reference[0]["p"], "p_max": reference[-1]["p"],
            "ideal": reference[0]["ideal_exponent"] or (d + 1) // 2,
        }
        for condition in CONDITIONS:
            points = series(rows, decoder, basis, d, condition)[:n]
            exponent, error, _ = fit_exponent(points) if points else (np.nan,) * 3
            entry[condition] = (exponent, error)
        out.append(entry)
    return out


def window_scan(rows, decoder, basis, windows=DEFAULT_WINDOWS) -> dict:
    """{(distance, condition): [(n, exponent, error)]} over a range of windows."""
    out: dict = {}
    for d in distances(rows, decoder, basis):
        for condition in CONDITIONS:
            found = []
            for n in windows:
                points = series(rows, decoder, basis, d, condition)[:n]
                if len(points) < max(3, n):
                    continue
                exponent, error, _ = fit_exponent(points)
                found.append((n, exponent, error))
            if found:
                out[(d, condition)] = found
    return out


def local_slopes(rows, decoder, basis, distance, condition,
                 width=DEFAULT_LOCAL_WIDTH) -> list[dict]:
    """Exponent in each sliding window of `width` points, ascending in p."""
    points = series(rows, decoder, basis, distance, condition)
    out = []
    for start in range(len(points) - width + 1):
        window = points[start:start + width]
        exponent, error, _ = fit_exponent(window)
        out.append({
            "p_min": window[0]["p"], "p_max": window[-1]["p"],
            "centre": float(np.sqrt(window[0]["p"] * window[-1]["p"])),
            "exponent": exponent, "error": error,
        })
    return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def diagnostics(rows, decoder, basis, windows, local, local_width, handle=sys.stderr):
    """The two diagnostics, as readable text, for one (decoder, basis)."""
    def say(text=""):
        print(text, file=handle)

    say(f"=== {decoder}, {basis or 'unknown'} basis ===")
    if windows:
        say("window sensitivity: the same fit over the lowest n points")
        say("   A real crossover moves the alpha=1 column and leaves the other")
        say("   two alone. If all three move together, the window is doing it.")
        scan = window_scan(rows, decoder, basis, windows)
        say(f"   {'d':>3} {'condition':>10} " +
            " ".join(f"{'n=' + str(w):>8}" for w in windows))
        for d in sorted({key[0] for key in scan}):
            for condition in CONDITIONS:
                found = {n: exponent for n, exponent, _ in scan.get((d, condition), [])}
                if not found:
                    continue
                cells = " ".join(
                    f"{found[w]:>8.2f}" if w in found else f"{'-':>8}" for w in windows
                )
                say(f"   {d:>3} {LABELS[condition]:>10} {cells}")
            spread = scan.get((d, "alpha_1"), [])
            if len(spread) > 1:
                values = [exponent for _, exponent, _ in spread]
                say(f"   {'':>3} {'-> alpha=1 moves by':>10} "
                    f"{max(values) - min(values):+.2f} over that range")
        say()

    if local:
        say(f"local slope, sliding {local_width}-point window")
        say("   A finite-range artefact has no reason to trend. A crossover to")
        say("   p^((d-1)/2) must fall monotonically as the window walks down.")
        for d in distances(rows, decoder, basis):
            for condition in local:
                found = local_slopes(rows, decoder, basis, d, condition, local_width)
                if len(found) < 2:
                    continue
                say(f"   d={d}, {LABELS[condition]}  "
                    f"(ideal {(d + 1) // 2}, broken {(d - 1) // 2})")
                say("     window centre: " + " ".join(
                    f"{entry['centre']:>8.1e}" for entry in found))
                say("     exponent:      " + " ".join(
                    f"{entry['exponent']:>8.2f}" for entry in found))
                trend = found[-1]["exponent"] - found[0]["exponent"]
                say(f"     -> {trend:+.2f} from the lowest window to the highest")
        say()


FIELDS = [
    "decoder", "basis", "distance", "n_points", "p_min", "p_max",
    "one_pass", "one_pass_err", "alpha_1", "alpha_1_err",
    "alpha_star", "alpha_star_err", "ideal_exponent",
]


def _number(value, spec):
    """A formatted cell, blank where the fit could not be made."""
    return f"{value:{spec}}" if np.isfinite(value) else ""


def write_csv(handle, table):
    """The exponent table, one row per (decoder, basis, distance)."""
    writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for entry in table:
        row = {
            "decoder": entry["decoder"], "basis": entry["basis"],
            "distance": entry["distance"], "n_points": entry["n"],
            "p_min": f"{entry['p_min']:.6e}", "p_max": f"{entry['p_max']:.6e}",
            "ideal_exponent": f"{entry['ideal']:.0f}",
        }
        for condition in CONDITIONS:
            exponent, error = entry[condition]
            row[condition] = _number(exponent, ".4f")
            row[f"{condition}_err"] = _number(error, ".4f")
        writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit the decay exponent of P_L and test it against the "
                    "distance-preserving value. Writes the table as CSV.",
    )
    parser.add_argument("csv", nargs="+", help="lep_scaling.csv files, any order")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="output CSV; stdout if omitted")
    parser.add_argument("-n", "--fit-points", type=int, default=DEFAULT_FIT_POINTS)
    parser.add_argument(
        "--windows", type=int, nargs="*", default=[],
        help="window sizes for the sensitivity scan, reported on stderr; "
             f"give the flag alone for {' '.join(map(str, DEFAULT_WINDOWS))}",
    )
    parser.add_argument(
        "--local", nargs="*", default=[], choices=CONDITIONS,
        help="conditions to report a sliding local slope for, on stderr; "
             "give the flag alone for alpha_1",
    )
    parser.add_argument("--local-width", type=int, default=DEFAULT_LOCAL_WIDTH)
    args = parser.parse_args(argv)
    # a bare flag asks for the defaults
    argv = sys.argv[1:] if argv is None else argv
    if "--windows" in argv and not args.windows:
        args.windows = list(DEFAULT_WINDOWS)
    if "--local" in argv and not args.local:
        args.local = ["alpha_1"]

    try:
        rows = read_scaling(*args.csv)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    table = []
    for decoder, basis in experiments(rows):
        table += exponent_table(rows, decoder, basis, args.fit_points)
        if args.windows or args.local:
            diagnostics(rows, decoder, basis, args.windows, args.local, args.local_width)
    if not table:
        parser.error("no distance has three one-pass points to fit")

    if args.out is None:
        write_csv(sys.stdout, table)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as handle:
            write_csv(handle, table)
        print(f"{len(table)} rows -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
