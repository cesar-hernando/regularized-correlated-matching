"""
Ratio curves with intervals built from per-decoder binomial fits, for the _v3 variant of the two-decoder ratio
figure: each decoder's logical error probability gets sinter's likelihood-ratio interval at maximum likelihood
factor 10 (96.8%; as in make_scaling_csv.py), and the two are combined in log space,

    rho_- = rho exp(-sqrt(ln(P_2 / P_2,low)^2 + ln(P_1,high / P_1)^2)),
    rho_+ = rho exp(+sqrt(ln(P_2,high / P_2)^2 + ln(P_1 / P_1,low)^2)),

with P_2 the two-pass decoder's rate and P_1 the one-pass reference's. The two fits are treated as independent:
this ignores the covariance from both decoders decoding the same shots.
`ratio` is kept as it is in the input; only `ratio_low` and `ratio_high` are replaced.

    python figures/binomial_ratio_intervals.py IN.csv OUT.csv
"""
import argparse
import csv
import math

import sinter

MAX_LIKELIHOOD_FACTOR = 10.0


def fit(shots, errors):
    """(low, best, high) of a logical error probability, as make_scaling_csv.py computes it."""
    result = sinter.fit_binomial(num_shots=int(shots), num_hits=int(errors),
                                 max_likelihood_factor=MAX_LIKELIHOOD_FACTOR)
    return result.low, result.best, result.high


def ratio_interval(rho, shots, errors, reference_errors):
    low2, best2, high2 = fit(shots, errors)
    low1, best1, high1 = fit(shots, reference_errors)
    down = math.hypot(math.log(best2 / low2), math.log(high1 / best1))
    up = math.hypot(math.log(high2 / best2), math.log(best1 / low1))
    return rho * math.exp(-down), rho * math.exp(up)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args()
    with open(args.source, newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        shots, e2, e1 = float(row["shots"]), float(row["errors"]), float(row["errors_reference"])
        if e2 > 0 and e1 > 0:
            row["ratio_low"], row["ratio_high"] = ratio_interval(float(row["ratio"]), shots, e2, e1)
    with open(args.target, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.target} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
