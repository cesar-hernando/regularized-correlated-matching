"""
Posterior edge weights of correlated matching (CM, alpha = 1) and regularized correlated
matching (RCM, alpha*), against p, at the points where alpha* is known.

When correlated matching's first pass selects an edge nu, every edge mu correlated with it
through a hyperedge of the detector error model gets the posterior weight
    w_post = log((1 - p') / p'),   p' = min(alpha P(mu | nu), 1/2),   P(mu | nu) = P(mu and nu) / P(nu),
where P(mu and nu) is the probability that one error flips both edges and P(nu) the edge's
prior probability. The posterior weight is computed for every correlated pair (nu, mu) of
the decoding graph, and its mean and median over all pairs are reported, next to the mean
and median prior weight w_prior = log((1 - p) / p) over all edges (with the rule of the installed
PyMatching fork).

The rule is checked against the decoder at every point: on sampled shots, MWPM decoding with
the recomputed reweights of the first-pass edges must select the same edges as CM and RCM.

Outputs, in this folder: `posterior_weights.csv` and `posterior_weights_vs_p.pdf`.

Example:
    python effective_distance_loss/posterior_weights.py --distances 3 5
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "figures"))

import enumerate_faults as faults  # noqa: E402
import failure_examples as examples  # noqa: E402
from lcd_figures import PAGE_WIDTH, USE_LATEX, paper_style, save  # noqa: E402
from RegularizedCM import build_decoders, check_pymatching_version, decomposed_dem  # noqa: E402

FIELDS = ("distance", "rounds", "p", "alpha_star", "edges", "correlated_pairs", "prior_mean", "prior_median",
          "cm_mean", "cm_median", "rcm_mean", "rcm_median", "checked_shots", "cm_reproduced", "rcm_reproduced")
SERIES = {
    # column prefix: (label, colour)
    "prior": ("prior", "#555555"),
    "cm": ("CM ($\\alpha = 1$)", "#dc4405"),
    "rcm": ("RCM ($\\alpha^*$)", "#006f62"),
}


def posterior_weights(priors, joint, alpha):
    """The posterior weight of every correlated pair (nu, mu), as an array."""
    return np.array([examples.weight_of(alpha * probability / priors[nu])
                     for (nu, mu), probability in joint.items() if nu in priors and priors[nu] > 0])


def reproduced(circuit, mwpm, cm, priors, joint, alpha, shots, seed):
    """Fraction of sampled shots with detections where the recomputed rule gives the decoder's edges."""
    detections = circuit.compile_detector_sampler(seed=seed).sample(shots)
    detections = detections[detections.any(axis=1)]
    agree = 0
    for syndrome in detections.astype(np.uint8):
        first = examples.matched_edges(mwpm, syndrome)
        new = examples.reweights(first, priors, joint, alpha)
        rows = np.array([[b, a, w] if a == -1 else [a, b, w] for (a, b), (_, w) in new.items()],
                        dtype=np.float64).reshape(-1, 3)
        mine = examples.matched_edges(mwpm, syndrome, edge_reweights=rows)
        theirs = examples.matched_edges(cm, syndrome, enable_correlations=True, alpha=alpha)
        agree += sorted(mine) == sorted(theirs)
    return agree / max(len(detections), 1), len(detections)


def plot(rows, path):
    distances = sorted({int(r["distance"]) for r in rows})
    figure, axes = plt.subplots(1, len(distances), squeeze=False, layout="constrained", figsize=(PAGE_WIDTH, 2.5))
    for ax, d in zip(axes[0], distances):
        points = sorted((r for r in rows if int(r["distance"]) == d), key=lambda r: float(r["p"]))
        p = [float(r["p"]) for r in points]
        for prefix, (label, color) in SERIES.items():
            # the mean alone: the median tracks it closely and doubles the key
            ax.plot(p, [float(r[f"{prefix}_mean"]) for r in points], color=color, lw=0.8, marker="o",
                    markersize=2.5, label=label)
        ax.set_xscale("log")
        # a panel spanning barely a decade (d = 7) otherwise labels its minor ticks too,
        # and those labels run into the decades
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_ylim(bottom=0)
        ax.grid(visible=True, linewidth=0.4, alpha=0.5)
        ax.set_title(f"$d={d}$", pad=3)
        ax.set_xlabel("Physical error rate, $p$")
    axes[0, 0].set_ylabel("Edge weight (nats)")
    # the key above the panels, a column per series, so it covers none of the curves
    handles, labels = axes[0, -1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=len(SERIES),
                  fontsize=6.5, handlelength=2.2, labelspacing=0.2, columnspacing=1.6,
                  frameon=False)
    save(figure, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+",
                        default=[faults.ROOT / "tasks/rcm_grid_search_tasks_si1000.txt"],
                        help="task lists naming the circuits; give several to cover the points "
                             "split across them (the d = 7 low-p points are in the extension)")
    parser.add_argument("--alpha-star", default=faults.ROOT / "data/mwpm/rcm_grid_search_si1000/analysis/alpha_star_vs_p.csv")
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--check-shots", type=int, default=2000, help="sampled shots per point for the rule check")
    parser.add_argument("--out-dir", default=HERE)
    args = parser.parse_args()
    check_pymatching_version()

    alpha_star = faults.read_alpha_star(args.alpha_star)
    listed = [t for path in args.tasks for t in faults.files.read_tasks(path)]
    tasks = sorted((t for t in listed
                    if t.distance in args.distances and faults.find(alpha_star, t.distance, t.p) is not None),
                   key=lambda t: (t.distance, t.p))
    rows = []
    for task in tasks:
        a_star = faults.find(alpha_star, task.distance, task.p)
        circuit, _, _ = faults.files.load_circuit(task.code, task.circuit, task.noise_model, task.distance,
                                                  task.rounds, task.p, task.basis)
        dem = decomposed_dem(circuit, task.decomposition)
        mwpm, cm = build_decoders(circuit, task.decomposition)
        priors = examples.graph_edges(mwpm)
        joint = examples.joint_probabilities(dem)
        prior = np.array([examples.weight_of(q) for q in priors.values() if q > 0])
        cm_weights, rcm_weights = posterior_weights(priors, joint, 1.0), posterior_weights(priors, joint, a_star)
        seed = task.distance * 10 ** 12 + round(task.p * 1e12)   # one reproducible seed per point
        cm_ok, shots = reproduced(circuit, mwpm, cm, priors, joint, 1.0, args.check_shots, seed)
        rcm_ok, _ = reproduced(circuit, mwpm, cm, priors, joint, a_star, args.check_shots, seed)
        rows.append({
            "distance": task.distance, "rounds": task.rounds, "p": task.p, "alpha_star": a_star,
            "edges": len(prior), "correlated_pairs": len(cm_weights),
            "prior_mean": prior.mean(), "prior_median": np.median(prior),
            "cm_mean": cm_weights.mean(), "cm_median": np.median(cm_weights),
            "rcm_mean": rcm_weights.mean(), "rcm_median": np.median(rcm_weights),
            "checked_shots": shots, "cm_reproduced": cm_ok, "rcm_reproduced": rcm_ok,
        })
        r = rows[-1]
        print(f"d={task.distance} p={task.p:.2e} alpha*={a_star:g}: prior {r['prior_mean']:.2f}/{r['prior_median']:.2f}, "
              f"CM {r['cm_mean']:.2f}/{r['cm_median']:.2f}, RCM {r['rcm_mean']:.2f}/{r['rcm_median']:.2f} "
              f"(mean/median); rule reproduces CM on {cm_ok:.1%}, RCM on {rcm_ok:.1%} of {shots} shots", flush=True)

    out = Path(args.out_dir)
    with open(out / "posterior_weights.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with paper_style(use_latex=USE_LATEX):
        plot(rows, out / "posterior_weights_vs_p.pdf")


if __name__ == "__main__":
    main()
