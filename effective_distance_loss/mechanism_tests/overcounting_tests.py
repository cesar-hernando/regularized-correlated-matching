"""
Tests of three explanations of correlated matching's (CM) effective-distance loss.

Every weight-(d - 1) / 2 fault set that CM (alpha = 1) decodes wrongly and MWPM correctly is
taken apart. With S1 the first-pass matching (correct by construction) and S2 CM's second
pass, both as edge sets over the X and Z graphs, this records per set:

    discounted / prior    how many edges of S1 and S2 CM prices below / at their prior weight
    sources               for each discounted edge mu of S2, the selected edges nu in S1 that
                          discount it (implied p = P(mu and nu) / P(nu), the cheapest wins)
    multi-use             a selected edge whose discount the matching S2 uses on two or more
                          edges at once (two mutually exclusive mechanisms both claimed)
    abandoned source      a discount of S2 whose source nu is no longer in S2
    exclusive price       S1 and S2 repriced so that each selected edge discounts at most one
                          edge (optimal assignment); "strict" also requires the source to stay
                          in the matching. If S2 then costs more than S1, the failure needed the
                          same source twice.
    negative evidence     the implied probabilities corrected for mechanisms with a further
                          component that the first pass did not select;
                          S1 and S2 repriced with them
    alpha_fix             the largest alpha on a grid at which RCM decodes the set correctly,
                          and the alpha at which S1 and S2 would cost the same

and, as controls, the same counts for code-capacity and phenomenological noise, plus
partner-multiplicity statistics of every model's decoding graph.

    python effective_distance_loss/mechanism_tests/overcounting_tests.py --model si1000 --d 5 --p 1e-4
"""

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import stim
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
EDL = HERE.parent
ROOT = EDL.parent
sys.path.insert(0, str(EDL))

import enumerate_faults as faults  # noqa: E402
import failure_examples as fx  # noqa: E402
from RegularizedCM import decomposed_dem  # noqa: E402
import pymatching  # noqa: E402

ALPHAS = np.geomspace(1.0, 1e-3, 61)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def si1000_file(d, p):
    name = f"QMEM_rotated_patch-True_d_X-{d}_d_Z-{d}_rounds-{d}_basis-Z_SI1000_p-{p:.6f}.stim"
    return ROOT / "circuits" / "si1000_circuits" / name


def load_dem(model, d, p):
    """Decomposed DEM of one model. Rotated models use the rotated coordinate rule."""
    if model == "si1000":
        return decomposed_dem(stim.Circuit.from_file(si1000_file(d, p)), "rotated-surface-code-coords")
    if model == "si1000stim":                    # Stim's own decomposition, as a robustness check
        return decomposed_dem(stim.Circuit.from_file(si1000_file(d, p)), "stim")
    if model in ("code_capacity", "phenomenological", "circuit_depolarizing"):
        knobs = {"code_capacity": dict(before_round_data_depolarization=p),
                 "phenomenological": dict(before_round_data_depolarization=p,
                                          before_measure_flip_probability=p),
                 "circuit_depolarizing": dict(before_round_data_depolarization=p,
                                              before_measure_flip_probability=p,
                                              after_reset_flip_probability=p,
                                              after_clifford_depolarization=p)}[model]
        circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d, **knobs)
        return decomposed_dem(circuit, "rotated-surface-code-coords")
    raise ValueError(model)


def context(dem):
    mwpm = pymatching.Matching.from_detector_error_model(dem)
    cm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    errors = [e for e in dem.flattened() if e.type == "error"]
    det, obs, probs = faults.mechanisms(dem)
    comps = [fx.components(e) for e in errors]
    links = collections.defaultdict(list)         # (nu, mu) -> mechanisms containing both
    for m, edges in enumerate(comps):
        for nu in edges:
            for mu in edges:
                if nu != mu:
                    links[(nu, mu)].append(m)
    partners = collections.defaultdict(dict)       # nu -> {mu: P(mu and nu)}
    for (nu, mu), q in fx.joint_probabilities(dem).items():
        partners[nu][mu] = q
    return dict(mwpm=mwpm, cm=cm, det=det, obs=obs, probs=probs, comps=comps, links=links,
                partners=partners, priors=fx.graph_edges(mwpm))


def multiplicity_stats(c):
    """Per edge: distinct correlated partners, and mechanisms it is a component of."""
    n_partners = [len(c["partners"].get(e, {})) for e in c["priors"]]
    n_mech = collections.Counter(e for edges in c["comps"] for e in edges)
    in_hyper = collections.Counter(e for edges in c["comps"] if len(edges) > 1 for e in edges)
    vals = [n_mech[e] for e in c["priors"]]
    hyper = [in_hyper[e] for e in c["priors"]]
    big = sum(1 for edges in c["comps"] if len(edges) > 2)
    return {"edges": len(c["priors"]), "mechanisms": len(c["comps"]),
            "mechanisms_with_2_components": sum(1 for e in c["comps"] if len(e) == 2),
            "mechanisms_with_3plus_components": big,
            "partners_mean": float(np.mean(n_partners)), "partners_median": float(np.median(n_partners)),
            "partners_max": int(max(n_partners)),
            "hyperedges_per_edge_mean": float(np.mean(hyper)),
            "edges_with_2plus_hyperedges": float(np.mean([h >= 2 for h in hyper])),
            "mechanisms_per_edge_mean": float(np.mean(vals))}


# ---------------------------------------------------------------------------
# one failing set
# ---------------------------------------------------------------------------

def implied_from(c, selected, alpha=1.0):
    """{mu: {nu: implied p}} for every edge correlated with a selected edge."""
    out = collections.defaultdict(dict)
    for nu in selected:
        for mu, q in c["partners"].get(nu, {}).items():
            out[mu][nu] = min(alpha * q / c["priors"][nu], fx.P_CAP)
    return out


def w_prior(c, e):
    return fx.weight_of(c["priors"][e])


def cm_cost(c, edges, imp):
    return sum(fx.weight_of(max(imp[e].values())) if e in imp else w_prior(c, e) for e in edges)


def exclusive_cost(c, edges, imp, allowed):
    """Cost when each source in `allowed` may discount at most one edge (best assignment)."""
    edges = list(edges)
    srcs = sorted({nu for e in edges for nu in imp.get(e, {}) if nu in allowed})
    base = sum(w_prior(c, e) for e in edges)
    if not srcs:
        return base
    gain = np.zeros((len(edges), len(srcs)))
    for i, e in enumerate(edges):
        for j, nu in enumerate(srcs):
            if nu in imp.get(e, {}):
                gain[i, j] = max(w_prior(c, e) - fx.weight_of(imp[e][nu]), 0.0)
    r, k = linear_sum_assignment(-gain)
    return base - gain[r, k].sum()


def mechanism_cost(c, edges):
    """Cost of an edge set under the hypergraph model itself, with no first-pass information.

    Edges that are the two components of one hyperedge may be paired and charged once, at
    the weight of P(nu and mu) (one mechanism), instead of twice; every other edge is
    charged its prior weight, as MWPM charges it. The best pairing is a maximum-weight
    matching. This is the cost CM's second pass approximates: a discount is earned only by
    an edge whose partner component is in the same explanation, and each edge is one
    mechanism's component only once.
    """
    import networkx as nx
    edges = list(edges)
    base = sum(w_prior(c, e) for e in edges)
    g = nx.Graph()
    for i, nu in enumerate(edges):
        for mu in edges[i + 1:]:
            q = c["partners"].get(nu, {}).get(mu)
            if q:
                save = w_prior(c, nu) + w_prior(c, mu) - fx.weight_of(q)
                if save > 0:
                    g.add_edge(nu, mu, weight=save)
    pairs = nx.max_weight_matching(g)
    return base - sum(g[a][b]["weight"] for a, b in pairs), len(pairs)


def negative_evidence_imp(c, imp, selected):
    """Implied p after removing mechanisms with a component the first pass did not select."""
    sel = set(selected)
    out = {}
    for mu, by_src in imp.items():
        new = {}
        for nu, q in by_src.items():
            ms = c["links"][(nu, mu)]
            tot = sum(c["probs"][m] for m in ms)
            ok = sum(c["probs"][m] for m in ms
                     if all(x in sel or x == mu for x in c["comps"][m]))
            if ok > 0:
                new[nu] = q * ok / tot
        if new:
            out[mu] = new
    return out


def analyse_set(c, members):
    syn = np.bitwise_xor.reduce(c["det"][list(members)], axis=0)
    truth = np.bitwise_xor.reduce(c["obs"][list(members)], axis=0)
    s1 = fx.matched_edges(c["mwpm"], syn)
    s2 = fx.matched_edges(c["cm"], syn, enable_correlations=True, alpha=1.0)
    imp = implied_from(c, s1)
    rows = np.array([[b, a, fx.weight_of(max(v.values()))] if a == -1 else
                     [a, b, fx.weight_of(max(v.values()))] for (a, b), v in imp.items()],
                    dtype=np.float64).reshape(-1, 3)
    check = fx.matched_edges(c["mwpm"], syn, edge_reweights=rows)

    def disc(edges):
        return [e for e in edges if e in imp and fx.weight_of(max(imp[e].values())) < w_prior(c, e) - 1e-9]

    d1, d2 = disc(s1), disc(s2)
    best = {e: max(imp[e], key=imp[e].get) for e in d2}
    use = collections.Counter(best.values())
    # can the discounts of S2 be given distinct sources at all?
    srcs = sorted({nu for e in d2 for nu in imp[e]})
    distinct_possible = True
    if d2:
        m = np.zeros((len(d2), len(srcs)))
        for i, e in enumerate(d2):
            for j, nu in enumerate(srcs):
                m[i, j] = 1.0 if nu in imp[e] else 0.0
        r, k = linear_sum_assignment(-m)
        distinct_possible = bool(m[r, k].sum() == len(d2))
    neg = negative_evidence_imp(c, imp, s1)

    # alpha at which RCM first decodes correctly, and at which S1 and S2 tie
    alpha_fix = None
    for a in ALPHAS:
        if not np.any(c["cm"].decode(syn, enable_correlations=True, alpha=float(a)) != truth):
            alpha_fix = float(a)
            break

    def cost_alpha(edges, a):
        return sum(fx.weight_of(min(a * max(imp[e].values()), fx.P_CAP)) if e in imp else w_prior(c, e)
                   for e in edges)

    alpha_tie = None
    for a in ALPHAS:
        if cost_alpha(s2, a) >= cost_alpha(s1, a):
            alpha_tie = float(a)
            break

    s1set, s2set = set(s1), set(s2)
    mech1, pairs1 = mechanism_cost(c, s1)
    mech2, pairs2 = mechanism_cost(c, s2)
    true_edges = set()
    for m in members:
        true_edges ^= set(c["comps"][m])
    return {
        "s1_is_true": s1set == true_edges,
        "s1_true_overlap": len(s1set & true_edges), "n_true_edges": len(true_edges),
        "true_hyperedges": sum(1 for m in members if len(c["comps"][m]) > 1),
        "cost_mech_s1": mech1, "cost_mech_s2": mech2,
        "mech_count_s1": len(s1) - pairs1, "mech_count_s2": len(s2) - pairs2,
        "members": [int(x) for x in members],
        "reconstruction_ok": set(check) == s2set,
        "len_s1": len(s1), "len_s2": len(s2),
        "disc_s1": len(d1), "disc_s2": len(d2),
        "prior_s1": len(s1) - len(d1), "prior_s2": len(s2) - len(d2),
        "new_edges_s2": len(s2set - s1set),
        "max_source_use": max(use.values()) if use else 0,
        "distinct_sources_possible": distinct_possible,
        "abandoned_sources": sum(1 for e in d2 if best[e] not in s2set),
        "cost_cm_s1": cm_cost(c, s1, imp), "cost_cm_s2": cm_cost(c, s2, imp),
        "cost_prior_s1": sum(w_prior(c, e) for e in s1), "cost_prior_s2": sum(w_prior(c, e) for e in s2),
        "cost_excl_s1": exclusive_cost(c, s1, imp, s1set),
        "cost_excl_s2": exclusive_cost(c, s2, imp, s1set),
        "cost_strict_s1": exclusive_cost(c, s1, imp, s1set),
        "cost_strict_s2": exclusive_cost(c, s2, imp, s1set & s2set),
        "cost_neg_s1": cm_cost(c, s1, neg), "cost_neg_s2": cm_cost(c, s2, neg),
        "alpha_fix": alpha_fix, "alpha_tie": alpha_tie,
    }


def count_failures(c, weight):
    if weight == 1:
        wm = np.any(c["mwpm"].decode_batch(c["det"]) != c["obs"], axis=1)
        wc = np.any(c["cm"].decode_batch(c["det"], enable_correlations=True, alpha=1.0) != c["obs"], axis=1)
        return [(int(i),) for i in np.flatnonzero(wc & ~wm)], int(wm.sum())
    found = fx.failing_sets(c["det"], c["obs"], c["mwpm"], c["cm"], weight)
    return found, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="si1000")
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--p", type=float, default=1e-4)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    args = ap.parse_args()
    t0 = time.time()
    c = context(load_dem(args.model, args.d, args.p))
    out = {"model": args.model, "d": args.d, "p": args.p, "multiplicity": multiplicity_stats(c)}
    if not args.stats_only:
        sets, _ = count_failures(c, (args.d - 1) // 2)
        out["failing_sets"] = len(sets)
        out["sets"] = [analyse_set(c, s) for s in sets]
    out["seconds"] = round(time.time() - t0, 1)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    name = f"{args.model}_d{args.d}_p{args.p:.0e}.json"
    with open(Path(args.out_dir) / name, "w") as fh:
        json.dump(out, fh, indent=1)
    s = out.get("sets", [])
    print(f"{name}: failing={out.get('failing_sets')} mult={out['multiplicity']}  [{out['seconds']}s]")
    if s:
        n = len(s)
        def frac(f):
            return f"{sum(1 for r in s if f(r))}/{n}"
        print("  reconstruction ok", frac(lambda r: r["reconstruction_ok"]))
        print("  disc_s2 > disc_s1", frac(lambda r: r["disc_s2"] > r["disc_s1"]))
        print("  prior_s2 < prior_s1", frac(lambda r: r["prior_s2"] < r["prior_s1"]),
              " ==", frac(lambda r: r["prior_s2"] == r["prior_s1"]))
        print("  some source used twice", frac(lambda r: r["max_source_use"] >= 2),
              " distinct sources impossible", frac(lambda r: not r["distinct_sources_possible"]))
        print("  abandoned source", frac(lambda r: r["abandoned_sources"] > 0))
        print("  fixed by exclusive", frac(lambda r: r["cost_excl_s2"] > r["cost_excl_s1"]),
              " strict", frac(lambda r: r["cost_strict_s2"] > r["cost_strict_s1"]),
              " negative evidence", frac(lambda r: r["cost_neg_s2"] > r["cost_neg_s1"]))
        print("  fixed by mechanism pricing", frac(lambda r: r["cost_mech_s2"] > r["cost_mech_s1"]),
              " mechanisms s1/s2:", collections.Counter((r["mech_count_s1"], r["mech_count_s2"])
                                                        for r in s).most_common(6))
        print("  first pass = true components", frac(lambda r: r["s1_is_true"]))
        for name, test in (("strict", lambda r: r["cost_strict_s2"] > r["cost_strict_s1"]),
                           ("mechanism", lambda r: r["cost_mech_s2"] > r["cost_mech_s1"])):
            fixed = [r for r in s if test(r)]
            un = [r for r in s if not test(r)]
            print(f"  {name}: first pass true among fixed {sum(r['s1_is_true'] for r in fixed)}/{len(fixed)},"
                  f" among unfixed {sum(r['s1_is_true'] for r in un)}/{len(un)};"
                  f" unfixed true hyperedges {collections.Counter(r['true_hyperedges'] for r in un)}")
        unfixed = [r for r in s if r["cost_strict_s2"] <= r["cost_strict_s1"]]
        print("  not fixed by strict: disc s1/s2, prior s1/s2:",
              collections.Counter((r["disc_s1"], r["disc_s2"], r["prior_s1"], r["prior_s2"])
                                  for r in unfixed).most_common(6))
        k = [(r["disc_s2"] - r["disc_s1"], r["alpha_fix"]) for r in s if r["alpha_fix"]]
        for dk in sorted({a for a, _ in k}):
            v = [b for a, b in k if a == dk]
            print(f"  extra discounted edges {dk:+d}: n={len(v)} median alpha_fix={np.median(v):.3f}")
        both = [(r["alpha_fix"], r["alpha_tie"]) for r in s if r["alpha_fix"] and r["alpha_tie"]]
        if both:
            print("  alpha_fix vs alpha_tie: log-corr",
                  round(float(np.corrcoef(np.log([a for a, _ in both]), np.log([b for _, b in both]))[0, 1]), 3),
                  " none fixed by any alpha:", sum(1 for r in s if r["alpha_fix"] is None))


if __name__ == "__main__":
    main()
