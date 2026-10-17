"""
Worked failures of correlated matching (CM) at d = 3, 5, 7: data and figures.

For each distance, one failing set (CM wrong, MWPM right) of each failure mode is taken from the
failing sets in `failing_sets/` (SI1000, p = 1e-4; exhaustive at d = 3, 5, sampled at d = 7):

    reuse      one first-pass edge discounts two or more edges of CM's matching through
               different (mutually exclusive) hyperedges
    abandoned  the second pass keeps a discount whose first-pass source it has dropped
    phantom    the first pass is logically right but selects edges other than the faults'

and described completely: the mechanisms (circuit fault, tick, Paulis, detectors), the first pass,
the reweighting (source edge and linking hyperedge of every discount), CM's second pass, RCM at
alpha*, the price of both matchings under prior, CM, RCM and mechanism-level weights, and the
unpaid discounts. Writes examples.json and fig_<d>_<mode>.pdf (PRL two-column width).

    python effective_distance_loss/failure_gallery/make_gallery.py
"""
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np
import stim

HERE = Path(__file__).resolve().parent
EDL = HERE.parent
sys.path.insert(0, str(EDL))
sys.path.insert(0, str(EDL / "mechanism_tests"))
sys.path.insert(0, str(EDL.parent / "figures"))

import enumerate_faults as faults  # noqa: E402
import failure_examples as fx  # noqa: E402
import overcounting_tests as ot  # noqa: E402
from RegularizedCM import decomposed_dem  # noqa: E402

P = 1e-4
ALPHA_STAR = {3: 0.2557, 5: 0.1732, 7: 0.2557}
RESULTS = {3: "si1000_d3_w1_p1.0e-04_exhaustive.json", 5: "si1000_d5_w2_p1.0e-04_exhaustive.json",
           7: "si1000_d7_w3_p1.0e-04_slab.json"}
MODES = ("reuse", "abandoned", "phantom")


def classify(r):
    if r["cost_excl_s2"] > r["cost_excl_s1"]:
        return "reuse"
    if r["cost_strict_s2"] > r["cost_strict_s1"]:
        return "abandoned"
    return "phantom" if not r["s1_is_true"] else "other"


def pick(rows, mode):
    """The clearest failing set of a mode: exact first pass (except phantom), fewest edges."""
    cand = [r for r in rows if classify(r) == mode and (mode == "phantom" or r["s1_is_true"])]
    if mode == "reuse":
        pure = [r for r in cand if r["abandoned_sources"] == 0]
        cand = pure or cand
    cand.sort(key=lambda r: (r["len_s2"] + r["len_s1"], r["members"]))
    return cand[0] if cand else None


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------

def context(d, decomposition="rotated-surface-code-coords"):
    circuit = stim.Circuit.from_file(ot.si1000_file(d, P))
    dem = decomposed_dem(circuit, decomposition)
    c = ot.context(dem)
    sources, shapes = faults.mechanism_labels(circuit, dem, P)
    coords = circuit.get_detector_coordinates()
    c.update(circuit=circuit, dem=dem, sources=sources, shapes=shapes, coords=coords,
             bases=fx.detector_bases(circuit, coords), reps=fx.representative_faults(circuit, P),
             errors=[e for e in dem.flattened() if e.type == "error"])
    return c


def det_text(c, k):
    x, y, t = c["coords"][k][:3]
    return f"D{k} ({x:g},{y:g},{t:g}, {'XZ'[c['bases'][k]]})"


def edge_text(c, e):
    a, b = e
    return f"D{b} - boundary" if a == -1 else f"D{a} - D{b}"


def mech_label(c, m):
    return f"{c['sources'][m]}, {c['shapes'][m]}"


def fault_text(c, m):
    rep = c["reps"].get(faults.symptoms(c["errors"][m].targets_copy()))
    if not rep:
        return "-"
    eq = f" (1 of {rep['equivalent_faults']} equivalent)" if rep["equivalent_faults"] > 1 else ""
    return f"{rep['stim_channel']} at tick {rep['tick']}: {'; '.join(rep['paulis'])}{eq}"


# ---------------------------------------------------------------------------
# one example
# ---------------------------------------------------------------------------

def describe(c, members, d):
    a_star = ALPHA_STAR[d]
    syn = np.bitwise_xor.reduce(c["det"][list(members)], axis=0)
    truth = np.bitwise_xor.reduce(c["obs"][list(members)], axis=0)
    s1 = fx.matched_edges(c["mwpm"], syn)
    s2 = fx.matched_edges(c["cm"], syn, enable_correlations=True, alpha=1.0)
    rcm = fx.matched_edges(c["cm"], syn, enable_correlations=True, alpha=a_star)
    imp = ot.implied_from(c, s1)                         # mu -> {nu: P(mu | nu)}
    true_edges = set()
    for m in members:
        true_edges ^= set(c["comps"][m])

    def w_cm(e):
        return fx.weight_of(max(imp[e].values())) if e in imp else ot.w_prior(c, e)

    def w_rcm(e):
        return fx.weight_of(min(a_star * max(imp[e].values()), fx.P_CAP)) if e in imp else ot.w_prior(c, e)

    def discount(e):
        if e not in imp or w_cm(e) >= ot.w_prior(c, e) - 1e-9:
            return None
        nu = max(imp[e], key=imp[e].get)
        links = c["links"][(nu, e)]
        return {"source": list(nu), "implied_p": imp[e][nu],
                "links": [{"mechanism": int(m), "p": float(c["probs"][m]), "label": mech_label(c, m),
                           "fault": fault_text(c, m), "true_fault": int(m) in members} for m in links]}

    def rows(edges, kind):
        out = []
        for e in edges:
            out.append({"edge": list(e), "text": edge_text(c, e), "w_prior": ot.w_prior(c, e),
                        "w_cm": w_cm(e), "w_rcm": w_rcm(e), "in_first_pass": e in s1, "true_component": e in true_edges,
                        "discount": discount(e) if kind != "first" else None})
        return out

    first = rows(s1, "first")
    second = rows(s2, "second")
    rcm_rows = rows(rcm, "rcm")
    # unpaid discounts in CM's matching
    groups = collections.defaultdict(list)
    for r in second:
        if r["discount"]:
            groups[tuple(r["discount"]["source"])].append(r)
    reused = {nu: rs for nu, rs in groups.items() if len(rs) >= 2}
    abandoned = [r for r in second if r["discount"] and tuple(r["discount"]["source"]) not in set(s2)]
    phantom_edges = [e for e in s1 if e not in true_edges]

    def tot(rs, key):
        return float(sum(r[key] for r in rs))

    mech1, _ = ot.mechanism_cost(c, s1)
    mech2, _ = ot.mechanism_cost(c, s2)
    true_list = [e for e in true_edges if e in c["priors"]]
    mech_true, _ = ot.mechanism_cost(c, true_list)
    imp_s1_true = {e: imp[e] for e in true_list if e in imp}
    cm_true = sum(fx.weight_of(max(imp_s1_true[e].values())) if e in imp_s1_true else ot.w_prior(c, e)
                  for e in true_list)
    # oracle: CM's second pass with discounts computed from the true fault components
    imp_true = ot.implied_from(c, [e for e in true_edges if e in c["priors"]])
    orows = np.array([[b, a, fx.weight_of(max(v.values()))] if a == -1 else [a, b, fx.weight_of(max(v.values()))]
                      for (a, b), v in imp_true.items()], dtype=np.float64).reshape(-1, 3)
    oracle_wrong = int(np.any(c["mwpm"].decode(syn, edge_reweights=orows) != truth))
    pred = {"MWPM": int(np.any(c["mwpm"].decode(syn) != truth)),
            "CM": int(np.any(c["cm"].decode(syn, enable_correlations=True, alpha=1.0) != truth)),
            "RCM": int(np.any(c["cm"].decode(syn, enable_correlations=True, alpha=a_star) != truth))}
    mechanisms = []
    for m in members:
        mechanisms.append({
            "index": int(m), "p": float(c["probs"][m]), "label": mech_label(c, m), "fault": fault_text(c, m),
            "detectors": [det_text(c, k) for k in np.flatnonzero(c["det"][m])],
            "components": [edge_text(c, e) for e in c["comps"][m]],
            "flips_observable": int(c["obs"][m].any())})
    return {
        "d": d, "p": P, "alpha_star": a_star, "members": [int(m) for m in members], "mechanisms": mechanisms,
        "syndrome": [det_text(c, k) for k in np.flatnonzero(syn)], "flips_observable": int(truth.any()),
        "first_pass": first, "second_pass": second, "rcm": rcm_rows,
        "true_edges": [list(e) for e in sorted(true_edges)],
        "totals": {"S1": {"prior": tot(first, "w_prior"), "cm": tot(first, "w_cm"), "rcm": tot(first, "w_rcm"),
                          "mechanism": mech1},
                   "S2": {"prior": tot(second, "w_prior"), "cm": tot(second, "w_cm"), "rcm": tot(second, "w_rcm"),
                          "mechanism": mech2},
                   "RCM": {"rcm": tot(rcm_rows, "w_rcm")},
                   "truth": {"prior": float(sum(ot.w_prior(c, e) for e in true_list)), "cm": cm_true,
                             "mechanism": mech_true}},
        "wrong": pred, "oracle_wrong": oracle_wrong,
        "reused": [{"source": list(nu), "source_text": edge_text(c, nu),
                    "edges": [r["text"] for r in rs],
                    "mechanisms": [l for r in rs for l in r["discount"]["links"]]} for nu, rs in reused.items()],
        "abandoned": [{"edge": r["text"], "source": edge_text(c, tuple(r["discount"]["source"]))} for r in abandoned],
        "phantom_edges": [edge_text(c, e) for e in phantom_edges],
        "phantom_edge_mechanisms": {edge_text(c, e): sorted({mech_label(c, m) for m in range(len(c["comps"]))
                                                             if e in c["comps"][m] and len(c["comps"][m]) > 1})
                                    for e in phantom_edges},
        "s1": [list(e) for e in s1], "s2": [list(e) for e in s2], "rcm_edges": [list(e) for e in rcm],
        # every edge the first pass reweights, for the reweighting panels
        "reweights": [{"edge": list(e), "w_prior": ot.w_prior(c, e), "w_cm": w_cm(e), "w_rcm": w_rcm(e)}
                      for e in imp if e in c["priors"]],
    }



def fault_class(c, members):
    return " + ".join(sorted(mech_label(c, m) for m in members))


def extra_picks(c, rows, chosen):
    """Additional sets (d = 5, and d = 3 pairs): for each fault class (most common first), one set per failure mode."""
    by_class = collections.defaultdict(list)
    for r in rows:
        by_class[fault_class(c, r["members"])].append(r)
    picks = []
    for cls, rs in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        for mode in MODES:
            r = pick(rs, mode)
            if r and tuple(r["members"]) not in chosen:
                picks.append((mode, r, cls, len(rs)))
                chosen.add(tuple(r["members"]))
    return picks


def d3_more(n_extra=3):
    """Further d = 3 examples, at weight (d + 1)/2 = 2 (CM fails on only two single faults, examples 1
    and 8): one failing pair per failure mode, then `n_extra` more across fault classes (most common
    first). CM fails on 1214 pairs that MWPM corrects. Returns (context, [(mode, members, label, class
    size, variant, weight)]).
    """
    cc = context(3)
    pairs = ot.fx.failing_sets(cc["det"], cc["obs"], cc["mwpm"], cc["cm"], 2)
    prow = [ot.analyse_set(cc, s) for s in pairs]
    out, chosen = [], set()
    for mode in MODES:
        r = pick(prow, mode)
        if r:
            chosen.add(tuple(r["members"]))
            out.append((cc, mode, r, fault_class(cc, r["members"]), len(prow), "SI1000, weight (d + 1)/2", 2))
    for mode, r, cls, size in extra_picks(cc, prow, chosen)[:n_extra]:
        out.append((cc, mode, r, cls, size, "SI1000, weight (d + 1)/2", 2))
    return out


def log_example(ex, mode):
    t = ex["totals"]
    print(f"d={ex['d']} {mode}: members {ex['members']}, CM S1/S2 {t['S1']['cm']:.2f}/{t['S2']['cm']:.2f}, "
          f"mech {t['S1']['mechanism']:.2f}/{t['S2']['mechanism']:.2f} (truth {t['truth']['mechanism']:.2f}), "
          f"reused {len(ex['reused'])}, abandoned {len(ex['abandoned'])}, phantom {len(ex['phantom_edges'])}, "
          f"RCM wrong {ex['wrong']['RCM']}, oracle wrong {ex['oracle_wrong']}", flush=True)


def main():
    import gallery_figure
    examples, extras = [], []
    for d in (3, 5, 7):
        rows = json.load(open(HERE / "failing_sets" / RESULTS[d]))["rows"]
        c = context(d)
        chosen = set()
        for mode in MODES:
            r = pick(rows, mode)
            if r is None:
                continue
            chosen.add(tuple(r["members"]))
            ex = describe(c, tuple(r["members"]), d)
            assert ex["wrong"]["CM"] and not ex["wrong"]["MWPM"]
            fig = f"fig_d{d}_{mode}.pdf"
            gallery_figure.draw(c, ex, HERE / fig)
            examples.append((mode, ex, fig))
            log_example(ex, mode)
            if d == 3:
                break
        # additional examples: the other d = 3 failure, and d = 5 across fault classes and modes
        if d == 3:
            more = [("reuse", r, fault_class(c, r["members"]), len(rows)) for r in rows
                    if tuple(r["members"]) not in chosen]
        elif d == 5:
            more = extra_picks(c, rows, chosen)
        else:
            more = []
        for mode, r, cls, size in more:
            ex = describe(c, tuple(r["members"]), d)
            assert ex["wrong"]["CM"] and not ex["wrong"]["MWPM"]
            ex["fault_class"], ex["class_size"] = cls, size
            ex["variant"], ex["weight"] = "SI1000", (d - 1) // 2
            fig = f"fig_d{d}_{mode}_{'_'.join(str(m) for m in ex['members'])}.pdf"
            gallery_figure.draw(c, ex, HERE / fig)
            extras.append((mode, ex, fig))
            log_example(ex, mode)
        if d == 3:
            for cx, mode, r, cls, size, variant, w in d3_more():
                ex = describe(cx, tuple(r["members"]), 3)
                assert ex["wrong"]["CM"] and not ex["wrong"]["MWPM"]
                ex["fault_class"], ex["class_size"], ex["variant"], ex["weight"] = cls, size, variant, w
                fig = f"fig_d3_w{w}_{mode}_{'_'.join(str(m) for m in ex['members'])}.pdf"
                gallery_figure.draw(cx, ex, HERE / fig)
                extras.append((mode, ex, fig))
                log_example(ex, mode)
    json.dump({"main": [{"mode": m, **ex} for m, ex, _ in examples],
               "additional": [{"mode": m, **ex} for m, ex, _ in extras]},
              open(HERE / "examples.json", "w"), indent=1)


if __name__ == "__main__":
    main()
