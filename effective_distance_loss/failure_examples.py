"""
Worked examples of correlated matching (CM, alpha = 1) failing on a weight-(d - 1) / 2 fault
set that MWPM decodes correctly.

For one point, every weight-(d - 1) / 2 set is decoded; among the sets CM gets wrong and MWPM
gets right, one example is taken from each of the most common failure classes (source and
shape of the mechanisms, see `enumerate_faults.py`). For each example this records
    the mechanisms      probability, class, a representative circuit fault (channel, tick,
                        Paulis and qubits), detectors with coordinates and basis, whether the
                        observable flips;
    the first pass      MWPM's matched edges with their prior weights w = log((1 - p) / p);
    the reweighting     every edge the first pass makes cheaper: for a selected edge nu and a
                        correlated edge mu, implied p = alpha * P(mu | nu), with
                        P(mu | nu) = P(mu and nu) / P(nu) from the hyperedges of the detector
                        error model, capped at 1/2, keeping the cheapest over all nu;
    the second pass     CM's matched edges, and the cost of both matchings before and after
                        the reweighting;
    the predictions     of MWPM, CM and RCM at alpha*.
The reweighting is recomputed here, and checked by decoding the syndrome with MWPM on the
reweighted graph (`edge_reweights`), which must reproduce CM's second pass.

Outputs, in this folder, named by distance and p (e.g. `failure_examples_d5_p1e-04`):
`<name>.json` with every quantity above, `<name>.md` with a readable account, and
`<name>_<k>.pdf`, which shows the faults, MWPM's, CM's and RCM's corrections in the (x, y, t)
decoding graph.

Example:
    python effective_distance_loss/failure_examples.py --distance 5 --p 1e-4
"""

import argparse
import collections
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import stim

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "figures"))

import enumerate_faults as faults  # noqa: E402
from lcd_figures import PAGE_WIDTH, paper_style, save  # noqa: E402
from RegularizedCM import build_decoders, check_pymatching_version, decomposed_dem  # noqa: E402
from RegularizedCM.decoders import detector_tag_components  # noqa: E402
from RegularizedCM.decompose_errors import detector_coord_to_basis_for_stim_surface_code_convention  # noqa: E402

P_CAP = 0.5
BASIS_NAMES = {0: "X", 1: "Z"}
BASIS_COLORS = {0: "#d62728", 1: "#1f77b4"}   # detectors of X stabilisers red, Z stabilisers blue


# ---------------------------------------------------------------------------
# the matching graph and its correlations
# ---------------------------------------------------------------------------

def edge_key(a, b):
    """An edge as a sorted detector pair; the boundary is -1."""
    a, b = int(a), -1 if b is None else int(b)
    return (min(a, b), max(a, b))


def weight_of(probability):
    probability = min(max(probability, 1e-300), P_CAP)
    return math.log((1 - probability) / probability)


def graph_edges(matching):
    """{edge: prior probability} of a PyMatching graph."""
    return {edge_key(u, v): data["error_probability"] for u, v, data in matching.edges()}


def components(error):
    """The graphlike components of one decomposed DEM error, as edges."""
    parts, current = [], []
    for target in error.targets_copy():
        if target.is_separator():
            parts.append(current)
            current = []
        elif target.is_relative_detector_id():
            current.append(target.val)
    parts.append(current)
    return [edge_key(p[0], p[1] if len(p) > 1 else None) for p in parts if p]


def joint_probabilities(dem):
    """{(nu, mu): probability that one error flips both edges}, from the hyperedges of `dem`."""
    joint = collections.defaultdict(float)
    for error in (i for i in dem.flattened() if i.type == "error"):
        probability = error.args_copy()[0]
        edges = components(error)
        for nu in edges:
            for mu in edges:
                if nu != mu:
                    q = joint[(nu, mu)]
                    joint[(nu, mu)] = q * (1 - probability) + probability * (1 - q)
    return joint


def reweights(selected, priors, joint, alpha):
    """{edge mu: (implied p, new weight)} for every edge correlated with a selected edge."""
    partners = collections.defaultdict(list)
    for (nu, mu), probability in joint.items():
        partners[nu].append((mu, probability))
    best = {}
    for nu in selected:
        for mu, probability in partners.get(nu, ()):
            implied = min(alpha * probability / priors[nu], P_CAP)
            if implied > best.get(mu, (0.0,))[0]:
                best[mu] = (implied, weight_of(implied))
    return best


# ---------------------------------------------------------------------------
# describing mechanisms
# ---------------------------------------------------------------------------

def detector_bases(circuit, coordinates):
    """0 (X stabiliser) or 1 (Z stabiliser) per detector, from tags or else coordinates."""
    try:
        return dict(enumerate(detector_tag_components(circuit)))
    except ValueError:
        return {d: detector_coord_to_basis_for_stim_surface_code_convention(c) for d, c in coordinates.items()}


def representative_faults(circuit, p):
    """{symptoms: description of the most probable circuit fault causing them}."""
    out = {}
    for explained in circuit.explain_detector_error_model_errors():
        key = faults.symptoms(term.dem_target for term in explained.dem_error_terms)
        best = None
        for location in explained.circuit_error_locations:
            targets = location.instruction_targets
            channel, probability = faults.source_label(targets.gate, targets.args[0] if targets.args else 0.0, p)
            if best is None or probability > best[0]:
                paulis = [f"{g.gate_target.pauli_type} on qubit {g.gate_target.value} {tuple(g.coords)}"
                          for g in location.flipped_pauli_product]
                if location.flipped_measurement and location.flipped_measurement.record_index is not None:
                    measured = location.flipped_measurement.observable
                    paulis.append("flip of the measurement of " + ", ".join(
                        f"qubit {g.gate_target.value} {tuple(g.coords)}" for g in measured))
                best = (probability, {"channel": channel, "stim_channel": targets.gate,
                                      "tick": location.tick_offset, "paulis": paulis,
                                      "equivalent_faults": len(explained.circuit_error_locations)})
        if best:
            out[key] = best[1]
    return out


def matched_edges(matching, syndrome, **decode_options):
    return [edge_key(a, None if b == -1 else b)
            for a, b in np.atleast_2d(matching.decode_to_edges_array(syndrome, **decode_options)).reshape(-1, 2)]


# ---------------------------------------------------------------------------
# examples
# ---------------------------------------------------------------------------

def failing_sets(detectors, observables, mwpm, cm, weight):
    """Every weight-w set (as an index tuple) that CM (alpha = 1) fails on and MWPM corrects."""
    n, found = len(detectors), []
    # all sets sharing their first w - 1 mechanisms are decoded in one batch
    for fixed in itertools.combinations(range(n), weight - 1):
        rest = np.arange(fixed[-1] + 1 if fixed else 0, n)
        if not len(rest):
            continue
        fixed = list(fixed)
        syndromes = np.bitwise_xor.reduce(detectors[fixed], axis=0) ^ detectors[rest] if fixed else detectors
        truth = np.bitwise_xor.reduce(observables[fixed], axis=0) ^ observables[rest] if fixed else observables
        wrong_mwpm = np.any(mwpm.decode_batch(syndromes) != truth, axis=1)
        wrong_cm = np.any(cm.decode_batch(syndromes, enable_correlations=True, alpha=1.0) != truth, axis=1)
        found += [(*fixed, int(rest[k])) for k in np.flatnonzero(wrong_cm & ~wrong_mwpm)]
    return found


def describe(members, context, alpha_star):
    """Everything about one failing set (see the module doc)."""
    c = context
    syndrome = np.bitwise_xor.reduce(c["detectors"][list(members)], axis=0)
    truth = np.bitwise_xor.reduce(c["observables"][list(members)], axis=0)

    first = matched_edges(c["mwpm"], syndrome)
    second = matched_edges(c["cm"], syndrome, enable_correlations=True, alpha=1.0)
    rcm = matched_edges(c["cm"], syndrome, enable_correlations=True, alpha=alpha_star)
    new = reweights(first, c["priors"], c["joint"], alpha=1.0)
    # PyMatching writes a boundary edge as (detector, -1)
    rows = np.array([[b, a, w] if a == -1 else [a, b, w] for (a, b), (_, w) in new.items()],
                    dtype=np.float64).reshape(-1, 3)
    check = matched_edges(c["mwpm"], syndrome, edge_reweights=rows)

    def cost(edges, after):
        return sum(new[e][1] if after and e in new else weight_of(c["priors"][e]) for e in edges)

    def prediction(matching, **options):
        return int(np.any(matching.decode(syndrome, **options) != truth))

    mechanisms = []
    for i in members:
        dets = [int(d) for d in np.flatnonzero(c["detectors"][i])]
        key = faults.symptoms(c["dem_errors"][i].targets_copy())
        mechanisms.append({
            "index": int(i),
            "probability": float(c["probabilities"][i]),
            "source": c["sources"][i],
            "shape": c["shapes"][i],
            "circuit_fault": c["representatives"].get(key),
            "detectors": [{"id": d, "coords": list(map(float, c["coordinates"][d])),
                           "basis": BASIS_NAMES[c["bases"][d]]} for d in dets],
            "components": [list(e) for e in components(c["dem_errors"][i])],
            "flips_observable": int(c["observables"][i].any()),
        })
    return {
        "members": [int(i) for i in members],
        "fault_class": " + ".join(sorted(f"{c['sources'][i]}, {c['shapes'][i]}" for i in members)),
        "mechanisms": mechanisms,
        "syndrome": [int(d) for d in np.flatnonzero(syndrome)],
        "flips_observable": int(truth.any()),
        "first_pass": [{"edge": list(e), "weight": weight_of(c["priors"][e])} for e in first],
        "reweighted": sorted(({"edge": list(e), "weight_before": weight_of(c["priors"][e]),
                               "implied_p": v[0], "weight_after": v[1]} for e, v in new.items()
                              if e in c["priors"]), key=lambda r: r["weight_after"] - r["weight_before"]),
        "second_pass": [{"edge": list(e), "weight_before": weight_of(c["priors"][e]),
                         "weight_after": new[e][1] if e in new else weight_of(c["priors"][e])} for e in second],
        "rcm_edges": [list(e) for e in rcm],
        "cost": {"first_pass_before": cost(first, False), "first_pass_after": cost(first, True),
                 "second_pass_before": cost(second, False), "second_pass_after": cost(second, True)},
        "reweighting_reproduces_cm": sorted(check) == sorted(second),
        "wrong": {"MWPM": prediction(c["mwpm"]),
                  "CM": prediction(c["cm"], enable_correlations=True, alpha=1.0),
                  "RCM": prediction(c["cm"], enable_correlations=True, alpha=alpha_star)},
    }


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def draw_case(case, context, alpha_star, path):
    """Faults, MWPM, CM and RCM in the whole (x, y, t) decoding graph, every round included."""
    coordinates, bases = context["coordinates"], context["bases"]
    xyz = {d: (c[0], c[1], c[2]) for d, c in coordinates.items()}
    fired = case["syndrome"]
    everything = np.array(list(xyz.values()))
    lows, highs = everything.min(axis=0) - (1.0, 1.0, 0.3), everything.max(axis=0) + (1.0, 1.0, 0.3)
    centre = everything.mean(axis=0)   # boundary edges point away from the middle of the patch

    panels = [
        ("faults", [tuple(e) for m in case["mechanisms"] for e in m["components"]], None),
        ("MWPM", [tuple(e["edge"]) for e in case["first_pass"]], case["wrong"]["MWPM"]),
        ("CM ($\\alpha=1$)", [tuple(e["edge"]) for e in case["second_pass"]], case["wrong"]["CM"]),
        (f"RCM ($\\alpha^*={alpha_star:g}$)", [tuple(e) for e in case["rcm_edges"]], case["wrong"]["RCM"]),
    ]
    span = highs - lows
    figure = plt.figure(figsize=(PAGE_WIDTH, PAGE_WIDTH * 0.8))
    figure.subplots_adjust(left=-0.05, right=1.05, bottom=-0.04, top=0.97, wspace=-0.05, hspace=0.0)
    for k, (title, edges, wrong) in enumerate(panels):
        ax = figure.add_subplot(2, 2, k + 1, projection="3d")
        ax.set_box_aspect((span[0], span[1], span[2] * 1.6), zoom=1.3)   # stretch t to separate rounds
        for a, b in context["priors"]:
            if a != -1:
                ax.plot(*zip(xyz[a], xyz[b]), color="0.85", lw=0.2, zorder=1)
        for d in xyz:
            ax.scatter(*xyz[d], s=1.0, color="0.6", depthshade=False)
        for d in fired:
            ax.scatter(*xyz[d], s=40, color=BASIS_COLORS[bases[d]], alpha=0.25, depthshade=False)
            ax.scatter(*xyz[d], s=9, color=BASIS_COLORS[bases[d]], depthshade=False)
        for a, b in edges:
            node = b if a == -1 else a
            color = BASIS_COLORS[bases[node]]
            if a == -1:
                direction = np.array(xyz[b]) - centre
                direction[2] = 0
                direction = direction / (np.linalg.norm(direction) + 1e-9) * 1.2
                ax.plot(*zip(xyz[b], np.array(xyz[b]) + direction), color=color, lw=1.6)
            else:
                ax.plot(*zip(xyz[a], xyz[b]), color=color, lw=1.6)
        ax.set_xlim(lows[0], highs[0]); ax.set_ylim(lows[1], highs[1]); ax.set_zlim(lows[2], highs[2])
        ax.view_init(elev=20, azim=-60)
        ax.set_axis_off()
        color = "black" if wrong is None else ("#c0392b" if wrong else "#1e8449")
        suffix = "" if wrong is None else (" -- wrong" if wrong else " -- correct")
        ax.set_title(title + suffix, color=color, fontsize=10, y=0.92)
    save(figure, path)
    plt.close(figure)


# ---------------------------------------------------------------------------
# write-up
# ---------------------------------------------------------------------------

def edge_text(edge, coordinates):
    """'D16 (2,4,1) - D22 (4,2,1)', or 'D29 (8,4,1) - boundary'."""
    def node(d):
        return "boundary" if d == -1 else f"D{d} ({','.join(f'{c:g}' for c in coordinates[d])})"
    a, b = edge
    return f"{node(b)} - boundary" if a == -1 else f"{node(a)} - {node(b)}"


def markdown(cases, header, coordinates, alpha_star, name):
    """A readable account of every example (see the module doc)."""
    lines = [f"# Correlated matching failures at weight {header['weight']}: worked examples", "",
             f"{header['code']}, {header['noise_model']}, Z memory, d = {header['distance']}, "
             f"{header['rounds']} rounds, p = {header['p']:g}, alpha* = {alpha_star:g}. "
             f"{header['failing_sets_cm_not_mwpm']} weight-{header['weight']} fault sets are decoded wrongly by "
             "CM (alpha = 1) and correctly by MWPM. Weights are w = log((1 - p) / p); the reweighted "
             "weight of an edge uses implied p = alpha P(mu | nu), and recomputing it reproduces "
             "CM's second pass in every example. Detectors are written D<id> (x,y,t) with their "
             "stabiliser type.", ""]
    for k, case in enumerate(cases, 1):
        lines += [f"## Example {k}: {case['fault_class']}", "",
                  f"One of {case['class_size']} failing sets of this class. "
                  f"Figure: `{name}_{k}.pdf` (red: X-stabiliser detectors and edges, blue: Z).", ""]
        for m in case["mechanisms"]:
            fault = m["circuit_fault"] or {}
            detectors = ", ".join(f"D{x['id']} ({','.join(f'{c:g}' for c in x['coords'])}, {x['basis']})"
                                  for x in m["detectors"])
            paulis = "; ".join(fault.get("paulis", [])) or "no Pauli listed"
            equivalent = fault.get("equivalent_faults", 1)
            alternatives = "" if equivalent == 1 else f" (one of {equivalent} equivalent circuit faults)"
            observable = "flipped" if m["flips_observable"] else "not flipped"
            lines.append(f"* **Mechanism {m['index']}** (p = {m['probability']:.2e}, {m['source']}, {m['shape']}): "
                         f"{fault.get('stim_channel', '?')} at tick {fault.get('tick', '?')}, {paulis}{alternatives}. "
                         f"Flips {detectors}; observable {observable}.")
        cost = case["cost"]
        second = {tuple(e["edge"]) for e in case["second_pass"]}
        lines += ["", f"**First pass (MWPM), correct**, cost {cost['first_pass_before']:.2f}:", ""]
        lines += [f"* {edge_text(e['edge'], coordinates)}: w = {e['weight']:.2f}" for e in case["first_pass"]]
        lines += ["", f"**Reweighting.** The first-pass edges are correlated with {len(case['reweighted'])} edges; "
                      "those CM's second pass then uses:", "",
                  "| edge | w before | w after | implied p |", "|---|---|---|---|"]
        lines += [f"| {edge_text(r['edge'], coordinates)} | {r['weight_before']:.2f} | {r['weight_after']:.2f} | "
                  f"{r['implied_p']:.3f} |" for r in case["reweighted"] if tuple(r["edge"]) in second]
        lines += ["", "**Second pass (CM), wrong**:", ""]
        lines += [f"* {edge_text(e['edge'], coordinates)}: w = {e['weight_before']:.2f} -> {e['weight_after']:.2f}"
                  for e in case["second_pass"]]
        lines += ["", "| matching | cost, prior weights | cost after reweighting |", "|---|---|---|",
                  f"| first pass (correct) | **{cost['first_pass_before']:.2f}** | {cost['first_pass_after']:.2f} |",
                  f"| second pass (wrong) | {cost['second_pass_before']:.2f} | **{cost['second_pass_after']:.2f}** |", "",
                  f"The wrong matching costs {cost['second_pass_before'] - cost['first_pass_before']:.2f} more "
                  f"before the reweighting and {cost['first_pass_after'] - cost['second_pass_after']:.2f} less after it. "
                  f"RCM at alpha* = {alpha_star:g} decodes this set "
                  f"{'wrongly' if case['wrong']['RCM'] else 'correctly'}.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default=ROOT / "tasks/rcm_grid_search_tasks_si1000.txt")
    parser.add_argument("--alpha-star", default=ROOT / "data/mwpm/rcm_grid_search_si1000/analysis/alpha_star_vs_p.csv")
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--p", type=float, default=1e-4)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--out-dir", default=HERE)
    args = parser.parse_args()
    check_pymatching_version()

    task = next((t for t in faults.files.read_tasks(args.tasks)
                 if t.distance == args.distance and math.isclose(t.p, args.p, rel_tol=1e-4)), None)
    alpha_star = faults.find(faults.read_alpha_star(args.alpha_star), args.distance, args.p)
    if task is None or alpha_star is None:
        raise SystemExit(f"no task with alpha* at d={args.distance}, p={args.p:g}")

    circuit, _, _ = faults.files.load_circuit(task.code, task.circuit, task.noise_model, task.distance,
                                              task.rounds, task.p, task.basis)
    dem = decomposed_dem(circuit, task.decomposition)
    detectors, observables, probabilities = faults.mechanisms(dem)
    mwpm, cm = build_decoders(circuit, task.decomposition)
    sources, shapes = faults.mechanism_labels(circuit, dem, task.p)
    coordinates = circuit.get_detector_coordinates()
    context = {
        "detectors": detectors, "observables": observables, "probabilities": probabilities,
        "mwpm": mwpm, "cm": cm, "sources": sources, "shapes": shapes, "coordinates": coordinates,
        "bases": detector_bases(circuit, coordinates), "priors": graph_edges(mwpm),
        "joint": joint_probabilities(dem), "representatives": representative_faults(circuit, task.p),
        "dem_errors": [i for i in dem.flattened() if i.type == "error"],
    }

    weight = (args.distance - 1) // 2
    found = failing_sets(detectors, observables, mwpm, cm, weight)
    by_class = collections.defaultdict(list)
    for members in found:
        by_class[" + ".join(sorted(f"{sources[i]}, {shapes[i]}" for i in members))].append(members)
    ranked = sorted(by_class.values(), key=len, reverse=True)
    chosen = [group[0] for group in ranked[:args.examples]]
    chosen += [m for group in ranked for m in group[1:]][:args.examples - len(chosen)]
    print(f"d={args.distance} p={args.p:g}: {len(found)} weight-{weight} sets fail with CM and not MWPM, "
          f"in {len(by_class)} classes")

    name = f"failure_examples_d{args.distance}_p{args.p:.0e}"
    cases = []
    with paper_style(use_latex=True):
        for k, members in enumerate(chosen, 1):
            case = describe(members, context, alpha_star)
            case["class_size"] = len(by_class[case["fault_class"]])
            cases.append(case)
            draw_case(case, context, alpha_star, Path(args.out_dir) / f"{name}_{k}.pdf")
            print(f"  example {k}: {case['fault_class']} ({case['class_size']} sets); cost first pass "
                  f"{case['cost']['first_pass_before']:.2f} -> {case['cost']['first_pass_after']:.2f}, "
                  f"CM {case['cost']['second_pass_before']:.2f} -> {case['cost']['second_pass_after']:.2f}; "
                  f"reweighting reproduces CM: {case['reweighting_reproduces_cm']}; RCM wrong: {case['wrong']['RCM']}")

    header = {"code": task.code, "noise_model": task.noise_model, "distance": task.distance,
              "rounds": task.rounds, "p": task.p, "alpha_star": alpha_star, "weight": weight,
              "failing_sets_cm_not_mwpm": len(found),
              "classes": {name: len(group) for name, group in sorted(by_class.items(), key=lambda x: -len(x[1]))}}
    with open(Path(args.out_dir) / f"{name}.json", "w") as handle:
        json.dump({**header, "examples": cases}, handle, indent=1)
    (Path(args.out_dir) / f"{name}.md").write_text(markdown(cases, header, coordinates, alpha_star, name))


if __name__ == "__main__":
    main()
