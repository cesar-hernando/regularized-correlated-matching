"""
Count the low-weight fault sets that MWPM, correlated matching (CM, alpha = 1) and regularized
correlated matching (RCM, alpha = alpha*) decode wrongly.

A fault is one error mechanism of the circuit's decomposed detector error model, and a
weight-w fault set is w distinct mechanisms occurring together. Each set's syndrome (the XOR
of its detectors) is decoded, and the decoder fails if its predicted observables differ from
the set's. A distance-d decoder should correct every set of weight (d - 1) / 2; at weight
(d + 1) / 2 some failures are unavoidable. For each (distance, p, weight) this reports
    failing_sets          how many weight-w sets the decoder fails on;
    failing_probability   the probability that exactly a failing set occurs, i.e. the sum
                          over failing sets S of prod_{i in S} p_i prod_{j not in S} (1 - p_j),
                          which is that weight's exact contribution to the logical error rate.

Sets are enumerated exhaustively when there are at most `--max-exhaustive` of them. Otherwise
they are sampled by Monte Carlo (for larger distances), with two samplers: one draws w
mechanisms uniformly, the other in proportion to their odds p / (1 - p). The uniform sampler
finds the many low-probability sets (such as those made of two-qubit gate faults), the odds
sampler the likely ones. Their draws are combined by multiple importance sampling (the balance
heuristic): a draw of set S counts with weight 1 / (n q_uniform(S) + n q_odds(S)), where n is
the number of draws per sampler and q the probability that one draw gives S. Every estimate
(failing sets, failing probability, the counts per failure class) is unbiased and at worst
about twice as noisy as with the better of the two samplers alone, and comes with a standard
error. `--sampler uniform` uses the uniform sampler alone instead, each draw counting with weight
1 / (n q_uniform(S)): the straightforward estimate. `--samples` gives n, separately for weights
(d - 1) / 2 and (d + 1) / 2 if two values
are given.

Points are the lines of a grid-search task list (circuit, decomposition) at which alpha* is
known, from the grid-search analysis. The decoders are MWPM, CM (alpha = 1) and RCM (alpha*).
With `--no-rcm`, every point of the task list at the requested distances runs, with MWPM and
CM only (for distances whose alpha* is not known yet).

Failure classes: at weight (d - 1) / 2, every failing set of MWPM, CM and RCM (every sampled
one, with its weight, when sampling) is also labelled by the mechanisms it contains
(`mechanism_labels`):
    source   the circuit noise channel contributing most probability to the mechanism:
             two-qubit gate, single-qubit gate/idle, idle during meas./reset, measurement
             flip or reset flip;
    shape    how it looks to matching: boundary (one detector), spacelike, timelike or
             spacetime (two detectors of one basis), or cross-basis (a hyperedge with
             components in both the X and the Z graph, the errors correlated matching uses).

Results go to `fault_counts.csv` and `failure_classes.csv`, rewritten after every
(distance, p, weight); running the command again skips what is already there.

Examples:
    python effective_distance_loss/enumerate_faults.py --distances 3 5 --n-workers 8
    python effective_distance_loss/enumerate_faults.py --distances 7 --no-rcm --samples 2e8 1e8 --n-workers 8
"""

import argparse
import collections
import csv
import itertools
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import stim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import tasks_and_results as files  # noqa: E402
from RegularizedCM import build_decoders, check_pymatching_version, decomposed_dem  # noqa: E402

HERE = Path(__file__).resolve().parent
POINT_FIELDS = ("code", "noise_model", "decomposition", "distance", "rounds", "p")
COUNT_FIELDS = POINT_FIELDS + (
    "weight", "weight_class", "method", "mechanisms", "sets", "samples", "total_probability",
    "decoder", "alpha", "failing_sets", "failing_sets_std", "failing_fraction",
    "failing_probability", "failing_probability_std", "only_this_fails", "only_mwpm_fails",
)
CLASS_FIELDS = POINT_FIELDS + (
    "weight", "decoder", "alpha", "subset", "kind", "fault_class", "failing_sets",
    "failing_probability",
)
BATCH = 1 << 16
"""Syndromes decoded per decode_batch call."""


# ---------------------------------------------------------------------------
# the mechanisms of a detector error model
# ---------------------------------------------------------------------------

def mechanisms(dem):
    """
    Detector masks (mechanisms x detectors), observable masks (mechanisms x observables) and
    probabilities of the error mechanisms of `dem`. The components of decomposed errors are
    merged back into one mechanism.
    """
    errors = [instruction for instruction in dem.flattened() if instruction.type == "error"]
    detectors = np.zeros((len(errors), dem.num_detectors), np.uint8)
    observables = np.zeros((len(errors), max(dem.num_observables, 1)), np.uint8)
    probabilities = np.empty(len(errors))
    for i, error in enumerate(errors):
        probabilities[i] = error.args_copy()[0]
        for target in error.targets_copy():
            if target.is_relative_detector_id():
                detectors[i, target.val] ^= 1
            elif target.is_logical_observable_id():
                observables[i, target.val] ^= 1
    return detectors, observables, probabilities


def source_label(gate, argument, p):
    """
    The noise channel of one circuit error location and the probability of its specific Pauli
    or flip. Written for SI1000 circuits; other channels keep their Stim name.
    """
    if gate == "DEPOLARIZE2":
        return "two-qubit gate", argument / 15
    if gate == "DEPOLARIZE1":
        return ("single-qubit gate/idle" if argument < p else "idle during meas./reset"), argument / 3
    if gate in ("M", "MR", "MX", "MY", "MZ", "MRX", "MRY", "MRZ"):
        return "measurement flip", argument
    if gate in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
        return "reset flip", argument
    return gate, argument


def symptoms(targets):
    """The detectors and observables a list of DEM targets flips, as a frozenset of names."""
    flipped = set()
    for target in targets:
        if target.is_relative_detector_id() or target.is_logical_observable_id():
            flipped ^= {str(target)}
    return frozenset(flipped)


def mechanism_labels(circuit, dem, p):
    """The source and shape label of every error mechanism of `dem` (see the module doc)."""
    contributions = collections.defaultdict(lambda: collections.defaultdict(float))
    for explained in circuit.explain_detector_error_model_errors():
        key = symptoms(term.dem_target for term in explained.dem_error_terms)
        for location in explained.circuit_error_locations:
            targets = location.instruction_targets
            name, probability = source_label(targets.gate, targets.args[0] if targets.args else 0.0, p)
            contributions[key][name] += probability

    coordinates = dem.get_detector_coordinates()
    sources, shapes = [], []
    for error in (i for i in dem.flattened() if i.type == "error"):
        targets = error.targets_copy()
        by_source = contributions.get(symptoms(targets)) or {"unknown": 1.0}
        sources.append(max(by_source, key=by_source.get))

        detectors = sorted({t.val for t in targets if t.is_relative_detector_id()})
        if any(t.is_separator() for t in targets):
            shapes.append("cross-basis")
        elif len(detectors) == 1:
            shapes.append("boundary")
        elif len(detectors) == 2:
            (*xy_a, t_a), (*xy_b, t_b) = (coordinates[d] for d in detectors)
            shapes.append("spacelike" if t_a == t_b else "timelike" if xy_a == xy_b else "spacetime")
        else:
            shapes.append("hyperedge")
    return sources, shapes


def weight_class_odds(odds, weight):
    """Sum over every weight-w set of the product of its odds (elementary symmetric polynomial)."""
    e = np.zeros(weight + 1)
    e[0] = 1.0
    for r in odds:
        e[1:] = e[1:] + r * e[:-1]
    return e[weight]


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------

_worker = {}


def init_worker(circuit_text, decomposition, alphas, weight, listed_alphas):
    circuit = stim.Circuit(circuit_text)
    detectors, observables, probabilities = mechanisms(decomposed_dem(circuit, decomposition))
    mwpm, cm = build_decoders(circuit, decomposition)
    _worker.update(
        detectors=detectors, observables=observables, odds=probabilities / (1 - probabilities),
        mwpm=mwpm, cm=cm, alphas=alphas, weight=weight,
        # decoders (0 = MWPM, i = CM at alphas[i-1]) whose failing sets are returned
        listed=[0] + [1 + alphas.index(a) for a in listed_alphas] if listed_alphas else [],
    )


class Tally:
    """
    Sums over the sets or draws decoded, per decoder (0 = MWPM, i >= 1 = CM at alphas[i-1]):
    the weight each failing one adds to the failing-set count and to the failing probability,
    their squares (for the sampling error), and the count weights of the failures only this
    decoder or only MWPM makes. `draws` counts every set or draw, including rejected ones.
    """

    FIELDS = ("draws", "fails", "sets", "sets_sq", "probability", "probability_sq", "only_this", "only_mwpm")

    def __init__(self, n_decoders):
        self.draws = 0
        self.fails = np.zeros(n_decoders, np.int64)
        for name in self.FIELDS[2:]:
            setattr(self, name, np.zeros(n_decoders))

    def add(self, other):
        for name in self.FIELDS:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self


def decode_into(tally, syndromes, observables, count_weights, probability_weights):
    """Decode a batch with every decoder, add the failures to `tally` and return them."""
    fails = [np.any(_worker["mwpm"].decode_batch(syndromes) != observables, axis=1)]
    for alpha in _worker["alphas"]:
        predictions = _worker["cm"].decode_batch(syndromes, enable_correlations=True, alpha=alpha)
        fails.append(np.any(predictions != observables, axis=1))
    fails = np.array(fails)
    tally.draws += syndromes.shape[0]
    tally.fails += fails.sum(axis=1)
    tally.sets += fails @ count_weights
    tally.sets_sq += fails @ count_weights ** 2
    tally.probability += fails @ probability_weights
    tally.probability_sq += fails @ probability_weights ** 2
    tally.only_this += (fails & ~fails[0]) @ count_weights
    tally.only_mwpm += (~fails & fails[0]) @ count_weights
    return fails


def gather(found, weight):
    """Concatenate (members, flags, count weights, probability weights) parts."""
    n_listed = len(_worker["listed"]) if _worker else 0
    if not found:
        return np.empty((0, weight), np.int64), np.empty((0, n_listed), bool), np.empty(0), np.empty(0)
    return tuple(np.concatenate(part) for part in zip(*found))


def keep_failing(found, fails, members, count_weights, probability_weights):
    """Record the sets any listed decoder fails on."""
    listed = _worker["listed"]
    if listed:
        flags = fails[listed].T
        keep = flags.any(axis=1)
        found.append((members[keep], flags[keep], count_weights[keep], probability_weights[keep]))


def exhaustive_task(first):
    """
    Every weight-w set whose smallest mechanism is `first`, each counting once. Returns the
    tally and the failing sets of the listed decoders (see `gather`).
    """
    detectors, observables, odds = _worker["detectors"], _worker["observables"], _worker["odds"]
    weight, n = _worker["weight"], len(_worker["odds"])
    tally, pending, found = Tally(1 + len(_worker["alphas"])), [], []

    def flush():
        syndromes, obs, set_odds, members = (np.concatenate(part) for part in zip(*pending))
        ones = np.ones(len(members))
        fails = decode_into(tally, syndromes, obs, ones, set_odds)
        keep_failing(found, fails, members, ones, set_odds)
        pending.clear()

    if weight == 1:
        pending.append((detectors[first:first + 1], observables[first:first + 1],
                        odds[first:first + 1], np.array([[first]])))
    else:
        # fix every mechanism but the largest, then take all choices of the largest at once
        for middle in itertools.combinations(range(first + 1, n - 1), weight - 2):
            fixed = [first, *middle]
            rest = np.arange(fixed[-1] + 1, n)
            members = np.column_stack([np.tile(fixed, (len(rest), 1)), rest])
            pending.append((
                np.bitwise_xor.reduce(detectors[fixed], axis=0) ^ detectors[rest],
                np.bitwise_xor.reduce(observables[fixed], axis=0) ^ observables[rest],
                np.prod(odds[fixed]) * odds[rest],
                members,
            ))
            if sum(len(part[0]) for part in pending) >= BATCH:
                flush()
    if pending:
        flush()
    return (tally, *gather(found, weight))


def sampled_task(task):
    """
    `samples` draws from each sampler (uniform and odds-weighted), weighted by the balance
    heuristic for `total` draws per sampler overall (see the module doc). With `uniform_only`,
    only the uniform sampler draws, and each draw counts with weight 1 / (total q_uniform(S)).
    Returns, per sampler, its tally and the failing draws of the listed decoders (see `gather`).
    """
    seed, samples, total, uniform_only = task
    detectors, observables, odds = _worker["detectors"], _worker["observables"], _worker["odds"]
    weight, n = _worker["weight"], len(odds)
    rng = np.random.default_rng(seed)
    log_odds, total_odds = np.log(odds), odds.sum()
    # log probability that one draw gives a given set (of distinct mechanisms)
    log_q_uniform = math.lgamma(weight + 1) - weight * math.log(n)
    results = []
    for draw_probabilities in (None,) if uniform_only else (None, odds / total_odds):
        tally, found = Tally(1 + len(_worker["alphas"])), []
        for start in range(0, samples, BATCH):
            size = min(BATCH, samples - start)
            draws = rng.choice(n, size=(size, weight), p=draw_probabilities)
            draws = draws[np.all(np.diff(np.sort(draws, axis=1), axis=1) != 0, axis=1)]
            tally.draws += size - len(draws)   # draws repeating a mechanism add nothing
            set_log_odds = log_odds[draws].sum(axis=1)
            if uniform_only:
                count_weights = np.full(len(draws), math.exp(-math.log(total) - log_q_uniform))
            else:
                log_q_odds = math.lgamma(weight + 1) + set_log_odds - weight * math.log(total_odds)
                count_weights = np.exp(-math.log(total) - np.logaddexp(log_q_uniform, log_q_odds))
            probability_weights = np.exp(set_log_odds) * count_weights
            fails = decode_into(tally, np.bitwise_xor.reduce(detectors[draws], axis=1),
                                np.bitwise_xor.reduce(observables[draws], axis=1),
                                count_weights, probability_weights)
            keep_failing(found, fails, draws, count_weights, probability_weights)
        results.append((tally, *gather(found, weight)))
    return tuple(results)


# ---------------------------------------------------------------------------
# one (distance, p, weight)
# ---------------------------------------------------------------------------

def decoder_rows(alphas, alpha_star):
    """(decoder name, alpha, index into MWPM + alphas) of MWPM, CM and, with an alpha*, RCM."""
    rows = [("MWPM", "", 0), ("CM", 1.0, 1 + alphas.index(1.0))]
    if alpha_star is not None:
        rows.append(("RCM", alpha_star, 1 + alphas.index(alpha_star)))
    return rows


def class_rows(point, weight, decoders, labels, members, flags, count_weights, probability_weights):
    """
    Failing sets and probability per failure class (see the module doc), for `decoders`
    (`decoder_rows`, in the column order of `flags`), from the failing sets found with the
    weight each adds to the failing-set count and to the failing probability.
    """
    sources, shapes = labels
    kinds = {
        "source": lambda i: sources[i],
        "shape": lambda i: shapes[i],
        "source, shape": lambda i: f"{sources[i]}, {shapes[i]}",
    }
    rows = []
    for column, (name, alpha, _) in enumerate(decoders):
        for subset in ("all", "MWPM correct"):
            if name == "MWPM" and subset != "all":
                continue
            selected = flags[:, column] & (~flags[:, 0] if subset == "MWPM correct" else True)
            for kind, label in kinds.items():
                sets, probability = collections.defaultdict(float), collections.defaultdict(float)
                for k in np.flatnonzero(selected):
                    fault_class = " + ".join(sorted(label(int(i)) for i in members[k]))
                    sets[fault_class] += count_weights[k]
                    probability[fault_class] += probability_weights[k]
                for fault_class in sorted(sets, key=lambda c: -sets[c]):
                    rows.append({**point, "weight": weight, "decoder": name, "alpha": alpha,
                                 "subset": subset, "kind": kind, "fault_class": fault_class,
                                 "failing_sets": sets[fault_class],
                                 "failing_probability": probability[fault_class]})
    return rows


def count_point(task, alpha_star, weight, args, point):
    """The fault-count rows and failure-class rows of one weight at one point."""
    circuit, _, _ = files.load_circuit(
        task.code, task.circuit, task.noise_model, task.distance, task.rounds, task.p, task.basis,
    )
    dem = decomposed_dem(circuit, task.decomposition)
    _, _, probabilities = mechanisms(dem)
    n = len(probabilities)
    odds = probabilities / (1 - probabilities)
    no_error = float(np.prod(1 - probabilities))
    sets = math.comb(n, weight)
    exhaustive = sets <= args.max_exhaustive
    samples = args.samples[0 if weight == (task.distance - 1) // 2 else -1]
    alphas = sorted({1.0} | ({alpha_star} if alpha_star is not None else set()))
    decoders = decoder_rows(alphas, alpha_star)
    classify = weight == (task.distance - 1) // 2
    listed_alphas = tuple(alpha for name, alpha, _ in decoders if name != "MWPM") if classify else ()

    start = time.monotonic()
    initargs = (str(circuit), task.decomposition, alphas, weight, listed_alphas)
    tallies, found = [], []
    with multiprocessing.Pool(args.n_workers, init_worker, initargs) as pool:
        if exhaustive:
            tallies = [Tally(1 + len(alphas))]
            order = np.argsort([-math.comb(n - 1 - i, weight - 1) for i in range(n)])  # big first
            for part, *failing in pool.imap_unordered(exhaustive_task, order.tolist()):
                tallies[0].add(part)
                found.append(failing)
            if tallies[0].draws != sets:
                raise RuntimeError(f"enumerated {tallies[0].draws} sets, expected {sets}")
        else:
            seeds = np.random.SeedSequence([args.seed, task.distance, round(task.p * 1e12), weight])
            chunks = [samples // args.n_workers] * args.n_workers
            chunks[0] += samples % args.n_workers
            uniform_only = args.sampler == "uniform"
            jobs = [(int(s.generate_state(1)[0]), c, samples, uniform_only)
                    for s, c in zip(seeds.spawn(len(chunks)), chunks)]
            tallies = [Tally(1 + len(alphas)) for _ in range(1 if uniform_only else 2)]   # uniform[, odds-weighted]
            for per_sampler in pool.imap_unordered(sampled_task, jobs):
                for tally, (part, *failing) in zip(tallies, per_sampler):
                    tally.add(part)
                    found.append(failing)

    counts = []
    for decoder, alpha, i in decoders:
        failing_sets = sum(t.sets[i] for t in tallies)
        failing_probability = no_error * sum(t.probability[i] for t in tallies)
        if exhaustive:
            failing_sets, failing_sets_std, failing_probability_std = round(failing_sets), 0.0, 0.0
        else:
            # the variance of a sum over independent draws, per sampler
            failing_sets_std = math.sqrt(sum(max(t.sets_sq[i] - t.sets[i] ** 2 / t.draws, 0.0) for t in tallies))
            failing_probability_std = no_error * math.sqrt(
                sum(max(t.probability_sq[i] - t.probability[i] ** 2 / t.draws, 0.0) for t in tallies))
        only_this = sum(t.only_this[i] for t in tallies)
        only_mwpm = sum(t.only_mwpm[i] for t in tallies)
        counts.append({
            **point,
            "weight": weight,
            "weight_class": "(d-1)/2" if weight == (task.distance - 1) // 2 else "(d+1)/2",
            "method": "exhaustive" if exhaustive else "sampled" if args.sampler == "mis" else "sampled (uniform)",
            "mechanisms": n,
            "sets": sets,
            "samples": sets if exhaustive else samples,
            "total_probability": no_error * weight_class_odds(odds, weight),
            "decoder": decoder,
            "alpha": alpha,
            "failing_sets": failing_sets,
            "failing_sets_std": failing_sets_std,
            "failing_fraction": failing_sets / sets,
            "failing_probability": failing_probability,
            "failing_probability_std": failing_probability_std,
            "only_this_fails": round(only_this) if exhaustive else only_this,
            "only_mwpm_fails": round(only_mwpm) if exhaustive else only_mwpm,
        })

    classes = []
    if classify:
        members, flags, count_weights, probability_weights = (
            np.concatenate(part) for part in zip(*found))
        classes = class_rows(point, weight, decoders, mechanism_labels(circuit, dem, task.p),
                             members, flags, count_weights, no_error * probability_weights)

    failing = ", ".join(f"{row['decoder']} {row['failing_sets']:.4g}" for row in counts)
    print(f"  weight {weight}: {counts[0]['method']}, {sets:.3g} sets, "
          f"{time.monotonic() - start:.0f} s | failing sets {failing}", flush=True)
    return counts, classes


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def read_alpha_star(path):
    """{(distance, p): alpha*} from the grid-search analysis."""
    with open(path, newline="") as handle:
        return {(int(r["distance"]), float(r["p"])): float(r["alpha_star"]) for r in csv.DictReader(handle)}


def find(table, distance, p):
    """The value of `table` at (distance, p), matching p to relative precision 1e-4."""
    return next((v for (d, q), v in table.items() if d == distance and math.isclose(q, p, rel_tol=1e-4)), None)


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    """Write `rows` atomically, sorted by point."""
    rows = sorted(rows, key=lambda r: (int(r["distance"]), float(r["p"]), int(r["weight"])))
    temporary = Path(path).with_suffix(".csv.tmp")
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default=ROOT / "tasks/rcm_grid_search_tasks_si1000.txt",
                        help="grid-search task list naming the circuits (default: the SI1000 list)")
    parser.add_argument("--alpha-star",
                        default=ROOT / "data/mwpm/rcm_grid_search_si1000/analysis/alpha_star_vs_p.csv",
                        help="alpha_star_vs_p.csv of the grid-search analysis; only its points run")
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--out-dir", default=HERE)
    parser.add_argument("--max-exhaustive", type=float, default=2e9,
                        help="enumerate every set up to this many sets, sample above (default 2e9)")
    parser.add_argument("--samples", type=float, nargs="+", default=[1e8],
                        help="draws per sampler when sampling; two values give weights "
                             "(d - 1) / 2 and (d + 1) / 2 separately (default 1e8)")
    parser.add_argument("--no-rcm", action="store_true",
                        help="run every task-list point at --distances with MWPM and CM only, "
                             "without alpha* (for distances not in the grid-search analysis)")
    parser.add_argument("--sampler", choices=["mis", "uniform"], default="mis",
                        help="mis: the uniform and odds-weighted samplers combined (module doc); "
                             "uniform: uniform draws of w distinct mechanisms only, each counting "
                             "1 / (draws * q_uniform), q_uniform = w! / M^w")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    args.samples = [int(s) for s in args.samples]
    check_pymatching_version()

    counts_path, classes_path = Path(args.out_dir) / "fault_counts.csv", Path(args.out_dir) / "failure_classes.csv"
    counts, classes = read_csv(counts_path), read_csv(classes_path)
    done = {(int(r["distance"]), float(r["p"]), int(r["weight"])) for r in counts}
    alpha_star = {} if args.no_rcm else read_alpha_star(args.alpha_star)

    tasks = sorted((t for t in files.read_tasks(args.tasks)
                    if t.distance in args.distances
                    and (args.no_rcm or find(alpha_star, t.distance, t.p) is not None)),
                   key=lambda t: (t.distance, t.p))
    which = "every point, MWPM and CM only," if args.no_rcm else "points with alpha*"
    print(f"{len(tasks)} {which} at distances {args.distances}; results in {args.out_dir}")

    # weight (d - 1) / 2 at every point first: it is cheaper and decides the effective distance
    for offset in (-1, 1):
        for task in tasks:
            weight = (task.distance + offset) // 2
            a_star = find(alpha_star, task.distance, task.p)
            print(f"d={task.distance} rounds={task.rounds} p={task.p:.4e} "
                  f"alpha*={'-' if a_star is None else f'{a_star:g}'}", flush=True)
            if any(d == task.distance and math.isclose(p, task.p, rel_tol=1e-9) and w == weight
                   for d, p, w in done):
                print(f"  weight {weight}: already done")
                continue
            point = {"code": task.code, "noise_model": task.noise_model, "decomposition": task.decomposition,
                     "distance": task.distance, "rounds": task.rounds, "p": task.p}
            new_counts, new_classes = count_point(task, a_star, weight, args, point)
            counts += new_counts
            classes += new_classes
            write_csv(classes_path, CLASS_FIELDS, classes)
            write_csv(counts_path, COUNT_FIELDS, counts)


if __name__ == "__main__":
    main()
