"""
Run points of the RCM grid search.

Every shot is decoded by uncorrelated MWPM and by correlated matching at each alpha, so all
decoders see exactly the same shots. Points are given either one at a time on the command
line, or as lines of a task list (format in `tasks_and_results.py`), run one after another.

A point's circuit is read from a file, which needs a noise model name, so any code that
matching can decode works. Without a file, Stim generates the memory circuit of `--code`
(rotated or unrotated surface code) with circuit-level depolarizing noise. By default, errors
are decomposed with the coordinate-aware method of the code.
Shots are sampled in chunks with independent seeds and decoded in parallel worker
processes. The result CSV is rewritten after every chunk; running the same command again
resumes from it, with fresh seeds, until the point's shots are done.

Examples:
    # one point
    python scripts/run_rcm_grid_search.py --distance 5 --p 1e-3 --shots 1000000 \\
        --results-dir data/mwpm/rcm_grid_search_circuit_level_depolarizing
    # tasks 0 to 3 of a task list
    python scripts/run_rcm_grid_search.py --tasks tasks/rcm_grid_search_tasks_si1000.txt \\
        --indices 0-3 --n-workers 8 --results-dir data/mwpm/rcm_grid_search_si1000
"""

import argparse
import multiprocessing
import time

import numpy as np
import stim

import tasks_and_results as files
from RegularizedCM import (
    DECOMPOSITIONS,
    DecoderCounts,
    build_decoders,
    check_pymatching_version,
    default_decomposition,
    sample_and_decode,
    split_shots,
)

DEFAULT_ALPHAS = tuple(float(a) for a in np.round(np.geomspace(0.03, 1.0, 19), 4))
"""19 log-spaced values from 0.03 to 1."""


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------

_worker = {}


def init_worker(circuit_text, decomposition, alphas):
    """Build the decoders once per worker process."""
    circuit = stim.Circuit(circuit_text)
    mwpm, cm = build_decoders(circuit, decomposition)
    _worker.update(circuit=circuit, mwpm=mwpm, cm=cm, alphas=alphas)


def run_chunk(task):
    """Sample and decode one (seed, shots) chunk with this worker's decoders."""
    seed, shots = task
    return sample_and_decode(
        _worker["circuit"], _worker["mwpm"], _worker["cm"], _worker["alphas"], seed, shots,
    )


# ---------------------------------------------------------------------------
# one point
# ---------------------------------------------------------------------------

def run_point(task, alphas, results_dir, chunk, n_workers, seed, deadline=None):
    """
    Sample and decode one task until its shots are done, resuming from its result file.
    Returns False if `deadline` (a `time.monotonic()` value) stopped it first.
    """
    if task.decomposition not in DECOMPOSITIONS:
        raise ValueError(
            f"unknown decomposition {task.decomposition!r}; choose one of {DECOMPOSITIONS}"
        )
    circuit, circuit_label, noise_model = files.load_circuit(
        task.code, task.circuit, task.noise_model, task.distance, task.rounds, task.p, task.basis,
    )
    point = {
        "code": task.code,
        "noise_model": noise_model,
        "decomposition": task.decomposition,
        "circuit": circuit_label,
        "basis": task.basis,
        "distance": task.distance,
        "rounds": task.rounds,
        "p": task.p,
    }
    out_file = files.result_path(
        results_dir, task.code, noise_model, task.basis, task.distance, task.rounds, task.p,
    )
    if out_file.exists():
        counts, runs = files.read_counts(out_file, alphas)
    else:
        counts, runs = DecoderCounts(alphas), 0
    remaining = task.shots - counts.shots

    print(f"\n{task.code}, {task.basis}-memory d={task.distance} rounds={task.rounds} "
          f"p={task.p:.6e}")
    print(f"circuit: {circuit_label} (noise model: {noise_model}, "
          f"decomposition: {task.decomposition})")
    print(f"results: {out_file} ({counts.shots:,} of {task.shots:,} shots done)")
    if remaining <= 0:
        print("nothing to do")
        return True

    runs += 1
    entropy = [seed, task.distance, task.rounds, round(task.p * 1e12), runs]
    chunks = split_shots(remaining, chunk, entropy)
    print(f"run {runs}: {remaining:,} shots in {len(chunks)} chunks, "
          f"{n_workers} workers, {len(alphas)} alphas", flush=True)

    start = time.monotonic()
    with multiprocessing.Pool(
        n_workers, initializer=init_worker, initargs=(str(circuit), task.decomposition, alphas),
    ) as pool:
        for chunk_counts in pool.imap_unordered(run_chunk, chunks):
            counts.add(chunk_counts)
            files.write_results(out_file, point, counts, runs)

            elapsed = time.monotonic() - start
            print(f"  {counts.shots:>14,} / {task.shots:,} shots | "
                  f"MWPM {counts.errors[0]} | CM(alpha={alphas[-1]:g}) {counts.errors[-1]} errors | "
                  f"{elapsed / 3600:.2f} h", flush=True)
            if deadline is not None and time.monotonic() > deadline:
                print("time limit reached: run the same command again to resume")
                return False

    print(f"done: {counts.shots:,} shots")
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_indices(text, count):
    """'0-2,5' -> [0, 1, 2, 5]. Without `text`, every index below `count`."""
    if not text:
        return list(range(count))
    indices = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        indices.extend(range(int(first), int(last or first) + 1))
    return indices


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    task_list = parser.add_argument_group("points from a task list")
    task_list.add_argument("--tasks", help="task list file")
    task_list.add_argument("--indices", help="task indices to run, e.g. 0-3,7 (default: all)")

    single = parser.add_argument_group("a single point")
    single.add_argument("--code", default="rotated_surface_code",
                        help="code Stim generates, or a name for the code of --circuit "
                             "(default: rotated_surface_code)")
    single.add_argument("--distance", type=int)
    single.add_argument("--rounds", type=int, default=0, help="default: the distance")
    single.add_argument("--p", type=float, help="physical error rate")
    single.add_argument("--basis", choices=("z", "x"), default="z", help="memory basis")
    single.add_argument("--shots", type=int, help="total shots")
    single.add_argument("--circuit", default=None,
                        help="Stim circuit file (default: Stim's circuit with circuit-level "
                             "depolarizing noise)")
    single.add_argument("--noise-model", default=None,
                        help="noise model of --circuit, e.g. si1000; required with --circuit, "
                             "and circuit-level-depolarizing without it")
    single.add_argument("--decomposition", default=None, choices=DECOMPOSITIONS,
                        help="how errors are decomposed for matching (default: the "
                             "coordinate-aware method of --code; required for other codes)")

    parser.add_argument("--alphas", default=",".join(map(str, DEFAULT_ALPHAS)),
                        help="comma-separated alphas (default: 19 log-spaced in [0.03, 1])")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--chunk", type=int, default=1_000_000, help="shots per chunk")
    parser.add_argument("--n-workers", type=int, default=1, help="worker processes")
    parser.add_argument("--max-seconds", type=float, default=0,
                        help="stop after this many seconds; resume later (0 = no limit)")
    parser.add_argument("--seed", type=int, default=12345)

    args = parser.parse_args()
    if not args.tasks and None in (args.distance, args.p, args.shots):
        parser.error("give --tasks, or --distance, --p and --shots")
    return args, parser


def selected_tasks(args, parser):
    """The tasks to run: from the task list, or the single point on the command line."""
    if args.tasks:
        tasks = files.read_tasks(args.tasks)
        indices = parse_indices(args.indices, len(tasks))
        if any(not 0 <= index < len(tasks) for index in indices):
            parser.error(f"{args.tasks} has tasks 0 to {len(tasks) - 1}")
        return [tasks[index] for index in indices]

    try:
        decomposition = args.decomposition or default_decomposition(args.code)
    except ValueError as error:
        parser.error(str(error))
    return [files.Task(
        code=args.code,
        basis=args.basis,
        distance=args.distance,
        rounds=args.rounds or args.distance,
        p=args.p,
        shots=args.shots,
        noise_model=args.noise_model,
        decomposition=decomposition,
        circuit=args.circuit,
    )]


def main():
    args, parser = parse_args()
    check_pymatching_version()

    alphas = tuple(sorted(float(a) for a in args.alphas.split(",")))
    deadline = time.monotonic() + args.max_seconds if args.max_seconds else None

    for task in selected_tasks(args, parser):
        try:
            finished = run_point(
                task, alphas, args.results_dir, args.chunk, args.n_workers, args.seed, deadline,
            )
        except ValueError as error:
            raise SystemExit(f"error: {error}") from None
        if not finished:
            break


if __name__ == "__main__":
    main()
