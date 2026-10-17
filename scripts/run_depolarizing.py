"""
Run the circuit-level depolarizing points of Fig. S2 (the points of data/mwpm/alpha_star_vs_p.csv) with the grid
runner, on Stim's generated memory circuits (four noise knobs = p, r = d), decoding every shot with MWPM and with
CM at alpha = 1 and at the point's alpha*. The result files keep the shot and failure counts of every decoder,
from which the ratio and its interval are computed.

Each point runs to TARGET failures of its rarest decoder (estimated from the rates in alpha_star_vs_p.csv, plus
10%), cheapest first, until the deadline; a point cut short resumes on the next run.

    python scripts/run_depolarizing.py --hours 14 --n-workers 15
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POINTS = ROOT / "data/mwpm/alpha_star_vs_p.csv"
RESULTS = ROOT / "data/mwpm/rcm_grid_search_depolarizing"
TARGET = 1000
RATE = {3: 2.48e5, 5: 3.19e5, 7: 5.68e4, 9: 1.47e4}   # shots/s per worker, measured, only to order the points


def plan():
    points = []
    for r in csv.DictReader(open(POINTS)):
        d, p = int(r["distance"]), float(r["p"])
        rarest = min(float(r[k]) for k in ("ler_mwpm", "ler_cm_hard", "ler_rcm_alpha_star"))
        shots = int(1.1 * TARGET / rarest)
        points.append((shots / RATE[d], d, p, float(r["alpha_star"]), shots))
    return sorted(points)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=float, default=14.0)
    parser.add_argument("--n-workers", type=int, default=15)
    args = parser.parse_args()
    deadline = time.time() + args.hours * 3600
    RESULTS.mkdir(parents=True, exist_ok=True)
    for _, d, p, alpha_star, shots in plan():
        left = deadline - time.time()
        if left < 120:
            print(f"deadline reached before d={d} p={p:g}", flush=True)
            break
        alphas = sorted({alpha_star, 1.0})
        print(f"d={d} p={p:g} alphas={alphas} target {shots:.3g} shots, {left / 3600:.1f} h left", flush=True)
        run = subprocess.run(
            [sys.executable, str(ROOT / "scripts/run_rcm_grid_search.py"), "--distance", str(d), "--p", str(p),
             "--shots", str(shots), "--alphas", ",".join(map(str, alphas)), "--n-workers", str(args.n_workers),
             "--max-seconds", str(int(left)), "--results-dir", str(RESULTS)],
            cwd=ROOT, capture_output=True, text=True)
        tail = [line for line in run.stdout.splitlines() if line.startswith(("done", "time limit", "nothing"))]
        print("   " + (tail[-1] if tail else run.stderr.strip().splitlines()[-1] if run.stderr else "?"), flush=True)
    print("stopped", time.strftime("%c"), flush=True)


if __name__ == "__main__":
    main()
