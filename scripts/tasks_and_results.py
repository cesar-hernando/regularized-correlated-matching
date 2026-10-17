"""
Task lists and result files of the RCM grid search, shared by the run, progress and analysis
scripts.

A task list has one point per line:
`code basis distance rounds p shots noise_model decomposition circuit`. With circuit `-`,
Stim generates the memory circuit of `code` (a key of `RegularizedCM.GENERATED_CODES`) with
circuit-level depolarizing noise; with a circuit file, `code` is just a name, so any code
works. `decomposition` is one of `RegularizedCM.DECOMPOSITIONS`. Blank lines and lines
starting with `#` are skipped.

A result file holds one point, with one row per decoder: MWPM first, then correlated
matching at each alpha in increasing order.
"""

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim

from RegularizedCM import DecoderCounts, circuit_level_depolarizing_memory, logical_error_rate

CIRCUIT_LEVEL_DEPOLARIZING = "circuit-level-depolarizing"
"""The noise model name of Stim's generated circuits."""

STIM_GENERATED = "stim-generated"
"""The `circuit` label of points that use Stim's generated circuit instead of a file."""

RESULT_FIELDS = (
    "code", "noise_model", "decomposition", "circuit", "basis", "distance", "rounds", "p",
    "decoder", "alpha", "shots", "errors", "ler", "ler_std", "only_this_fails", "only_mwpm_fails", "decode_seconds",
    "us_per_shot", "runs",
)
_INT_FIELDS = {"distance", "rounds", "shots", "errors", "only_this_fails", "only_mwpm_fails", "runs"}
_FLOAT_FIELDS = {"p", "ler", "ler_std", "decode_seconds", "us_per_shot"}


# ---------------------------------------------------------------------------
# tasks and circuits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """One point of a grid search. `circuit` is a file path, or None for Stim's circuit."""

    code: str
    basis: str
    distance: int
    rounds: int
    p: float
    shots: int
    noise_model: str
    decomposition: str
    circuit: str | None


def read_tasks(path) -> list[Task]:
    """The tasks of a task list."""
    tasks = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        code, basis, distance, rounds, p, shots, noise_model, decomposition, circuit = (
            line.split()
        )
        tasks.append(Task(
            code=code,
            basis=basis,
            distance=int(distance),
            rounds=int(rounds),
            p=float(p),
            shots=int(shots),
            noise_model=noise_model,
            decomposition=decomposition,
            circuit=None if circuit == "-" else circuit,
        ))
    return tasks


def load_circuit(code, circuit_file, noise_model, distance, rounds, p, basis):
    """
    The circuit of one point, with its `circuit` label and noise model.

    With `circuit_file`, the circuit is read from that file and `noise_model` must name its
    noise. Without it, Stim generates the circuit of `code` with circuit-level depolarizing
    noise, and `noise_model` must be None or `circuit-level-depolarizing`.
    """
    if circuit_file:
        if not noise_model:
            raise ValueError("a circuit file needs a noise model")
        return stim.Circuit.from_file(circuit_file), Path(circuit_file).name, noise_model

    if noise_model not in (None, CIRCUIT_LEVEL_DEPOLARIZING):
        raise ValueError(
            f"Stim's generated circuit has noise model {CIRCUIT_LEVEL_DEPOLARIZING}, "
            f"not {noise_model}; use a circuit file for other noise models"
        )
    circuit = circuit_level_depolarizing_memory(code, distance, rounds, p, basis)
    return circuit, STIM_GENERATED, CIRCUIT_LEVEL_DEPOLARIZING


# ---------------------------------------------------------------------------
# result files
# ---------------------------------------------------------------------------

def result_path(results_dir, code, noise_model, basis, distance, rounds, p) -> Path:
    """The result file of one point."""
    name = f"{code}_{noise_model}_{basis}_d{distance}_r{rounds}_p{p:.6e}.csv"
    return Path(results_dir) / name


def write_results(path, point, counts, runs):
    """
    Write the counts of one point atomically, so an interrupted job never leaves a broken
    file. `point` holds the columns from `code` to `p` of `RESULT_FIELDS`.
    """
    decoders = [("mwpm", "")] + [("cm", alpha) for alpha in counts.alphas]
    rows = []
    for i, (decoder, alpha) in enumerate(decoders):
        ler, ler_std = logical_error_rate(counts.errors[i], counts.shots)
        rows.append({
            **point,
            "decoder": decoder,
            "alpha": alpha,
            "shots": counts.shots,
            "errors": counts.errors[i],
            "ler": ler,
            "ler_std": ler_std,
            "only_this_fails": counts.only_this_fails[i],
            "only_mwpm_fails": counts.only_mwpm_fails[i],
            "decode_seconds": round(counts.decode_seconds[i], 3),
            "us_per_shot": round(counts.decode_seconds[i] / counts.shots * 1e6, 4),
            "runs": runs,
        })

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_results(path) -> list[dict]:
    """Typed rows of a result file. `alpha` is None for the MWPM row."""
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            for key in _INT_FIELDS:
                row[key] = int(row[key])
            for key in _FLOAT_FIELDS:
                row[key] = float(row[key])
            row["alpha"] = float(row["alpha"]) if row["alpha"] else None
            rows.append(row)
    return rows


def read_counts(path, alphas):
    """The counts and number of runs stored in a result file, which must use `alphas`."""
    rows = read_results(path)
    stored_alphas = [row["alpha"] for row in rows[1:]]
    if (
        rows[0]["decoder"] != "mwpm"
        or len(stored_alphas) != len(alphas)
        or not np.allclose(stored_alphas, alphas)
    ):
        raise ValueError(f"{path} was written with different alphas")

    counts = DecoderCounts(alphas)
    counts.shots = rows[0]["shots"]
    for i, row in enumerate(rows):
        counts.errors[i] = row["errors"]
        counts.only_this_fails[i] = row["only_this_fails"]
        counts.only_mwpm_fails[i] = row["only_mwpm_fails"]
        counts.decode_seconds[i] = row["decode_seconds"]
    return counts, rows[0]["runs"]
