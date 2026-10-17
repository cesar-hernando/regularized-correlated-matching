"""
Stim circuits for memory experiments.

Stim can generate the codes in `GENERATED_CODES`. Any other code that matching can decode
works too: build or load its `stim.Circuit` and pass it to `decoders.build_decoders`.
"""

import stim

GENERATED_CODES = {
    "rotated_surface_code": "surface_code:rotated_memory_{basis}",
    "unrotated_surface_code": "surface_code:unrotated_memory_{basis}",
}
"""Codes Stim can generate, mapped to Stim's circuit names."""


def circuit_level_depolarizing_memory(code, distance, rounds, p, basis="z") -> stim.Circuit:
    """
    Stim's memory circuit for `code` (a key of `GENERATED_CODES`) with circuit-level
    depolarizing noise: depolarization after Clifford gates and on data qubits before each
    round, and bit flips after resets and before measurements, all with probability p.
    """
    if code not in GENERATED_CODES:
        raise ValueError(
            f"Stim cannot generate {code!r}; choose one of {sorted(GENERATED_CODES)}, "
            f"or use a circuit file"
        )
    return stim.Circuit.generated(
        GENERATED_CODES[code].format(basis=basis),
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,
    )
