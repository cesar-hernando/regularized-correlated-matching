"""
Benchmarking RCM against uncorrelated MWPM on shared shots.

Every shot is decoded by MWPM and by correlated matching at each alpha, so all decoders are
compared on exactly the same shots. Counts are stored per decoder: index 0 is MWPM and
index i >= 1 is correlated matching at alphas[i - 1].
"""

import time

import numpy as np


class DecoderCounts:
    """Logical failures of MWPM and of correlated matching at each alpha, on shared shots."""

    def __init__(self, alphas):
        size = len(alphas) + 1
        self.alphas = tuple(alphas)
        self.shots = 0
        self.errors = np.zeros(size, dtype=np.int64)
        self.only_this_fails = np.zeros(size, dtype=np.int64)
        """Shots where this decoder fails and MWPM does not."""
        self.only_mwpm_fails = np.zeros(size, dtype=np.int64)
        """Shots where MWPM fails and this decoder does not."""
        self.decode_seconds = np.zeros(size)

    def add(self, other):
        """Add the counts of `other`, which must use the same alphas."""
        if other.alphas != self.alphas:
            raise ValueError("cannot add counts over different alphas")
        self.shots += other.shots
        self.errors += other.errors
        self.only_this_fails += other.only_this_fails
        self.only_mwpm_fails += other.only_mwpm_fails
        self.decode_seconds += other.decode_seconds


def sample_and_decode(circuit, mwpm, cm, alphas, seed, shots) -> DecoderCounts:
    """
    Sample `shots` shots of `circuit` with `seed` and decode them with MWPM and with
    correlated matching at every alpha. The decoders come from `decoders.build_decoders`.
    A decoder fails on a shot if it predicts any logical observable wrongly.
    """
    sampler = circuit.compile_detector_sampler(seed=seed)
    detections, observables = sampler.sample(
        shots, separate_observables=True, bit_packed=True,
    )
    actual = np.unpackbits(
        observables, axis=1, count=circuit.num_observables, bitorder="little",
    )

    counts = DecoderCounts(alphas)
    counts.shots = shots

    start = time.perf_counter()
    prediction = mwpm.decode_batch(detections, bit_packed_shots=True)
    counts.decode_seconds[0] = time.perf_counter() - start
    mwpm_fails = np.any(prediction != actual, axis=1)
    counts.errors[0] = np.count_nonzero(mwpm_fails)

    for i, alpha in enumerate(alphas, start=1):
        start = time.perf_counter()
        prediction = cm.decode_batch(
            detections, bit_packed_shots=True, enable_correlations=True, alpha=alpha,
        )
        counts.decode_seconds[i] = time.perf_counter() - start
        cm_fails = np.any(prediction != actual, axis=1)
        counts.errors[i] = np.count_nonzero(cm_fails)
        counts.only_this_fails[i] = np.count_nonzero(cm_fails & ~mwpm_fails)
        counts.only_mwpm_fails[i] = np.count_nonzero(mwpm_fails & ~cm_fails)

    return counts


def split_shots(total, chunk, entropy) -> list[tuple[int, int]]:
    """
    Split `total` shots into chunks of at most `chunk` shots, as `(seed, shots)` pairs.
    Seeds are drawn from `entropy` (a list of integers), so they are reproducible and
    different for every chunk.
    """
    full_chunks, last = divmod(total, chunk)
    sizes = [chunk] * full_chunks + ([last] if last else [])
    seeds = np.random.SeedSequence(entropy).generate_state(len(sizes), dtype=np.uint64)
    return [(int(seed), size) for seed, size in zip(seeds, sizes, strict=True)]
