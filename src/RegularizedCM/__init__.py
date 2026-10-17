"""
Regularized correlated matching (RCM) for surface-code decoding.

RCM is correlated matching in which the conditional probability of every reweighted edge is
damped by alpha in (0, 1]: `implied_p = alpha * P(e_mu | e_nu)`. alpha = 1 is standard
correlated matching; smaller alpha restores the code distance deep below threshold.

Modules:
    circuits          Stim circuits for surface-code memory experiments
    decoders          uncorrelated MWPM and (regularized) correlated matching decoders
    benchmark         decoding the same shots with MWPM and with RCM at many alphas
    statistics        logical error rates, their ratios and McNemar tests
    models            models of the error-rate ratios and of the optimal alpha*(p)
    decompose_errors  X/Z decomposition of detector error models (adapted from Tesseract)
"""

from .benchmark import DecoderCounts, sample_and_decode, split_shots
from .circuits import GENERATED_CODES, circuit_level_depolarizing_memory
from .decoders import (
    DECOMPOSITIONS,
    DEFAULT_DECOMPOSITIONS,
    build_decoders,
    check_pymatching_version,
    decomposed_dem,
    default_decomposition,
)
from .models import cm_ratio, optimal_alpha, rcm_ratio
from .statistics import error_rate_ratio, logical_error_rate, mcnemar_p_value

__all__ = [
    "DECOMPOSITIONS",
    "DEFAULT_DECOMPOSITIONS",
    "GENERATED_CODES",
    "DecoderCounts",
    "build_decoders",
    "check_pymatching_version",
    "circuit_level_depolarizing_memory",
    "cm_ratio",
    "decomposed_dem",
    "default_decomposition",
    "error_rate_ratio",
    "logical_error_rate",
    "mcnemar_p_value",
    "optimal_alpha",
    "rcm_ratio",
    "sample_and_decode",
    "split_shots",
]
