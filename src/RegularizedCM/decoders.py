"""
Uncorrelated MWPM and (regularized) correlated matching decoders.

RCM is implemented in the regularized PyMatching fork and chosen at decode time:
`matcher.decode_batch(detections, enable_correlations=True, alpha=alpha)`, where the matcher
was built with `enable_correlations=True`. One matcher therefore decodes every alpha.

Matching needs every error to flip at most two detectors, so errors that flip more (for
example Y errors in a surface code) are first decomposed into graphlike components; see
`decomposed_dem`.
"""

import pymatching
import stim

from .decompose_errors import (
    decompose_errors_for_stim_surface_code_coords,
    decompose_errors_for_stim_unrotated_surface_code_coords,
    decompose_errors_using_detector_assignment,
    decompose_errors_using_last_coordinate_index,
)

REQUIRED_PYMATCHING = "2.3.1+rcm.1"

DECOMPOSITIONS = (
    "rotated-surface-code-coords",
    "unrotated-surface-code-coords",
    "stim",
    "detector-tags",
    "last-coordinate",
)
"""The named decomposition methods of `decomposed_dem`."""

DEFAULT_DECOMPOSITIONS = {
    "rotated_surface_code": "rotated-surface-code-coords",
    "unrotated_surface_code": "unrotated-surface-code-coords",
}
"""
The default decomposition of each code Stim generates: the coordinate-aware decompositions,
which give lower logical error rates than Stim's decomposer with both MWPM and correlated
matching.
"""


def check_pymatching_version():
    """Fail loudly if the regularized PyMatching fork is not installed."""
    if pymatching.__version__ != REQUIRED_PYMATCHING:
        raise RuntimeError(
            f"pymatching {pymatching.__version__} is installed, but regularized correlated "
            f"matching needs the fork {REQUIRED_PYMATCHING}."
        )


def default_decomposition(code) -> str:
    """The default decomposition of `code`, from `DEFAULT_DECOMPOSITIONS`."""
    if code not in DEFAULT_DECOMPOSITIONS:
        raise ValueError(
            f"no default decomposition for code {code!r}; choose one of {DECOMPOSITIONS}"
        )
    return DEFAULT_DECOMPOSITIONS[code]


def detector_tag_components(circuit: stim.Circuit) -> list[int]:
    """
    The component of every detector from its tag: 0 for X and 1 for Z, from tags such as
    `DETECTOR[stabiliser=X]` or `DETECTOR[Z]`.
    """
    components = []
    for instruction in circuit.flattened():
        if instruction.name == "DETECTOR":
            basis = instruction.tag.split("=")[-1].strip().upper()
            if basis not in ("X", "Z"):
                raise ValueError(f"detector tag {instruction.tag!r} does not name X or Z")
            components.append(0 if basis == "X" else 1)
    return components


def decomposed_dem(
    circuit: stim.Circuit, decomposition="rotated-surface-code-coords",
) -> stim.DetectorErrorModel:
    """
    The detector error model of `circuit`, with errors decomposed into graphlike components.

    `decomposition` is one of the following. Use the rule of the circuit's code, e.g. with
    `default_decomposition(code)`:
        "rotated-surface-code-coords"    the default: X/Z split from detector coordinates in
                                         the layout of Stim's rotated surface-code circuits.
        "unrotated-surface-code-coords"  the same for Stim's unrotated surface-code circuits.
                                         Both are in `decompose_errors.py` (Tesseract).
        "stim"                           Stim's own decomposer; works for any code matching
                                         can decode.
        "detector-tags"                  X/Z split from detector tags such as `stabiliser=X`.
        "last-coordinate"                split by the last entry of each detector's coordinates.
        a function                       `component(detector_index) -> int`, for custom codes.
    """
    if decomposition == "stim":
        return circuit.detector_error_model(decompose_errors=True)

    dem = circuit.detector_error_model(decompose_errors=False)
    if decomposition == "rotated-surface-code-coords":
        return decompose_errors_for_stim_surface_code_coords(dem)
    if decomposition == "unrotated-surface-code-coords":
        return decompose_errors_for_stim_unrotated_surface_code_coords(dem)
    if decomposition == "last-coordinate":
        return decompose_errors_using_last_coordinate_index(dem)
    if decomposition == "detector-tags":
        components = detector_tag_components(circuit)
        return decompose_errors_using_detector_assignment(
            dem, detector_component_func=components.__getitem__,
        )
    if callable(decomposition):
        return decompose_errors_using_detector_assignment(
            dem, detector_component_func=decomposition,
        )
    raise ValueError(
        f"unknown decomposition {decomposition!r}; choose one of {DECOMPOSITIONS} or a function"
    )


def build_decoders(circuit: stim.Circuit, decomposition="rotated-surface-code-coords"):
    """
    Uncorrelated MWPM and correlated matching for `circuit`, both on its decomposed detector
    error model (see `decomposed_dem`). Returns `(mwpm, cm)`.
    """
    dem = decomposed_dem(circuit, decomposition)
    mwpm = pymatching.Matching.from_detector_error_model(dem)
    cm = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    return mwpm, cm
