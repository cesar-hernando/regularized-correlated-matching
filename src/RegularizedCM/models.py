"""
Models of the logical error rate ratio to uncorrelated MWPM, and of the optimal damping.
"""

import numpy as np


def rcm_ratio(p, a, b):
    """RCM at the optimal alpha over MWPM: `1 + a + b p`."""
    return 1.0 + a + b * p


def cm_ratio(p, a, b, c):
    """
    Standard correlated matching over MWPM: `1 + a + b p + c ln(1/p) / p`. The last term comes
    from the fault configurations of weight (d-1)/2 that correlated matching fails to decode.
    """
    return 1.0 + a + b * p + c * np.log(1.0 / p) / p


def optimal_alpha(p, alpha0, p0, k):
    """The optimal damping `alpha*(p) = alpha0 / (1 + (p0/p)^k)`."""
    return alpha0 / (1.0 + (p0 / p) ** k)
