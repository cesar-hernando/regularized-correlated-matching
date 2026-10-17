"""Statistics of logical error rates measured on shared shots."""

import numpy as np
from scipy.stats import binomtest


def logical_error_rate(errors, shots):
    """The logical error rate and its binomial standard error."""
    rate = errors / shots
    return rate, np.sqrt(rate * (1 - rate) / shots)


def error_rate_ratio(errors, reference_errors, shots):
    """
    The ratio of a decoder's logical error rate to a reference decoder's, over the same
    shots, with its standard error propagated from both binomial errors.
    """
    rate, std = logical_error_rate(errors, shots)
    reference, reference_std = logical_error_rate(reference_errors, shots)
    ratio = rate / reference
    return ratio, ratio * np.sqrt((std / rate) ** 2 + (reference_std / reference) ** 2)


def mcnemar_p_value(only_this_fails, only_reference_fails):
    """
    Exact two-sided McNemar p-value for two decoders on the same shots, from the number of
    shots where only one of them fails.
    """
    discordant = int(only_this_fails + only_reference_fails)
    if discordant == 0:
        return 1.0
    return binomtest(int(only_this_fails), discordant, 0.5).pvalue
