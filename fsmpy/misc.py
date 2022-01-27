import numpy as np

from .utils._measure_input_validation import check_sets_cardinality
from .sets import FuzzySet


def fuzzy_divergence(A: FuzzySet, B: FuzzySet):
    """ Fuzzy Divergence proposed by J. Fan, W. Xie, from the related article: 
    "Distance measure and induced fuzzy entropy"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)
    ab_diff = A.membership_values - B.membership_values
    ba_diff = B.membership_values - A.membership_values

    return np.sum(
        2.0 -
        (1.0 - A.membership_values + B.non_membership_values) * np.exp(ab_diff) -
        (1.0 - B.membership_values + A.membership_values) * np.exp(ba_diff)
    )


def fuzzy_index(A: FuzzySet, coeff: int):
    """ Fuzzy Index  T. Chaira, A.R. Ray, from the related article: 
    "Threshold selection using fuzzy set theory"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    coeff: int
        Coefficient of the index.
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    n = len(A)
    multiplier = 2.0 / float(np.power(n, 1.0 / coeff))

    return multiplier * np.sum(
        np.minimum(
            A.membership_values, 1.0 - A.membership_values
        ) ** coeff
    ) ** 1.0 / coeff


