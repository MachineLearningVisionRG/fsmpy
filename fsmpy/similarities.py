import numpy as np
from typing import Iterable
import warnings

from .utils._measure_input_validation import check_weights, check_p, check_sets_cardinality
from .sets import FuzzySet, IntuitionisticFuzzySet
from . import HUNG_YANG_1_SIMILARITY_1, HUNG_YANG_1_SIMILARITY_2, HUNG_YANG_1_SIMILARITY_3
from . import HUNG_YANG_2_SIMILARITY_1, HUNG_YANG_2_SIMILARITY_2, HUNG_YANG_2_SIMILARITY_3
from . import HUNG_YANG_3_SIMILARITY_1, HUNG_YANG_3_SIMILARITY_2, HUNG_YANG_3_SIMILARITY_3, HUNG_YANG_3_SIMILARITY_4, HUNG_YANG_3_SIMILARITY_5, HUNG_YANG_3_SIMILARITY_6, HUNG_YANG_3_SIMILARITY_7
from . import HUNG_YANG_4_SIMILARITY_1, HUNG_YANG_4_SIMILARITY_2, HUNG_YANG_4_SIMILARITY_3
from . import LIANG_SHI_SIMILARITY_1, LIANG_SHI_SIMILARITY_2, LIANG_SHI_SIMILARITY_3
from . import IANCU_SIMILARITY_1, IANCU_SIMILARITY_2, IANCU_SIMILARITY_3, IANCU_SIMILARITY_4, IANCU_SIMILARITY_5, IANCU_SIMILARITY_6, IANCU_SIMILARITY_7, IANCU_SIMILARITY_8, IANCU_SIMILARITY_9, IANCU_SIMILARITY_10, \
    IANCU_SIMILARITY_11, IANCU_SIMILARITY_12, IANCU_SIMILARITY_13, IANCU_SIMILARITY_14, IANCU_SIMILARITY_15, IANCU_SIMILARITY_16, IANCU_SIMILARITY_17, IANCU_SIMILARITY_18, IANCU_SIMILARITY_19, IANCU_SIMILARITY_20
from . import DENG_JIANG_FU_MONOTONIC_TYPE_1_1, DENG_JIANG_FU_MONOTONIC_TYPE_1_2, \
    DENG_JIANG_FU_MONOTONIC_TYPE_1_3, DENG_JIANG_FU_MONOTONIC_TYPE_1_4, DENG_JIANG_FU_MONOTONIC_TYPE_2_1, \
    DENG_JIANG_FU_MONOTONIC_TYPE_2_2, DENG_JIANG_FU_MONOTONIC_TYPE_2_3, DENG_JIANG_FU_MONOTONIC_TYPE_2_4, \
    DENG_JIANG_FU_MONOTONIC_TYPE_3_1, DENG_JIANG_FU_MONOTONIC_TYPE_3_2, DENG_JIANG_FU_MONOTONIC_TYPE_3_3


def dengfeng_chuntian(A: FuzzySet, B: FuzzySet, p: int = 1, weights: Iterable = None):
    """ Similarity proposed by L. Dengfeng and C. Chuntian, from the related article: 
    "New similarity measures of intuitionistic fuzzy sets and application to pattern recognition"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    p : int
        Positive integer >= 1.
    weights : list of floats
        List of weights for each membership/non-membership value.
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)
    check_p(p)
    check_weights(weights, len(A))

    fA = (A.membership_values + 1 - A.non_membership_values) / 2.0
    fB = (B.membership_values + 1 - B.non_membership_values) / 2.0
    n = len(A)
    if weights is None:
        return 1 - 1 / (n ** (1 / p)) * (np.sum(np.absolute(fA - fB) ** p) ** (1 / p))
    else:
        return 1 - np.power(np.sum(weights * np.power(np.absolute(fA - fB), p)), (1.0 / float(p)))


def liang_shi(A: FuzzySet, B: FuzzySet,  similarity_type: str = LIANG_SHI_SIMILARITY_1, p: int = 1, weights: Iterable = None , omegas: Iterable = [0.5, 0.3, 0.2]):
    """ Similarity proposed by Z. Liang and P. Shi, from the related article: 
    "Similarity measures on intuitionistic fuzzy sets""

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    similarity_type : str, optional 
        Type of computed similarity:
        
        >>> LIANG_SHI_SIMILARITY_1
        >>> LIANG_SHI_SIMILARITY_2
        >>> LIANG_SHI_SIMILARITY_3

    p : int
        Positive integer >= 1.
    weights : list of floats
        List of weights for each membership/non-membership value.
    omegas: Iterable
        An iterable with 3 elements, with their sum equal to 1.
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)
    check_p(p)
    check_weights(weights, len(A), similarity_type, LIANG_SHI_SIMILARITY_3)

    if similarity_type == LIANG_SHI_SIMILARITY_3:
        if weights is None:
            weights = np.full(len(A), 1 / float(len(A)))

        omegas = np.array(omegas)
        if len(omegas.shape) == 1:
            omegas = np.expand_dims(omegas, axis=1)

        if not np.isclose(np.sum(omegas), 1.0):
            raise ValueError(
                "Sum of omegas parameter must be equal to 1, not {}".format(np.sum(omegas))
            )

    if similarity_type == LIANG_SHI_SIMILARITY_1:
        n = len(A)
        delta_memberships = np.absolute(A.membership_values - B.membership_values)
        delta_non_memberships = np.absolute(A.non_membership_values - B.non_membership_values)

        return 1.0 - 1.0 / np.power(n, 1.0 / p) * np.power(
                np.sum(
                    np.power((delta_memberships + delta_non_memberships) / 2.0, p)
                ),
                1.0 / p
            )
    elif similarity_type == LIANG_SHI_SIMILARITY_2:
        n = len(A)
        mA = (A.membership_values + 1.0 - A.non_membership_values) / 2.0
        mB = (B.membership_values + 1.0 - B.non_membership_values) / 2.0
        mA1 = (A.membership_values + mA) / 2.0
        mB1 = (B.membership_values + mB) / 2.0
        mA2 = (mA + 1.0 - A.non_membership_values) / 2.0
        mB2 = (mB + 1.0 - B.non_membership_values) / 2.0
        fS1 = np.absolute(mA1 - mB1) / 2.0
        fS2 = np.absolute(mA2 - mB2) / 2.0
        return 1.0 - (1.0 / np.power(n, 1.0 / p)) * np.power(
            np.sum(
                np.power(fS1 + fS2, p)
            ),
            1.0 / p
        )
    elif similarity_type == LIANG_SHI_SIMILARITY_3:
        delta_memberships = np.absolute(A.membership_values - B.membership_values)
        delta_non_memberships = np.absolute(A.non_membership_values - B.non_membership_values)

        f1 = (delta_memberships + delta_non_memberships) / 2.0
        mA = (A.membership_values + 1.0 - A.non_membership_values) / 2.0
        mB = (B.membership_values + 1.0 - B.non_membership_values) / 2.0
        f2 = np.absolute(mA - mB)
        iA = mA - A.membership_values
        iB = mB - B.membership_values
        f3 = np.maximum(iA, iB) - np.minimum(iA, iB)
        fs = np.array([f1, f2, f3])
        return 1.0 - np.power(
            np.sum(
                weights * np.power(np.sum(omegas * fs), p)
            ),
            1.0 / p
        )
    else:
        raise ValueError(
            "similarity_type parameter must be LIANG_SHI_SIMILARITY_1, LIANG_SHI_SIMILARITY_2, LIANG_SHI_SIMILARITY_3 or LIANG_SHI_SIMILARITY_4.")


def park_kwun_lim(A: IntuitionisticFuzzySet, B: IntuitionisticFuzzySet, p: int = 1, weights: Iterable = None):
    """ Similarity proposed by A.H. Park, A.S. Park, Y.C. Kwun and K.M. Lim, from the related article:
    "New Similarity Measures on Intuitionistic Fuzzy Sets".

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    p : int
        Positive integer >= 1.
    weights : list of floats
        List of weights for each membership/non-membership value.
    
        
    Returns
    -------
    numpy.float64
    The similarity between the two sets provided.

    """
    check_sets_cardinality(A, B)
    check_p(p)

    n = len(A)
    delta_memberships = np.abs(A.membership_values - B.membership_values)
    delta_non_memberships = np.abs(A.non_membership_values - B.non_membership_values)
    delta_hesitations = np.abs(A.hesitation_degrees - B.hesitation_degrees)
    if weights is None:
        return 1.0 - (1.0 / np.power(n, 1.0 / p)) * np.power(
            np.sum(
                np.power(delta_memberships + delta_non_memberships + delta_hesitations, p)
            ) / 2.0,
            1.0 / p
        )
    else:
        check_weights(weights, len(A))
        return 1.0 - np.power(
            np.sum(
                weights * np.power(delta_memberships + delta_non_memberships + delta_hesitations, p)
            ) / 2.0,
            1.0 / p
        )


def mitchell(A: FuzzySet, B: FuzzySet, p: int = 1, weights: Iterable = None):
    """ Similarity proposed by H.B. Mitchell, from the related article: 
    "On the Dengfeng-Chuntian similarity measure and its application to pattern recognition"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    p : int
        Positive integer >= 1.
    weights : list of floats
        List of weights for each membership/non-membership value.
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)
    check_p(p)
    check_weights(weights, len(A))

    if weights is None:
        weights = np.full(len(A), 1.0 / float(len(A)))
    
    D_m = 1.0 - np.power(
        np.sum(
            weights * 
            np.power(np.absolute(A.membership_values - B.membership_values), p)
        ), (1.0 / p)
    )
    D_f = 1.0 - np.power(
        np.sum(
            weights * 
            np.power(np.absolute(A.non_membership_values - B.non_membership_values), p)
        ), (1.0 / p)
    )

    return (D_m + D_f) / 2.0


def julian_hung_lin(A: FuzzySet, B: FuzzySet, p: int = 1, weights: Iterable = None):
    """ Similarity proposed by P. Julian, K.C. Hung and S.J. Lin, from the related article: 
    "On the Mitchell similarity measure and its application to pattern recognition"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    p : int
        Positive integer >= 1.
    weights : list of floats
        List of weights for each membership/non-membership value.
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)
    check_p(p)
    check_weights(weights, len(A))

    if weights is None:
        weights = np.full(len(A), 1 / float(len(A)))
    
    fA = 1.0 - A.non_membership_values
    fB = 1.0 - B.non_membership_values
    return 1.0 - np.power(
        np.sum(
            weights * 
            np.power(np.absolute(A.membership_values - B.membership_values), p)
        ), 1.0 / p
    ) + np.power(
        np.sum(
            weights * 
            np.power(np.absolute(fA - fB), p)
        ), 1.0 / p
    )


def hung_yang_1(A: FuzzySet, B: FuzzySet, similarity_type: str = HUNG_YANG_1_SIMILARITY_1, weights: Iterable = None):
    """ Similarity proposed by W.L. Hung and M.S. Yang, from the related article: 
    "Similarity measures of intuitionistic fuzzy sets based on Hausdorff similarity"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    similarity_type: str, optional
        Type of computed similarity:
        
        >>> HUNG_YANG_1_SIMILARITY_1
        >>> HUNG_YANG_1_SIMILARITY_2
        >>> HUNG_YANG_1_SIMILARITY_3
    
    weights: List of weights for each membership/non-membership value.
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)

    if weights is not None:
        weights = np.array(weights)

    if weights is None:
        D = np.sum(np.maximum(np.absolute(A.membership_values - B.membership_values),
                   np.absolute(A.non_membership_values - B.non_membership_values))) / float(len(A))
    else:
        check_weights(weights, len(A))
        D = np.sum(weights * np.maximum(np.absolute(A.membership_values - B.membership_values),
                   np.absolute(A.non_membership_values - B.non_membership_values)))

    if similarity_type == HUNG_YANG_1_SIMILARITY_1:
        return 1 - D
    elif similarity_type == HUNG_YANG_1_SIMILARITY_2:
        tp = np.exp(-1)
        return (np.exp(-D) - tp) / float(1 - tp)
    elif similarity_type == HUNG_YANG_1_SIMILARITY_3:
        return (1 - D) / float(1 + D)
    else:
        raise ValueError(
            "similarity_type parameter must be HUNG_YANG_1_SIMILARITY_1, HUNG_YANG_1_SIMILARITY_2 or HUNG_YANG_1_SIMILARITY_3.")


def ye(A: FuzzySet, B: FuzzySet, weights: Iterable = None):
    """ Similarity proposed by J. Ye, from the related article: 
    "Cosine similarity measures for intuitionistic fuzzy sets and their applications"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    weights : list of floats
        List of weights for each membership/non-membership value.
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)

    n = len(A)
    nominator = (
        A.membership_values * B.membership_values + 
        A.non_membership_values * B.non_membership_values
    )
    denominator = (
        np.sqrt(A.membership_values ** 2.0 + A.non_membership_values ** 2.0) *
        np.sqrt(B.membership_values ** 2.0 + B.non_membership_values ** 2.0)
    )

    if weights is None:
        return 1.0 / n * np.sum(nominator / denominator)
    else:
        check_weights(weights, len(A))
        return sum(weights * nominator / denominator)


def hwang_yang(A: FuzzySet, B: FuzzySet):
    """ Similarity proposed by C.M. Hwang and M.S. Yang, from the related article: 
    "Modified cosine similarity measure between intuitionistic fuzzy sets"

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

    n = len(A)
    
    C1 = ye(A, B)

    fA = 1.0 + A.membership_values - A.non_membership_values
    fB = 1.0 + B.membership_values - B.non_membership_values
    
    nominator = fA * fB + (4.0 * A.non_membership_values * B.non_membership_values)
    
    denominator = (
        np.sqrt(fA ** 2.0 + (2.0 * A.non_membership_values) ** 2.0) *
        np.sqrt(fB ** 2.0 + (2.0 * B.non_membership_values) ** 2.0)
    )
    C2 = 1.0 / n * np.sum(nominator / denominator)

    nominator = (
        ((1.0 - A.membership_values) * (1.0 - B.membership_values)) + 
        ((1.0 - A.non_membership_values) * (1.0 - B.non_membership_values))
    )
    denominator = (
        np.sqrt((1.0 - A.membership_values) ** 2.0 + (1.0 - B.membership_values) ** 2.0) *
        np.sqrt((1.0 - A.non_membership_values) ** 2.0 + (1.0 - B.non_membership_values) ** 2.0)
    )
    C3 = 1.0 / n * np.sum(nominator / denominator)

    return (C1 + C2 + C3) / 3.0


def hung_yang_2(A: IntuitionisticFuzzySet, B: IntuitionisticFuzzySet, similarity_type: str = HUNG_YANG_2_SIMILARITY_1, a: int = 1):
    """ Similarity proposed by W.L. Hung and M.S. Yang, from the related article: 
    "On the J-divergence of intuitionistic fuzzy sets with its applications to pattern recognition"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    similarity_type : str, optional
        Type of computed similarity:

        >>> HUNG_YANG_2_SIMILARITY_1
        >>> HUNG_YANG_2_SIMILARITY_2
        >>> HUNG_YANG_2_SIMILARITY_3

    a: case of divergence measure. Positive integer >= 1.
    
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    Raises:
        ValueError if a is < 1.
    """
    def _log(arr):
        invalids = np.logical_or(
            arr <= 0, 
            np.isnan(arr), 
            np.isinf(arr)
        )
        invalid_indxs = np.where(invalids)
        valid_indxs = np.where(np.logical_not(invalids))
        arr[invalid_indxs] = 0.0
        arr[valid_indxs] = np.log(arr[valid_indxs])
        return arr
    
    check_sets_cardinality(A, B)
    
    n = len(A)
    if a < 1 and not isinstance(a, int):
        raise ValueError("a parameter must be an integer >= 1.")

    if a == 1:
        U_a = np.log(2.0)

        LABm = (
            (A.membership_values + B.membership_values) * _log((A.membership_values + B.membership_values) / 2) -
            A.membership_values * _log(A.membership_values) - 
            B.membership_values * _log(B.membership_values)
        )

        LABv = (
            (A.non_membership_values + B.non_membership_values) * _log((A.non_membership_values + B.non_membership_values) / 2) - 
            A.non_membership_values *  _log(A.non_membership_values) - 
            B.non_membership_values * _log(B.non_membership_values)
        )

        LABp = (
            (A.hesitation_degrees + B.hesitation_degrees) * _log((A.hesitation_degrees + B.hesitation_degrees) / 2) -
            A.hesitation_degrees * _log(A.hesitation_degrees) - 
            B.hesitation_degrees * _log(B.hesitation_degrees)
        )
        ja = -0.5 * (LABm + LABv + LABp)
    else:
        U_a = 1.0 / (float(a) - 1.0) * (1 - 1 / float(2.0 ** (a - 1.0)))
        TmAB = (
                np.power((A.membership_values + B.membership_values) / 2.0, a) - 
            0.5 * (
                np.power(A.membership_values, a) + 
                np.power(B.membership_values, a)
            )
        )
        TvAB = (
            np.power((A.non_membership_values + B.non_membership_values) / 2.0, a) - 
            0.5 * (
                np.power(A.non_membership_values, a) +
                np.power(B.non_membership_values, a)
            )
        )
        TpAB = (
            np.power((A.hesitation_degrees + B.hesitation_degrees) / 2.0, a) - 
            0.5 * (
                np.power(A.hesitation_degrees, a) +
                np.power(B.hesitation_degrees, a)
            )
        )
        ja = -1 / (a - 1) * (TmAB + TvAB + TpAB)

    J_a = np.sum(ja) / float(n)

    if (similarity_type == HUNG_YANG_2_SIMILARITY_1):
        return (U_a - J_a) / U_a
    elif (similarity_type == HUNG_YANG_2_SIMILARITY_2):
        return (np.exp(-J_a) - np.exp(-U_a)) / float(1 - np.exp(-U_a))
    elif (similarity_type == HUNG_YANG_2_SIMILARITY_3):
        return (U_a - J_a) / np.array((1 + J_a) * U_a)
    else:
        raise ValueError(
            "similarity_type parameter must be HUNG_YANG_2_SIMILARITY_1, HUNG_YANG_2_SIMILARITY_2 or HUNG_YANG_2_SIMILARITY_3.")


def zhang_fu(A: FuzzySet, B: FuzzySet):
    """ Similarity proposed by C. Zhang and H. Fu, from the related article: 
    "Similarity measures on three kinds of fuzzy sets"

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
    
    n = len(A)
    dA = A.membership_values + (1.0 - A.membership_values - A.non_membership_values) * A.membership_values
    aA = A.non_membership_values + (1.0 - A.membership_values - A.non_membership_values) * A.non_membership_values

    dB = B.membership_values + (1.0 - B.membership_values - B.non_membership_values) * B.membership_values
    aB = B.non_membership_values + (1.0 - B.membership_values - B.non_membership_values) * B.non_membership_values
    
    return 1.0 - (1.0 / (2.0 * n)) * np.sum(np.abs(dA - dB) + np.abs(aA - aB))


def hung_yang_3(A: FuzzySet, B: FuzzySet, similarity_type: str = HUNG_YANG_3_SIMILARITY_1):
    """ Similarity proposed by W.L. Hung and M.S. Yang, from the related article: 
    "On similarity measures between intuitionistic fuzzy sets"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    similarity_type : str, optional
        Type of computed similarity:

        >>> HUNG_YANG_3_SIMILARITY_1
        >>> HUNG_YANG_3_SIMILARITY_2
        >>> HUNG_YANG_3_SIMILARITY_3
        >>> HUNG_YANG_3_SIMILARITY_4
        >>> HUNG_YANG_3_SIMILARITY_5
        >>> HUNG_YANG_3_SIMILARITY_6
        >>> HUNG_YANG_3_SIMILARITY_7
        
    Returns
    -------
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)

    n = float(len(A))

    if similarity_type == HUNG_YANG_3_SIMILARITY_1:
        num = np.minimum(A.membership_values, B.membership_values) + \
            np.minimum(A.non_membership_values, B.non_membership_values)
        denom = np.maximum(A.membership_values, B.membership_values) + \
            np.maximum(A.non_membership_values, B.non_membership_values)
        D = np.sum(np.divide(num, denom.astype(float)))
        return D / n
    elif similarity_type == HUNG_YANG_3_SIMILARITY_2:
        ptr2 = 0.5 * np.absolute(A.membership_values - B.membership_values) + \
            np.absolute(A.non_membership_values - B.non_membership_values)
        D = np.sum(np.subtract(1, ptr2))
        return D / n
    elif similarity_type == HUNG_YANG_3_SIMILARITY_3:
        num = np.minimum(A.membership_values, B.membership_values) + \
            np.minimum(A.non_membership_values, B.non_membership_values)
        denom = np.maximum(A.membership_values, B.membership_values) + \
            np.maximum(A.non_membership_values, B.non_membership_values)
        D1 = np.sum(num)
        D2 = np.sum(denom)
        return D1 / float(D2)
    elif similarity_type == HUNG_YANG_3_SIMILARITY_4:
        dif_m = np.absolute(A.membership_values - B.membership_values)
        dif_v = np.absolute(A.non_membership_values - B.non_membership_values)
        return 1 - 0.5 * (dif_m.max() + dif_v.max())
    elif similarity_type == HUNG_YANG_3_SIMILARITY_5:
        num = np.absolute(A.membership_values - B.membership_values) + \
            np.absolute(A.non_membership_values - B.non_membership_values)
        denom = np.absolute(A.membership_values + B.membership_values) + \
            np.absolute(A.non_membership_values + B.non_membership_values)
        D1 = np.sum(num)
        D2 = np.sum(denom)
        return 1 - (D1 / float(D2))
    elif similarity_type == HUNG_YANG_3_SIMILARITY_6:
        num = 1 - np.exp(-0.5 * sum(np.absolute(A.membership_values - B.membership_values) +
                         np.absolute(A.non_membership_values - B.non_membership_values)))
        denum = 1 - np.exp(-n)
        return 1 - num / denum
    elif similarity_type == HUNG_YANG_3_SIMILARITY_7:
        num = 1 - np.exp(
            -0.5 * np.sum(np.absolute(np.sqrt(A.membership_values) - np.sqrt(B.membership_values)) + np.absolute(np.sqrt(A.non_membership_values) - np.sqrt(B.non_membership_values))))
        denum = 1 - np.exp(-n)
        return 1 - num / denum
    else:
        raise ValueError(
            "similarity_type parameter must be HUNG_YANG_3_SIMILARITY_1, HUNG_YANG_3_SIMILARITY_2, HUNG_YANG_3_SIMILARITY_3, HUNG_YANG_3_SIMILARITY_4, HUNG_YANG_3_SIMILARITY_,5, HUNG_YANG_3_SIMILARITY_6 or HUNG_YANG_3_SIMILARITY_7.")


def chen_1(A: FuzzySet, B: FuzzySet, weights: Iterable = None):
    """ Similarity proposed by S.M. Chen, from the related article: 
    "Measures of similarity between vague sets"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    weights : list of floats
        List of weights for each membership/non-membership value.
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)
    check_weights(weights, len(A))

    if weights is None:
        weights = np.full(len(A), 1.0 / len(A))

    sA = A.membership_values - A.non_membership_values
    sB = B.membership_values - B.non_membership_values
    
    return np.sum(
        weights * (
            1.0 - np.absolute((sA - sB) / 2.0)
        )
    ) / np.sum(weights)


def hung_yang_4(A: FuzzySet, B: FuzzySet, similarity_type: str = HUNG_YANG_4_SIMILARITY_1, p: int = 1):
    """ Similarity proposed by W.L. Hung and M.S. Yang, from the related article: 
    "Similarity measures of intuitionistic fuzzy sets based on Lp metric"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    similarity_type : str, optional
        Type of computed similarity:

        >>> HUNG_YANG_4_SIMILARITY_1
        >>> HUNG_YANG_4_SIMILARITY_2 
        >>> HUNG_YANG_4_SIMILARITY_3

    p : int
        Positive integer >= 1.
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)
    check_p(p)
    
    D = np.sum(np.power((np.absolute(np.power(A.membership_values - B.membership_values, p)) +
               np.power(np.absolute(A.non_membership_values - B.non_membership_values), p)), (1.0 / p)))

    D = D / float(len(A))
    f1 = 2 ** (1.0 / float(p))

    if similarity_type == HUNG_YANG_4_SIMILARITY_1:
        return (f1 - D) / float(f1)
    elif similarity_type == HUNG_YANG_4_SIMILARITY_2:
        return (np.exp(-D) - np.exp(-f1)) / float(1 - np.exp(-f1))
    elif similarity_type == HUNG_YANG_4_SIMILARITY_3:
        return (f1 - D) / float(f1 * (1.0 + D))
    else:
        raise ValueError(
            "similarity_type parameter must be HUNG_YANG_4_SIMILARITY_1, HUNG_YANG_4_SIMILARITY_2 or HUNG_YANG_4_SIMILARITY_3.")


def hong_kim(A: FuzzySet, B: FuzzySet, weights: Iterable = None, a: int = 1, b: int = 0, c: int = 0):
    """ Similarity proposed by D.H. Hong and C.Kim, from the related article: 
    "A note on similarity measures between vague sets and between elements"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    weights : list of floats
        List of weights for each membership/non-membership value.
    a, b, c: int
        Must satisfy the condition: a >= c >= 0 >= b.
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)
    check_weights(weights, len(A))
    n = len(A)
    
    if not (a >= c >= 0 >= b):
        raise ValueError(
            "Parameters a, b and c must satisfy the condition a >= c >= 0 >= 0, got {} {} {}".format(a, b, c))

    if weights is None:
        D = np.sum(
            1 - (
                np.abs(A.membership_values - B.membership_values) +
                np.abs(A.non_membership_values - B.non_membership_values)
            ) / 2.0
        )
        return D / float(n)
    else:
        denom = float(a + b + c)

        num = (a * np.absolute(A.membership_values - B.membership_values)) + (b * np.absolute(A.non_membership_values - B.non_membership_values)
                                                                              ) + (c * np.absolute(B.membership_values + B.non_membership_values - A.membership_values - A.non_membership_values))

        D = np.sum(weights * (1 - num / denom))

        return D


def chen_2(A: FuzzySet, B: FuzzySet, weights: Iterable = None, a: int = 1, b: int = 0, c: int = 0) -> float:
    """ Similarity proposed by S.M. Chen, from the related article: 
        "Similarity measure between vague sets and between elements"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    weights : list of floats
        List of weights for each membership/non-membership value.
    a, b, c: int
        Must satisfy the condition: a >= c >= 0 >= b.
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)
    check_weights(weights, len(A))

    if not (a >= c >= 0 >= b):
        raise ValueError(
            "a, b and c parameters must meet the condition: a >= c >= 0 >= b.")

    if weights is None:
        weights = np.full(len(A), 1 / float(len(A)))

    denom = a - b
    num = (a * (A.membership_values - B.membership_values)) + (b * (A.non_membership_values - B.non_membership_values)
                                                               ) - (c * (A.membership_values - B.membership_values + A.non_membership_values - B.non_membership_values))
    D = np.sum(weights * (1 - np.absolute(num) / float(denom)))
    D = np.array(D)

    return D / float(np.sum(weights))


def liu(A: IntuitionisticFuzzySet, B: IntuitionisticFuzzySet, p: int = 1, weights: Iterable = None, a: float = 0.4, b: float = 0.3, c: float = 0.3):
    """ Similarity proposed by H.W. Liu, from the related article: 
    "New similarity measures between intuitionistic fuzzy sets and between elements"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    p : int
        Positive integer >= 1.
    weights : list of floats
        List of weights for each membership/non-membership value.
    a, b, c: float
        Sum of those parameters must be equal to 1.
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)
    check_p(p)
    check_weights(weights, len(A))
    n = len(A)

    if weights is None:
        D = np.sum((np.power(np.absolute(A.membership_values - B.membership_values), p)) + (np.power(np.absolute(A.non_membership_values - B.non_membership_values), p)) + (
            np.power(np.absolute(A.hesitation_degrees - B.hesitation_degrees), p)))
        return 1 - (D / float(2.0 * n)) ** (1.0 / p)
    else:
        prts = (a * np.power(np.absolute(A.membership_values - B.membership_values), p)) + (b * np.power(np.absolute(A.non_membership_values - B.non_membership_values), p)) + (
            c * np.power(np.absolute(A.hesitation_degrees - B.hesitation_degrees), p))
        D = np.sum(weights * prts)
        return 1 - D ** (1.0 / p)


def iancu(A: FuzzySet, B: FuzzySet, similarity_type: int = IANCU_SIMILARITY_1, lamda: int = 1):
    """ Similarities proposed by I. Iancu, from the related article: 
    "Intuitionistic fuzzy similarity measures based on Frank t-norms family"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    similarity_type : int, optional
        Type of computed similarity:

        >>> IANCU_SIMILARITY_1 IANCU_SIMILARITY_2 ..., IANCU_SIMILARITY_20
    
    lambda: float
        Frank family of t-operator parameter. Different cases of input:
        0, 1, Inf and other. Used in all similarity_type cases *except*
        
        >>> IANCU_SIMILARITY_1, IANCU_SIMILARITY_2,  IANCU_SIMILARITY_3 
        >>> IANCU_SIMILARITY_4, IANCU_SIMILARITY_18
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)
    def tOperator(a: Iterable, b: Iterable, lamda: float):
        if (lamda == np.inf):
            return np.maximum(0, a + b - 1)
        elif (lamda == 0):
            return np.minimum(a, b)
        elif (lamda == 1):
            return a * b
        else:
            return np.log(1.0 + (
                (
                    (np.power(lamda, 1 - a) - 1.0) * (np.power(lamda, 1.0 - b) - 1.0)
                ) / (lamda - 1.0)
            ))
    delta_A = A.membership_values - A.non_membership_values
    delta_B = B.membership_values - B.non_membership_values
    n = len(A)
    if similarity_type == IANCU_SIMILARITY_1:  # S11
        return (
            (n + np.minimum(np.sum(delta_A), np.sum(delta_B))) /
            (n + np.maximum(np.sum(delta_A), np.sum(delta_B)))
        )
    elif similarity_type == IANCU_SIMILARITY_2:  # S11c
        nominator = n - np.maximum(np.sum(delta_A), np.sum(delta_B))
        denom = n - np.minimum(np.sum(delta_A), np.sum(delta_B))
        return nominator / np.array(denom).astype("float")
    elif similarity_type == IANCU_SIMILARITY_3:  # S19
        nominator = n + np.minimum(np.sum(delta_A),
                                   np.sum(delta_B))
        return nominator / (2.0 * n)
    elif similarity_type == IANCU_SIMILARITY_4:  # S19c
        nominator = n - np.maximum(np.sum(delta_A),
                                   np.sum(delta_B))
        return nominator / (2.0 * n)
    elif similarity_type == IANCU_SIMILARITY_5:  # S1
        return (
            (
                n + np.sum(
                    tOperator(A.membership_values, B.membership_values, lamda) +
                    tOperator(A.non_membership_values, B.non_membership_values, lamda) -
                    A.non_membership_values - B.non_membership_values
                    )
             ) / 
             (
                 n + np.max(np.sum(delta_A), np.sum(delta_B))
             )
        )
    elif similarity_type == IANCU_SIMILARITY_6:  # S1c
        nom = n + np.sum(tOperator(A.membership_values, B.membership_values, lamda) + tOperator(
            A.non_membership_values, B.non_membership_values, lamda) - A.membership_values - B.membership_values)
        denom = n - np.minimum(np.sum(delta_A),
                               np.sum(delta_B))
        return nom / np.array(denom).astype("float")
    elif similarity_type == IANCU_SIMILARITY_7:  # S5
        nom = n + np.sum(tOperator(A.membership_values, B.membership_values, lamda) + tOperator(
            A.non_membership_values, B.non_membership_values, lamda) - A.non_membership_values - B.non_membership_values)
        denom = n + np.sum(A.membership_values + B.membership_values - tOperator(A.membership_values,
                           B.membership_values, lamda) - tOperator(A.non_membership_values, B.non_membership_values, lamda))
        return nom / np.array(denom).astype("float")
    elif similarity_type == IANCU_SIMILARITY_8:  # S5c
        nom = n + np.sum(tOperator(A.membership_values, B.membership_values, lamda) + tOperator(
            A.non_membership_values, B.non_membership_values, lamda) - A.membership_values - B.membership_values)
        denom = n + np.sum(A.non_membership_values + B.non_membership_values - tOperator(A.membership_values,
                           B.membership_values, lamda) - tOperator(A.non_membership_values, B.non_membership_values, lamda))
        return nom / np.array(np.sum(denom)).astype("float")
    elif similarity_type == IANCU_SIMILARITY_9:  # S14
        nom = n + np.minimum(np.sum(delta_A),
                             np.sum(delta_B))
        denom = n + np.sum(A.membership_values + B.membership_values - tOperator(A.membership_values,
                                                                                 B.membership_values, lamda) - tOperator(A.non_membership_values, B.non_membership_values, lamda))
        return nom / float(denom)
    elif similarity_type == IANCU_SIMILARITY_10:  # S14c
        nom = n - np.maximum(np.sum(delta_A),
                             np.sum(delta_B))
        denom = n + np.sum(A.non_membership_values + B.non_membership_values - tOperator(A.membership_values,
                                                                                         B.membership_values, lamda) - tOperator(A.non_membership_values, B.non_membership_values, lamda))
        return nom / float(denom)
    elif similarity_type == IANCU_SIMILARITY_11:  # S18
        nom = n + np.sum(tOperator(A.membership_values, B.membership_values, lamda) + tOperator(
            A.non_membership_values, B.non_membership_values, lamda) - A.non_membership_values - B.non_membership_values)
        denom = 2.0 * n
        return nom / denom
    elif similarity_type == IANCU_SIMILARITY_12:  # S18c
        nom = n - np.sum(tOperator(A.membership_values, B.membership_values, lamda) + tOperator(
            A.non_membership_values, B.non_membership_values, lamda) - A.membership_values - B.membership_values)
        denom = 2.0 * n
        return nom / denom
    elif similarity_type == IANCU_SIMILARITY_13:  # S2
        nom = 2.0 * n + np.sum(
            2.0 * tOperator(A.membership_values, B.membership_values, lamda) + 2.0 * tOperator(A.non_membership_values, B.non_membership_values, lamda) - A.membership_values - B.membership_values - A.non_membership_values - B.non_membership_values)
        denom = 2.0 * n + np.sum(tOperator(A.membership_values, B.membership_values, lamda) + tOperator(A.non_membership_values, B.non_membership_values, lamda)) - np.minimum(
            np.sum(A.membership_values + B.non_membership_values), np.sum(B.membership_values + A.non_membership_values))
        return nom / denom
    elif similarity_type == IANCU_SIMILARITY_14:  # S6
        nom = 2.0 * n + np.sum(
            2.0 * tOperator(A.membership_values, B.membership_values, lamda) + 2.0 * tOperator(A.non_membership_values, B.non_membership_values, lamda) - A.membership_values - B.membership_values - A.non_membership_values - B.non_membership_values)
        denom = 2.0 * n
        return nom / denom
    elif similarity_type == IANCU_SIMILARITY_15:  # S13
        nom = np.minimum(np.sum(A.membership_values + B.non_membership_values), np.sum(B.membership_values + A.non_membership_values)) - np.sum(
            tOperator(A.membership_values, B.membership_values, lamda) + tOperator(A.non_membership_values, B.non_membership_values, lamda))
        denom = np.sum(A.membership_values + B.membership_values + A.non_membership_values + B.non_membership_values - 2.0 * tOperator(
            A.membership_values, B.membership_values, lamda) - 2 * tOperator(A.non_membership_values, B.non_membership_values, lamda))
        return nom / denom
    elif similarity_type == IANCU_SIMILARITY_16:  # S15
        nom = 2.0 * n + np.sum(tOperator(A.membership_values, B.membership_values, lamda) + tOperator(A.non_membership_values, B.non_membership_values, lamda)) - np.maximum(np.sum(A.membership_values + B.non_membership_values),
                                                                                                                                                                             np.sum(B.membership_values + A.non_membership_values))
        denom = 2.0 * n
        return nom / denom
    elif similarity_type == IANCU_SIMILARITY_17:  # S2"
        nom = np.sum(A.membership_values + B.membership_values + A.non_membership_values + B.non_membership_values - 2.0 * tOperator(
            A.membership_values, B.non_membership_values, lamda) - 2.0 * tOperator(B.membership_values, A.non_membership_values, lamda))
        denom = n - np.sum(tOperator(A.membership_values, B.non_membership_values, lamda) + tOperator(B.membership_values, A.non_membership_values, lamda)) + np.maximum(np.sum(A.membership_values + B.non_membership_values),
                                                                                                                                                                         np.sum(B.membership_values + A.non_membership_values))
        return nom / denom.astype(float)
    elif similarity_type == IANCU_SIMILARITY_18:  # S6"
        nom = np.sum(A.membership_values + B.membership_values + A.non_membership_values + B.non_membership_values - 2.0 * tOperator(
            A.membership_values, B.non_membership_values, lamda) - 2.0 * tOperator(B.membership_values, A.non_membership_values, lamda))
        denom = 2.0 * n
        return nom / denom
    elif similarity_type == IANCU_SIMILARITY_19:  # S13"
        nom = n + np.sum(tOperator(A.membership_values, B.non_membership_values, lamda) + tOperator(B.membership_values, A.non_membership_values, lamda)) - np.maximum(np.sum(A.membership_values + B.non_membership_values),
                                                                                                                                                                       np.sum(B.membership_values + A.non_membership_values))
        denom = 2.0 * n + np.sum(
            2.0 * tOperator(A.membership_values, B.non_membership_values, lamda) + 2.0 * tOperator(B.membership_values, A.non_membership_values, lamda) - A.membership_values - B.membership_values - A.non_membership_values - B.non_membership_values)
        return nom / denom
    elif similarity_type == IANCU_SIMILARITY_20:  # S15"
        nom = n + np.minimum(np.sum(A.membership_values + B.non_membership_values), np.sum(B.membership_values + A.non_membership_values)) - np.sum(
            tOperator(A.membership_values, B.non_membership_values, lamda) + tOperator(B.membership_values, A.non_membership_values, lamda))
        denom = 2.0 * n
        return nom / denom
    else:
        raise ValueError(
            "similarity_type parameter must be IANCU_SIMILARITY_1, IANCU_SIMILARITY_2, IANCU_SIMILARITY_3, ..., or IANCU_SIMILARITY_20.")


def song_wang_lei_xue(A: IntuitionisticFuzzySet, B: IntuitionisticFuzzySet, weights: Iterable = None):
    """ Similarity proposed by Y. Song, X. Wang, L. Lei, A. Xue, from the related article: 
    "A novel similarity measure on intuitionistic fuzzy sets with its applications"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    weights : list of floats
        List of weights for each membership/non-membership value.
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)
    n = len(A)

    if weights is None:
        ff = (1.0 / (2.0 * n)) * np.sum(
            np.sqrt(A.membership_values * B.membership_values) + (2 * np.sqrt(A.non_membership_values * B.non_membership_values)) + np.sqrt(A.hesitation_degrees * B.hesitation_degrees) + np.sqrt((1 - A.non_membership_values) * (1 - B.non_membership_values)))
        return ff
    else:
        check_weights(weights, len(A))
        return 1.0 / 2.0 * np.sum(
            weights * (np.sqrt(A.membership_values * B.membership_values) + 2.0 * np.sqrt(A.non_membership_values * B.non_membership_values) + np.sqrt(A.hesitation_degrees * B.hesitation_degrees) + np.sqrt((1 - A.non_membership_values) * (1 - B.non_membership_values))))


def intarapaiboon(A: FuzzySet, B: FuzzySet):
    """ Similarity proposed by P. Intarapaiboon, from the related article: 
    "A hierarchy-based similarity measure for intuitionistic fuzzy sets"

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
    n = len(A)
    delta_memberships = np.abs(A.membership_values - B.membership_values)
    delta_non_memberships = np.abs(A.non_membership_values - B.non_membership_values)
    return 1 - (1.0 / (2.0 * n) * np.sum(delta_memberships + delta_non_memberships))


def deng_jiang_fu(A: FuzzySet, B: FuzzySet, similarity_type: int = DENG_JIANG_FU_MONOTONIC_TYPE_1_1, p: int = None, u:float = None, v: float = None):
    """ Similarity proposed by G. Deng, Y. Jiang, J. Fu, from the related article: 
    "Monotonic similarity measures between intuitionistic fuzzy sets and their relationship with entropy and inclusion measure"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    similarity_type : int, optional
        Type of computed similarity:

        >>> DENG_JIANG_FU_MONOTONIC_TYPE_1_1
        >>> DENG_JIANG_FU_MONOTONIC_TYPE_1_2
        >>> DENG_JIANG_FU_MONOTONIC_TYPE_1_3
        >>> DENG_JIANG_FU_MONOTONIC_TYPE_1_4
        >>> DENG_JIANG_FU_MONOTONIC_TYPE_2_1
        >>> DENG_JIANG_FU_MONOTONIC_TYPE_2_2
        >>> DENG_JIANG_FU_MONOTONIC_TYPE_2_3
        >>> DENG_JIANG_FU_MONOTONIC_TYPE_2_4
        >>> DENG_JIANG_FU_MONOTONIC_TYPE_3_1
        >>> DENG_JIANG_FU_MONOTONIC_TYPE_3_2
        >>> DENG_JIANG_FU_MONOTONIC_TYPE_3_3

    p: float 
        must be >= 1. Used in all types *except* 
        
        >>> DENG_JIANG_FU_MONOTONIC_TYPE_1_3 DENG_JIANG_FU_MONOTONIC_TYPE_2_3 DENG_JIANG_FU_MONOTONIC_TYPE_3_1 DENG_JIANG_FU_MONOTONIC_TYPE_3_2 DENG_JIANG_FU_MONOTONIC_TYPE_3_3
    u: float
        Must be positive. Used only in 
        
        >>> DENG_JIANG_FU_MONOTONIC_TYPE_3_2
    v: float
        Must be positive. Used only in
        
        >>> DENG_JIANG_FU_MONOTONIC_TYPE_3_2
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)
    check_p(p, similarity_type, DENG_JIANG_FU_MONOTONIC_TYPE_1_3, DENG_JIANG_FU_MONOTONIC_TYPE_2_3, DENG_JIANG_FU_MONOTONIC_TYPE_3_1, DENG_JIANG_FU_MONOTONIC_TYPE_3_2, DENG_JIANG_FU_MONOTONIC_TYPE_3_3)

    if p is not None and similarity_type not in [DENG_JIANG_FU_MONOTONIC_TYPE_1_3, DENG_JIANG_FU_MONOTONIC_TYPE_2_3, DENG_JIANG_FU_MONOTONIC_TYPE_3_1, DENG_JIANG_FU_MONOTONIC_TYPE_3_2, DENG_JIANG_FU_MONOTONIC_TYPE_3_3]:
        warnings.warn("Ignoring parameter p (not used in provided similarity_type).")
    if u is not None and similarity_type != DENG_JIANG_FU_MONOTONIC_TYPE_3_2:
        warnings.warn("Ignoring parameter u (not used in provided similarity_type).")
    if v is not None and similarity_type != DENG_JIANG_FU_MONOTONIC_TYPE_3_2:
        warnings.warn("Ignoring parameter v (not used in provided similarity_type).")
    
    if similarity_type == DENG_JIANG_FU_MONOTONIC_TYPE_1_1:
        n = len(A)
        delta_memberships = np.abs(A.membership_values - B.membership_values)
        delta_non_memberships = np.abs(A.non_membership_values - B.non_membership_values)
        min_memberships = np.minimum(A.membership_values, B.membership_values)
        min_non_memberships = np.minimum(A.non_membership_values, B.non_membership_values)
        max_memberships = np.maximum(A.membership_values, B.membership_values)
        max_non_memberships = np.maximum(A.non_membership_values, B.non_membership_values)
        return 1.0 / n * np.sum(
            (min_memberships + min_non_memberships) / 
            (max_memberships + max_non_memberships + 
            delta_memberships + delta_non_memberships)
        )
    elif similarity_type == DENG_JIANG_FU_MONOTONIC_TYPE_1_2:
        n = len(A)
        min_memberships = np.minimum(A.membership_values, B.membership_values)
        min_non_memberships = np.minimum(A.non_membership_values, B.non_membership_values)
        max_memberships = np.maximum(A.membership_values, B.membership_values)
        max_non_memberships = np.maximum(A.non_membership_values, B.non_membership_values)
        return (
            np.sum(
                (min_memberships + min_non_memberships) /
                (2.0 * (max_memberships + max_non_memberships))
            ) /
            (
                n - np.sum(
                    (min_memberships + min_non_memberships) / 
                    (2.0 * (max_memberships + max_non_memberships)))
            )
        )
    elif similarity_type == DENG_JIANG_FU_MONOTONIC_TYPE_1_3:
        n = len(A)
        delta_memberships = np.abs(A.membership_values - B.membership_values)
        delta_non_memberships = np.abs(A.non_membership_values - B.non_membership_values)
        max_memberships = np.maximum(A.membership_values, B.membership_values)
        max_non_memberships = np.maximum(A.non_membership_values, B.non_membership_values)
        return 1 - np.power(
            np.sum(
                np.power(
                    (delta_memberships + delta_non_memberships) /
                    (max_memberships + max_non_memberships), p
                )
            ) / n, 1.0 / p
        )
    elif similarity_type == DENG_JIANG_FU_MONOTONIC_TYPE_1_4:
        n = len(A)
        delta_memberships = np.abs(A.membership_values - B.membership_values)
        delta_non_memberships = np.abs(A.non_membership_values - B.non_membership_values)
        max_memberships = np.maximum(A.membership_values, B.membership_values)
        max_non_memberships = np.maximum(A.non_membership_values, B.non_membership_values)
        return (
            (n - np.sum(
                (delta_memberships + delta_non_memberships) /
                (max_memberships + max_non_memberships)
            )) / 
            (n + np.sum(
                (delta_memberships + delta_non_memberships) /
                (max_memberships + max_non_memberships)
            ))
        )
    elif similarity_type == DENG_JIANG_FU_MONOTONIC_TYPE_2_1:
        n = len(A)
        return (1.0 / n) * np.sum(
            (1.0 - np.abs(A.non_membership_values - B.non_membership_values)) /
            (1.0 + np.abs(A.membership_values - B.membership_values))
        )
    elif similarity_type == DENG_JIANG_FU_MONOTONIC_TYPE_2_2:
        return (
            np.sum(1.0 - np.abs(A.non_membership_values - B.non_membership_values)) /
            np.sum(1.0 + np.abs(A.membership_values - B.membership_values))
        )
    elif similarity_type == DENG_JIANG_FU_MONOTONIC_TYPE_2_3:
        n = len(A)
        delta_memberships = np.abs(A.membership_values - B.membership_values)
        delta_non_memberships = np.abs(A.non_membership_values - B.non_membership_values)
        return 1.0 - np.power(
            np.sum(np.power(
                (delta_memberships + delta_non_memberships) / 2.0, p
            )) / n, 1.0 / p
        )
    elif similarity_type == DENG_JIANG_FU_MONOTONIC_TYPE_2_4:
        delta_memberships = np.abs(A.membership_values - B.membership_values)
        delta_non_memberships = np.abs(A.non_membership_values - B.non_membership_values)
        return (
            np.sum(2.0 - delta_memberships - delta_non_memberships) / 
            np.sum(2.0 + delta_memberships + delta_non_memberships)
        )
    elif similarity_type == DENG_JIANG_FU_MONOTONIC_TYPE_3_1:
        n = len(A)
        delta_memberships = np.abs(A.membership_values - B.membership_values)
        delta_non_memberships = np.abs(A.non_membership_values - B.non_membership_values)
        return (
            (
                np.e ** 2.0 - 
                np.power(
                    np.sum(
                        np.power(delta_memberships, p)
                    ) / n, 1.0 / p
                ) -
                np.power(
                    np.sum(
                        np.power(delta_non_memberships, p)
                    ) / n, 1.0 / p
                ) -
                1.0
            ) /
            (np.e ** 2.0 - 1.0)
        )
    elif similarity_type == DENG_JIANG_FU_MONOTONIC_TYPE_3_2:
        n = len(A)
        delta_memberships = np.abs(A.membership_values - B.membership_values)
        delta_non_memberships = np.abs(A.non_membership_values - B.non_membership_values)
        return 1.0 - (
            (
                np.power(
                    np.power(
                        np.sum(np.power(delta_memberships, p)) / n, p
                    ), u
                ) + 
                np.power(
                    np.power(
                        np.sum(np.power(delta_non_memberships, p)) / n, p
                    ), v
                )
            ) / 2.0
        )
    elif similarity_type == DENG_JIANG_FU_MONOTONIC_TYPE_3_3:
        n = len(A)
        delta_memberships = np.abs(A.membership_values - B.membership_values)
        delta_non_memberships = np.abs(A.non_membership_values - B.non_membership_values)
        return 1.0 - (
            np.sin(np.pi / 2.0 * np.power(
                np.sum(np.power(delta_memberships, p)) / n, 1.0 / p
            )) + 
            np.sin(np.pi / 2.0 * np.power(
                np.sum(np.power(delta_non_memberships, p)) / n, 1.0 / p
            )) 
        ) / 2.0
    else:
        raise ValueError("similarity_type parameter must be DENG_JIANG_FU_MONOTONIC_TYPE_1_1...4, DENG_JIANG_FU_MONOTONIC_TYPE_2_1...4 or DENG_JIANG_FU_MONOTONIC_TYPE_3_1...3.")


def nguyen(A: IntuitionisticFuzzySet, B: IntuitionisticFuzzySet):
    """ Similarity proposed by H. Nguyen, from the related article: 
    "A novel similarity/dissimilarity measure for intuitionistic fuzzy sets and its application in pattern recognition"

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

    def k_single(m, v, p):
        return np.sqrt(
            np.power(m, 2.0) +
            np.power(v, 2.0) +
            np.power(1.0 - p, 2.0)
        )
    def k(set: IntuitionisticFuzzySet):
        return 1.0 / (len(set) * np.sqrt(2)) * np.sum(np.sqrt(
            np.power(set.membership_values, 2.0) +
            np.power(set.non_membership_values, 2.0) +
            np.power(1.0 - set.hesitation_degrees, 2.0)
        ))
    def membership_knowledge(set: IntuitionisticFuzzySet):
        memberships, non_memberships, hestitations = set.membership_values, set.non_membership_values, set.hesitation_degrees
        K = 1.0 / (len(set) * np.sqrt(2)) * np.array([
            k_single(m, v, p) if m >= v else -k_single(m, v, p)
            for m, v, p in zip(memberships, non_memberships, hestitations)
        ])
        return 1.0 / len(set) * np.sum(K)
        
    _Ka = membership_knowledge(A)
    _Kb = membership_knowledge(B)
    Ka = k(A)
    Kb = k(B)
    return 1.0 - np.abs(Ka - Kb) if _Ka * _Kb >= 0 else np.abs(Ka - Kb) - 1.0


def chen_cheng_lan(A: IntuitionisticFuzzySet, B: IntuitionisticFuzzySet, weights=None):
    """ Similarity proposed by S.M. Chen, S.H. Cheng, T.-C. Lan, from the related article: 
    "A novel similarity measure between intuitionistic fuzzy sets based on the centroid points of transformed fuzzy numbers with applications to pattern recognition"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    weights : list of floats
        List of weights for each membership/non-membership value.
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)
    check_weights(weights, len(A))

    if weights is None:
        weights = np.full(len(A), 1.0 / len(A))
    
    delta_memberships = A.membership_values - B.membership_values
    delta_non_memberships = A.non_membership_values - B.non_membership_values
    return np.sum(
        weights * (
            1.0 - 
            np.abs(2.0 * delta_memberships - delta_non_memberships) / 3.0 * (1.0 - (A.hesitation_degrees + B.hesitation_degrees) / 2.0) -
            np.abs(2.0 * delta_non_memberships - delta_memberships) / 3.0 * ((A.hesitation_degrees + B.hesitation_degrees) / 2.0)
        )
    )


def muthukumar_krishnanb(A: FuzzySet, B: FuzzySet, weights=None):
    """ Similarity proposed by P. Muthukumar, G. S. S. Krishnan, from the related article: 
    "A similarity measure of intuitionistic fuzzy soft sets and itsapplication in medical diagnosis"

    Parameters
    ----------
    A : FuzzySet
        A fuzzy set.
    B : FuzzySet
        A fuzzy set.
    weights : list of floats
        List of weights for each membership/non-membership value.
        
    Returns
    -------
    numpy.float64
        The similarity between the two sets provided.
    """
    check_sets_cardinality(A, B)

    if weights is None:
        return (
            np.sum(
                A.membership_values * B.membership_values + 
                A.non_membership_values * B.non_membership_values
            ) / 
            np.sum(
                np.maximum(np.power(A.membership_values, 2.0), np.power(B.membership_values, 2.0)) + 
                np.maximum(np.power(A.non_membership_values, 2.0), np.power(B.non_membership_values, 2.0))
            )
        )
    else:
        check_weights(weights, len(A))
        return (
            np.sum(
                weights * 
                (A.membership_values * B.membership_values + 
                A.non_membership_values * B.non_membership_values)
            ) / 
            np.sum(
                np.maximum(np.power(A.membership_values, 2.0), np.power(B.membership_values, 2.0)) + 
                np.maximum(np.power(A.non_membership_values, 2.0), np.power(B.non_membership_values, 2.0))
            )
        ) / np.sum(weights)
