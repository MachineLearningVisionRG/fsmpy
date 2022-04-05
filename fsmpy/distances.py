from typing import Iterable
import numpy as np
import math

from .utils._measure_input_validation import check_weights, check_p, check_sets_cardinality
from . import DISTANCE_HAMMING, DISTANCE_EUCLIDEAN, DISTANCE_NORMALIZED_HAMMING, DISTANCE_NORMALIZED_EUCLIDEAN
from . import WANGXIN_DISTANCE_1, WANGXIN_DISTANCE_2
from .sets import IntuitionisticFuzzySet


def atanassov(A: IntuitionisticFuzzySet, B: IntuitionisticFuzzySet, distance_type: str = DISTANCE_HAMMING) -> np.float64:
    """ Distance proposed by K.T. Atanassov, from the related article: "Intuitionistic fuzzy sets"

    Parameters
    ----------
    A : IntuitionisticFuzzySet
        A fuzzy set.
    B : IntuitionisticFuzzySet
        A fuzzy set.
    distance_type: str, optional
        Type of computed distance: 

        >>> DISTANCE_HAMMING,
        >>> DISTANCE_EUCLIDEAN
        >>> DISTANCE_NORMALIZED_HAMMING
        >>> DISTANCE_NORMALIZED_EUCLIDEAN
        
    Returns
    -------
    numpy.float64
        The distance between the two sets provided.
    """
    check_sets_cardinality(A, B)
    if distance_type == DISTANCE_HAMMING:
        return (
            np.sum(
                np.absolute(A.membership_values - B.membership_values) + 
                np.absolute(A.non_membership_values - B.non_membership_values)
            )
            ) / 2.0

    elif distance_type == DISTANCE_EUCLIDEAN:
        return np.sqrt(
                np.sum(
                    np.power(A.membership_values - B.membership_values, 2.0) + 
                    np.power(A.non_membership_values - B.non_membership_values, 2.0)
                ) / 2.0
            )

    elif distance_type == DISTANCE_NORMALIZED_HAMMING:
        return np.sum(
                np.absolute(A.membership_values - B.membership_values) + 
                np.absolute(A.non_membership_values - B.non_membership_values)
            ) / (2.0 * len(A))

    elif distance_type == DISTANCE_NORMALIZED_EUCLIDEAN:
        return np.sqrt(
            np.sum(
                np.power(A.membership_values - B.membership_values, 2.0) + 
                np.power(A.non_membership_values - B.non_membership_values, 2.0)
            ) / (2.0 * len(A))
        )

    else:
        raise ValueError(
            "Invalid distance type provided. Please check the available flags.")


def szmidt_kacprzyk(A: IntuitionisticFuzzySet, B: IntuitionisticFuzzySet, distance_type: str = DISTANCE_HAMMING) -> np.float64:
    """ Distances proposed by E. Szmidt and A. Kacprzyk, from the related article: "Distances between intuitionistic fuzzy sets"

    Parameters
    ----------
    A : IntuitionisticFuzzySet
        A fuzzy set.
    B : IntuitionisticFuzzySet
        A fuzzy set.
    distance_type: str, optional
        Type of computed distance: 
        
        >>> DISTANCE_HAMMING
        >>> DISTANCE_EUCLIDEAN
        >>> DISTANCE_NORMALIZED_HAMMING
        >>> DISTANCE_NORMALIZED_EUCLIDEAN
        
    Returns
    -------
    numpy.float64
        The distance between the two sets provided.
    """
    check_sets_cardinality(A, B)
    if distance_type == DISTANCE_HAMMING:
        return np.sum(
            np.absolute(A.membership_values - B.membership_values) +
            np.absolute(A.non_membership_values - B.non_membership_values) +
            np.absolute(A.hesitation_degrees - B.hesitation_degrees)
        ) / 2.0

    elif distance_type == DISTANCE_EUCLIDEAN:
        return np.sqrt(
            (
                np.sum(
                    np.power(A.membership_values - B.membership_values, 2.0) +
                    np.power(A.non_membership_values - B.non_membership_values, 2.0) +
                    np.power(A.hesitation_degrees - B.hesitation_degrees, 2.0))
            ) / 2.0
        )

    elif distance_type == DISTANCE_NORMALIZED_HAMMING:
        return np.sum(
            np.absolute(A.membership_values - B.membership_values) + 
            np.absolute(A.non_membership_values - B.non_membership_values) + 
            np.absolute(A.hesitation_degrees - B.hesitation_degrees)
        ) / (2.0 * len(A))

    elif distance_type == DISTANCE_NORMALIZED_EUCLIDEAN:
        return np.sqrt(
            np.sum(
                np.power(A.membership_values - B.membership_values, 2.0) + 
                np.power(A.non_membership_values - B.non_membership_values, 2.0) + 
                np.power(A.hesitation_degrees - B.hesitation_degrees, 2.0)
            ) / (2.0 * len(A))
        )

    else:
        raise ValueError(
            "Invalid distance type provided. Please check the available flags.")


def wang_xin(A: IntuitionisticFuzzySet, B: IntuitionisticFuzzySet, distance_type: int = WANGXIN_DISTANCE_1, weights: Iterable = None, p: int = 1) -> np.float64:
    """ Distances proposed by W. Wang and X. Xin, from the related article: 
        "Distance measure between intuitionistic fuzzy sets"

    Parameters
    ----------
    A : IntuitionisticFuzzySet
        A fuzzy set.
    B : IntuitionisticFuzzySet
        A fuzzy set.
        distance_type: int, optional
        Type of computed distance: 
    
        >>> WANGXIN_DISTANCE_1
        >>> WANGXIN_DISTANCE_2
    
    weights : list of floats
        List of weights for each membership/non-membership value.
    p : int
        Positive integer >= 1.
        
    Returns
    -------
    numpy.float64
        The distance between the two sets provided.
    """
    check_sets_cardinality(A, B)
    check_p(p, distance_type, measure_types_required=[WANGXIN_DISTANCE_2])
    check_weights(weights, len(A), distance_type, measure_types_required=[WANGXIN_DISTANCE_1])
    n = len(A)

    m_diff = np.absolute(A.membership_values - B.membership_values)
    v_diff = np.absolute(A.non_membership_values - B.non_membership_values)

    if distance_type == WANGXIN_DISTANCE_1:
        if weights is None:
            return np.sum(
                (m_diff + v_diff) / 4.0 +
                np.maximum(m_diff, v_diff) / 2.0
            ) / n
        else:
            return np.sum(
                weights *
                ((m_diff + v_diff) / 4.0 +
                 np.maximum(m_diff, v_diff) / 2.0)
            ) / np.sum(weights)

    elif distance_type == WANGXIN_DISTANCE_2:
        multiplier = (1. / math.pow(n, 1. / p))
        summation = np.sum(
            np.power((m_diff + v_diff) / 2., p)
        )
        return multiplier * np.power(summation, (1.0 / p))


def yang_chiclana(A: IntuitionisticFuzzySet, B: IntuitionisticFuzzySet,  distance_type: str = DISTANCE_HAMMING) -> np.float64:
    """ Distances proposed by Y. Yang and F. Chiclana, from the related article: "Consistency of 2D and 3D distances of intuitionistic fuzzy sets"

    Parameters
    ----------
    A : IntuitionisticFuzzySet
        A fuzzy set.
    B : IntuitionisticFuzzySet
        A fuzzy set.
    distance_type: str, optional
        Type of computed distance: 

        >>> DISTANCE_HAMMING
        >>> DISTANCE_EUCLIDEAN
        >>> DISTANCE_NORMALIZED_HAMMING or 
        >>> DISTANCE_NORMALIZED_EUCLIDEAN
        
    Returns
    -------
    numpy.float64
        The distance between the two sets provided.
    """
    check_sets_cardinality(A, B)
    n = len(A)
    delta_memberships = np.absolute(A.membership_values - B.membership_values)
    delta_non_memberships = np.absolute(A.non_membership_values - B.non_membership_values)
    delta_hesitations = np.absolute(A.hesitation_degrees - B.hesitation_degrees)

    if distance_type == DISTANCE_HAMMING:
        return np.sum(
            np.amax((delta_memberships, delta_non_memberships, delta_hesitations), axis=0)
        )
    elif distance_type == DISTANCE_EUCLIDEAN:
        return np.sqrt(
            np.sum(
                np.amax((delta_memberships ** 2, delta_non_memberships ** 2, delta_hesitations ** 2), axis=0)
            )
        )
    elif distance_type == DISTANCE_NORMALIZED_HAMMING:
        return np.sum(
            np.amax((delta_memberships, delta_non_memberships, delta_hesitations), axis=0)
        ) / n
    elif distance_type == DISTANCE_NORMALIZED_EUCLIDEAN:
        return np.sqrt(
            np.sum(
                np.amax((delta_memberships ** 2, delta_non_memberships ** 2, delta_hesitations ** 2), axis=0)
            ) / n
        )
    else:
        raise ValueError(
            "Invalid distance type provided. Please check the available flags.")


def grzegorzewski(A: IntuitionisticFuzzySet, B: IntuitionisticFuzzySet, distance_type: str = DISTANCE_HAMMING) -> np.float64:
    """ Distances proposed by P. Grzegorzewski from the related article: 
    "Distances between intuitionistic fuzzy sets and/or interval-valued fuzzy sets based on the Hausdorff metric"

    Parameters
    ----------
    A : IntuitionisticFuzzySet
        A fuzzy set.
    B : IntuitionisticFuzzySet
        A fuzzy set.
    distance_type: str, optional
        Type of computed distance: 
        >>> DISTANCE_HAMMING
        >>> DISTANCE_EUCLIDEAN
        >>> DISTANCE_NORMALIZED_HAMMING** or 
        >>> DISTANCE_NORMALIZED_EUCLIDEAN**.
        
    Returns
    -------
    numpy.float64
        The distance between the two sets provided.
    """
    check_sets_cardinality(A, B)
    m_diff = A.membership_values - B.membership_values
    v_diff = A.non_membership_values - B.non_membership_values

    if distance_type == DISTANCE_HAMMING:
        return np.sum(np.maximum(np.absolute(m_diff), np.absolute(v_diff)))
    elif distance_type == DISTANCE_EUCLIDEAN:
        return np.sqrt(np.sum(np.maximum(m_diff ** 2, v_diff ** 2)))
    elif distance_type == DISTANCE_NORMALIZED_HAMMING:
        return np.sum(np.maximum(np.absolute(m_diff), np.absolute(v_diff))) / float(len(A))
    elif distance_type == DISTANCE_NORMALIZED_EUCLIDEAN:
        return np.sqrt(np.sum(np.maximum((m_diff) ** 2, v_diff ** 2)) / float(len(A)))
    else:
        raise ValueError(
            "Invalid distance type provided. Please check the available flags.")


def vlachos_sergiadis(A: IntuitionisticFuzzySet, B: IntuitionisticFuzzySet) -> np.float64:
    """ Distance proposed by I.K. Vlachos, G.D. Sergiadis from the related article: "Intuitionistic fuzzy information - Applications to pattern recognition"

    Parameters
    ----------
    A : IntuitionisticFuzzySet
        A fuzzy set.
    B : IntuitionisticFuzzySet
        A fuzzy set.
        
    Returns
    -------
    numpy.float64
        The distance between the two sets provided.
    """
    check_sets_cardinality(A, B)
    def _log(arr):
        invalids = np.logical_or(
            arr <= 0, 
            np.isnan(arr), 
            np.isinf(arr)
        )
        invalid_indxs = np.where(invalids)
        valid_indxs = np.where(np.logical_not(invalids))
        arr[invalid_indxs] = 0
        arr[valid_indxs] = np.log(arr[valid_indxs])
        return arr

    def iifs(setA: IntuitionisticFuzzySet, setB: IntuitionisticFuzzySet):
        return np.sum(
            setA.membership_values *
            _log(
                setA.membership_values /
                (0.5 * (setA.membership_values + setB.membership_values))
            ) +
            setA.non_membership_values *
            _log(
                setA.non_membership_values /
                (0.5 * (setA.non_membership_values + setB.non_membership_values))
            )
        )
    with np.errstate(divide='ignore'):
        return iifs(A, B) + iifs(B, A)
