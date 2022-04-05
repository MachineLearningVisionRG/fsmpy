import numpy as np
from typing import Iterable, Tuple
from collections.abc import Callable
from tqdm import tqdm

from ..sets import FuzzySet, IntuitionisticFuzzySet
from .classifiers import classify, FuzzyTextClassifier
from .image_processing import threshold


def calculate_documents_membership(data: Iterable, membership_weight: float, non_membership_weight: float, 
                                   means: Iterable[float] = None, stds: Iterable[float] = None) -> Tuple[list, np.ndarray, np.ndarray]:
    """ Calculates the Fuzzy Set of each class from token counts.
    Proposed by P. Intarapaiboon from the related article: 
    "Text classification using similarity measures on intuitionistic fuzzy sets".

    Parameters
    ----------
    data : Iterable
        Token counts of a document dataset.
    membership_weight : float
        Weight used to calculate the membership values of each token's membership value.
    non_membership_weight : float
        Weight used to calculate the non-membership values of each token's membership value.
    
        
    Returns
    -------
    list[IntuitionisticFuzzySet]
        List of IntuitionisticFuzzySets for each class.
    np.ndarray : means
        Mean values for each token in data.
    np.ndarray : stds
        Standard deviation values for each token in data.
    
    See also
    --------
    sklearn.feature_extraction.text.CountVectorizer
    """
    if means is None and stds is None:
        means = np.mean(data, axis=0, dtype=np.float64)  # mean of each sample
        stds = np.std(data, axis=0, dtype=np.float64, ddof=1)  # std of each sample
    else:
        if len(means) != len(data[0]) or len(stds) != len(data[0]):
            raise ValueError(f"Parameters means and stds must have the same size as the number of samples in data ({len(means)} and {len(stds)} != {len(data[0])}).")

    # calculate membership values of each word for each document
    z = (data - means) / stds
    np.nan_to_num(z, copy=False, nan=0)

    m = membership_weight / (1. + np.exp(-z))
    v = non_membership_weight / (1. + np.exp(z))
    p = 1 - m - v
    
    sets = [
        IntuitionisticFuzzySet(*args)
        for args in zip(m, v, p)
    ]

    return sets, means, stds


def confidence_degree(predicted_class_distance: float, other_classes_distance: Iterable[float]) -> np.float64:
    """ Degree of Confidence proposd by A.G. Hatzimichailidis, G.A. Papakostas, V.G. Kaburlasos
    from the related article: 
    "A Novel Distance Measure of Intuitionistic Fuzzy Sets and Its Application to Pattern Recognition Problems"

    Parameters
    ----------
    predicted_class_distance : float
        Distance calculated between the sample and the predicted class.
    other_classes_distance: Iterable[float]
        An Iterable consisted of the distances calculated between the sample and every other class.
        
    Returns
    -------
    numpy.float64
        Degree of Confidence.
    """
    return np.sum(np.abs(predicted_class_distance - other_classes_distance))


def compactness(A: FuzzySet, shape : tuple) -> np.float64:
    """ Image geometry proposd by S.K. Pal, A. Rosenfeld, from the related article: 
    "Image enhancement and thresholding by optimization of fuzzy compactness"

    Parameters
    ----------
    A : FuzzySet
        Fuzzy set of an image, containing a 2-d np.ndarray as an image.
    shape : Tuple[float, float]
        Shape of the original image.
        
    Returns
    -------
    numpy.float64
        Compactness of the image.

    Raises
    ------
    ValueError
        In case the shape propvided is not a tuple of 2 values.
    """
    if len(shape) != 2:
        ValueError("Shape provided must have 2 elements (width and height), not {}.".format(len(shape)))
    
    shaped_memberships = A.membership_values.reshape(shape)
    
    a = np.sum(A.membership_values)
    p = np.sum(
        np.absolute(
            shaped_memberships[:, :-1] -
            shaped_memberships[:, 1:]
        ) +
        np.absolute(
            shaped_memberships[:, :-1] -
            shaped_memberships[1:, :]
        )
    )
    return a / (p ** 2)


def check_similarity_conditions(measure_caller: Callable, **measure_kwargs):
    """ Checks if measures satisfies the required conditions.

    Checks whether the measure provided satisfies the following conditions:

    .. math:: S(A, B) >= 0\ and\ S(A, B) <= 1
    .. math:: S(A, B) == S(B, A)
    .. math:: S(A, A) == 1

    The algorithm conducts two tests. The first one creates a list from `np.arange(start=0.0, stop=1.01, step=0.01)`
    and 10.000 creates two sets A and B with a random size (1-1000) of (non)membership values with random values picked from
    the range above, checking if the conditions apply.

    The second test creates 4 sets with random sizes (1-1000), with random values using `np.random.rand(random_size)`
     and checks for the conditions again. 
    The first two sets only contain membership and non-membership values, while the 2 last sets contain
    membership values and hesitation degrees. Checks if the condition applies when the measure
    is applied on all combination of the 4 sets.

    Parameters
    ----------
    measure_caller : Callable
        The measure to be tested.
    **measure_kwargs : measure arguments
        Passed to the measure_caller.

    Returns
    -------
    numpy.float64
        Compactness of the image.

    """
    value_range = np.arange(start=0.0, stop=1.01, step=0.01)
    for _ in tqdm(range(10000), desc="Test 1 progress:"):
        random_size = np.random.randint(1, 1000)
        A = IntuitionisticFuzzySet(np.random.choice(value_range, random_size), np.random.choice(value_range, random_size))
        B = IntuitionisticFuzzySet(np.random.choice(value_range, random_size), np.random.choice(value_range, random_size))
        assert measure_caller(A, B, **measure_kwargs) >= 0.0 and measure_caller(A, B, **measure_kwargs) <= 1.0

        assert measure_caller(A, B, **measure_kwargs) == measure_caller(B, A, **measure_kwargs)

        assert measure_caller(A, A, **measure_kwargs) == 1.0 and measure_caller(B, B, **measure_kwargs) == 1.0

    for _ in tqdm(range(10000), desc="Test 2 progress:"):
        random_size = np.random.randint(1, 1000)
        A = IntuitionisticFuzzySet(np.random.rand(random_size), np.random.rand(random_size))
        B = IntuitionisticFuzzySet(np.random.rand(random_size), np.random.rand(random_size))
        C = IntuitionisticFuzzySet(np.random.rand(random_size), None, np.random.rand(random_size))
        D = IntuitionisticFuzzySet(np.random.rand(random_size), None, np.random.rand(random_size))

        from itertools import combinations
        for a, b in combinations([A, B, C, D], 2):    
            assert measure_caller(a, b, **measure_kwargs) >= 0.0 and measure_caller(a, b, **measure_kwargs) <= 1.0

            assert measure_caller(a, b, **measure_kwargs) == measure_caller(b, a, **measure_kwargs)

            assert measure_caller(a, a, **measure_kwargs) == 1.0 and measure_caller(b, b, **measure_kwargs) == 1.0

    print("Similarity passes all tests!")


__all__ = (
    "classify",
    "FuzzyTextClassifier",
    "threshold",
    "calculate_documents_membership",
    "confidence_degree",
    "compactness",
    "check_similarity_conditions"
)