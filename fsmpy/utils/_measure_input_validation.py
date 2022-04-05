import numpy as np
from typing import Union, Iterable

from ..sets import FuzzySet, IntuitionisticFuzzySet


def check_weights(weights: Iterable, set_cardinality: int, actual_measure_type: Union[Iterable, str, int]=None, sum_1=False, measure_types_required=None) -> np.ndarray:
    """ Input validation for measures that require weights.

    Checks if the weights have the same size as the sets' cardinality and if the values are [0, 1].
    Validation is performed only if there is no requirement for a specific measure type or if the measure type provided (actual_measure_type)
    requires the weights parameter.

        Args:
            weights: Input weights.
            set_cardinality: Cardinality (length) of the sets.
            actual_value: The type of measure that was provided.
            measure_types_required: In case the weights are used for a specific measure type (and all).
        Returns:
            The converted weights.
        Raises:
            ValueError if weights size != set_cardinality or if weights values are not [0, 1].
    """
    if weights is None:
        return weights
    
    weights = np.array(weights)
    if measure_types_required is not None and actual_measure_type in measure_types_required:
        if weights.size != set_cardinality:
            raise ValueError(
                "Weight parameter must have the same size as sets A and B!({} vs {})".format(weights.size, set_cardinality))

        if sum_1 and not np.isclose(np.sum(weights), 1.0):
            raise ValueError(
                f"Sum of weights must be [0, 1], not {np.sum(weights)}")
        elif any(np.logical_or(weights < 0, weights > 1)):
            raise ValueError("Weights much be [0, 1].")
    return weights


def check_p(p: Union[int, float], actual_measure_type: Union[Iterable, str, int] = None, measure_types_required=None) -> None:
    """ Input validation for measures that require the parameter p.

    Checks if the p is an int and if it is >=1.

        Args:
            p: Input p.
            actual_value: The type of measure that was provided.
            measure_types_required: In case the weights are used for a specific measure type (and all).
        Raises:
            ValueError if p is not an integer or if it is < 1.
    """
    if measure_types_required is not None and actual_measure_type in measure_types_required:
        if not np.issubdtype(type(p), int):
            raise ValueError(
                "p parameter must be an integer, not {}".format(type(p)))
        elif p < 1:
            raise ValueError(
                "p parameter must be >= 1, not {}".format(p))


def check_sets_cardinality(A: FuzzySet, B: FuzzySet) -> None:
    """ Checks if sets have the same cardinality

        Args:
            A: FuzzySet.
            B: FuzzySet.

        Raises:
            ValueError if the two sets have different cardinalities
    """
    if len(A) != len(B):
        raise ValueError("A and B sets must be have the same sizes.({} and {})".format(len(A), len(B)))
