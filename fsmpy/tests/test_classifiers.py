import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from fsmpy.utils.classifiers import FuzzyTextClassifier
from fsmpy.utils import calculate_documents_membership
from fsmpy.sets import IntuitionisticFuzzySet
from fsmpy.similarities import intarapaiboon


def test_fuzzy_classifier():
    # Example 2 text classification from study "Text classification using similarity measures on
    # intuitionistic fuzzy sets" by Peerasak Intarapaiboon
    X = [
        [15, 2, 7],
        [10, 3, 5],
        [2, 9, 6],
        [3, 11, 5],
        [3, 9, 4]
    ]
    sets, _, _ = calculate_documents_membership(X, 0.9, 0.9)

    cls = FuzzyTextClassifier(intarapaiboon, is_distance = False)
    cls.fit(np.array(sets), np.array([1, 1, 2, 2, 2]))

    assert_almost_equal(
        cls.class_patterns[1].membership_values,
        [0.66, 0.23, 0.55],
        decimal=2
    )
    assert_almost_equal(
        cls.class_patterns[1].non_membership_values,
        [0.24, 0.67, 0.35],
        decimal=2
    )

    assert_almost_equal(
        cls.class_patterns[2].membership_values,
        [0.30, 0.60, 0.38],
        decimal=2
    )
    assert_almost_equal(
        cls.class_patterns[2].non_membership_values,
        [0.60, 0.30, 0.52],
        decimal=2
    )

    y_pred = cls.predict([
        IntuitionisticFuzzySet([0.58, 0.39, 0.39], [0.32, 0.51, 0.51])
    ])
    assert_equal(y_pred, [1])

    y_proba = cls.predict_proba([
        IntuitionisticFuzzySet([0.58, 0.39, 0.39], [0.32, 0.51, 0.51])
    ])
    assert_almost_equal(y_proba, [[0.96, 0.91]], decimal=2) # fails
