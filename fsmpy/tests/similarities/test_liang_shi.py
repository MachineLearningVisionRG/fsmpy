from numpy.testing import assert_almost_equal

from fsmpy.sets import IntuitionisticFuzzySet
from fsmpy.similarities import liang_shi
from fsmpy import LIANG_SHI_SIMILARITY_1, LIANG_SHI_SIMILARITY_2, LIANG_SHI_SIMILARITY_3

def test_liang_shi_1():
    A1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.9, 0.9, 0.1])
    A2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.7, 0.2])
    A3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.8, 0.2, 0.6])
    B = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.6, 0.8, 0.2])

    # Example 1
    assert_almost_equal(liang_shi(A1, B, similarity_type=LIANG_SHI_SIMILARITY_1, p=1), 0.83, decimal=2)
    assert_almost_equal(liang_shi(A2, B, similarity_type=LIANG_SHI_SIMILARITY_1, p=1), 0.93, decimal=2)
    assert_almost_equal(liang_shi(A3, B, similarity_type=LIANG_SHI_SIMILARITY_1, p=1), 0.60, decimal=2)


def test_liang_shi_2():
    A1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.9, 0.9, 0.1])
    A2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.7, 0.2])
    A3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.8, 0.2, 0.6])
    B = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.6, 0.8, 0.2])

    # Example 2 # fails
    assert_almost_equal(liang_shi(A1, B, similarity_type=LIANG_SHI_SIMILARITY_2, p=1), 0.92, decimal=2)
    assert_almost_equal(liang_shi(A2, B, similarity_type=LIANG_SHI_SIMILARITY_2, p=1), 0.97, decimal=2)
    assert_almost_equal(liang_shi(A3, B, similarity_type=LIANG_SHI_SIMILARITY_2, p=1), 0.77, decimal=2)


def test_liang_shi_3():
    A1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.9, 0.9, 0.1])
    A2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.7, 0.2])
    A3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.8, 0.2, 0.6])
    B = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.6, 0.8, 0.2])

    # Example 3
    assert_almost_equal(liang_shi(A1, B, similarity_type=LIANG_SHI_SIMILARITY_3, p=1, omegas=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), 0.89, decimal=2)
    assert_almost_equal(liang_shi(A2, B, similarity_type=LIANG_SHI_SIMILARITY_3, p=1, omegas=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), 0.95, decimal=2)
    assert_almost_equal(liang_shi(A3, B, similarity_type=LIANG_SHI_SIMILARITY_3, p=1, omegas=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), 0.72, decimal=2)
