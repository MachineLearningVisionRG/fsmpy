from numpy.testing import assert_almost_equal

from fsmpy.sets import IntuitionisticFuzzySet
from fsmpy.similarities import hung_yang_2
from fsmpy import HUNG_YANG_2_SIMILARITY_1, HUNG_YANG_2_SIMILARITY_2, HUNG_YANG_2_SIMILARITY_3


def test_hung_yang_2_1():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_2(A1, B, a=2, similarity_type=HUNG_YANG_2_SIMILARITY_1), 1.000, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, a=2, similarity_type=HUNG_YANG_2_SIMILARITY_1), 0.980, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B, a=2, similarity_type=HUNG_YANG_2_SIMILARITY_1), 0.860, decimal=3)
    assert_almost_equal(hung_yang_2(A1, B, a=1.5, similarity_type=HUNG_YANG_2_SIMILARITY_1), 1.000, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, a=1.5, similarity_type=HUNG_YANG_2_SIMILARITY_1), 0.979, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B, a=1.5, similarity_type=HUNG_YANG_2_SIMILARITY_1), 0.850, decimal=3)
    assert_almost_equal(hung_yang_2(A1, B, a=1, similarity_type=HUNG_YANG_2_SIMILARITY_1), 1.000, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, a=1, similarity_type=HUNG_YANG_2_SIMILARITY_1), 0.979, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B, a=1, similarity_type=HUNG_YANG_2_SIMILARITY_1), 0.854, decimal=3)

    # Example 2
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.6, 0.6, 0.6])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4], [0.2, 0.2, 0.2])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3], [0.4, 0.4, 0.6])

    assert_almost_equal(hung_yang_2(A1, B, similarity_type=HUNG_YANG_2_SIMILARITY_1), 0.974, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, similarity_type=HUNG_YANG_2_SIMILARITY_1), 0.928, decimal=3)
    assert_almost_equal(hung_yang_2(A1, B, similarity_type=HUNG_YANG_2_SIMILARITY_1, a=1.5), 0.974, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, similarity_type=HUNG_YANG_2_SIMILARITY_1, a=1.5), 0.928, decimal=3)

    assert_almost_equal(hung_yang_2(A1, B, similarity_type=HUNG_YANG_2_SIMILARITY_3), 0.957, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, similarity_type=HUNG_YANG_2_SIMILARITY_3), 0.882, decimal=3)

    # Example 3      Division with zero
    A1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.1, 0.1, 0.9], [0.8, 0.4, 0.0])
    A2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.3, 0.8], [0.0, 0.0, 0.2])
    A3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.2, 0.8, 0.4], [0.1, 0.1, 0.2])
    B = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.4, 0.2, 0.8], [0.2, 0.2, 0.2])

    assert_almost_equal(hung_yang_2(A1, B, similarity_type=HUNG_YANG_2_SIMILARITY_1), 0.843, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, similarity_type=HUNG_YANG_2_SIMILARITY_1), 0.927, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B, similarity_type=HUNG_YANG_2_SIMILARITY_1), 0.797, decimal=3)


def test_hung_yang_2_2():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_2(A1, B, a=2, similarity_type=HUNG_YANG_2_SIMILARITY_2), 1.000, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, a=2, similarity_type=HUNG_YANG_2_SIMILARITY_2), 0.975, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B, a=2, similarity_type=HUNG_YANG_2_SIMILARITY_2), 0.828, decimal=3)
    assert_almost_equal(hung_yang_2(A1, B, a=1.5, similarity_type=HUNG_YANG_2_SIMILARITY_2), 1.000, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, a=1.5, similarity_type=HUNG_YANG_2_SIMILARITY_2), 0.972, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B, a=1.5, similarity_type=HUNG_YANG_2_SIMILARITY_2), 0.811, decimal=3)
    assert_almost_equal(hung_yang_2(A1, B, a=1, similarity_type=HUNG_YANG_2_SIMILARITY_2), 1.000, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, a=1, similarity_type=HUNG_YANG_2_SIMILARITY_2), 0.971, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B, a=1, similarity_type=HUNG_YANG_2_SIMILARITY_2), 0.808, decimal=3)

    # Example 3      Division with zero
    A1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.1, 0.1, 0.9], [0.8, 0.4, 0.0])
    A2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.3, 0.8], [0.0, 0.0, 0.2])
    A3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.2, 0.8, 0.4], [0.1, 0.1, 0.2])
    B = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.4, 0.2, 0.8], [0.2, 0.2, 0.2])

    assert_almost_equal(hung_yang_2(A1, B, similarity_type=HUNG_YANG_2_SIMILARITY_2), 0.964, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, similarity_type=HUNG_YANG_2_SIMILARITY_2), 0.901, decimal=3)

    # Example 3      Division with zero
    A1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.1, 0.1, 0.9], [0.8, 0.4, 0.0])
    A2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.3, 0.8], [0.0, 0.0, 0.2])
    A3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.2, 0.8, 0.4], [0.1, 0.1, 0.2])
    B = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.4, 0.2, 0.8], [0.2, 0.2, 0.2])

    assert_almost_equal(hung_yang_2(A1, B, similarity_type=HUNG_YANG_2_SIMILARITY_2), 0.794, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, similarity_type=HUNG_YANG_2_SIMILARITY_2), 0.902, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B, similarity_type=HUNG_YANG_2_SIMILARITY_2), 0.737, decimal=3)



def test_hung_yang_2_3():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_2(A1, B, a=2, similarity_type=HUNG_YANG_2_SIMILARITY_3), 1.000, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, a=2, similarity_type=HUNG_YANG_2_SIMILARITY_3), 0.970, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B, a=2, similarity_type=HUNG_YANG_2_SIMILARITY_3), 0.804, decimal=3)
    assert_almost_equal(hung_yang_2(A1, B, a=1.5, similarity_type=HUNG_YANG_2_SIMILARITY_3), 1.000, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, a=1.5, similarity_type=HUNG_YANG_2_SIMILARITY_3), 0.967, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B, a=1.5, similarity_type=HUNG_YANG_2_SIMILARITY_3), 0.782, decimal=3)
    assert_almost_equal(hung_yang_2(A1, B, a=1, similarity_type=HUNG_YANG_2_SIMILARITY_3), 1.000, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, a=1, similarity_type=HUNG_YANG_2_SIMILARITY_3), 0.964, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B, a=1, similarity_type=HUNG_YANG_2_SIMILARITY_3), 0.776, decimal=3)

    # Example 3      Division with zero
    A1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.1, 0.1, 0.9], [0.8, 0.4, 0.0])
    A2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.3, 0.8], [0.0, 0.0, 0.2])
    A3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.2, 0.8, 0.4], [0.1, 0.1, 0.2])
    B = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.4, 0.2, 0.8], [0.2, 0.2, 0.2])

    assert_almost_equal(hung_yang_2(A1, B, similarity_type=HUNG_YANG_2_SIMILARITY_3), 0.761, decimal=2)
    assert_almost_equal(hung_yang_2(A2, B, similarity_type=HUNG_YANG_2_SIMILARITY_3), 0.883, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B, similarity_type=HUNG_YANG_2_SIMILARITY_3), 0.698, decimal=3)
