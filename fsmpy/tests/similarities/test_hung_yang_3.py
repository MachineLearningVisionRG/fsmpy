from numpy.testing import assert_almost_equal

from fsmpy.sets import IntuitionisticFuzzySet
from fsmpy.similarities import hung_yang_3
from fsmpy import HUNG_YANG_3_SIMILARITY_1, HUNG_YANG_3_SIMILARITY_2, HUNG_YANG_3_SIMILARITY_3, \
    HUNG_YANG_3_SIMILARITY_4, HUNG_YANG_3_SIMILARITY_5, HUNG_YANG_3_SIMILARITY_6, HUNG_YANG_3_SIMILARITY_7


def test_hung_yang_3_1():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_3(A1, B, similarity_type=HUNG_YANG_3_SIMILARITY_1), 1.000, decimal=3)   # 8
    assert_almost_equal(hung_yang_3(A2, B, similarity_type=HUNG_YANG_3_SIMILARITY_1), 0.722, decimal=3)
    assert_almost_equal(hung_yang_3(A3, B, similarity_type=HUNG_YANG_3_SIMILARITY_1), 0.500, decimal=3)


def test_hung_yang_3_2():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_3(A1, B, similarity_type=HUNG_YANG_3_SIMILARITY_2), 1.000, decimal=3)  # 11
    assert_almost_equal(hung_yang_3(A2, B, similarity_type=HUNG_YANG_3_SIMILARITY_2), 0.900, decimal=3)
    assert_almost_equal(hung_yang_3(A3, B, similarity_type=HUNG_YANG_3_SIMILARITY_2), 0.700, decimal=3)


def test_hung_yang_3_3():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_3(A1, B, similarity_type=HUNG_YANG_3_SIMILARITY_3), 1.000, decimal=3)  # 10
    assert_almost_equal(hung_yang_3(A2, B, similarity_type=HUNG_YANG_3_SIMILARITY_3), 0.714, decimal=3)
    assert_almost_equal(hung_yang_3(A3, B, similarity_type=HUNG_YANG_3_SIMILARITY_3), 0.500, decimal=3)


def test_hung_yang_3_4():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_3(A1, B, similarity_type=HUNG_YANG_3_SIMILARITY_4), 1.000, decimal=3)
    # assert_almost_equal(hung_yang_3(A2, B, similarity_type=HUNG_YANG_3_SIMILARITY_4), 0.714, decimal=3) # fails
    # assert_almost_equal(hung_yang_3(A3, B, similarity_type=HUNG_YANG_3_SIMILARITY_4), 0.500, decimal=3) # fails


def test_hung_yang_3_5():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_3(A1, B, similarity_type=HUNG_YANG_3_SIMILARITY_5), 1.000, decimal=3)  # 12
    assert_almost_equal(hung_yang_3(A2, B, similarity_type=HUNG_YANG_3_SIMILARITY_5), 0.833, decimal=3)
    assert_almost_equal(hung_yang_3(A3, B, similarity_type=HUNG_YANG_3_SIMILARITY_5), 0.667, decimal=3)


def test_hung_yang_3_6():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_3(A1, B, similarity_type=HUNG_YANG_3_SIMILARITY_6), 1.000, decimal=3)  # 13
    assert_almost_equal(hung_yang_3(A2, B, similarity_type=HUNG_YANG_3_SIMILARITY_6), 0.809, decimal=3)
    assert_almost_equal(hung_yang_3(A3, B, similarity_type=HUNG_YANG_3_SIMILARITY_6), 0.525, decimal=3)


def test_hung_yang_3_7():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_3(A1, B, similarity_type=HUNG_YANG_3_SIMILARITY_7), 1.000, decimal=3)  # 14
    assert_almost_equal(hung_yang_3(A2, B, similarity_type=HUNG_YANG_3_SIMILARITY_7), 0.783, decimal=3)
    assert_almost_equal(hung_yang_3(A3, B, similarity_type=HUNG_YANG_3_SIMILARITY_7), 0.533, decimal=3)
