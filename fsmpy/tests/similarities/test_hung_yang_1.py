from numpy.testing import assert_almost_equal

from fsmpy.sets import IntuitionisticFuzzySet
from fsmpy.similarities import hung_yang_1
from fsmpy import HUNG_YANG_1_SIMILARITY_1, HUNG_YANG_1_SIMILARITY_2, HUNG_YANG_1_SIMILARITY_3


def test_hung_yang_1_1():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_1(A1, B, similarity_type=HUNG_YANG_1_SIMILARITY_1), 1.00, decimal=2)
    assert_almost_equal(hung_yang_1(A2, B, similarity_type=HUNG_YANG_1_SIMILARITY_1), 0.933, decimal=3)
    assert_almost_equal(hung_yang_1(A3, B, similarity_type=HUNG_YANG_1_SIMILARITY_1), 0.800, decimal=3)

    # Example 2
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.3, 0.3, 0.3])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])

    assert_almost_equal(hung_yang_1(A1, B, similarity_type=HUNG_YANG_1_SIMILARITY_1), 0.900, decimal=3)
    assert_almost_equal(hung_yang_1(A2, B, similarity_type=HUNG_YANG_1_SIMILARITY_1), 0.833, decimal=3)

    # Example 3
    A1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.9, 0.9, 0.1])
    A2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.7, 0.2])
    A3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.8, 0.2, 0.6])
    B = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.6, 0.8, 0.2])

    assert_almost_equal(hung_yang_1(A1, B, similarity_type=HUNG_YANG_1_SIMILARITY_1), 0.833, decimal=3)
    assert_almost_equal(hung_yang_1(A2, B, similarity_type=HUNG_YANG_1_SIMILARITY_1), 0.933, decimal=3)
    assert_almost_equal(hung_yang_1(A3, B, similarity_type=HUNG_YANG_1_SIMILARITY_1), 0.567, decimal=3)


def test_hung_yang_1_2():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_1(A1, B, similarity_type=HUNG_YANG_1_SIMILARITY_2), 1.00, decimal=2)
    assert_almost_equal(hung_yang_1(A2, B, similarity_type=HUNG_YANG_1_SIMILARITY_2), 0.898, decimal=3)
    assert_almost_equal(hung_yang_1(A3, B, similarity_type=HUNG_YANG_1_SIMILARITY_2), 0.713, decimal=3)

    # Example 2
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.3, 0.3, 0.3])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])

    assert_almost_equal(hung_yang_1(A1, B, similarity_type=HUNG_YANG_1_SIMILARITY_2), 0.849, decimal=3)
    assert_almost_equal(hung_yang_1(A2, B, similarity_type=HUNG_YANG_1_SIMILARITY_2), 0.757, decimal=3)

    # Example 3
    A1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.9, 0.9, 0.1])
    A2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.7, 0.2])
    A3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.8, 0.2, 0.6])
    B = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.6, 0.8, 0.2])

    assert_almost_equal(hung_yang_1(A1, B, similarity_type=HUNG_YANG_1_SIMILARITY_3), 0.714, decimal=2)
    assert_almost_equal(hung_yang_1(A2, B, similarity_type=HUNG_YANG_1_SIMILARITY_3), 0.875, decimal=3)
    assert_almost_equal(hung_yang_1(A3, B, similarity_type=HUNG_YANG_1_SIMILARITY_3), 0.395, decimal=3)


def test_hung_yang_1_3():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_1(A1, B, similarity_type=HUNG_YANG_1_SIMILARITY_3), 1.00, decimal=2)
    assert_almost_equal(hung_yang_1(A2, B, similarity_type=HUNG_YANG_1_SIMILARITY_3), 0.875, decimal=3)
    assert_almost_equal(hung_yang_1(A3, B, similarity_type=HUNG_YANG_1_SIMILARITY_3), 0.667, decimal=3)

    # Example 2
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.3, 0.3, 0.3])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])

    assert_almost_equal(hung_yang_1(A1, B, similarity_type=HUNG_YANG_1_SIMILARITY_3), 0.818, decimal=3)
    assert_almost_equal(hung_yang_1(A2, B, similarity_type=HUNG_YANG_1_SIMILARITY_3), 0.714, decimal=3)

    # Example 3
    A1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.9, 0.9, 0.1])
    A2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.7, 0.2])
    A3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.8, 0.2, 0.6])
    B = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.6, 0.8, 0.2])

    assert_almost_equal(hung_yang_1(A1, B, similarity_type=HUNG_YANG_1_SIMILARITY_2), 0.757, decimal=3)
    assert_almost_equal(hung_yang_1(A2, B, similarity_type=HUNG_YANG_1_SIMILARITY_2), 0.898, decimal=3)
    assert_almost_equal(hung_yang_1(A3, B, similarity_type=HUNG_YANG_1_SIMILARITY_2), 0.444, decimal=3)

