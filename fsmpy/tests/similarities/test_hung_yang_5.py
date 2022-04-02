from numpy.testing import assert_almost_equal

from fsmpy.sets import IntuitionisticFuzzySet, IntuitionisticFuzzySet
from fsmpy.similarities import hung_yang_4
from fsmpy import HUNG_YANG_4_SIMILARITY_1, HUNG_YANG_4_SIMILARITY_2, HUNG_YANG_4_SIMILARITY_3


def test_hung_yang_4_1():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_4(A1, B, similarity_type=HUNG_YANG_4_SIMILARITY_1, p=2), 1.000, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, similarity_type=HUNG_YANG_4_SIMILARITY_1, p=2), 0.933, decimal=3)
    assert_almost_equal(hung_yang_4(A3, B, similarity_type=HUNG_YANG_4_SIMILARITY_1, p=2), 0.800, decimal=3)

    # Example 2
    A1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.1, 0.1, 0.9])
    A2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.3, 0.8])
    A3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.2, 0.8, 0.4])
    B = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.4, 0.2, 0.8])

    assert_almost_equal(hung_yang_4(A1, B, similarity_type=HUNG_YANG_4_SIMILARITY_1, p=2), 0.833, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, similarity_type=HUNG_YANG_4_SIMILARITY_1, p=2), 0.933, decimal=3)
    assert_almost_equal(hung_yang_4(A3, B, similarity_type=HUNG_YANG_4_SIMILARITY_1, p=2), 0.598, decimal=3)

    # Example 3
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])

    assert_almost_equal(hung_yang_4(A1, B, similarity_type=HUNG_YANG_4_SIMILARITY_1, p=2), 0.900, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, similarity_type=HUNG_YANG_4_SIMILARITY_1, p=2), 0.859, decimal=3)


def test_hung_yang_4_2():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_4(A1, B, similarity_type=HUNG_YANG_4_SIMILARITY_2, p=2), 1.000, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, similarity_type=HUNG_YANG_4_SIMILARITY_2, p=2), 0.881, decimal=3)
    assert_almost_equal(hung_yang_4(A3, B, similarity_type=HUNG_YANG_4_SIMILARITY_2, p=2), 0.675, decimal=3)

    # Example 2
    A1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.1, 0.1, 0.9])
    A2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.3, 0.8])
    A3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.2, 0.8, 0.4])
    B = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.4, 0.2, 0.8])

    assert_almost_equal(hung_yang_4(A1, B, similarity_type=HUNG_YANG_4_SIMILARITY_2, p=2), 0.723, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, similarity_type=HUNG_YANG_4_SIMILARITY_2, p=2), 0.881, decimal=3)
    assert_almost_equal(hung_yang_4(A3, B, similarity_type=HUNG_YANG_4_SIMILARITY_2, p=2), 0.427, decimal=3)

    # Example 3
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])

    assert_almost_equal(hung_yang_4(A1, B, similarity_type=HUNG_YANG_4_SIMILARITY_2, p=2), 0.826, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, similarity_type=HUNG_YANG_4_SIMILARITY_2, p=2), 0.761, decimal=3)



def test_hung_yang_4_3():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_4(A1, B, similarity_type=HUNG_YANG_4_SIMILARITY_3, p=2), 1.000, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, similarity_type=HUNG_YANG_4_SIMILARITY_3, p=2), 0.853, decimal=3)
    assert_almost_equal(hung_yang_4(A3, B, similarity_type=HUNG_YANG_4_SIMILARITY_3, p=2), 0.624, decimal=3)

    # Example 2
    A1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.1, 0.1, 0.9])
    A2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.3, 0.8])
    A3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.2, 0.8, 0.4])
    B = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.4, 0.2, 0.8])

    assert_almost_equal(hung_yang_4(A1, B, similarity_type=HUNG_YANG_4_SIMILARITY_3, p=2), 0.674, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, similarity_type=HUNG_YANG_4_SIMILARITY_3, p=2), 0.853, decimal=3)
    assert_almost_equal(hung_yang_4(A3, B, similarity_type=HUNG_YANG_4_SIMILARITY_3, p=2), 0.381, decimal=3)

    # Example 3
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])

    assert_almost_equal(hung_yang_4(A1, B, similarity_type=HUNG_YANG_4_SIMILARITY_3, p=2), 0.788, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, similarity_type=HUNG_YANG_4_SIMILARITY_3, p=2), 0.716, decimal=3)

