from numpy.testing import assert_almost_equal

from fsmpy.sets import IntuitionisticFuzzySet
from fsmpy.similarities import iancu
from fsmpy import IANCU_SIMILARITY_1, IANCU_SIMILARITY_2, IANCU_SIMILARITY_3, IANCU_SIMILARITY_4, \
    IANCU_SIMILARITY_5, IANCU_SIMILARITY_6, IANCU_SIMILARITY_7, IANCU_SIMILARITY_8, IANCU_SIMILARITY_9, \
    IANCU_SIMILARITY_10, IANCU_SIMILARITY_11, IANCU_SIMILARITY_12, IANCU_SIMILARITY_13, IANCU_SIMILARITY_14,\
    IANCU_SIMILARITY_15, IANCU_SIMILARITY_16, IANCU_SIMILARITY_17, IANCU_SIMILARITY_18, IANCU_SIMILARITY_19, \
    IANCU_SIMILARITY_20


def test_iancu_1():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_1), 1.000, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_1), 1.000, decimal=3)
    assert_almost_equal(iancu(A3, B, similarity_type=IANCU_SIMILARITY_1), 1.000, decimal=3)
    
    # Example 2
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_1), 0.933, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_1), 0.933, decimal=3)

    
def test_iancu_2():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_2), 1.000, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_2), 1.000, decimal=3)
    assert_almost_equal(iancu(A3, B, similarity_type=IANCU_SIMILARITY_2), 1.000, decimal=3)
    
    # Example 2
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_2), 0.938, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_2), 0.938, decimal=3)
    
    
def test_iancu_5():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_5), 1.000, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_5), 0.933, decimal=3)
    assert_almost_equal(iancu(A3, B, similarity_type=IANCU_SIMILARITY_5), 0.800, decimal=3)
    
    # Example 2
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_5), 0.867, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_5), 0.833, decimal=3)
    
    
def test_iancu_6():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_6), 1.000, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_6), 0.933, decimal=3)
    assert_almost_equal(iancu(A3, B, similarity_type=IANCU_SIMILARITY_6), 0.800, decimal=3)
    
    # Example 2
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_6), 0.875, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_6), 0.844, decimal=3)
    
    
def test_iancu_7():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_7), 1.000, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_7), 0.875, decimal=3)
    assert_almost_equal(iancu(A3, B, similarity_type=IANCU_SIMILARITY_7), 0.667, decimal=3)
    
    # Example 2
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_7), 0.813, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_7), 0.758, decimal=3)
    
    
def test_iancu_8():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_8), 1.000, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_8), 0.875, decimal=3)
    assert_almost_equal(iancu(A3, B, similarity_type=IANCU_SIMILARITY_8), 0.667, decimal=3)
    
    # Example 2
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_8), 0.824, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_8), 0.771, decimal=3)
    
    
def test_iancu_9():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_9), 1.000, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_9), 0.938, decimal=3)
    assert_almost_equal(iancu(A3, B, similarity_type=IANCU_SIMILARITY_9), 0.833, decimal=3)

    # Example 2
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_9), 0.875, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_9), 0.848, decimal=3)

    
def test_iancu_10():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_10), 1.000, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_10), 0.938, decimal=3)
    assert_almost_equal(iancu(A3, B, similarity_type=IANCU_SIMILARITY_10), 0.833, decimal=3)

    # Example 2
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_10), 0.882, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_10), 0.857, decimal=3)

    
def test_iancu_13():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_13), 1.000, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_13), 0.966, decimal=3)
    assert_almost_equal(iancu(A3, B, similarity_type=IANCU_SIMILARITY_13), 0.889, decimal=3)

    # Example 2
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_13), 0.931, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_13), 0.912, decimal=3)

    
def test_iancu_14():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_14), 1.000, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_14), 0.933, decimal=3)
    assert_almost_equal(iancu(A3, B, similarity_type=IANCU_SIMILARITY_14), 0.800, decimal=3)
    
    # Example 2
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_14), 0.900, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_14), 0.867, decimal=3)
    
    
def test_iancu_16():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_16), 1.000, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_16), 0.967, decimal=3)
    assert_almost_equal(iancu(A3, B, similarity_type=IANCU_SIMILARITY_16), 0.900, decimal=3)
    
    # Example 2
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_16), 0.933, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_16), 0.917, decimal=3)
    
