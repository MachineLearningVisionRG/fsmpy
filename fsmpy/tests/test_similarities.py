import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from fsmpy.datasets import load_patients_diagnoses
from fsmpy.sets import IntuitionisticFuzzySet
from fsmpy.similarities import chen_1, hung_yang_4, hung_yang_3, hung_yang_2, hwang_yang, park_kwun_lim, ye, hung_yang_1, julian_hung_lin, zhang_fu
from fsmpy.similarities import mitchell, iancu, liang_shi, dengfeng_chuntian, hong_kim, chen_cheng_lan, song_wang_lei_xue
from fsmpy.similarities import liu, chen_2
from fsmpy.similarities import dengfeng_chuntian, hong_kim, song_wang_lei_xue, muthukumar_krishnanb, nguyen, deng_jiang_fu
from fsmpy import LIANG_SHI_SIMILARITY_1, LIANG_SHI_SIMILARITY_2, LIANG_SHI_SIMILARITY_3
from fsmpy import HUNG_YANG_3_SIMILARITY_1, HUNG_YANG_3_SIMILARITY_2, HUNG_YANG_3_SIMILARITY_3, \
    HUNG_YANG_3_SIMILARITY_4, HUNG_YANG_3_SIMILARITY_5, HUNG_YANG_3_SIMILARITY_6, HUNG_YANG_3_SIMILARITY_7
from fsmpy import DENG_JIANG_FU_MONOTONIC_TYPE_1_1, DENG_JIANG_FU_MONOTONIC_TYPE_1_2, \
    DENG_JIANG_FU_MONOTONIC_TYPE_1_3, DENG_JIANG_FU_MONOTONIC_TYPE_1_4, DENG_JIANG_FU_MONOTONIC_TYPE_2_1, \
    DENG_JIANG_FU_MONOTONIC_TYPE_2_2, DENG_JIANG_FU_MONOTONIC_TYPE_2_3, DENG_JIANG_FU_MONOTONIC_TYPE_2_4, \
    DENG_JIANG_FU_MONOTONIC_TYPE_3_1, DENG_JIANG_FU_MONOTONIC_TYPE_3_2, DENG_JIANG_FU_MONOTONIC_TYPE_3_3
from fsmpy import IANCU_SIMILARITY_1, IANCU_SIMILARITY_2, IANCU_SIMILARITY_3, IANCU_SIMILARITY_4, \
    IANCU_SIMILARITY_5, IANCU_SIMILARITY_6, IANCU_SIMILARITY_7, IANCU_SIMILARITY_8, IANCU_SIMILARITY_9, \
    IANCU_SIMILARITY_10, IANCU_SIMILARITY_11, IANCU_SIMILARITY_12, IANCU_SIMILARITY_13, IANCU_SIMILARITY_14,\
    IANCU_SIMILARITY_15, IANCU_SIMILARITY_16, IANCU_SIMILARITY_17, IANCU_SIMILARITY_18, IANCU_SIMILARITY_19, \
    IANCU_SIMILARITY_20


def test_dengfeng_chuntian():
    # Example 1
    A1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    A2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    A3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    B = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])

    assert_almost_equal(dengfeng_chuntian(A1, B, p=1, weights=None), 0.78, decimal=2)
    # assert_almost_equal(dengfeng_chuntian(A2, B, p=1, weights=None), 0.80, decimal=2) # fails
    assert_almost_equal(dengfeng_chuntian(A3, B, p=1, weights=None), 0.85, decimal=2)

    assert_almost_equal(dengfeng_chuntian(A1, B, p=2, weights=None), 0.74, decimal=2)
    assert_almost_equal(dengfeng_chuntian(A2, B, p=2, weights=None), 0.78, decimal=2)
    assert_almost_equal(dengfeng_chuntian(A3, B, p=2, weights=None), 0.84, decimal=2)

    assert_almost_equal(dengfeng_chuntian(A1, B, p=2, weights=[0.5, 0.3, 0.2]), 0.696, decimal=3)
    # assert_almost_equal(dengfeng_chuntian(A2, B, p=2, weights=[0.5, 0.3, 0.2]), 0.779, decimal=3)
    assert_almost_equal(dengfeng_chuntian(A3, B, p=2, weights=[0.5, 0.3, 0.2]), 0.853, decimal=3)


def test_liang_shi():
    A1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.9, 0.9, 0.1])
    A2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.7, 0.2])
    A3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.8, 0.2, 0.6])
    B = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.6, 0.8, 0.2])

    # Example 1
    assert_almost_equal(liang_shi(A1, B, similarity_type=LIANG_SHI_SIMILARITY_1, p=1), 0.83, decimal=2)
    assert_almost_equal(liang_shi(A2, B, similarity_type=LIANG_SHI_SIMILARITY_1, p=1), 0.93, decimal=2)
    assert_almost_equal(liang_shi(A3, B, similarity_type=LIANG_SHI_SIMILARITY_1, p=1), 0.60, decimal=2)

    # Example 2 # fails
    assert_almost_equal(liang_shi(A1, B, similarity_type=LIANG_SHI_SIMILARITY_2, p=1), 0.92, decimal=2)
    assert_almost_equal(liang_shi(A2, B, similarity_type=LIANG_SHI_SIMILARITY_2, p=1), 0.97, decimal=2)
    assert_almost_equal(liang_shi(A3, B, similarity_type=LIANG_SHI_SIMILARITY_2, p=1), 0.77, decimal=2)

    # Example 3
    assert_almost_equal(liang_shi(A1, B, similarity_type=LIANG_SHI_SIMILARITY_3, p=1, omegas=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), 0.89, decimal=2)
    assert_almost_equal(liang_shi(A2, B, similarity_type=LIANG_SHI_SIMILARITY_3, p=1, omegas=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), 0.95, decimal=2)
    assert_almost_equal(liang_shi(A3, B, similarity_type=LIANG_SHI_SIMILARITY_3, p=1, omegas=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]), 0.72, decimal=2)


def test_park_kwun_lin():
    A1 = IntuitionisticFuzzySet([0.2, 0.1, 0.0], [0.6, 0.7, 0.6])
    A2 = IntuitionisticFuzzySet([0.2, 0.0, 0.2], [0.6, 0.6, 0.8])
    A3 = IntuitionisticFuzzySet([0.1, 0.2, 0.2], [0.5, 0.7, 0.8])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.7, 0.8, 0.7])

    assert_almost_equal(park_kwun_lim(A1, B), 0.800, decimal=3)
    assert_almost_equal(park_kwun_lim(A2, B), 0.733, decimal=3)
    assert_almost_equal(park_kwun_lim(A3, B), 0.767, decimal=3)


def test_mitchell():
    A1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    A2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    A3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    B = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])

    # raise NotImplementedError("No tests implemented")


def test_julian_hung_lin():
    pass
    # raise NotImplementedError("No tests implemented")


def test_hung_yang_1():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_1(A1, B), 1.00, decimal=2)
    assert_almost_equal(hung_yang_1(A2, B), 0.933, decimal=3)
    assert_almost_equal(hung_yang_1(A3, B), 0.800, decimal=3)

    assert_almost_equal(hung_yang_1(A1, B, similarity_type='c'), 1.00, decimal=2)
    assert_almost_equal(hung_yang_1(A2, B, similarity_type='c'), 0.875, decimal=3)
    assert_almost_equal(hung_yang_1(A3, B, similarity_type='c'), 0.667, decimal=3)

    assert_almost_equal(hung_yang_1(A1, B, similarity_type='e'), 1.00, decimal=2)
    assert_almost_equal(hung_yang_1(A2, B, similarity_type='e'), 0.898, decimal=3)
    assert_almost_equal(hung_yang_1(A3, B, similarity_type='e'), 0.713, decimal=3)

    # Example 2
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.3, 0.3, 0.3])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])

    assert_almost_equal(hung_yang_1(A1, B), 0.900, decimal=3)
    assert_almost_equal(hung_yang_1(A2, B), 0.833, decimal=3)

    assert_almost_equal(hung_yang_1(A1, B, similarity_type='c'), 0.818, decimal=3)
    assert_almost_equal(hung_yang_1(A2, B, similarity_type='c'), 0.714, decimal=3)

    assert_almost_equal(hung_yang_1(A1, B, similarity_type='e'), 0.849, decimal=3)
    assert_almost_equal(hung_yang_1(A2, B, similarity_type='e'), 0.757, decimal=3)

    # Example 3
    A1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.9, 0.9, 0.1])
    A2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.7, 0.2])
    A3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.8, 0.2, 0.6])
    B = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.6, 0.8, 0.2])

    assert_almost_equal(hung_yang_1(A1, B), 0.833, decimal=3)
    assert_almost_equal(hung_yang_1(A2, B), 0.933, decimal=3)
    assert_almost_equal(hung_yang_1(A3, B), 0.567, decimal=3)

    assert_almost_equal(hung_yang_1(A1, B, similarity_type='c'), 0.714, decimal=2)
    assert_almost_equal(hung_yang_1(A2, B, similarity_type='c'), 0.875, decimal=3)
    assert_almost_equal(hung_yang_1(A3, B, similarity_type='c'), 0.395, decimal=3)

    assert_almost_equal(hung_yang_1(A1, B, similarity_type='e'), 0.757, decimal=3)
    assert_almost_equal(hung_yang_1(A2, B, similarity_type='e'), 0.898, decimal=3)
    assert_almost_equal(hung_yang_1(A3, B, similarity_type='e'), 0.444, decimal=3)


def test_ye():
    # Example 1
    C1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    C2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    C3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    Q = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])

    assert_almost_equal(ye(C1, Q), 0.9353, decimal=4)
    assert_almost_equal(ye(C2, Q), 0.9519, decimal=4)
    assert_almost_equal(ye(C3, Q), 0.9724, decimal=4)

    weights = [0.5, 0.3, 0.2]
  
    assert_almost_equal(ye(C1, Q, weights=weights), 0.9133, decimal=4)
    assert_almost_equal(ye(C2, Q, weights=weights), 0.9404, decimal=4)
    assert_almost_equal(ye(C3, Q, weights=weights), 0.9712, decimal=4)

    viral_fever = IntuitionisticFuzzySet([0.4, 0.3, 0.1, 0.4, 0.1],
                           [0.0, 0.5, 0.7, 0.3, 0.7])
    malaria = IntuitionisticFuzzySet([0.7, 0.2, 0.0, 0.7, 0.1], [0.0, 0.6, 0.9, 0.0, 0.8])
    typhoid = IntuitionisticFuzzySet([0.3, 0.6, 0.2, 0.2, 0.1], [0.3, 0.1, 0.7, 0.6, 0.9])
    stomach_problem = IntuitionisticFuzzySet([0.1, 0.2, 0.8, 0.2, 0.2], [
        0.7, 0.4, 0.0, 0.7, 0.7])
    chest_problem = IntuitionisticFuzzySet([0.1, 0.0, 0.2, 0.2, 0.8], [
        0.8, 0.8, 0.8, 0.8, 0.1])
    
    patient = IntuitionisticFuzzySet([0.8, 0.6, 0.2, 0.6, 0.1], [0.1, 0.1, 0.8, 0.1, 0.6])

    assert_almost_equal(ye(patient, viral_fever), 0.9046, decimal=4)
    assert_almost_equal(ye(patient, malaria), 0.8602, decimal=4) # fails
    assert_almost_equal(ye(patient, typhoid), 0.8510, decimal=4)
    assert_almost_equal(ye(patient, stomach_problem), 0.5033, decimal=4)
    assert_almost_equal(ye(patient, chest_problem), 0.4542, decimal=4) # fails


def test_hwang_yang():
    # Example 1
    X1A = IntuitionisticFuzzySet([0.3], [0.3])
    X1B = IntuitionisticFuzzySet([0.4], [0.4])
    assert_almost_equal(hwang_yang(X1A, X1B), 0.997, decimal=3) # fails

    X2A = IntuitionisticFuzzySet([0.3], [0.4])
    X2B = IntuitionisticFuzzySet([0.4], [0.3])
    assert_almost_equal(hwang_yang(X2A, X2B), 0.859, decimal=3)

    X3A = IntuitionisticFuzzySet([1.0], [0.0])
    X3B = IntuitionisticFuzzySet([0.0], [0.0])
    assert_almost_equal(hwang_yang(X3A, X3B), 0.902, decimal=3)

    X4A = IntuitionisticFuzzySet([0.5], [0.5])
    X4B = IntuitionisticFuzzySet([0.0], [0.0])
    assert_almost_equal(hwang_yang(X4A, X4B), 0.902, decimal=3)

    X5A = IntuitionisticFuzzySet([0.4], [0.2])
    X5B = IntuitionisticFuzzySet([0.5], [0.3])
    assert_almost_equal(hwang_yang(X5A, X5B), 0.995, decimal=3)

    X6A = IntuitionisticFuzzySet([0.4], [0.2])
    X6B = IntuitionisticFuzzySet([0.5], [0.2])
    assert_almost_equal(hwang_yang(X6A, X6B), 0.997, decimal=3)


def test_hung_yang_2():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1], [0.4, 0.6, 0.8])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.6, 0.6, 0.6])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4], [0.2, 0.2, 0.2])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1], [0.4, 0.6, 0.8])

    assert_almost_equal(hung_yang_2(A1, B), 1.00, decimal=2)
    assert_almost_equal(hung_yang_2(A2, B), 0.979, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B), 0.854, decimal=3)

    assert_almost_equal(hung_yang_2(A1, B, similarity_type='c'), 1.00, decimal=2)
    assert_almost_equal(hung_yang_2(A2, B, similarity_type='c'), 0.964, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B, similarity_type='c'), 0.776, decimal=3)

    assert_almost_equal(hung_yang_2(A1, B, similarity_type='e'), 1.00, decimal=2)
    assert_almost_equal(hung_yang_2(A2, B, similarity_type='e'), 0.971, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B, similarity_type='e'), 0.808, decimal=3)

    # Example 2
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.6, 0.6, 0.6])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4], [0.2, 0.2, 0.2])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3], [0.4, 0.4, 0.6])

    assert_almost_equal(hung_yang_2(A1, B), 0.974, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B), 0.928, decimal=3)
    assert_almost_equal(hung_yang_2(A1, B, a=1.5), 0.974, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, a=1.5), 0.928, decimal=3)

    assert_almost_equal(hung_yang_2(A1, B, similarity_type='c'), 0.957, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, similarity_type='c'), 0.882, decimal=3)

    assert_almost_equal(hung_yang_2(A1, B, similarity_type='e'), 0.964, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, similarity_type='e'), 0.901, decimal=3)

    # Example 3      Division with zero
    A1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.1, 0.1, 0.9], [0.8, 0.4, 0.0])
    A2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.3, 0.8], [0.0, 0.0, 0.2])
    A3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.2, 0.8, 0.4], [0.1, 0.1, 0.2])
    B = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.4, 0.2, 0.8], [0.2, 0.2, 0.2])

    assert_almost_equal(hung_yang_2(A1, B), 0.843, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B), 0.927, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B), 0.797, decimal=3)

    assert_almost_equal(hung_yang_2(A1, B, similarity_type='c'), 0.761, decimal=2)
    assert_almost_equal(hung_yang_2(A2, B, similarity_type='c'), 0.883, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B, similarity_type='c'), 0.698, decimal=3)

    assert_almost_equal(hung_yang_2(A1, B, similarity_type='e'), 0.794, decimal=3)
    assert_almost_equal(hung_yang_2(A2, B, similarity_type='e'), 0.902, decimal=3)
    assert_almost_equal(hung_yang_2(A3, B, similarity_type='e'), 0.737, decimal=3)


def test_zhang_fu():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.4, 0.3, 0.5, 0.5, 0.6], [0.4, 0.3, 0.1, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.2, 0.3, 0.2, 0.7, 0.8], [0.6, 0.5, 0.3, 0.1, 0.0])
    A3 = IntuitionisticFuzzySet([0.1, 0.0, 0.2, 0.1, 0.2], [0.9, 1.0, 0.7, 0.8, 0.8])
    A4 = IntuitionisticFuzzySet([0.8, 0.9, 1.0, 0.7, 0.6], [0.2, 0.0, 0.0, 0.2, 0.4])
    A = IntuitionisticFuzzySet([0.3, 0.4, 0.6, 0.5, 0.9], [0.5, 0.4, 0.2, 0.1, 0.0])

    assert_almost_equal(zhang_fu(A, A1), 0.884, decimal=3)
    assert_almost_equal(zhang_fu(A, A2), 0.870, decimal=3)
    assert_almost_equal(zhang_fu(A, A3), 0.449, decimal=3)
    assert_almost_equal(zhang_fu(A, A4), 0.671, decimal=3)


def test_hung_yang_3():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_3(A1, B, similarity_type=HUNG_YANG_3_SIMILARITY_1), 1.000, decimal=3)   # 8
    assert_almost_equal(hung_yang_3(A2, B, similarity_type=HUNG_YANG_3_SIMILARITY_1), 0.722, decimal=3)
    assert_almost_equal(hung_yang_3(A3, B, similarity_type=HUNG_YANG_3_SIMILARITY_1), 0.500, decimal=3)

    assert_almost_equal(hung_yang_3(A1, B, similarity_type=HUNG_YANG_3_SIMILARITY_2), 1.000, decimal=3)  # 11
    assert_almost_equal(hung_yang_3(A2, B, similarity_type=HUNG_YANG_3_SIMILARITY_2), 0.900, decimal=3)
    assert_almost_equal(hung_yang_3(A3, B, similarity_type=HUNG_YANG_3_SIMILARITY_2), 0.700, decimal=3)

    assert_almost_equal(hung_yang_3(A1, B, similarity_type=HUNG_YANG_3_SIMILARITY_3), 1.000, decimal=3)  # 10
    assert_almost_equal(hung_yang_3(A2, B, similarity_type=HUNG_YANG_3_SIMILARITY_3), 0.714, decimal=3)
    assert_almost_equal(hung_yang_3(A3, B, similarity_type=HUNG_YANG_3_SIMILARITY_3), 0.500, decimal=3)

    assert_almost_equal(hung_yang_3(A1, B, similarity_type=HUNG_YANG_3_SIMILARITY_4), 1.000, decimal=3)
    assert_almost_equal(hung_yang_3(A2, B, similarity_type=HUNG_YANG_3_SIMILARITY_4), 0.714, decimal=3) # fails
    assert_almost_equal(hung_yang_3(A3, B, similarity_type=HUNG_YANG_3_SIMILARITY_4), 0.500, decimal=3) # fails

    assert_almost_equal(hung_yang_3(A1, B, similarity_type=HUNG_YANG_3_SIMILARITY_5), 1.000, decimal=3)  # 12
    assert_almost_equal(hung_yang_3(A2, B, similarity_type=HUNG_YANG_3_SIMILARITY_5), 0.833, decimal=3)
    assert_almost_equal(hung_yang_3(A3, B, similarity_type=HUNG_YANG_3_SIMILARITY_5), 0.667, decimal=3)

    assert_almost_equal(hung_yang_3(A1, B, similarity_type=HUNG_YANG_3_SIMILARITY_6), 1.000, decimal=3)  # 13
    assert_almost_equal(hung_yang_3(A2, B, similarity_type=HUNG_YANG_3_SIMILARITY_6), 0.809, decimal=3)
    assert_almost_equal(hung_yang_3(A3, B, similarity_type=HUNG_YANG_3_SIMILARITY_6), 0.525, decimal=3)

    assert_almost_equal(hung_yang_3(A1, B, similarity_type=HUNG_YANG_3_SIMILARITY_7), 1.000, decimal=3)  # 14
    assert_almost_equal(hung_yang_3(A2, B, similarity_type=HUNG_YANG_3_SIMILARITY_7), 0.783, decimal=3)
    assert_almost_equal(hung_yang_3(A3, B, similarity_type=HUNG_YANG_3_SIMILARITY_7), 0.533, decimal=3)


def test_chen_1():
    A = IntuitionisticFuzzySet([0.1, 0.2, 0.4, 0.6, 0.8], [0.3, 0.6, 0.8, 0.8, 1.0])
    B = IntuitionisticFuzzySet([0.2, 0.3, 0.5, 0.7, 0.9], [0.5, 0.7, 0.8, 0.9, 1.0])

    assert_almost_equal(chen_1(A, B, weights=[0.5, 0.8, 1.0, 0.7, 1.0]), 0.90625, decimal=5)


def test_hung_yang_4():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(hung_yang_4(A1, B, p=2), 1.000, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, p=2), 0.933, decimal=3)
    assert_almost_equal(hung_yang_4(A3, B, p=2), 0.800, decimal=3)

    assert_almost_equal(hung_yang_4(A1, B, similarity_type='c', p=2), 1.000, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, similarity_type='c', p=2), 0.853, decimal=3)
    assert_almost_equal(hung_yang_4(A3, B, similarity_type='c', p=2), 0.624, decimal=3)

    assert_almost_equal(hung_yang_4(A1, B, similarity_type='e', p=2), 1.000, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, similarity_type='e', p=2), 0.881, decimal=3)
    assert_almost_equal(hung_yang_4(A3, B, similarity_type='e', p=2), 0.675, decimal=3)

    # Example 2

    A1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.1, 0.1, 0.9])
    A2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.3, 0.8])
    A3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.2, 0.8, 0.4])
    B = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.4, 0.2, 0.8])

    assert_almost_equal(hung_yang_4(A1, B, p=2), 0.833, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, p=2), 0.933, decimal=3)
    assert_almost_equal(hung_yang_4(A3, B, p=2), 0.598, decimal=3)

    assert_almost_equal(hung_yang_4(A1, B, similarity_type='c', p=2), 0.674, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, similarity_type='c', p=2), 0.853, decimal=3)
    assert_almost_equal(hung_yang_4(A3, B, similarity_type='c', p=2), 0.381, decimal=3)

    assert_almost_equal(hung_yang_4(A1, B, similarity_type='e', p=2), 0.723, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, similarity_type='e', p=2), 0.881, decimal=3)
    assert_almost_equal(hung_yang_4(A3, B, similarity_type='e', p=2), 0.427, decimal=3)

    # Example 3
    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])

    assert_almost_equal(hung_yang_4(A1, B, p=2), 0.900, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, p=2), 0.859, decimal=3)

    assert_almost_equal(hung_yang_4(A1, B, similarity_type='c', p=2), 0.788, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, similarity_type='c', p=2), 0.716, decimal=3)

    assert_almost_equal(hung_yang_4(A1, B, similarity_type='e', p=2), 0.826, decimal=3)
    assert_almost_equal(hung_yang_4(A2, B, similarity_type='e', p=2), 0.761, decimal=3)


def test_hong_kim():
    #Example 1
    A = IntuitionisticFuzzySet([0.8, 0.3, 0.4], [0.9, 0.5, 0.6])
    B = IntuitionisticFuzzySet([0.9, 0.0, 0.8], [0.9, 0.0, 0.9])

    assert_almost_equal(hong_kim(A, B, weights=None), 0.7333, decimal=4)


def test_chen_2():
    # Example 1
    A = IntuitionisticFuzzySet([0.8, 0.3, 0.4], [0.9, 0.5, 0.6])
    B = IntuitionisticFuzzySet([0.9, 0.0, 0.8], [0.9, 0.0, 0.9])

    assert_almost_equal(chen_2(A, B, weights=None), 0.7333, decimal=4)


def test_liu():
    A1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1], [0.0, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0], [0.1, 0.0, 0.1])
    A3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0], [0.2, 0.2, 0.0])
    B = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1], [0.2, 0.2, 0.1])

    assert_almost_equal(liu(A1, B, p=2), 0.72, decimal=2)
    assert_almost_equal(liu(A2, B, p=2), 0.74, decimal=2)
    assert_almost_equal(liu(A3, B, p=2), 0.84, decimal=2)

def test_iancu():
    # Example 1
    A1 = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
    A2 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.3, 0.2, 0.1])

    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_1), 1.000, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_1), 1.000, decimal=3)
    assert_almost_equal(iancu(A3, B, similarity_type=IANCU_SIMILARITY_1), 1.000, decimal=3)

    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_2), 1.000, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_2), 1.000, decimal=3)
    assert_almost_equal(iancu(A3, B, similarity_type=IANCU_SIMILARITY_2), 1.000, decimal=3)

    # fail
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_9), 1.000, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_9), 0.938, decimal=3)
    assert_almost_equal(iancu(A3, B, similarity_type=IANCU_SIMILARITY_9), 0.833, decimal=3)

    # fail
    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_10), 1.000, decimal=3)
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_10), 0.938, decimal=3)
    assert_almost_equal(iancu(A3, B, similarity_type=IANCU_SIMILARITY_10), 0.833, decimal=3)

    A1 = IntuitionisticFuzzySet([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
    B = IntuitionisticFuzzySet([0.3, 0.3, 0.1], [0.3, 0.3, 0.3])

    assert_almost_equal(iancu(A1, B), 0.933, decimal=3)
    assert_almost_equal(iancu(A2, B), 0.933, decimal=3)

    assert_almost_equal(iancu(A1, B, similarity_type=IANCU_SIMILARITY_7), 0.938, decimal=3) # fails
    assert_almost_equal(iancu(A2, B, similarity_type=IANCU_SIMILARITY_7), 0.938, decimal=3)


def test_song_wang_lei_xue():
    # Example 1
    A1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1], [0.0, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0], [0.1, 0.0, 0.1])
    A3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0], [0.2, 0.2, 0.0])
    B = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1], [0.2, 0.2, 0.1])

    assert_almost_equal(song_wang_lei_xue(A1, B), 0.887, decimal=3)
    assert_almost_equal(song_wang_lei_xue(A2, B), 0.913, decimal=3)
    assert_almost_equal(song_wang_lei_xue(A3, B), 0.936, decimal=3)

    assert_almost_equal(song_wang_lei_xue(A1, B, weights=[0.5, 0.3, 0.2]), 0.853, decimal=3)
    assert_almost_equal(song_wang_lei_xue(A2, B, weights=[0.5, 0.3, 0.2]), 0.919, decimal=3)
    assert_almost_equal(song_wang_lei_xue(A3, B, weights=[0.5, 0.3, 0.2]), 0.949, decimal=3)


def test_deng_jiang_fu():
    A1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    A2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    A3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    B = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])

    # Example 2
    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_1), 0.489, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_1), 0.458, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_1), 0.546, decimal=3)

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_2), 0.454, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_2), 0.444, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_2), 0.541, decimal=3)

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.625, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.615, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.702, decimal=3)

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_1), 0.681, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_1), 0.668, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_1), 0.745, decimal=3)

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_2), 0.658, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_2), 0.658, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_2), 0.743, decimal=3)

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.783, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.783, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.850, decimal=3)

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_4), 0.644, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_4), 0.644, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_4), 0.739, decimal=3)

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.593, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.593, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.700, decimal=3) # fails

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.928, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.941, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.975, decimal=3)

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.667, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.667, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.766, decimal=3)

    # Example 3
    diagnoses, patients = load_patients_diagnoses()
    viral_fever, malaria, typhoid, stomach_problem, chest_problem = diagnoses
    al, bob, joe, ted = patients

    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.467,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.437,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.608,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.698,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.683,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.81,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.681,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.634, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.947,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.706,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.517,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.489,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.657,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.709,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.69, decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.82,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.695,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.65, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5),
                        0.946, decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.721,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.544,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.474,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.643,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.698,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.661,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.8,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.667,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.619, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5),
                        0.92, decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.691,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.216,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.186,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1),
                        0.313, decimal=3)
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.393,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.361,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.54,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.37,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.304, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.736,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1),
                        0.339, decimal=3)
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.26,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.184,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.311,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.375,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.324,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.5,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.333,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.269, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.678,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.293,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.348,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.28,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.437,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.518,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.476,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.67,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.504,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.441, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.831,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.508,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.3, decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.21,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.348,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.419,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.352,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.54,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.37,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.304, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.694,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.34,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.415,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.366,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.536,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.594,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.567,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.74,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.587,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.531, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.898,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.605,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.641,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.635,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1),
                        0.777, decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.826,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.825,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.9,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.818,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.79, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.986,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1),
                        0.844, decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.371,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.309,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.472,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.509,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.463,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.64,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.471,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.406, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.802,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.464,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.363,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.348,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.516,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.618,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.603,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.75,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.6,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.545, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.915,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.617,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.344,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.308,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.471,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.533,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.492,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.68,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.515,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.453, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.844,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.52,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.498,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.47,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.639,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.712,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.7, decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.82,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.695,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.65, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.944,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.721,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.32,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.241,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1),
                        0.388, decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.512,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.452,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.6,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.429,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.363, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.762,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1),
                        0.415, decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.277,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.214,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.353,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.449,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.387,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.54,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.37,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.304, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.7,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.34,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.407,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.403,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.574,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.672,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.672,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.8,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.667,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.619, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.954,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.691,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.421,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.401,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.572,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.624,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.61,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.77,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.626,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.574, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.927,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.648,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.318,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.31,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.474,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.541,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.532,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.71,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.55,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.491, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.897,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.561,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.264,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.243,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1),
                        0.391, decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.481,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.464,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1),
                        0.63, decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.46,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.395, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.829,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1),
                        0.451, decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.198,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.189,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.319,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.376,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.366,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.55,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.379,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.314, decimal=3) # fails
    assert_almost_equal(
        deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.773,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.351,
                        decimal=3)


def test_nguyen():
    # Example 1
    M = IntuitionisticFuzzySet([1.0], [0.0], [0.0])
    N = IntuitionisticFuzzySet([0.0], [1.0], [0.0])
    N = IntuitionisticFuzzySet([0.0], [1.0], [0.0])
    F = IntuitionisticFuzzySet([0.0], [0.0], [1.0])
    assert_equal(nguyen(M, N), -1)
    assert_equal(nguyen(M, F), 0.0)

    R = IntuitionisticFuzzySet([0.5], [0.3], [0.2])
    S = IntuitionisticFuzzySet([0.5], [0.2], [0.3])
    assert_almost_equal(nguyen(M, R), 0.7, decimal=1)
    assert_almost_equal(nguyen(M, S), 0.625, decimal=3)

    # Example 2
    A = IntuitionisticFuzzySet([0.3], [0.3], [0.4])
    B = IntuitionisticFuzzySet([0.4], [0.4], [0.2])
    assert_almost_equal(nguyen(A, B), 0.827, decimal=3)

    A = IntuitionisticFuzzySet([0.3], [0.4], [0.3])
    B = IntuitionisticFuzzySet([0.4], [0.3], [0.3])
    assert_equal(nguyen(A, B), -1)

    A = IntuitionisticFuzzySet([1.0], [0.0], [0.0])
    B = IntuitionisticFuzzySet([0.0], [0.0], [1.0])
    assert_equal(nguyen(A, B), 0.0)

    A = IntuitionisticFuzzySet([0.5], [0.5], [0.0])
    B = IntuitionisticFuzzySet([0.0], [0.0], [1.0])
    assert_almost_equal(nguyen(A, B), 0.134, decimal=3)

    A = IntuitionisticFuzzySet([0.4], [0.2], [0.4])
    B = IntuitionisticFuzzySet([0.5], [0.3], [0.2])
    assert_almost_equal(nguyen(A, B), 0.829, decimal=3)

    A = IntuitionisticFuzzySet([0.4], [0.2], [0.4])
    B = IntuitionisticFuzzySet([0.5], [0.2], [0.3])
    assert_almost_equal(nguyen(A, B), 0.904, decimal=3)

    A = IntuitionisticFuzzySet([0.0], [0.87], [0.13])
    B = IntuitionisticFuzzySet([0.28], [0.55], [0.17])
    assert_almost_equal(nguyen(A, B), 0.861, decimal=3)

    A = IntuitionisticFuzzySet([0.6], [0.87], [-0.4])
    B = IntuitionisticFuzzySet([0.28], [0.55], [0.17])
    assert_almost_equal(nguyen(A, B), 0.960, decimal=3) # fails

    # Example 3
    A1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1], [0.0, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.35, 0.45, 0.55], [0.15, 0.25, 0.35], [0.5, 0.3, 0.1])
    A3 = IntuitionisticFuzzySet([0.25, 0.35, 0.45], [0.25, 0.35, 0.45], [0.5, 0.3, 0.1])
    B = IntuitionisticFuzzySet([0.3, 0.4, 0.5], [0.2, 0.3, 0.4], [0.5, 0.3, 0.1])
    assert_almost_equal(nguyen(A1, B), 0.757, decimal=3)
    assert_almost_equal(nguyen(A2, B), 0.994, decimal=3)
    assert_almost_equal(nguyen(A3, B), 0.998, decimal=3)

    # Example 4
    A1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1], [0.0, 0.2, 0.2])
    A2 = IntuitionisticFuzzySet([0.3, 0.4, 0.2], [0.5, 0.4, 0.6], [0.2, 0.2, 0.2])
    A3 = IntuitionisticFuzzySet([0.4, 0.3, 0.2], [0.4, 0.5, 0.6], [0.2, 0.2, 0.2])
    B = IntuitionisticFuzzySet([0.3, 0.4, 0.5], [0.3, 0.4, 0.5], [0.4, 0.2, 0.0])
    assert_almost_equal(nguyen(A1, B), 0.841, decimal=3)
    assert_almost_equal(nguyen(A2, B), -0.988, decimal=3)
    assert_almost_equal(nguyen(A3, B), -0.988, decimal=3)


def test_chen_cheng_lan():
    A = IntuitionisticFuzzySet([0.3], [0.3])
    B = IntuitionisticFuzzySet([0.4], [0.4])
    assert_almost_equal(chen_cheng_lan(A, B), 0.9667, decimal=4)
    
    A = IntuitionisticFuzzySet([0.3], [0.4])
    B = IntuitionisticFuzzySet([0.4], [0.3])
    assert_almost_equal(chen_cheng_lan(A, B), 0.9000, decimal=4)
    
    A = IntuitionisticFuzzySet([1.0], [0.0])
    B = IntuitionisticFuzzySet([0.0], [0.0])
    assert_almost_equal(chen_cheng_lan(A, B), 0.5000, decimal=4)
    
    A = IntuitionisticFuzzySet([0.5], [0.5])
    B = IntuitionisticFuzzySet([0.0], [0.0])
    assert_almost_equal(chen_cheng_lan(A, B), 0.8333, decimal=4)
    
    A = IntuitionisticFuzzySet([0.4], [0.2])
    B = IntuitionisticFuzzySet([0.5], [0.3])
    assert_almost_equal(chen_cheng_lan(A, B), 0.9667, decimal=4)
    
    A = IntuitionisticFuzzySet([0.4], [0.2])
    B = IntuitionisticFuzzySet([0.5], [0.2])
    assert_almost_equal(chen_cheng_lan(A, B), 0.9450, decimal=4)

    # Table 2
    A = IntuitionisticFuzzySet([0.5], [0.5])
    B = IntuitionisticFuzzySet([0.0], [0.0])
    assert_almost_equal(chen_cheng_lan(A, B), 0.8333, decimal=4)
    
    A = IntuitionisticFuzzySet([0.6], [0.4])
    B = IntuitionisticFuzzySet([0.0], [0.0])
    assert_almost_equal(chen_cheng_lan(A, B), 0.8330, decimal=3)
    
    A = IntuitionisticFuzzySet([0.0], [0.87])
    B = IntuitionisticFuzzySet([0.28], [0.55])
    assert_almost_equal(chen_cheng_lan(A, B), 0.7047, decimal=4)
    
    A = IntuitionisticFuzzySet([0.6], [0.27])
    B = IntuitionisticFuzzySet([0.28], [0.55])
    assert_almost_equal(chen_cheng_lan(A, B), 0.6953, decimal=4)
    
    # The examples below fails, most likely due to the rounding process of the authors
    # Example 7.1
    P1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    P2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    P3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    Q = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])

    assert_almost_equal(chen_cheng_lan(P1,Q), 0.7100, decimal=4)
    assert_almost_equal(chen_cheng_lan(P2,Q), 0.7133, decimal=4)
    assert_almost_equal(chen_cheng_lan(P3,Q), 0.8117, decimal=4)

    # Example 7.2
    P1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.1, 0.1, 0.9])
    P2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.3, 0.8])
    P3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.2, 0.8, 0.4])
    Q = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.4, 0.2, 0.8])

    assert_almost_equal(chen_cheng_lan(P1,Q), 0.8544, decimal=4)
    assert_almost_equal(chen_cheng_lan(P2,Q), 0.9356, decimal=4)
    assert_almost_equal(chen_cheng_lan(P3,Q), 0.5333, decimal=4)

    # Example 7.3
    P1 = IntuitionisticFuzzySet([0.5, 0.7, 0.4, 0.7], [0.3, 0.0, 0.5, 0.3])
    P2 = IntuitionisticFuzzySet([0.5, 0.6, 0.2, 0.7], [0.2, 0.1, 0.7, 0.3])
    P3 = IntuitionisticFuzzySet([0.5, 0.7, 0.4, 0.7], [0.4, 0.1, 0.6, 0.2])
    Q = IntuitionisticFuzzySet([0.4, 0.7, 0.3, 0.7], [0.3, 0.1, 0.6, 0.3])

    assert_almost_equal(chen_cheng_lan(P1,Q), 0.9413, decimal=4)
    assert_almost_equal(chen_cheng_lan(P2,Q), 0.9150, decimal=4)
    assert_almost_equal(chen_cheng_lan(P3,Q), 0.9504, decimal=4)


def test_muthukumar_krishnanb():
    # all tests fail 
    F = IntuitionisticFuzzySet([0.3, 0.5, 0.6, 0.5, 0.7, 0.9, 0.7, 0.8, 0.6, 0.7, 0.7, 0.3], [0.0, 0.1, 0.3, 0.0, 0.1, 0.0, 0.1, 0.2, 0.2, 0.0, 0.2, 0.0])
    G = IntuitionisticFuzzySet([0.8, 0.7, 0.5, 0.4, 0.9, 0.9, 0.8, 0.7, 0.5, 0.9, 0.6, 0.8], [0.1, 0.2, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.3, 0.1, 0.1, 0.1])
    assert_almost_equal(muthukumar_krishnanb(F, G), 0.81448, decimal=5)

    F = IntuitionisticFuzzySet([0.6, 0.4, 0.8, 0.5, 0.7, 0.6, 0.8, 0.6, 0.9], [0.2, 0.5, 0.1, 0.3, 0.1, 0.3, 0.2, 0.0, 0.0])
    G = IntuitionisticFuzzySet([0.5, 0.7, 0.6, 0.6, 0.4, 0.5, 0.9, 0.5, 0.8], [0.3, 0.0, 0.3, 0.2, 0.0, 0.1, 0.0, 0.1, 0.0])
    H = IntuitionisticFuzzySet([0.4, 0.6, 0.5, 0.3, 0.7, 0.5, 0.2, 0.5, 0.1], [0.4, 0.2, 0.1, 0.2, 0.1, 0.4, 0.0, 0.0, 0.8])
    assert_almost_equal(muthukumar_krishnanb(F, G), 0.8029, decimal=4)
    assert_almost_equal(muthukumar_krishnanb(G, H), 0.4907, decimal=4)
    assert_almost_equal(muthukumar_krishnanb(F, H), 0.4843, decimal=4)

    M = IntuitionisticFuzzySet([0.6, 0.4, 0.8, 0.5, 0.7, 0.6, 0.8, 0.6, 0.9], [0.2, 0.5, 0.1, 0.3, 0.1, 0.3, 0.2, 0.0, 0.0])
    P1 = IntuitionisticFuzzySet([0.5, 0.7, 0.6, 0.6, 0.4, 0.5, 0.9, 0.5, 0.8], [0.3, 0.0, 0.3, 0.2, 0.0, 0.1, 0.0, 0.1, 0.0])
    P2 = IntuitionisticFuzzySet([0.2, 0.6, 0.5, 0.3, 0.7, 0.4, 0.2, 0.5, 0.1], [0.4, 0.2, 0.1, 0.2, 0.1, 0.4, 0.0, 0.0, 0.8])
    P3 = IntuitionisticFuzzySet([0.5, 0.5, 0.3, 0.1, 0.3, 0.6, 0.3, 0.0, 0.2], [0.4, 0.0, 0.6, 0.8, 0.0, 0.2, 0.5, 0.2, 0.4])
    P4 = IntuitionisticFuzzySet([0.3, 0.6, 0.2, 0.4, 0.2, 0.5, 0.3, 0.4, 0.2], [0.5, 0.0, 0.6, 0.5, 0.4, 0.0, 0.1, 0.0, 0.6])
    P5 = IntuitionisticFuzzySet([0.5, 0.4, 0.6, 0.0, 0.3, 0.4, 0.1, 0.2, 0.4], [0.0, 0.0, 0.2, 0.2, 0.0, 0.0 ,0.5, 0.0, 0.4])
    P6 = IntuitionisticFuzzySet([0.4, 0.6, 0.5, 0.3, 0.7, 0.5, 0.2, 0.5, 0.1], [0.4, 0.2, 0.1, 0.2, 0.1, 0.4, 0.0, 0.0, 0.8])
    P7 = IntuitionisticFuzzySet([0.3, 0.7, 0.6, 0.5, 0.9, 0.7, 0.6, 0.7, 0.7], [0.0, 0.1, 0.2, 0.1, 0.0, 0.0, 0.3, 0.1, 0.2])
    P8 = IntuitionisticFuzzySet([0.8, 0.9, 0.5, 0.7, 0.9, 0.9, 0.5, 0.8, 0.6], [0.1, 0.0, 0.3, 0.2, 0.0, 0.1, 0.2, 0.0, 0.1])
    P9 = IntuitionisticFuzzySet([0.5, 0.8, 0.3, 0.4, 0.7, 0.8, 0.0, 0.4, 0.0], [0.0, 0.2, 0.0, 0.1, 0.0, 0.1, 0.8, 0.3, 0.7])
    P10 = IntuitionisticFuzzySet([0.7, 0.4, 0.6, 0.5, 0.7, 0.6, 0.8, 0.6, 0.9], [0.2, 0.5, 0.1, 0.3, 0.1, 0.0, 0.2, 0.0, 0.0])
    P11 = IntuitionisticFuzzySet([0.4, 0.7, 0.6, 0.6, 0.4, 0.5, 0.7, 0.5, 0.8], [0.3, 0.0, 0.3, 0.2, 0.0, 0.1, 0.2, 0.1, 0.0])
    P12 = IntuitionisticFuzzySet([0.6, 0.5, 0.5, 0.3, 0.5, 0.4, 0.2, 0.5, 0.1], [0.4, 0.0, 0.1, 0.2, 0.1, 0.4, 0.0, 0.0, 0.8])
    P13 = IntuitionisticFuzzySet([0.5, 0.6, 0.4, 0.5, 0.3, 0.2, 0.5, 0.4, 0.2], [0.3, 0.0, 0.3, 0.4, 0.2, 0.1, 0.0, 0.0, 0.5])
    P14 = IntuitionisticFuzzySet([0.0, 0.4, 0.5, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3], [0.5, 0.3, 0.2, 0.1, 0.2, 0.1, 0.1, 0.3, 0.5])
    P15 = IntuitionisticFuzzySet([0.4, 0.2, 0.0, 0.0, 0.5, 0.4, 0.5, 0.2, 0.4], [0.0, 0.3, 0.2, 0.3, 0.2, 0.3, 0.3, 0.3, 0.4])
    
    assert_almost_equal(muthukumar_krishnanb(P1, M), 0.8092, decimal=4)
    assert_almost_equal(muthukumar_krishnanb(P2, M), 0.4733, decimal=4)
    assert_almost_equal(muthukumar_krishnanb(P3, M), 0.3906, decimal=4)
    assert_almost_equal(muthukumar_krishnanb(P4, M), 0.4047, decimal=4)
    assert_almost_equal(muthukumar_krishnanb(P5, M), 0.4232, decimal=4)
    assert_almost_equal(muthukumar_krishnanb(P6, M), 0.5064, decimal=4)
    assert_almost_equal(muthukumar_krishnanb(P7, M), 0.7305, decimal=4)
    assert_almost_equal(muthukumar_krishnanb(P8, M), 0.7279, decimal=4)
    assert_almost_equal(muthukumar_krishnanb(P9, M), 0.4497, decimal=4)
    assert_almost_equal(muthukumar_krishnanb(P10, M), 0.9323, decimal=4)
    assert_almost_equal(muthukumar_krishnanb(P11, M), 0.8000, decimal=4)
    assert_almost_equal(muthukumar_krishnanb(P12, M), 0.4738, decimal=4)
    assert_almost_equal(muthukumar_krishnanb(P13, M), 0.5112, decimal=4)
    assert_almost_equal(muthukumar_krishnanb(P14, M), 0.4755, decimal=4)
    assert_almost_equal(muthukumar_krishnanb(P15, M), 0.4625, decimal=4)

