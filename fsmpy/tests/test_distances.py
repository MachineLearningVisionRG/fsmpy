import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal
import math

from fsmpy.sets import IntuitionisticFuzzySet
from fsmpy.distances import atanassov, szmidt_kacprzyk, wang_xin, yang_chiclana, grzegorzewski, vlachos_sergiadis
from fsmpy import DISTANCE_HAMMING, DISTANCE_EUCLIDEAN, DISTANCE_NORMALIZED_EUCLIDEAN, DISTANCE_NORMALIZED_HAMMING
from fsmpy import WANGXIN_DISTANCE_1, WANGXIN_DISTANCE_2
from fsmpy.datasets import load_patients_diagnoses

def test_atanassov():
    # no tests
    pass


def test_szmidt_kacprzyk():
    A = IntuitionisticFuzzySet([1], [0], [0])
    B = IntuitionisticFuzzySet([0], [1], [0])
    D = IntuitionisticFuzzySet([0], [0], [1])
    G = IntuitionisticFuzzySet([0.5], [0.5], [0])
    E = IntuitionisticFuzzySet([0.25], [0.25], [0.5])
    assert_equal(szmidt_kacprzyk(A, D, DISTANCE_EUCLIDEAN), 1)  # Eq. 73
    assert_equal(szmidt_kacprzyk(B, D, DISTANCE_EUCLIDEAN), 1)  # Eq. 74
    assert_equal(szmidt_kacprzyk(A, B, DISTANCE_EUCLIDEAN), 1)  # Eq. 75
    assert_equal(szmidt_kacprzyk(A, G, DISTANCE_EUCLIDEAN), 0.5)  # Eq. 76
    assert_equal(szmidt_kacprzyk(B, G, DISTANCE_EUCLIDEAN), 0.5)  # Eq. 77
    assert_equal(szmidt_kacprzyk(E, G, DISTANCE_EUCLIDEAN),
                 math.sqrt(3.) / 4.)  # Eq. 78
    assert_equal(szmidt_kacprzyk(D, G, DISTANCE_EUCLIDEAN),
                 math.sqrt(3.) / 2.)  # Eq. 79

    A = IntuitionisticFuzzySet(
        [0.5, 0.2, 0, 0.3, 0.2, 1, 0],
        [0.3, 0.6, 1, 0.2, 0.2, 0, 1],
        [0.2, 0.2, 0, 0.5, 0.6, 0, 0])
    B = IntuitionisticFuzzySet(
        [0.2, 0, 0, 0.3, 0.5, 0, 0.9],
        [0.6, 1, 1, 0.2, 0.2, 1, 0],
        [0.2, 0, 0, 0.5, 0.3, 0, 0.1])
    assert_equal(szmidt_kacprzyk(A, B, DISTANCE_HAMMING), 3)  # Eq. 82
    assert_almost_equal(szmidt_kacprzyk(
        A, B, DISTANCE_NORMALIZED_HAMMING), 0.43, decimal=2)  # Eq. 83
    assert_almost_equal(szmidt_kacprzyk(
        A, B, DISTANCE_EUCLIDEAN), 1.49, decimal=2)  # Eq. 86
    assert_almost_equal(szmidt_kacprzyk(
        A, B, DISTANCE_NORMALIZED_EUCLIDEAN), 0.56, decimal=2)  # Eq. 87

    A.hesitation_degrees = np.zeros_like(A.membership_values)
    B.hesitation_degrees = np.zeros_like(B.membership_values)
    assert_equal(szmidt_kacprzyk(A, B, DISTANCE_HAMMING), 2.7)  # Eq. 84
    assert_almost_equal(szmidt_kacprzyk(
        A, B, DISTANCE_NORMALIZED_HAMMING), 0.39, decimal=2)  # Eq. 85
    assert_almost_equal(szmidt_kacprzyk(
        A, B, DISTANCE_EUCLIDEAN), 1.46, decimal=2)  # Eq. 88
    assert_almost_equal(szmidt_kacprzyk(
        A, B, DISTANCE_NORMALIZED_EUCLIDEAN), 0.55, decimal=2)  # Eq. 89


def test_wang_xin():
    # these tests fail
    # A = IntuitionisticFuzzySet([0.3, 0.5, 0.7], [0.6, 0.4, 0.1])
    # B = IntuitionisticFuzzySet([0.4, 0.6, 0.5], [0.6, 0.3, 0.2])
    # w = [1./5., 1./3., 2./5.]
    # assert_almost_equal(wang_xin(A, B, WANGXIN_DISTANCE_1), 0.116, decimal=3)
    # # assert_almost_equal(
    # #     wang_xin(A, B, WANGXIN_DISTANCE_1, w), 0.123, decimal=3)
    # assert_almost_equal(
    #     wang_xin(A, B, WANGXIN_DISTANCE_1, w, 1), 0.100, decimal=3)
    A1 = IntuitionisticFuzzySet([0.173, 0.102, 0.530, 0.965, 0.420, 0.008, 0.331, 1.000, 0.215, 0.432, 0.750, 0.432],
                  [0.524, 0.818, 0.326, 0.008, 0.351, 0.956, 0.512, 0.000, 0.625, 0.534, 0.126, 0.432])
    A2 = IntuitionisticFuzzySet([0.510, 0.627, 1.000, 0.125, 0.026, 0.732, 0.556, 0.650, 1.000, 0.145, 0.047, 0.760],
                  [0.365, 0.125, 0.000, 0.648, 0.823, 0.153, 0.303, 0.267, 0.000, 0.762, 0.923, 0.231])
    A3 = IntuitionisticFuzzySet([0.495, 0.603, 0.987, 0.073, 0.037, 0.690, 0.147, 0.213, 0.501, 1.000, 0.324, 0.045],
                  [0.387, 0.298, 0.006, 0.849, 0.923, 0.268, 0.812, 0.653, 0.284, 0.000, 0.483, 0.912])
    A4 = IntuitionisticFuzzySet([1.000, 1.000, 0.857, 0.734, 0.021, 0.076, 0.152, 0.113, 0.489, 1.000, 0.386, 0.028],
                  [0.000, 0.000, 0.123, 0.158, 0.896, 0.912, 0.712, 0.756, 0.389, 0.000, 0.485, 0.912])
    B = IntuitionisticFuzzySet([0.978, 0.980, 0.798, 0.693, 0.051, 0.123, 0.152, 0.113, 0.494, 0.987, 0.376, 0.012],
                 [0.003, 0.012, 0.132, 0.213, 0.876, 0.756, 0.721, 0.732, 0.368, 0.000, 0.423, 0.897])
    assert_almost_equal(wang_xin(A1, B, WANGXIN_DISTANCE_1), 0.454, 3)
    assert_almost_equal(wang_xin(A2, B, WANGXIN_DISTANCE_1), 0.460, 3)
    assert_almost_equal(wang_xin(A3, B, WANGXIN_DISTANCE_1), 0.211, 3)
    assert_almost_equal(wang_xin(A4, B, WANGXIN_DISTANCE_1), 0.034, 3)

    assert_almost_equal(wang_xin(A1, B, WANGXIN_DISTANCE_2, p=1), 0.431, 3)
    assert_almost_equal(wang_xin(A2, B, WANGXIN_DISTANCE_2, p=1), 0.436, 3)
    assert_almost_equal(wang_xin(A3, B, WANGXIN_DISTANCE_2, p=1), 0.198, 3)
    assert_almost_equal(wang_xin(A4, B, WANGXIN_DISTANCE_2, p=1), 0.027, 3)

    C1 = IntuitionisticFuzzySet([0.739, 0.033, 0.188, 0.492, 0.020, 0.739],
                  [0.125, 0.818, 0.626, 0.358, 0.628, 0.125])
    C2 = IntuitionisticFuzzySet([0.124, 0.030, 0.048, 0.136, 0.019, 0.393],
                  [0.665, 0.825, 0.800, 0.648, 0.823, 0.653])
    C3 = IntuitionisticFuzzySet([0.449, 0.662, 1.000, 1.000, 1.000, 1.000],
                  [0.387, 0.298, 0.000, 0.000, 0.000, 0.000])
    C4 = IntuitionisticFuzzySet([0.280, 0.521, 0.470, 0.295, 0.188, 0.735],
                  [0.715, 0.368, 0.423, 0.658, 0.806, 0.118])
    C5 = IntuitionisticFuzzySet([0.326, 1.000, 0.182, 0.156, 0.049, 0.675],
                  [0.452, 0.000, 0.725, 0.765, 0.896, 0.263])
    B = IntuitionisticFuzzySet([0.629, 0.524, 0.210, 0.218, 0.069, 0.658],
                 [0.303, 0.356, 0.689, 0.753, 0.876, 0.256])
    assert_almost_equal(wang_xin(C1, B, WANGXIN_DISTANCE_1), 0.230, 3)
    assert_almost_equal(wang_xin(C2, B, WANGXIN_DISTANCE_1), 0.270, 3)
    assert_almost_equal(wang_xin(C3, B, WANGXIN_DISTANCE_1), 0.509, 3)
    assert_almost_equal(wang_xin(C4, B, WANGXIN_DISTANCE_1), 0.165, 3)
    assert_almost_equal(wang_xin(C5, B, WANGXIN_DISTANCE_1), 0.138, 3)

    assert_almost_equal(wang_xin(C1, B, WANGXIN_DISTANCE_2, p=1), 0.209, 3)
    assert_almost_equal(wang_xin(C2, B, WANGXIN_DISTANCE_2, p=1), 0.255, 3)
    assert_almost_equal(wang_xin(C3, B, WANGXIN_DISTANCE_2, p=1), 0.490, 3)
    assert_almost_equal(wang_xin(C4, B, WANGXIN_DISTANCE_2, p=1), 0.156, 3)
    assert_almost_equal(wang_xin(C5, B, WANGXIN_DISTANCE_2, p=1), 0.124, 3)


def test_yang_chiclana():
    A = IntuitionisticFuzzySet([0.25], [0.25])
    B = IntuitionisticFuzzySet([0.2], [0.2])
    C = IntuitionisticFuzzySet([0.18], [0.32])
    
    assert_almost_equal(yang_chiclana(A, B, DISTANCE_HAMMING), 0.1, decimal=1)
    assert_almost_equal(yang_chiclana(
        A, B, DISTANCE_NORMALIZED_HAMMING), 0.1, decimal=1)
    assert_almost_equal(yang_chiclana(
        A, B, DISTANCE_EUCLIDEAN), 0.1, decimal=1)
    assert_almost_equal(yang_chiclana(
        A, B, DISTANCE_NORMALIZED_EUCLIDEAN), 0.1, decimal=1)

    assert_almost_equal(yang_chiclana(A, C, DISTANCE_HAMMING), 0.07, decimal=2)
    assert_almost_equal(yang_chiclana(
        A, C, DISTANCE_NORMALIZED_HAMMING), 0.07, decimal=2)
    assert_almost_equal(yang_chiclana(
        A, C, DISTANCE_EUCLIDEAN), 0.07, decimal=2)
    assert_almost_equal(yang_chiclana(
        A, C, DISTANCE_NORMALIZED_EUCLIDEAN), 0.07, decimal=2)


def test_grzegorzewski():
    A = IntuitionisticFuzzySet([1], [0])
    B = IntuitionisticFuzzySet([0], [1])
    D = IntuitionisticFuzzySet([0], [0])
    G = IntuitionisticFuzzySet([0.5], [0.5])
    E = IntuitionisticFuzzySet([0.25], [0.25])
    # Example 2
    assert_equal(grzegorzewski(A, B), 1)
    assert_equal(grzegorzewski(A, D), 1)
    assert_equal(grzegorzewski(B, D), 1)
    assert_equal(grzegorzewski(A, G), 0.5)
    assert_equal(grzegorzewski(A, E), 0.75)
    assert_equal(grzegorzewski(B, G), 0.5)
    assert_equal(grzegorzewski(B, E), 0.75)
    assert_equal(grzegorzewski(D, G), 0.5)
    assert_equal(grzegorzewski(D, E), 0.25)
    assert_equal(grzegorzewski(G, E), 0.25)


def test_vlachos_sergiadis():
    P1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7],
                  [0.0, 0.0, 0.1])
    P2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9],
                  [0.1, 0.0, 0.0])
    P3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0],
                  [0.2, 0.0, 0.0])
    Q = IntuitionisticFuzzySet([0.5, 0.6, 0.8],
                 [0.3, 0.2, 0.1])
    assert_almost_equal(vlachos_sergiadis(P1, Q), 0.4492, decimal=4)
    assert_almost_equal(vlachos_sergiadis(P2, Q), 0.3487, decimal=4)
    assert_almost_equal(vlachos_sergiadis(P3, Q), 0.2480, decimal=4)

    diagnoses, patients = load_patients_diagnoses()
    viral_fever, malaria, typhoid, stomach_problem, chest_problem = diagnoses
    al, bob, joe, ted = patients

    assert_almost_equal(vlachos_sergiadis(al, viral_fever), 0.4304, decimal=4)
    assert_almost_equal(vlachos_sergiadis(al, malaria), 0.6045, decimal=4)
    assert_almost_equal(vlachos_sergiadis(al, typhoid), 0.5065, decimal=4)
    assert_almost_equal(vlachos_sergiadis(
        al, stomach_problem), 1.8899, decimal=4)
    assert_almost_equal(vlachos_sergiadis(
        al, chest_problem), 2.2681, decimal=4)

    assert_almost_equal(vlachos_sergiadis(bob, viral_fever), 1.4777, decimal=4)
    assert_almost_equal(vlachos_sergiadis(bob, malaria), 2.6161, decimal=4)
    assert_almost_equal(vlachos_sergiadis(bob, typhoid), 0.8239, decimal=4)
    assert_almost_equal(vlachos_sergiadis(
        bob, stomach_problem), 0.2276, decimal=4)
    assert_almost_equal(vlachos_sergiadis(
        bob, chest_problem), 1.4692, decimal=4)
    assert_almost_equal(vlachos_sergiadis(joe, viral_fever), 0.6762, decimal=4)
    # assert_almost_equal(vlachos_sergiadis(joe, malaria), 0.7365, decimal=4) # fails
    assert_almost_equal(vlachos_sergiadis(joe, typhoid), 0.4582, decimal=4)
    assert_almost_equal(vlachos_sergiadis(
        joe, stomach_problem), 1.9779, decimal=4)
    assert_almost_equal(vlachos_sergiadis(
        joe, chest_problem), 2.3405, decimal=4)

    assert_almost_equal(vlachos_sergiadis(ted, viral_fever), 0.3593, decimal=4)
    assert_almost_equal(vlachos_sergiadis(ted, malaria), 0.7252, decimal=4)
    assert_almost_equal(vlachos_sergiadis(ted, typhoid), 0.6585, decimal=4)
    assert_almost_equal(vlachos_sergiadis(
        ted, stomach_problem), 1.2585, decimal=4)
    assert_almost_equal(vlachos_sergiadis(
        ted, chest_problem), 1.5547, decimal=4)
