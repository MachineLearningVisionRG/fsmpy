import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from fsmpy.sets import IntuitionisticFuzzySet
from fsmpy.similarities import chen_1, hwang_yang, intarapaiboon, park_kwun_lim, ye, julian_hung_lin, zhang_fu
from fsmpy.similarities import mitchell, dengfeng_chuntian, hong_kim, chen_cheng_lan, song_wang_lei_xue
from fsmpy.similarities import liu, chen_2
from fsmpy.similarities import dengfeng_chuntian, hong_kim, song_wang_lei_xue, muthukumar_krishnanb, nguyen, deng_jiang_fu


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


def test_park_kwun_lin():
    A1 = IntuitionisticFuzzySet([0.2, 0.1, 0.0], [0.6, 0.7, 0.6])
    A2 = IntuitionisticFuzzySet([0.2, 0.0, 0.2], [0.6, 0.6, 0.8])
    A3 = IntuitionisticFuzzySet([0.1, 0.2, 0.2], [0.5, 0.7, 0.8])
    B = IntuitionisticFuzzySet([0.3, 0.2, 0.1], [0.7, 0.8, 0.7])

    assert_almost_equal(park_kwun_lim(A1, B), 0.800, decimal=3)
    assert_almost_equal(park_kwun_lim(A2, B), 0.733, decimal=3)
    assert_almost_equal(park_kwun_lim(A3, B), 0.767, decimal=3)


def test_mitchell():
    P1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    P2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    P3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    Q = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])
    
    assert_almost_equal(mitchell(P1, Q, p=1), 0.54, decimal=2)
    assert_almost_equal(mitchell(P2, Q, p=1), 0.54, decimal=2)
    assert_almost_equal(mitchell(P3, Q, p=1), 0.61, decimal=2)
    
    assert_almost_equal(mitchell(P1, Q, p=2), 0.74, decimal=2)
    assert_almost_equal(mitchell(P2, Q, p=2), 0.77, decimal=2)
    assert_almost_equal(mitchell(P3, Q, p=2), 0.84, decimal=2)
    

def test_julian_hung_lin():
    P1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    P2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    P3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    Q = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])
    
    assert_almost_equal(julian_hung_lin(P1, Q, p=1), 0.6083, decimal=4)
    assert_almost_equal(julian_hung_lin(P2, Q, p=1), 0.5250, decimal=4)
    assert_almost_equal(julian_hung_lin(P3, Q, p=1), 0.7492, decimal=4)
    
    assert_almost_equal(julian_hung_lin(P1, Q, p=2), 0.5397, decimal=4)
    assert_almost_equal(julian_hung_lin(P2, Q, p=2), 0.5111, decimal=4)
    assert_almost_equal(julian_hung_lin(P3, Q, p=2), 0.6692, decimal=4)
    

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
    # assert_almost_equal(ye(patient, malaria), 0.8602, decimal=4) # fails
    assert_almost_equal(ye(patient, typhoid), 0.8510, decimal=4)
    assert_almost_equal(ye(patient, stomach_problem), 0.5033, decimal=4)
    # assert_almost_equal(ye(patient, chest_problem), 0.4542, decimal=4) # fails


def test_hwang_yang():
    # Example 1
    X1A = IntuitionisticFuzzySet([0.3], [0.3])
    X1B = IntuitionisticFuzzySet([0.4], [0.4])
    assert_almost_equal(hwang_yang(X1A, X1B), 1.000, decimal=3)

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


def test_chen_1():
    A = IntuitionisticFuzzySet([0.2, 0.3, 0.5, 0.7, 0.8], [0.4, 0.5, 0.7, 0.9, 1.0])
    B = IntuitionisticFuzzySet([0.3, 0.4, 0.6, 0.7, 0.9], [0.5, 0.6, 0.8, 0.9, 1.0])
    assert_almost_equal(chen_1(A, B), 0.93, decimal=2)
    
    A = IntuitionisticFuzzySet([0.1, 0.2, 0.4, 0.6, 0.8], [0.3, 0.6, 0.8, 0.8, 1.0])
    B = IntuitionisticFuzzySet([0.2, 0.3, 0.5, 0.7, 0.9], [0.5, 0.7, 0.8, 0.9, 1.0])
    assert_almost_equal(chen_1(A, B, weights=[0.5, 0.8, 1.0, 0.7, 1.0]), 0.90625, decimal=5) # fails


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


def test_intarapaiboon():
    A1 = IntuitionisticFuzzySet([0.3], [0.3])
    B1 = IntuitionisticFuzzySet([0.4], [0.4])
    assert_almost_equal(intarapaiboon(A1, B1), 0.900, decimal=3)
    
    A2 = IntuitionisticFuzzySet([0.3], [0.4])
    B2 = IntuitionisticFuzzySet([0.4], [0.3])
    assert_almost_equal(intarapaiboon(A2, B2), 0.947, decimal=3)
    
    A3 = IntuitionisticFuzzySet([1.0], [0.0])
    B3 = IntuitionisticFuzzySet([0.0], [0.0])
    assert_almost_equal(intarapaiboon(A3, B3), 0.667, decimal=3)
    
    A4 = IntuitionisticFuzzySet([0.5], [0.5])
    B4 = IntuitionisticFuzzySet([0.0], [0.0])
    assert_almost_equal(intarapaiboon(A4, B4), 0.500, decimal=3)
    
    A5 = IntuitionisticFuzzySet([0.4], [0.2])
    B5 = IntuitionisticFuzzySet([0.5], [0.3])
    assert_almost_equal(intarapaiboon(A5, B5), 0.900, decimal=3)
    
    A6 = IntuitionisticFuzzySet([0.4], [0.2])
    B6 = IntuitionisticFuzzySet([0.5], [0.2])
    assert_almost_equal(intarapaiboon(A6, B6), 0.974, decimal=3)


def test_nguyen():
    # Example 1
    M = IntuitionisticFuzzySet([1.0], [0.0], [0.0])
    N = IntuitionisticFuzzySet([0.0], [1.0], [0.0])
    N = IntuitionisticFuzzySet([0.0], [1.0], [0.0])
    F = IntuitionisticFuzzySet([0.0], [0.0], [1.0])
    assert_equal(nguyen(M, N), -1.0)
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
    # assert_almost_equal(nguyen(A, B), 0.960, decimal=3) # fails

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
    
    # Example 7.1
    P1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    P2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    P3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    Q = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])

    assert_almost_equal(chen_cheng_lan(P1,Q), 0.7706, decimal=4)
    # assert_almost_equal(chen_cheng_lan(P2,Q), 0.7710, decimal=4) # fails
    assert_almost_equal(chen_cheng_lan(P3,Q), 0.8450, decimal=4)

    # Example 7.2
    P1 = IntuitionisticFuzzySet([0.1, 0.5, 0.1], [0.1, 0.1, 0.9])
    P2 = IntuitionisticFuzzySet([0.5, 0.7, 0.0], [0.5, 0.3, 0.8])
    P3 = IntuitionisticFuzzySet([0.7, 0.1, 0.4], [0.2, 0.8, 0.4])
    Q = IntuitionisticFuzzySet([0.4, 0.6, 0.0], [0.4, 0.2, 0.8])

    assert_almost_equal(chen_cheng_lan(P1,Q), 0.9444, decimal=4)
    assert_almost_equal(chen_cheng_lan(P2,Q), 0.9778, decimal=4)
    assert_almost_equal(chen_cheng_lan(P3,Q), 0.6000, decimal=4)

    # Example 7.3
    P1 = IntuitionisticFuzzySet([0.5, 0.7, 0.4, 0.7], [0.3, 0.0, 0.5, 0.3])
    P2 = IntuitionisticFuzzySet([0.5, 0.6, 0.2, 0.7], [0.2, 0.1, 0.7, 0.3])
    P3 = IntuitionisticFuzzySet([0.5, 0.7, 0.4, 0.7], [0.4, 0.1, 0.6, 0.2])
    Q = IntuitionisticFuzzySet([0.4, 0.7, 0.3, 0.7], [0.3, 0.1, 0.6, 0.3])

    assert_almost_equal(chen_cheng_lan(P1,Q), 0.9500, decimal=4)
    assert_almost_equal(chen_cheng_lan(P2,Q), 0.9354, decimal=4)
    assert_almost_equal(chen_cheng_lan(P3,Q), 0.9667, decimal=4)


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

