from numpy.testing import assert_almost_equal

from fsmpy.sets import IntuitionisticFuzzySet
from fsmpy.datasets import load_patients_diagnoses
from fsmpy.similarities import deng_jiang_fu
from fsmpy import DENG_JIANG_FU_MONOTONIC_TYPE_1_1, DENG_JIANG_FU_MONOTONIC_TYPE_1_2, \
    DENG_JIANG_FU_MONOTONIC_TYPE_1_3, DENG_JIANG_FU_MONOTONIC_TYPE_1_4, DENG_JIANG_FU_MONOTONIC_TYPE_2_1, \
    DENG_JIANG_FU_MONOTONIC_TYPE_2_2, DENG_JIANG_FU_MONOTONIC_TYPE_2_3, DENG_JIANG_FU_MONOTONIC_TYPE_2_4, \
    DENG_JIANG_FU_MONOTONIC_TYPE_3_1, DENG_JIANG_FU_MONOTONIC_TYPE_3_2, DENG_JIANG_FU_MONOTONIC_TYPE_3_3


def test_deng_jiang_fu_1_1():
    A1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    A2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    A3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    B = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])

    # Example 2
    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_1), 0.489, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_1), 0.458, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_1), 0.546, decimal=3)

    # Example 3
    diagnoses, patients = load_patients_diagnoses()
    viral_fever, malaria, typhoid, stomach_problem, chest_problem = diagnoses
    al, bob, joe, ted = patients

    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.467,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.517,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.544,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.216,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.26,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.348,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.3, decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.415,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.641,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.371,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.363,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.344,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.498,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.32,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.277,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.198,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.264,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.318,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.421,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_1, ), 0.407,
                        decimal=3)
    

def test_deng_jiang_fu_1_2():
    A1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    A2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    A3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    B = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_2), 0.454, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_2), 0.444, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_2), 0.541, decimal=3)

    # Example 3
    diagnoses, patients = load_patients_diagnoses()
    viral_fever, malaria, typhoid, stomach_problem, chest_problem = diagnoses
    al, bob, joe, ted = patients

    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.437,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.489,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.474,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.186,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.184,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.28,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.21,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.366,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.635,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.309,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.348,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.308,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.241,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.214,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.47,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.189,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.243,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.31,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.401,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_2, ), 0.403,
                        decimal=3)


def test_deng_jiang_fu_1_3():
    A1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    A2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    A3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    B = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.625, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.615, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.702, decimal=3)

    # Example 3
    diagnoses, patients = load_patients_diagnoses()
    viral_fever, malaria, typhoid, stomach_problem, chest_problem = diagnoses
    al, bob, joe, ted = patients

    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.608,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.657,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.643,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1),
                        0.313, decimal=3)
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.311,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.437,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.348,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.536,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1),
                        0.777, decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.516,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.472,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.471,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.639,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1),
                        0.388, decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.353,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.574,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.572,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.474,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1),
                        0.391, decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_1_3, p=1), 0.319,
                        decimal=3)
    
    
def test_deng_jiang_fu_2_1():
    A1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    A2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    A3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    B = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_1), 0.681, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_1), 0.668, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_1), 0.745, decimal=3)

    diagnoses, patients = load_patients_diagnoses()
    viral_fever, malaria, typhoid, stomach_problem, chest_problem = diagnoses
    al, bob, joe, ted = patients

    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.698,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.709,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.698,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.393,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.375,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.518,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.419,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.594,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.826,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.509,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.618,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.533,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.712,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.512,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.449,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.672,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.624,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.541,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.481,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_1, ), 0.376,
                        decimal=3)


def test_deng_jiang_fu_2_2():
    A1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    A2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    A3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    B = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_2), 0.658, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_2), 0.658, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_2), 0.743, decimal=3)

    diagnoses, patients = load_patients_diagnoses()
    viral_fever, malaria, typhoid, stomach_problem, chest_problem = diagnoses
    al, bob, joe, ted = patients
    
    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.683,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.69, decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.661,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.361,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.324,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.476,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.352,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.567,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.825,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.463,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.603,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.492,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.7, decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.452,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.387,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.672,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.61,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.532,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.464,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_2, ), 0.366,
                        decimal=3)


def test_deng_jiang_fu_2_3():
    A1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    A2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    A3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    B = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.783, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.783, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.850, decimal=3)

    diagnoses, patients = load_patients_diagnoses()
    viral_fever, malaria, typhoid, stomach_problem, chest_problem = diagnoses
    al, bob, joe, ted = patients

    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.81,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.82,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.8,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.54,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.5,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.67,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.54,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.74,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.9,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.64,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.75,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.68,
                        decimal=3)    
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.82,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.6,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.54,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.8,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.77,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.71,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1),
                        0.63, decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_3, p=1), 0.55,
                        decimal=3)


def test_deng_jiang_fu_2_4():
    A1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    A2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    A3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    B = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_4), 0.644, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_4), 0.644, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_2_4), 0.739, decimal=3)

    diagnoses, patients = load_patients_diagnoses()
    viral_fever, malaria, typhoid, stomach_problem, chest_problem = diagnoses
    al, bob, joe, ted = patients

    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.681,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.695,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.667,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.37,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.333,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.504,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.37,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.587,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.818,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.471,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.6,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.515,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.695,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.429,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.37,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.667,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.626,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.55,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.46,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_2_4, ), 0.379,
                        decimal=3)
    
    
def test_deng_jiang_fu_3_1():
    A1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    A2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    A3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    B = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.593, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.593, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.700, decimal=3) # fails

    diagnoses, patients = load_patients_diagnoses()
    viral_fever, malaria, typhoid, stomach_problem, chest_problem = diagnoses
    al, bob, joe, ted = patients

    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.634, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.65, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.619, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.304, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.269, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.441, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.304, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.531, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.79, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.406, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.545, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.453, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.65, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.363, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.304, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.619, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.574, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.491, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.395, decimal=3) # fails
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_1, p=1), 0.314, decimal=3) # fails


def test_deng_jiang_fu_3_2():
    A1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    A2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    A3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    B = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.928, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.941, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.975, decimal=3)

    diagnoses, patients = load_patients_diagnoses()
    viral_fever, malaria, typhoid, stomach_problem, chest_problem = diagnoses
    al, bob, joe, ted = patients

    assert_almost_equal(
        deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.947,
        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5),
                        0.946, decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5),
                        0.92, decimal=3)
    assert_almost_equal(
        deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.736,
        decimal=3)
    assert_almost_equal(
        deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.678,
        decimal=3)
    assert_almost_equal(
        deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.831,
        decimal=3)
    assert_almost_equal(
        deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.694,
        decimal=3)
    assert_almost_equal(
        deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.898,
        decimal=3)
    assert_almost_equal(
        deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.986,
        decimal=3)
    assert_almost_equal(
        deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.802,
        decimal=3)
    assert_almost_equal(
        deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.915,
        decimal=3)
    assert_almost_equal(
        deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.844,
        decimal=3)
    assert_almost_equal(
        deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.944,
        decimal=3)
    assert_almost_equal(
        deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.762,
        decimal=3)
    assert_almost_equal(
        deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.7,
        decimal=3)
    assert_almost_equal(
        deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.954,
        decimal=3)
    assert_almost_equal(
        deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.927,
        decimal=3)
    assert_almost_equal(
        deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.897,
        decimal=3)
    assert_almost_equal(
        deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.829,
        decimal=3)
    assert_almost_equal(
        deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_2, p=2, u=0.5, v=0.5), 0.773,
        decimal=3)


def test_deng_jiang_fu_3_3():
    A1 = IntuitionisticFuzzySet([1.0, 0.8, 0.7], [0.0, 0.0, 0.1])
    A2 = IntuitionisticFuzzySet([0.8, 1.0, 0.9], [0.1, 0.0, 0.0])
    A3 = IntuitionisticFuzzySet([0.6, 0.8, 1.0], [0.2, 0.0, 0.0])
    B = IntuitionisticFuzzySet([0.5, 0.6, 0.8], [0.3, 0.2, 0.1])

    assert_almost_equal(deng_jiang_fu(A1, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.667, decimal=3)
    assert_almost_equal(deng_jiang_fu(A2, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.667, decimal=3)
    assert_almost_equal(deng_jiang_fu(A3, B, DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.766, decimal=3)

    diagnoses, patients = load_patients_diagnoses()
    viral_fever, malaria, typhoid, stomach_problem, chest_problem = diagnoses
    al, bob, joe, ted = patients
    assert_almost_equal(deng_jiang_fu(al, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.706,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.721,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.691,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(al, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1),
                        0.339, decimal=3)
    assert_almost_equal(deng_jiang_fu(al, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.293,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.508,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.34,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.605,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1),
                        0.844, decimal=3)
    assert_almost_equal(deng_jiang_fu(bob, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.464,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.617,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.52,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.721,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1),
                        0.415, decimal=3)
    assert_almost_equal(deng_jiang_fu(joe, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.34,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, viral_fever, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.691,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, malaria, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.648,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, typhoid, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.561,
                        decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, stomach_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1),
                        0.451, decimal=3)
    assert_almost_equal(deng_jiang_fu(ted, chest_problem, similarity_type=DENG_JIANG_FU_MONOTONIC_TYPE_3_3, p=1), 0.351,
                        decimal=3)
