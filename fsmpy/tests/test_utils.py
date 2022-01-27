from numpy.testing import assert_almost_equal, assert_equal


from fsmpy.sets import FuzzySet, IntuitionisticFuzzySet
from fsmpy.utils import calculate_documents_membership
from fsmpy.utils.classifiers import classify
from fsmpy.distances import wang_xin

def test_calculate_documents_membership():
    X = [
        [15, 2, 7],
        [10, 3, 5],
        [2, 9, 6],
        [3, 11, 5],
        [3, 9, 4]
    ]
    sets, mean_feature_words, std_feature_words = calculate_documents_membership(
        X, 0.9, 0.9)
    
    assert_almost_equal(
        mean_feature_words,
        [6.6, 6.8, 5.4],
        decimal=2
    )
    assert_almost_equal(
        std_feature_words,
        [5.68, 4.02, 1.14],
        decimal=2
    )

    assert_almost_equal(
        sets[0].membership_values,
        [0.73, 0.21, 0.72],
        decimal=2
    )
    assert_almost_equal(
        sets[1].membership_values,
        [0.58, 0.25, 0.37],
        decimal=2
    )
    assert_almost_equal(
        sets[2].membership_values,
        [0.28, 0.57, 0.57],
        decimal=2
    )
    assert_almost_equal(
        sets[3].membership_values,
        [0.31, 0.67, 0.37],
        decimal=2
    )
    assert_almost_equal(
        sets[4].membership_values,
        [0.31, 0.57, 0.20],
        decimal=2
    )

    assert_almost_equal(
        sets[0].non_membership_values,
        [0.17, 0.69, 0.18],
        decimal=2
    )
    assert_almost_equal(
        sets[1].non_membership_values,
        [0.32, 0.65, 0.53],
        decimal=2
    )
    assert_almost_equal(
        sets[2].non_membership_values,
        [0.62, 0.33, 0.33],
        decimal=2
    )
    assert_almost_equal(
        sets[3].non_membership_values,
        [0.59, 0.23, 0.53],
        decimal=2
    )
    assert_almost_equal(
        sets[4].non_membership_values,
        [0.59, 0.33, 0.70],
        decimal=2
    )

def test_confidence_degree():
    # the distance proposed in 
    # A Novel Distance Measure of Intuitionistic Fuzzy Sets and Its Application to Pattern Recognition Problems
    # is required for the tests
    P1 = FuzzySet([0.173, 0.102, 0.530, 0.965, 0.420, 0.008, 0.331, 1.000, 0.215, 0.432, 0.750, 0.432],
    [0.173, 0.102, 0.530, 0.965, 0.420, 0.008, 0.331, 1.000, 0.215, 0.432, 0.750, 0.432])
    P2 = FuzzySet([0.510, 0.627, 1.000, 0.125, 0.026, 0.732, 0.556, 0.650, 1.000, 0.145, 0.047, 0.760],
    [0.365, 0.125, 0.000, 0.648, 0.823, 0.153, 0.303, 0.267, 0.000, 0.762, 0.923, 0.231])
    P3 = FuzzySet([0.495, 0.603, 0.987, 0.073, 0.037, 0.690, 0.147, 0.213, 0.501, 1.000, 0.324, 0.045],
    [0.387, 0.298, 0.006, 0.849, 0.923, 0.268, 0.812, 0.653, 0.284, 0.000, 0.483, 0.912])
    P4 = FuzzySet([1.000, 1.000, 0.857, 0.734, 0.021, 0.076, 0.152, 0.113, 0.489, 1.000, 0.386, 0.028],
    [0.000, 0.000, 0.123, 0.158, 0.896, 0.912, 0.712, 0.756, 0.389, 0.000, 0.485, 0.912])
    S = FuzzySet([0.978, 0.980, 0.798, 0.693, 0.051, 0.123, 0.152, 0.113, 0.494, 0.987, 0.376, 0.012],
    [0.003, 0.012, 0.132, 0.213, 0.876, 0.756, 0.721, 0.732, 0.368, 0.000, 0.423, 0.897])


    