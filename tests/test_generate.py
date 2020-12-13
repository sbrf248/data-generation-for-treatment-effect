import pytest
import numpy as np
from numpy.testing import assert_equal

from generator import _calc_odd_even_n_feature
from generator import generate_features
from generator import _propensity_function
from generator import generate_treatment
from generator import _assign_variance
from generator import generate_outcome
from generator import compute_conditional_mean_effect


odd_even_n_feature_test_cases = [
    (1, 1, 0), # n_feature=1: (1) => n_odd=1: (1), n_even=0: ()
    (5, 3, 2), # n_feature=5: (1, 2, 3, 4, 5) => n_odd=3: (1, 3, 5), n_even=2: (2, 4)
    (6, 3, 3)  # n_feature=6: (1, 2, 3, 4, 5, 6) => n_odd=3: (1, 3, 5), n_even=3: (2, 4, 6)
]

@pytest.mark.parametrize(
    'n_feature, odd_expected, even_expected', 
    odd_even_n_feature_test_cases
)
def test_calc_odd_even_n_feature(
    n_feature: int, 
    odd_expected: int, 
    even_expected: int
):

    odd_actual, even_actual = _calc_odd_even_n_feature(n_feature)
    assert odd_actual == odd_expected
    assert even_actual == even_actual


def test_generate_features():

    x = generate_features(100, 50)
    assert x.shape == (100, 50)


test_feature = np.array([
    [-0.5, 0.0, 0.5, 1.0, -0.5, 0.0, 0.5, 1.0, -0.5],
    [0.5, 1.0, -0.5, 0.0, 0.5, 1.0, -0.5, 0.0, 0.5]
])

propensity_function_test_cases = [
    (0, test_feature, np.array([0.5, 0.5])),
    (1, test_feature, np.array([0.5, 0.5])),
    (8, test_feature, np.array([
        np.exp(2.25 / np.sqrt(2)) / (1 + np.exp(2.25 / np.sqrt(2))), 
        np.exp(0.75 / np.sqrt(2)) / (1 + np.exp(0.75 / np.sqrt(2)))
    ]))
]

@pytest.mark.parametrize('scenario, x, expected', propensity_function_test_cases)
def test_propensity_function(scenario: int, x: np.ndarray, expected: np.ndarray):

    actual = _propensity_function(scenario)(x)
    assert_equal(actual, expected)


def test_generate_treatment(x: np.ndarray = test_feature):

    t = generate_treatment(0, x)
    assert t.shape == (2,)


assign_variance_test_cases = [
    (0, 1.0),
    (1, 1/4),
    (2, 1.0),
    (3, 1/4),
    (4, 1.0),
    (5, 1.0),
    (6, 4.0),
    (7, 4.0),
    (8, 1.0),
    (9, 1/4),
    (10, 1.0),
    (11, 1/4),
    (12, 1.0),
    (13, 1.0),
    (14, 4.0),
    (15, 4.0)
]

@pytest.mark.parametrize('scenario, expected', assign_variance_test_cases)
def test_assign_variance(scenario: int, expected: float):

    actual = _assign_variance(scenario)
    assert actual == expected


def test_generate_outcome(x: np.ndarray = test_feature):

    t = generate_treatment(0, x)
    y = generate_outcome(0, x, t)
    assert y.shape == (2,)


conditional_mean_effect_test_cases = [
    (
        0, test_feature, 
        np.array([4.5 / np.sqrt(2), 1.5 / np.sqrt(2)]), 
        np.array([4.5 / np.sqrt(2), 1.5 / np.sqrt(2)])
    ),
    (
        1, test_feature,
        np.array([-4.0, -4.0]),
        np.array([1.0, 1.0])
    )
]

@pytest.mark.parametrize('scenario, x, treatment_expected, control_expected', conditional_mean_effect_test_cases)
def test_compute_conditional_mean_effect(
    scenario: int, 
    x: np.ndarray, 
    treatment_expected: np.ndarray, 
    control_expected: np.ndarray
):

    treatment_actual, control_actual = compute_conditional_mean_effect(scenario, x)
    assert_equal(treatment_actual, treatment_expected)
    assert_equal(control_actual, control_expected)