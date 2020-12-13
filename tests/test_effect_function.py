import pytest
import numpy as np
from numpy.testing import assert_equal

from effect_function import EffectFunction
from effect_function import MeanEffectFunction
from effect_function import TreatmentEffectFunction


test_feature = np.array([
    [-0.5, 0.0, 0.5, 1.0, -0.5, 0.0, 0.5, 1.0, -0.5],
    [0.5, 1.0, -0.5, 0.0, 0.5, 1.0, -0.5, 0.0, 0.5]
])

expected_results = [
    np.array([0, 0]),
    np.array([-5, -5]),
    np.array([-5, -3]),
    np.array([6, 3]),
    np.array([-1.5, -1.5]),
    np.array([-1.0, 0.0]),
    np.array([-3.875, -3.875]),
    np.array([4.5 / np.sqrt(2), 1.5 / np.sqrt(2)])
]


@pytest.mark.parametrize('x, expected', [(test_feature, expected_results[0])])
def test_effect_func0(x: np.ndarray, expected: np.ndarray):

    actual = EffectFunction._func0(x)
    assert_equal(actual, expected)


@pytest.mark.parametrize('x, expected', [(test_feature, expected_results[1])])
def test_effect_func1(x: np.ndarray, expected: np.ndarray):

    actual = EffectFunction._func1(x)
    assert_equal(actual, expected)


@pytest.mark.parametrize('x, expected', [(test_feature, expected_results[2])])
def test_effect_func2(x: np.ndarray, expected: np.ndarray):

    actual = EffectFunction._func2(x)
    assert_equal(actual, expected)


@pytest.mark.parametrize('x, expected', [(test_feature, expected_results[3])])
def test_effect_func3(x: np.ndarray, expected: np.ndarray):

    actual = EffectFunction._func3(x)
    assert_equal(actual, expected)


@pytest.mark.parametrize('x, expected', [(test_feature, expected_results[4])])
def test_effect_func4(x: np.ndarray, expected: np.ndarray):

    actual = EffectFunction._func4(x)
    assert_equal(actual, expected)


@pytest.mark.parametrize('x, expected', [(test_feature, expected_results[5])])
def test_effect_func5(x: np.ndarray, expected: np.ndarray):

    actual = EffectFunction._func5(x)
    assert_equal(actual, expected)


@pytest.mark.parametrize('x, expected', [(test_feature, expected_results[6])])
def test_effect_func6(x: np.ndarray, expected: np.ndarray):

    actual = EffectFunction._func6(x)
    assert_equal(actual, expected)


@pytest.mark.parametrize('x, expected', [(test_feature, expected_results[7])])
def test_effect_func7(x: np.ndarray, expected: np.ndarray):

    actual = EffectFunction._func7(x)
    assert_equal(actual, expected)


mean_effect_function_assignment_test_cases = [
    (0, test_feature, expected_results[7]),
    (1, test_feature, expected_results[4]),
    (2, test_feature, expected_results[3]),
    (3, test_feature, expected_results[6]),
    (4, test_feature, expected_results[2]),
    (5, test_feature, expected_results[0]),
    (6, test_feature, expected_results[1]),
    (7, test_feature, expected_results[5]),
    (8, test_feature, expected_results[7]),
    (9, test_feature, expected_results[4]),
    (10, test_feature, expected_results[3]),
    (11, test_feature, expected_results[6]),
    (12, test_feature, expected_results[2]),
    (13, test_feature, expected_results[0]),
    (14, test_feature, expected_results[1]),
    (15, test_feature, expected_results[5])
]

@pytest.mark.parametrize('scenario, x, expected', mean_effect_function_assignment_test_cases)
def test_mean_effect_function(scenario: int, x: np.ndarray, expected: np.ndarray):

    mef = MeanEffectFunction(scenario)
    actual = mef(x)
    assert_equal(actual, expected)


treatment_effect_function_assignment_test_cases = [
    (0, test_feature, expected_results[0]),
    (1, test_feature, expected_results[1]),
    (2, test_feature, expected_results[2]),
    (3, test_feature, expected_results[3]),
    (4, test_feature, expected_results[4]),
    (5, test_feature, expected_results[5]),
    (6, test_feature, expected_results[6]),
    (7, test_feature, expected_results[7]),
    (8, test_feature, expected_results[0]),
    (9, test_feature, expected_results[1]),
    (10, test_feature, expected_results[2]),
    (11, test_feature, expected_results[3]),
    (12, test_feature, expected_results[4]),
    (13, test_feature, expected_results[5]),
    (14, test_feature, expected_results[6]),
    (15, test_feature, expected_results[7])
]

@pytest.mark.parametrize('scenario, x, expected', treatment_effect_function_assignment_test_cases)
def test_treatment_effect_function(scenario: int, x: np.ndarray, expected: np.ndarray):

    tef = TreatmentEffectFunction(scenario)
    actual = tef(x)
    assert_equal(actual, expected)