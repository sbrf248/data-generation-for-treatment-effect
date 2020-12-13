import numpy as np
import math
from typing import Tuple, Callable

from effect_function import MeanEffectFunction, TreatmentEffectFunction

"""
An implementation of a data generation process for treatment effect estimation
from [Powers, 2018]

[Powers, 2018] Scott Powers, Junyang Qian, Kenneth Jung, Alejandro Schuler, Nigam H. Shah,
Trevor Hastie, Robert Tibshirani. (2018). "Some methods for heterogeneous treatment effect 
estimation in high-dimensions." Statics in Medicine, 37(11): 1767--1787.
"""

def _calc_odd_even_n_feature(n_feature: int) -> Tuple[int, int]:

    return math.ceil(n_feature / 2), math.floor(n_feature / 2)


def generate_features(n_data: int, n_feature: int) -> np.ndarray:

    result = np.empty((n_data, n_feature), dtype=np.float)

    n_odd, n_even = _calc_odd_even_n_feature(n_feature)

    gaussian_features = np.random.normal(0, 1, (n_data, n_odd))
    result[:, 0::2] = gaussian_features

    binom_features = np.random.binomial(1, 0.5, (n_data, n_even))
    result[:, 1::2] = binom_features
    
    return result


def _propensity_function(scenario: int) -> Callable[[np.ndarray], np.ndarray]:

    if scenario < 8:
        return lambda x: np.ones(x.shape[0]) * 0.5
    else: 
        mu = MeanEffectFunction(scenario)
        tau = TreatmentEffectFunction(scenario)
        return lambda x: np.exp((mu(x) - tau(x)) / 2) / (1 + np.exp((mu(x) - tau(x)) / 2))


def generate_treatment(scenario: int, x: np.ndarray) -> np.ndarray:

    prob = _propensity_function(scenario)(x)

    return np.random.binomial(1, prob)


def _assign_variance(scenario: int) -> float:

    scenario = scenario % 8
    variances = [1.0, 1/4, 1.0, 1/4, 1.0, 1.0, 4.0, 4.0]

    return variances[scenario]


def generate_outcome(scenario: int, x: np.ndarray, t: np.ndarray) -> np.ndarray:

    mu = MeanEffectFunction(scenario)
    tau = TreatmentEffectFunction(scenario)
    variance = _assign_variance(scenario)
    outcome = np.random.normal(mu(x) + (t - 0.5) * tau(x), variance, x.shape[0])

    return outcome


def compute_conditional_mean_effect(scenario: int, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    mu = MeanEffectFunction(scenario)
    tau = TreatmentEffectFunction(scenario)

    treatment = mu(x) + tau(x) / 2
    control = mu(x) - tau(x) / 2

    return treatment, control


def generate_data(
    scenario: int,
    n_data: int, 
    n_feature: int, 
    p_prob: float = 0.5
):

    assert scenario < 16, 'Scenario should be between 0 and 15.'
    
    assert n_feature >= 10, \
        'The number of features should be at least 10 so that all effect functions can be used.'

    # Generate n_data patients with n_feature dimensions.
    x = generate_features(n_data, n_feature)
    # Assign each patient to treatment (t=1) or control (t=0) group.
    t = generate_treatment(scenario, x)
    # Generate outcome y^{(1)} or y^{(0)} for each patient, which includes observational noises.
    y = generate_outcome(scenario, x, t)
    # Compute true outcomes of treated or controlled cases.
    response_given_treatment, response_given_control = compute_conditional_mean_effect(scenario, x)
    
    return x, t, y, response_given_treatment, response_given_control

