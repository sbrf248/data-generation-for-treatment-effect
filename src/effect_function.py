import numpy as np
from typing import Callable

class EffectFunction(object):
    def __init__(self, scenario: int):

        func_id = self._scenario_to_func_id(scenario)
        self.effect_func = self.set_effect_func(func_id)


    def __call__(self, x: np.ndarray) -> np.ndarray:

        return self.effect_func(x)


    def set_effect_func(self, func_id: int) -> Callable[[np.ndarray], np.ndarray]:

        candidates = [
            self._func0,
            self._func1,
            self._func2,
            self._func3,
            self._func4,
            self._func5,
            self._func6,
            self._func7
        ]

        return candidates[func_id]


    @staticmethod
    def _scenario_to_func_id(scneario: int) -> int:

        pass


    @staticmethod
    def _func0(x: np.ndarray) -> np.ndarray:

        return np.zeros(x.shape[0])


    @staticmethod
    def _func1(x: np.ndarray) -> np.ndarray:

        return 5 * (x[:, 0] > 1) - 5


    @staticmethod
    def _func2(x: np.ndarray) -> np.ndarray:

        return 2 * x[:, 0] - 4


    @staticmethod
    def _func3(x: np.ndarray) -> np.ndarray:

        return \
            x[:, 1] * x[:, 3] * x[:, 5] \
            + 2 * x[:, 1] * x[:, 3] * (1 - x[:, 5]) \
            + 3 * x[:, 1] * (1 - x[:, 3]) * x[:, 5] \
            + 4 * x[:, 1] * (1 - x[:, 3]) * (1 - x[:, 5]) \
            + 5 * (1 - x[:, 1]) * x[:, 3] * x[:, 5] \
            + 6 * (1 - x[:, 1]) * x[:, 3] * (1 - x[:, 5]) \
            + 7 * (1 - x[:, 1]) * (1 - x[:, 3]) * x[:, 5] \
            + 8 * (1 - x[:, 1]) * (1 - x[:, 3]) * (1 - x[:, 5])


    @staticmethod    
    def _func4(x: np.ndarray) -> np.ndarray:

        return np.sum(x[:, [0, 2, 4, 6, 7, 8]], axis=1) - 2


    @staticmethod
    def _func5(x: np.ndarray) -> np.ndarray:

        return \
            4 * (x[:, 0] > 1) * (x[:, 2] > 0) \
            + 4 * (x[:, 4] > 1) * (x[:, 6] > 0) \
            + 2 * x[:, 7] * x[:, 8]


    @staticmethod
    def _func6(x: np.ndarray) -> np.ndarray:

        return (
            np.sum(x[:, [0, 2, 4, 6, 8]]**2, axis=1) 
            + np.sum(x[:, [1, 3, 5, 7]], axis=1) - 11
        ) / 2


    @classmethod
    def _func7(self, x: np.ndarray) -> np.ndarray:

        return (self._func3(x) + self._func4(x)) / np.sqrt(2)


class MeanEffectFunction(EffectFunction):
    @staticmethod
    def _scenario_to_func_id(scneario: int) -> int:

        mean_effect_function_ids = [7, 4, 3, 6, 2, 0, 1, 5]
        return mean_effect_function_ids[scneario % 8]


class TreatmentEffectFunction(EffectFunction):
    @staticmethod
    def _scenario_to_func_id(scneario: int) -> int:
        
        treatment_effect_function_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        return treatment_effect_function_ids[scneario % 8]