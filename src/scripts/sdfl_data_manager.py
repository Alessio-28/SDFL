from typing import Any
import numpy as np

from ..test import problems
from ..sdfl.core.parameters import Parameters
from .constants import *

class SDFLData:
    functions: problems.Problem
    max_eval: int
    min_step: np.float64
    params: Parameters

    def __init__(self: SDFLData, functions: problems.Problem, max_eval: int, min_step: np.float64, params: Parameters) -> None:
        self.functions = functions
        self.max_eval = max_eval
        self.min_step = min_step
        self.params = params

    @staticmethod
    def to_dict(data: SDFLData) -> dict[str, Any]:
        return {
            KEY_STARTING_POINT: data.starting_point.tolist(),
            KEY_STARTING_STEP: data.starting_step.tolist(),
            KEY_MAX_EVAL: data.max_eval,
            KEY_MIN_STEP: data.min_step,
            KEY_THETA: data.params.theta,
            KEY_GAMMA: data.params.gamma,
            KEY_C: data.params.c,
            KEY_ETA: data.params.eta,
            KEY_EPSILON: data.params.epsilon
        }

    @staticmethod
    def to_SDFLData(data_dict: dict[str, Any]) -> SDFLData:
        if not SDFLData.validate_data_dict(data_dict):
            raise ValueError("Invalid dictionary")
        return SDFLData(
            starting_point=np.array(data_dict[KEY_STARTING_POINT], dtype=np.float64),
            starting_step=np.array(data_dict[KEY_STARTING_STEP], dtype=np.float64),
            max_eval=int(data_dict[KEY_MAX_EVAL]),
            min_step=np.float64(data_dict[KEY_MIN_STEP]),
            params=Parameters(
                theta=np.float64(data_dict[KEY_THETA]),
                gamma=np.float64(data_dict[KEY_GAMMA]),
                c=np.float64(data_dict[KEY_C]),
                eta=np.float64(data_dict[KEY_ETA]),
                epsilon=np.float64(data_dict[KEY_EPSILON])
            )
        )

