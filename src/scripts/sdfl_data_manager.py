from typing import Any
import numpy as np
import numpy.typing as npt

from ..sdfl.core.parameters import Parameters
from ..sdfl.core.typing import Point
from ..sdfl.core.sdfl import _validate_sdfl_args

from .constants import *

class SDFLData:
    starting_point: Point
    starting_step: npt.NDArray[np.float64]
    max_eval: int
    min_step: np.float64
    params: Parameters

    def __init__(self: SDFLData, starting_point: Point, starting_step: npt.NDArray[np.float64], max_eval: int, min_step: np.float64, params: Parameters) -> None:
        _validate_sdfl_args(starting_point, starting_step, max_eval, min_step)
        self.starting_point = starting_point
        self.starting_step = starting_step
        self.max_eval = max_eval
        self.min_step = min_step
        self.params = params

    @staticmethod
    def to_dict(data: SDFLData) -> dict[str, Any]: # pyright: ignore[reportExplicitAny]
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

    @classmethod
    def to_SDFLData(cls, data_dict: dict[str, Any]) -> SDFLData: # pyright: ignore[reportExplicitAny]
        return cls(
            np.array(data_dict[KEY_STARTING_POINT], dtype = np.float64),
            np.array(data_dict[KEY_STARTING_STEP], dtype = np.float64),
            int(data_dict[KEY_MAX_EVAL]),
            np.float64(data_dict[KEY_MIN_STEP]),
            Parameters(
                data_dict[KEY_THETA],
                data_dict[KEY_GAMMA],
                data_dict[KEY_C],
                data_dict[KEY_ETA],
                data_dict[KEY_EPSILON]
            )
        )
