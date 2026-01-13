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

    @staticmethod
    def validate_data_dict(data_dict: dict[str, Any]) -> bool:
        ARRAY = (list, np.ndarray)
        INT = (np.integer,)
        INT_OR_FLOAT = (np.integer, np.floating)
        valid_data_dict = {
            KEY_STARTING_POINT: (ARRAY,        True),
            KEY_STARTING_STEP:  (ARRAY,        True),
            KEY_MAX_EVAL:       (INT,          False),
            KEY_MIN_STEP:       (INT_OR_FLOAT, False),
            KEY_THETA:          (INT_OR_FLOAT, False),
            KEY_GAMMA:          (INT_OR_FLOAT, False),
            KEY_C:              (INT_OR_FLOAT, False),
            KEY_ETA:            (INT_OR_FLOAT, False),
            KEY_EPSILON:        (INT_OR_FLOAT, False),
        }

        if set(valid_data_dict.keys()) != set(data_dict.keys()):
            return False

        for k, (allowed_types, is_iterable) in valid_data_dict.items():
            v = data_dict[k]

            if not any([np.issubdtype(type(v), t) for t in allowed_types]):
                return False

            if is_iterable:
                if isinstance(v, np.ndarray):
                    if not any([np.issubdtype(v.dtype, t) for t in INT_OR_FLOAT]):
                        return False
                elif not all([any([np.issubdtype(type(x), t) for t in INT_OR_FLOAT]) for x in v]):
                    return False

        return True
