from typing import Any
import numpy as np
import numpy.typing as npt
import json

from ..sdfl.core.parameters import Parameters
from ..sdfl.core.typing import Point
from ..sdfl._utils._sdfl_requirements import _check_sdfl_arguments_requirements

_N: int = 2
DEFAULT_STARTING_POINT: Point = np.array([1] * _N, dtype = np.float64)
DEFAULT_STARTING_STEP: npt.NDArray[np.float64] = np.array([1] * _N, dtype = np.float64)
DEFAULT_LIMIT_EVAL: int = 10_000
DEFAULT_LIMIT_STEP: np.float64 = np.float64(1e-8)
DEFAULT_THETA: np.float64 = np.float64(0.5)
DEFAULT_GAMMA: np.float64 = np.float64(2.5)
DEFAULT_C: np.float64 = np.float64(1)
DEFAULT_ETA: np.float64 = np.float64(1)
DEFAULT_EPSILON: np.float64 = np.float64(1)

DEFAULT_PARAMS: Parameters = Parameters(DEFAULT_THETA, DEFAULT_GAMMA, DEFAULT_C, DEFAULT_ETA, DEFAULT_EPSILON)

KEY_STARTING_POINT: str = "starting_point"
KEY_STARTING_STEP: str = "starting_step"
KEY_LIMIT_EVAL: str = "limit_eval"
KEY_LIMIT_STEP: str = "limit_step"
KEY_THETA: str = "theta"
KEY_GAMMA: str = "gamma"
KEY_C: str = "c"
KEY_ETA: str = "eta"
KEY_EPSILON: str = "epsilon"

KEY_PARAMS: str = "params"

DATA_JSON: str = "data.json"

class SDFLData:
    starting_point: Point
    starting_step: npt.NDArray[np.float64]
    limit_eval: int
    limit_step: np.float64
    params: Parameters

    def __init__(self: SDFLData, starting_point: Point = DEFAULT_STARTING_POINT, starting_step: npt.NDArray[np.float64] = DEFAULT_STARTING_STEP, limit_eval: int = DEFAULT_LIMIT_EVAL, limit_step: np.float64 = DEFAULT_LIMIT_STEP, params: Parameters = DEFAULT_PARAMS) -> None:
        _check_sdfl_arguments_requirements(starting_point, starting_step, limit_eval, limit_step)
        self.starting_point = starting_point
        self.starting_step = starting_step
        self.limit_eval = limit_eval
        self.limit_step = limit_step
        self.params = params

    @staticmethod
    def to_dict(data: SDFLData) -> dict[str, Any]:
        return {
            KEY_STARTING_POINT: data.starting_point,
            KEY_STARTING_STEP: data.starting_step,
            KEY_LIMIT_EVAL: data.limit_eval,
            KEY_LIMIT_STEP: data.limit_step,
            KEY_PARAMS: data.params
        }

    @classmethod
    def to_sdfl_data(cls, data_dict: dict[str, Any]) -> SDFLData:
        return cls(
            data_dict[KEY_STARTING_POINT],
            data_dict[KEY_STARTING_STEP],
            data_dict[KEY_LIMIT_EVAL],
            data_dict[KEY_LIMIT_STEP],
            data_dict[KEY_PARAMS],
        )

def export_data(data: SDFLData) -> None:
    data_dict: dict[str, int | np.float64 | npt.NDArray[np.float64]] = {
        KEY_STARTING_POINT: data.starting_point.tolist(),
        KEY_STARTING_STEP: data.starting_step.tolist(),
        KEY_LIMIT_EVAL: data.limit_eval,
        KEY_LIMIT_STEP: data.limit_step,
        KEY_THETA: data.params.theta,
        KEY_GAMMA: data.params.gamma,
        KEY_C: data.params.c,
        KEY_ETA: data.params.eta,
        KEY_EPSILON: data.params.epsilon,
    }
    with open(DATA_JSON, "w") as p:
        json.dump(data_dict, p, indent = 4, separators = (",", ": "))

def import_data() -> SDFLData:
    try:
        with open(DATA_JSON, "r") as p:
            data_dict = json.load(p)
        data: SDFLData = SDFLData(
            np.array(data_dict[KEY_STARTING_POINT], dtype = np.float64),
            np.array(data_dict[KEY_STARTING_STEP], dtype = np.float64),
            int(data_dict[KEY_LIMIT_EVAL]),
            np.float64(data_dict[KEY_LIMIT_STEP]),
            Parameters(
                np.float64(data_dict[KEY_THETA]),
                np.float64(data_dict[KEY_GAMMA]),
                np.float64(data_dict[KEY_C]),
                np.float64(data_dict[KEY_ETA]),
                np.float64(data_dict[KEY_EPSILON])
            )
        )
        return data
    except OSError:
        return _create_default_data_json()

def _create_default_data_json() -> SDFLData:
    data: SDFLData = SDFLData()
    export_data(data)
    return data
