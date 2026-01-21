import json
from typing import Any
import numpy as np
import numpy.typing as npt

from . import sdfl_data
from . import constants
from ..test import problems
from ..sdfl.core import parameters

DATA_JSON: str = "data.json"

def export_data(data_dict: dict[str, Any]) -> None:
    with open(DATA_JSON, "w") as p:
        json.dump(data_dict, p, indent=4, separators=(",", ": "))

def import_data() -> dict[str, Any]:
    try:
        with open(DATA_JSON, "r") as p:
            data_dict = json.load(p)
        if validate_data_json(data_dict):
            return data_dict
        else:
            raise ValueError(f"Invalid {DATA_JSON} file.")
    except OSError:
        return create_default_data_json()


def create_default_data_json() -> dict[str, Any]:
    data = {
        constants.KEY_MAX_EVAL: constants.DEFAULT_MAX_EVAL,
        constants.KEY_MIN_STEP: constants.DEFAULT_MIN_STEP,
        constants.KEY_THETA:    constants.DEFAULT_THETA,
        constants.KEY_GAMMA:    constants.DEFAULT_GAMMA,
        constants.KEY_C:        constants.DEFAULT_C,
        constants.KEY_ETA:      constants.DEFAULT_ETA,
        constants.KEY_EPSILON:  constants.DEFAULT_EPSILON
    }
    export_data(data)
    return data

def validate_data_json(data_dict: dict[str, Any]) -> bool:
    INT = (np.integer,)
    INT_OR_FLOAT = (np.integer, np.floating)
    valid_data_dict = {
        constants.KEY_MAX_EVAL: INT,         
        constants.KEY_MIN_STEP: INT_OR_FLOAT,
        constants.KEY_THETA:    INT_OR_FLOAT,
        constants.KEY_GAMMA:    INT_OR_FLOAT,
        constants.KEY_C:        INT_OR_FLOAT,
        constants.KEY_ETA:      INT_OR_FLOAT,
        constants.KEY_EPSILON:  INT_OR_FLOAT,
    }

    if set(valid_data_dict.keys()) != set(data_dict.keys()):
        return False

    for k, allowed_types in valid_data_dict.items():
        v = data_dict[k]

        if not any([np.issubdtype(type(v), t) for t in allowed_types]):
            return False

    return True

def dict_to_SDFLData(f: problems.Problem, data_dict: dict[str, Any], starting_step: npt.NDArray[np.float64] | None = None) -> sdfl_data.SDFLData:
    data = sdfl_data.SDFLData(
        function=f,
        max_eval=data_dict[constants.KEY_MAX_EVAL],
        min_step=data_dict[constants.KEY_MIN_STEP],
        params=parameters.Parameters(
            theta=data_dict[constants.KEY_THETA],
            gamma=data_dict[constants.KEY_GAMMA],
            c=data_dict[constants.KEY_C],
            eta=data_dict[constants.KEY_ETA],
            epsilon=data_dict[constants.KEY_EPSILON],
        ),
        starting_step=starting_step
    )
    return data
