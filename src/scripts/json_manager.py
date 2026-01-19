import json
from typing import Any
import numpy as np
from ..sdfl.core.parameters import Parameters
from . import sdfl_data_manager as sdfl_data
from . import constants

DATA_JSON: str = "data.json"

# DATA_JSON_SCHEMA: str = '''{
#     "max_eval": 10000,
#     "min_step": 1e-08,
#     "theta": 0.5,
#     "gamma": 2.5,
#     "c": 1.0,
#     "eta": 1.0,
#     "epsilon": 1.0
# }'''

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

# class JSONContent:
#     max_eval: int
#     min_step: np.float64
#     theta: np.float64
#     gamma: np.float64
#     c: np.float64
#     eta: np.float64
#     epsilon: np.float64

#     def __init__(self: JSONContent, max_eval: int, min_step: np.float64, theta: np.float64, gamma: np.float64, c: np.float64, eta: np.float64, epsilon: np.float64) -> None:
#         self.max_eval = max_eval
#         self.min_step = min_step
#         self.theta = theta
#         self.gamma = gamma
#         self.c = c
#         self.eta = eta
#         self.epsilon = epsilon

