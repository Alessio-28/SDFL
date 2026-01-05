import numpy as np
import pathlib
import json

from ..sdfl.core.parameters import Parameters

DEFAULT_THETA: np.float64 = np.float64(0.5)
DEFAULT_GAMMA: np.float64 = np.float64(2.5)
DEFAULT_C: np.float64 = np.float64(1)
DEFAULT_ETA: np.float64 = np.float64(1)
DEFAULT_EPSILON: np.float64 = np.float64(1)
DEFAULT_STARTING_POINT: np.float64 = np.float64(1)
DEFAULT_STARTING_STEP: np.float64 = np.float64(1)
DEFAULT_LIMIT_EVAL: int = 1_000
DEFAULT_LIMIT_STEP: np.float64 = np.float64(1e-8)

KEY_STARTING_POINT: str = "starting_point"
KEY_STARTING_STEP: str = "starting_step"
KEY_LIMIT_EVAL: str = "limit_eval"
KEY_LIMIT_STEP: str = "min_step"
KEY_THETA: str = "theta"
KEY_GAMMA: str = "gamma"
KEY_C: str = "c"
KEY_ETA: str = "eta"
KEY_EPSILON: str = "epsilon"


DATA_JSON: pathlib.PurePath = pathlib.PurePath("./parameters.json")

def export_parameters(params: Parameters) -> None:
    param_dict: dict[str, np.float64] = {
        KEY_THETA: params.theta,
        KEY_GAMMA: params.gamma,
        KEY_C: params.c,
        KEY_ETA: params.eta,
        KEY_EPSILON: params.epsilon,
    }
    with open(DATA_JSON, "w") as p:
        json.dump(param_dict, p, indent = 4, separators = (",", ": "))

def import_parameters() -> Parameters:
    try:
        with open(DATA_JSON) as p:
            data = json.load(p)
        params: Parameters = Parameters(
            theta   = data[KEY_THETA],
            gamma   = data[KEY_GAMMA],
            c       = data[KEY_C],
            eta     = data[KEY_ETA],
            epsilon = data[KEY_EPSILON]
        )
        return params
    except OSError:
        return _create_default_parameters_json()

def _create_default_parameters_json() -> Parameters:
    params: Parameters = Parameters(
        theta   = DEFAULT_THETA,
        gamma   = DEFAULT_GAMMA,
        c       = DEFAULT_C,
        eta     = DEFAULT_ETA,
        epsilon = DEFAULT_EPSILON
    )
    export_parameters(params)
    return params
