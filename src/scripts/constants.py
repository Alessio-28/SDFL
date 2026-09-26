from typing import TypedDict

import numpy as np


class JSONProblemParams(TypedDict):
    theta: np.float64
    gamma: np.float64
    c: np.float64
    eta: np.float64
    epsilon: np.float64


class JSONProblemData(TypedDict):
    max_eval: int
    min_step: np.float64
    params: JSONProblemParams


DEFAULT_JSON_DATA: JSONProblemData = {
    "max_eval": 5000,
    "min_step": np.float64(1e-6),
    "params": {
        "theta": np.float64(0.5),
        "gamma": np.float64(2.0001),
        "c": np.float64(1e-3),
        "eta": np.float64(1e-5),
        "epsilon": np.float64(0.1),
    },
}
