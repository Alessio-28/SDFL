import numpy as np

from .._utils._sdfl_requirements import _are_parameters_in_range

_THETA_LOWER_BOUND: int = 0
_THETA_UPPER_BOUND: int = 1
_GAMMA_LOWER_BOUND: int = 2
_C_LOWER_BOUND: int = 0
_ETA_LOWER_BOUND: int = 0
_EPSILON_LOWER_BOUND: int = 0

class Parameters:
    theta: np.float64   # in (0, 1)
    gamma: np.float64   # > 2
    c: np.float64       # > 0
    eta: np.float64     # > 0
    epsilon: np.float64 # > 0

    def __init__(self: Parameters, theta: np.float64, gamma: np.float64, c: np.float64, eta: np.float64, epsilon: np.float64) -> None:
        if _are_parameters_in_range(theta, gamma, c, eta, epsilon):
            str_error: str = (
                "Invalid parameter values: "
                f"{_THETA_LOWER_BOUND} < theta < {_THETA_UPPER_BOUND}, "
                f"gamma > {_GAMMA_LOWER_BOUND}, "
                f"c > {_C_LOWER_BOUND}, "
                f"eta > {_ETA_LOWER_BOUND}, "
                f"epsilon > {_EPSILON_LOWER_BOUND}"
            )
            raise ValueError(str_error)

        self.theta   = theta
        self.gamma   = gamma
        self.c       = c
        self.eta     = eta
        self.epsilon = epsilon


def _compute_bound_coeff(param: Parameters) -> np.float64:
    return -param.gamma * param.c * param.epsilon

def _compute_bound(coeff: np.float64, step_size: np.float64) -> np.float64:
    return coeff * step_size * step_size
