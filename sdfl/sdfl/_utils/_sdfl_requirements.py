import numpy as np
import numpy.typing as npt

from ..core.typing import Point
from ..core import parameters

def _check_starting_point_and_step_requirements(starting_point: Point, starting_step: npt.NDArray[np.float64]) -> None:
    if len(starting_point.shape) != 1:
        raise ValueError("starting_point must be a 1-dimensional array")
    if len(starting_step.shape) != 1:
        raise ValueError("starting_step must be a 1-dimensional array")
    if starting_point.size != starting_step.size:
        raise ValueError("starting_point and starting_step must have the same dimension")

def _check_limit_eval_requirements(limit_eval: int) -> None:
    if limit_eval <= 0:
        raise ValueError("limit_eval must be a positive integer")

def _check_limit_step_requirements(limit_step: np.float64) -> None:
    if limit_step <= 0:
        raise ValueError("limit_step must be a positive real number")

def _check_sdfl_arguments_requirements(starting_point: Point, starting_step: npt.NDArray[np.float64], limit_eval: int, limit_step: np.float64) -> None:
    _check_starting_point_and_step_requirements(starting_point, starting_step)
    _check_limit_eval_requirements(limit_eval)
    _check_limit_step_requirements(limit_step)

def _are_parameters_in_range(theta: np.float64, gamma: np.float64, c: np.float64, eta: np.float64, epsilon: np.float64) -> bool:
    return (theta <= parameters._THETA_LOWER_BOUND or
            theta >= parameters._THETA_UPPER_BOUND or
            gamma <= parameters._GAMMA_LOWER_BOUND or
            c <= parameters._C_LOWER_BOUND or
            eta <= parameters._ETA_LOWER_BOUND or
            epsilon <= parameters._EPSILON_LOWER_BOUND) # pyright: ignore[reportReturnType]
