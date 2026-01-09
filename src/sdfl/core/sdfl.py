import numpy as np
import numpy.typing as npt
from enum import Enum
from typing import override
import logging

from .typing import Point, ObjectiveFunction
from .parameters import Parameters, _compute_bound_coeff, _compute_bound
from .._utils._logging import _enable_default_logging, _disable_default_logging

sdfl_logger: logging.Logger = logging.getLogger(__name__)

def SDFL(obj_fun: ObjectiveFunction, starting_point: Point, starting_step: npt.NDArray[np.float64], param: Parameters, max_eval: int, min_step: np.float64, verbose: bool = False) -> SDFLResult:
    _validate_sdfl_args(starting_point, starting_step, max_eval, min_step)

    n: int = starting_point.size

    f_wrapper: _FunctionWrapper = _FunctionWrapper(obj_fun)
    F: ObjectiveFunction = f_wrapper.eval

    accepted_step: npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64) # \alpha
    # init_step: npt.NDArray[np.float64] = np.zeros(n, dtype = np.float64) # \bar{\alpha}
    tentative_step: npt.NDArray[np.float64] = starting_step.copy()         # \tilde{\alpha}
    max_tentative_step: np.float64 = np.max(tentative_step)

    bound_coeff: np.float64 = _compute_bound_coeff(param)
    eta: np.float64 = param.eta
    theta: np.float64 = param.theta

    current_point: Point = starting_point.copy()
    fun_eval_at_cur_point: np.float64 = F(current_point)
    prev_dir_res: _DirectionResult = _DirectionResult.FAILURE

    if verbose:
        _enable_default_logging(sdfl_logger)
        sdfl_logger.info("x = %s\nF(x) = %g\nStep = %s\n", current_point, fun_eval_at_cur_point, tentative_step)

    try:
        while f_wrapper.nfev < max_eval and max_tentative_step >= min_step:
            new_point_found: bool = False
            np.maximum(tentative_step, eta * max_tentative_step, out = tentative_step)

            for i in range(n):
                if prev_dir_res != _DirectionResult.FAILURE:
                    fun_eval_at_cur_point = F(current_point)

                dir_res: _DirectionResult
                fun_eval_at_direction: np.float64
                (dir_res, fun_eval_at_direction) = _compute_direction(F, current_point, fun_eval_at_cur_point, tentative_step[i], i, bound_coeff)

                if dir_res == _DirectionResult.FAILURE:
                    accepted_step[i] = 0
                else:
                    accepted_step[i] = _line_search(F, current_point, fun_eval_at_direction, dir_res.value, tentative_step[i], i, bound_coeff)
                    new_point_found = True
                prev_dir_res = dir_res

            if verbose:
                sdfl_logger.info("x = %s\nF(x) = %g\nStep = %s\n", current_point, fun_eval_at_cur_point, tentative_step)

            if new_point_found:
                np.maximum(accepted_step, tentative_step, out = tentative_step)
            else:
                tentative_step *= theta
            max_tentative_step = np.max(tentative_step)

        result: SDFLResult = SDFLResult(current_point, fun_eval_at_cur_point, f_wrapper.nfev)

        if verbose:
            sdfl_logger.info("Result:\n%s\n", result)
    finally:
        _disable_default_logging(sdfl_logger)

    return result

def _compute_direction(obj_fun: ObjectiveFunction, point: Point, fun_eval_at_point: np.float64, step_size: np.float64, index: int, bound_coeff: np.float64) -> tuple[_DirectionResult, np.float64]:
    elem: np.float64 = point[index]
    F_bound: np.float64 = fun_eval_at_point + _compute_bound(bound_coeff, step_size)

    # Try POSITIVE direction
    point[index] = elem + step_size
    fun_eval_at_direction: np.float64 = obj_fun(point)

    if fun_eval_at_direction > F_bound:
        # Try NEGATIVE direction
        point[index] = elem - step_size
        fun_eval_at_direction = obj_fun(point)

        # Restore changes
        point[index] = elem
        if fun_eval_at_direction > F_bound:
            return (_DirectionResult.FAILURE, fun_eval_at_direction)
        else:
            return (_DirectionResult.NEGATIVE, fun_eval_at_direction)

    # Restore changes
    point[index] = elem
    return (_DirectionResult.POSITIVE, fun_eval_at_direction)

def _line_search(obj_fun: ObjectiveFunction, point: Point, fun_eval_at_point: np.float64, direction_sign: int, step_size: np.float64, index: int, bound_coeff: np.float64) -> np.float64:
    elem: np.float64 = point[index]
    step: np.float64 = step_size * direction_sign
    step_aux: np.float64 = step * 2

    iter2: int = 1
    bound: np.float64 = _compute_bound(bound_coeff, step_size)

    F_a: np.float64 = fun_eval_at_point
    point[index] = elem + step_aux
    F_b: np.float64 = obj_fun(point)
    while F_b - F_a <= bound * iter2 * iter2:
        iter2 *= 2
        point[index] = elem + iter2 * step_aux

        F_a, F_b = F_b, obj_fun(point)

    # Restore changes
    point[index] = elem + iter2 * step

    return step_size * iter2

class _FunctionWrapper:
    obj_fun: ObjectiveFunction
    nfev: int

    def __init__(self: _FunctionWrapper, obj_fun: ObjectiveFunction) -> None:
        self.obj_fun = obj_fun
        self.nfev = 0

    def eval(self: _FunctionWrapper, x: Point) -> np.float64:
        self.nfev += 1
        return self.obj_fun(x)

class _DirectionResult(Enum):
    POSITIVE =  1
    NEGATIVE = -1
    FAILURE  =  0

class SDFLResult:
    x: Point
    fun: np.float64
    nfev: int

    def __init__(self: SDFLResult, x: Point, fun: np.float64, nfev: int) -> None:
        self.x = x
        self.fun = fun
        self.nfev = nfev

    @override
    def __str__(self: SDFLResult) -> str:
        str_repr : str = (
            f"x = {self.x}\n"
            f"f(x) = {self.fun}\n"
            f"nfev = {self.nfev}\n"
        )
        return str_repr

def _validate_sdfl_args(starting_point: Point, starting_step: npt.NDArray[np.float64], max_eval: int, min_step: np.float64) -> None:
    if len(starting_point.shape) != 1:
        raise ValueError("starting_point must be a 1-dimensional array")
    if len(starting_step.shape) != 1:
        raise ValueError("starting_step must be a 1-dimensional array")
    if starting_point.size != starting_step.size:
        raise ValueError("starting_point and starting_step must have the same size")
    if np.all(starting_step <= 0):
        raise ValueError("starting_step must be an array of positive real numbers")
    if max_eval <= 0:
        raise ValueError("max_eval must be a positive integer")
    if min_step <= 0:
        raise ValueError("min_step must be a positive real number")
