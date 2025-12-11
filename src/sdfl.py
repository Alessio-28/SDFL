import numpy as np
from numpy import float64
from numpy.typing import NDArray
from collections.abc import Callable
from typing import Tuple
from enum import Enum
from dataclasses import dataclass

################ Logging #######################################
import logging
LOGGING : bool = False

_fh : logging.FileHandler 
_sdfl_log : logging.Logger
# _dir_log : logging.Logger
# _line_log : logging.Logger

def _setup_logging() -> None:
    global _fh
    global _sdfl_log
    # global _dir_log
    # global _line_log

    _fh = logging.FileHandler(filename = "sdfl.log", mode = "w")

    _fh.setLevel(logging.DEBUG)

    _sdfl_log = logging.getLogger(name = __name__)
    # _dir_log = logging.getLogger(name = f"{_sdfl_log.name}.direction")
    # _line_log = logging.getLogger(name = f"{_sdfl_log.name}.line")

    _sdfl_log.setLevel(logging.DEBUG)
    # _dir_log.setLevel(logging.DEBUG)
    # _line_log.setLevel(logging.DEBUG)

    _sdfl_log.addHandler(_fh)
    # _dir_log.addHandler(_fh)
    # _line_log.addHandler(_fh)
################################################################

type Point = NDArray[float64]
type ObjectiveFunction = Callable[[Point], float64]

@dataclass(frozen = True)
class Parameters:
    theta   : float64 # in (0, 1)
    gamma   : float64 # > 2
    c       : float64 # > 0
    eta     : float64 # > 0
    epsilon : float64 # > 0

class _FunctionWrapper:
    _obj_func : ObjectiveFunction
    _evaluations : int

    def __init__(self : _FunctionWrapper, obj_func : ObjectiveFunction) -> None:
        self._obj_func = obj_func
        self._evaluations = 0

    def eval(self : _FunctionWrapper, x : Point) -> float64:
        self._evaluations += 1
        return self._obj_func(x)

class _DirectionResult(Enum):
    POSITIVE =  1
    NEGATIVE = -1
    FAILURE  =  0

def _compute_bound_coeff(param : Parameters) -> float64:
    return -param.gamma * param.c * param.epsilon

def _compute_bound(coeff : float64, step_size : float64) -> float64:
    return coeff * step_size * step_size

def _compute_direction(obj_func : ObjectiveFunction, point : Point, func_eval_at_point : float64, step_size : float64, index : int, bound_coeff : float64) -> Tuple[_DirectionResult, float64]:
    elem : float64 = point[index]
    F_bound : float64 = func_eval_at_point + _compute_bound(bound_coeff, step_size)

    # Try POSITIVE direction
    point[index] = elem + step_size
    func_eval_at_direction : float64 = obj_func(point)
    if func_eval_at_direction > F_bound:
        # Try NEGATIVE direction
        point[index] = elem - step_size
        func_eval_at_direction = obj_func(point)

        # Restore changes
        point[index] = elem
        if func_eval_at_direction > F_bound:
            return (_DirectionResult.FAILURE, func_eval_at_direction)
        else:
            return (_DirectionResult.NEGATIVE, func_eval_at_direction)

    # Restore changes
    point[index] = elem
    return (_DirectionResult.POSITIVE, func_eval_at_direction)

def _line_search(obj_func : ObjectiveFunction, point : Point, func_eval_at_point : float64, direction_sign : int, step_size : float64, index : int, bound_coeff : float64) -> float64:
    elem : float64 = point[index]
    step : float64 = step_size * direction_sign
    step_aux : float64 = step * 2

    iter2 : int = 1
    bound : float64 = _compute_bound(bound_coeff, step_size)

    F_a : float64 = func_eval_at_point
    point[index] = elem + step_aux
    F_b : float64 = obj_func(point)
    while F_b - F_a <= bound * iter2 * iter2:
        iter2 *= 2
        point[index] = elem + iter2 * step_aux

        F_a, F_b = F_b, obj_func(point)

    # Restore changes
    point[index] = elem + iter2 * step

    return step_size * iter2

def SDFL(obj_func : ObjectiveFunction, starting_point : Point, starting_step : NDArray[float64], param : Parameters, evaluations : int) -> Point:
    LIMIT_EVAL : int = evaluations
    LIMIT_STEP : float64 = float64(1e-8)
    n : int = starting_point.size

    f_wrapper : _FunctionWrapper = _FunctionWrapper(obj_func)
    F : ObjectiveFunction = f_wrapper.eval

    accepted_step  : NDArray[float64] = np.zeros(n, dtype = float64) # \alpha
    # init_step      : NDArray[float64] = np.zeros(n, dtype = float64) # \bar{\alpha}
    tentative_step : NDArray[float64] = starting_step.copy()         # \tilde{\alpha}
    max_tentative_step : float64 = np.max(tentative_step)

    bound_coeff : float64 = _compute_bound_coeff(param)
    eta   : float64 = param.eta
    theta : float64 = param.theta

    current_point : Point = starting_point.copy()

    func_eval_at_cur_point : float64 = F(current_point)
    prev_dir_res : _DirectionResult = _DirectionResult.FAILURE

    ############ Logging #######################################
    if LOGGING:
        _sdfl_log.debug("########## SDFL: Start ##########")
    ############################################################

    while f_wrapper._evaluations < LIMIT_EVAL: # and max_tentative_step >= LIMIT_STEP:
        new_point_found : bool = False
        np.maximum(tentative_step, eta * max_tentative_step, out = tentative_step)

        for i in range(n):

            if prev_dir_res != _DirectionResult.FAILURE:
                func_eval_at_cur_point = F(current_point)

            #### Logging #######################################
            if LOGGING:
                _sdfl_log.debug(f"x = {current_point}")
                _sdfl_log.debug(f"F(x) = {func_eval_at_cur_point}")
                _sdfl_log.debug(f"Step = {tentative_step}\n")
            ####################################################

            dir_res : _DirectionResult
            func_eval_at_direction : float64
            (dir_res , func_eval_at_direction) = _compute_direction(F, current_point, func_eval_at_cur_point, tentative_step[i], i, bound_coeff)

            if dir_res == _DirectionResult.FAILURE:
                accepted_step[i] = 0
            else:
                accepted_step[i] = _line_search(F, current_point, func_eval_at_direction, dir_res.value, tentative_step[i], i, bound_coeff)
                new_point_found = True
            prev_dir_res = dir_res

        ######## Logging #######################################
        if LOGGING:
            _sdfl_log.debug(f"New point found: {new_point_found}\n")
        ########################################################

        if new_point_found:
            np.maximum(accepted_step, tentative_step, out = tentative_step)
        else:
            tentative_step *= theta
        max_tentative_step = np.max(tentative_step)

    ############ Logging #######################################
    if LOGGING:
        _sdfl_log.debug(f"Minimum: {current_point}")
        _sdfl_log.debug("########## SDFL: End ##########\n")
    ############################################################

    return current_point
