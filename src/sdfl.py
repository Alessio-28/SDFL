import numpy as np
from numpy import float64
from numpy.typing import NDArray
from collections.abc import Callable
from typing import cast
from enum import Enum

import logging
from logging.handlers import QueueHandler, QueueListener
from os import mkdir
from os.path import exists, isdir
from queue import Queue

LOGGING : bool = False
LOG_DIR : str = "log"
LOG_FILE : str = "sdfl.log"
MODE : str = "a"
LEVEL = logging.INFO 

_sdfl_log : logging.Logger
_sdfl_listener : QueueListener

def _setup_logging() -> None:
    global _sdfl_log
    global _sdfl_listener

    if not (exists(LOG_DIR) and isdir(LOG_DIR)):
        mkdir(LOG_DIR)

    queue : Queue[logging.LogRecord] = Queue(-1)

    sdfl_fh : logging.FileHandler  = logging.FileHandler(filename = f"./{LOG_DIR}/{LOG_FILE}", mode = MODE)
    sdfl_fh.setLevel(LEVEL)

    _sdfl_listener = QueueListener(queue, sdfl_fh)

    _sdfl_log = logging.getLogger(name = __name__)
    _sdfl_log.setLevel(LEVEL)
    _sdfl_log.addHandler(QueueHandler(queue))

    logging._srcfile = None # pyright: ignore[reportPrivateUsage]
    logging.logProcesses = False
    logging.logThreads = False
    logging.logMultiprocessing = False

    # np.set_printoptions(precision = 4, suppress = True)

type Point = NDArray[float64]
type ObjectiveFunction = Callable[[Point], float64]

class Parameters:
    theta   : float64 # in (0, 1)
    gamma   : float64 # > 2
    c       : float64 # > 0
    eta     : float64 # > 0
    epsilon : float64 # > 0

    _THETA_LOWER_BOUND   : int = 0
    _THETA_UPPER_BOUND   : int = 1
    _GAMMA_LOWER_BOUND   : int = 2
    _C_LOWER_BOUND       : int = 0
    _ETA_LOWER_BOUND     : int = 0
    _EPSILON_LOWER_BOUND : int = 0

    def __init__(self : Parameters, theta : float64, gamma : float64, c : float64, eta : float64, epsilon : float64) -> None:
        if not (self._THETA_LOWER_BOUND < theta < self._THETA_UPPER_BOUND) or gamma <= self._GAMMA_LOWER_BOUND or c <= self._C_LOWER_BOUND or eta <= self._ETA_LOWER_BOUND or epsilon <= self._EPSILON_LOWER_BOUND:
            str_error : str = (
                "Invalid parameter values: "
                f"{self._THETA_LOWER_BOUND} < theta < {self._THETA_UPPER_BOUND}, "
                f"gamma > {self._GAMMA_LOWER_BOUND}, "
                f"c > {self._C_LOWER_BOUND}, "
                f"eta > {self._ETA_LOWER_BOUND}, "
                f"epsilon > {self._EPSILON_LOWER_BOUND}"
            )
            raise ValueError(str_error)

        self.theta   = theta
        self.gamma   = gamma
        self.c       = c
        self.eta     = eta
        self.epsilon = epsilon


class _FunctionWrapper:
    obj_func : ObjectiveFunction
    evaluations : int

    def __init__(self : _FunctionWrapper, obj_func : ObjectiveFunction) -> None:
        self.obj_func = obj_func
        self.evaluations = 0

    def eval(self : _FunctionWrapper, x : Point) -> float64:
        self.evaluations += 1
        return self.obj_func(x)

class _DirectionResult(Enum):
    POSITIVE =  1
    NEGATIVE = -1
    FAILURE  =  0

def _compute_bound_coeff(param : Parameters) -> float64:
    return -param.gamma * param.c * param.epsilon

def _compute_bound(coeff : float64, step_size : float64) -> float64:
    return coeff * step_size * step_size

def _compute_direction(obj_func : ObjectiveFunction, point : Point, func_eval_at_point : float64, step_size : float64, index : int, bound_coeff : float64) -> tuple[_DirectionResult, float64]:
    elem : float64 = cast(float64, point[index])
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

def SDFL(obj_func : ObjectiveFunction, starting_point : Point, starting_step : NDArray[float64], param : Parameters, limit_evaluations : int, limit_step : float64) -> Point:
    if len(starting_point.shape) != 1:
        raise ValueError("starting_point must be a 1-dimensional array")
    if len(starting_step.shape) != 1:
        raise ValueError("starting_step must be a 1-dimensional array")
    if starting_point.size != starting_step.size:
        raise ValueError("starting_point and starting_step must have the same size")

    LIMIT_EVAL : int = limit_evaluations
    LIMIT_STEP : float64 = limit_step
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

    if LOGGING:
        _sdfl_listener.start()
        _sdfl_log.info("#################### SDFL: Start ####################")

    while f_wrapper.evaluations < LIMIT_EVAL and max_tentative_step >= LIMIT_STEP:
        new_point_found : bool = False
        _ = np.maximum(tentative_step, eta * max_tentative_step, out = tentative_step)

        for i in range(n):
            if prev_dir_res != _DirectionResult.FAILURE:
                func_eval_at_cur_point = F(current_point)

            if LOGGING:
                _sdfl_log.info(f"x = %s\nF(x) = %g\nStep = %s\n", current_point, func_eval_at_cur_point, tentative_step)

            dir_res : _DirectionResult
            func_eval_at_direction : float64
            (dir_res , func_eval_at_direction) = _compute_direction(F, current_point, func_eval_at_cur_point, tentative_step[i], i, bound_coeff)

            if dir_res == _DirectionResult.FAILURE:
                accepted_step[i] = 0
            else:
                accepted_step[i] = _line_search(F, current_point, func_eval_at_direction, dir_res.value, tentative_step[i], i, bound_coeff)
                new_point_found = True
            prev_dir_res = dir_res

        if LOGGING:
            _sdfl_log.info("New point found: %s\n", new_point_found)

        if new_point_found:
            _ = np.maximum(accepted_step, tentative_step, out = tentative_step)
        else:
            tentative_step *= theta
        max_tentative_step = np.max(tentative_step)

    if LOGGING:
        _sdfl_log.info(f"Minimum: %s\n#################### SDFL: End ####################\n", current_point)
        _sdfl_listener.stop()

    return current_point
