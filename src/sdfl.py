import numpy as np
from numpy import float64
from numpy.typing import NDArray
from collections.abc import Callable
from typing import Tuple
from enum import Enum
from dataclasses import dataclass

############# Logging ###############################################
#####################################################################
# import logging
#
# fh : logging.FileHandler  = logging.FileHandler(filename = "sdfl.log", mode = "w")
#
# fh.setLevel(logging.DEBUG)
#
# sdfl_log : logging.Logger = logging.getLogger(name = __name__)
# dir_log  : logging.Logger = logging.getLogger(name = f"{sdfl_log.name}.direction")
# line_log : logging.Logger = logging.getLogger(name = f"{sdfl_log.name}.line")
#
# sdfl_log.setLevel(logging.DEBUG)
# dir_log.setLevel(logging.DEBUG)
# line_log.setLevel(logging.DEBUG)
#
# sdfl_log.addHandler(fh)
# dir_log.addHandler(fh)
# line_log.addHandler(fh)
#####################################################################
#####################################################################

type Point = NDArray[float64]
type ObjectiveFunction = Callable[[Point], float64]

@dataclass(frozen = True)
class Parameters:
    theta   : float64
    gamma   : float64
    c       : float64
    eta     : float64
    epsilon : float64

class __FunctionWrapper:
    __obj_func : ObjectiveFunction
    __evaluations : int

    def __init__(self : __FunctionWrapper, obj_func : ObjectiveFunction) -> None:
        self.__obj_func = obj_func
        self.__evaluations = 0

    def eval(self : __FunctionWrapper, x : Point) -> float64:
        self.__evaluations += 1
        return self.__obj_func(x)

    def get_obj_func(self : __FunctionWrapper) -> ObjectiveFunction:
        return self.__obj_func

    def get_nF(self : __FunctionWrapper) -> int:
        return self.__evaluations


class __DirectionResult(Enum):
    POSITIVE =  1
    NEGATIVE = -1
    FAILURE  =  0

def __compute_bound_coeff(param : Parameters) -> float64:
    return -param.gamma * param.c * param.epsilon

def __compute_bound(coeff : float64, step_size : float64) -> float64:
    return coeff * step_size * step_size

def __compute_direction(F : ObjectiveFunction, y : Point, F_y : float64, step_size : float64, index : int, bound_coeff : float64) -> Tuple[__DirectionResult, float64]:
    F_bound : float64 = F_y + __compute_bound(bound_coeff, step_size)

    # Try POSITIVE direction
    y[index] += step_size
    F_dir : float64 = F(y)
    if F_dir > F_bound:
        # Try NEGATIVE direction
        y[index] -= 2 * step_size
        F_dir = F(y)

        # Restore changes
        y[index] += step_size
        if F_dir > F_bound:
            return (__DirectionResult.FAILURE, F_dir)
        else:
            return (__DirectionResult.NEGATIVE, F_dir)

    # Restore changes
    y[index] -= step_size
    return (__DirectionResult.POSITIVE, F_dir)

def __line_search(F : ObjectiveFunction, y : Point, F_dir : float64, direction_sign : int, step_size : float64, index : int, bound_coeff : float64) -> float64:
    iter2 : int = 1
    step  : float64 = step_size * direction_sign
    bound : float64 = __compute_bound(bound_coeff, step_size)

    F_a : float64 = F_dir
    y[index] += 2 * step
    F_b : float64 = F(y)
    while F_b - F_a <= bound * (iter2 * iter2):
        iter2 *= 2
        y[index] += iter2 * step

        F_a, F_b = F_b, F(y)

    # Restore changes
    y[index] -= step * iter2

############# Logging ##########################
    # line_log.debug(f"y_i: {y[index]}, Step size: {step_size * iter2}, Step: {step_size}, iter2: {iter2}")
###############################################

    return step_size * iter2

def SDFL(obj_func : ObjectiveFunction, starting_point : Point, starting_step : NDArray[float64], param : Parameters) -> Point:

############# Logging ##########################
    # sdfl_log.debug("SDFL: Start")
    # log_counter : int = 0
################################################

    LIMIT : float64 = float64(1e-8)
    direction_sign : int = 0
    n : int = starting_point.size

    f_wrapper : __FunctionWrapper = __FunctionWrapper(obj_func)
    F : ObjectiveFunction = f_wrapper.eval

    # Creare una classe?
    accepted_step  : NDArray[float64] = np.zeros(n, dtype = float64) # \alpha_k^i
    init_step      : NDArray[float64] = np.zeros(n, dtype = float64) # \bar{\alpha}_k^i
    tentative_step : NDArray[float64] = starting_step.copy()         # \tilde{\alpha}_k^i
    max_tentative_step : float64 = np.max(tentative_step)

    bound_coeff : float64 = __compute_bound_coeff(param)
    eta   : float64 = param.eta
    theta : float64 = param.theta

    F_y : float64 = float64(0)
    minimum : Point = starting_point.copy()
    y : Point = starting_point.copy()

    prev_dir_res : __DirectionResult = __DirectionResult.POSITIVE
    while max_tentative_step > LIMIT:
        new_point_found : bool = False
        np.maximum(tentative_step, eta * max_tentative_step, out = init_step)

        for i in range(n):

############# Logging ##########################
            # sdfl_log.debug(f"Pre:  Iter: [{log_counter}|{i}], y: {y}, direction: {prev_dir_res.name}, F_y: {F_y}")
################################################

            if prev_dir_res != __DirectionResult.FAILURE:
                F_y = F(y)

            dir_res : __DirectionResult
            F_dir : float64
            (dir_res , F_dir) = __compute_direction(F, y, F_y, init_step[i], i, bound_coeff)

            if dir_res == __DirectionResult.FAILURE:
                accepted_step[i] = 0
            else:
                direction_sign = dir_res.value
                accepted_step[i] = __line_search(F, y, F_dir, direction_sign, init_step[i], i, bound_coeff)
                new_point_found = True
            prev_dir_res = dir_res

############# Logging ##########################
        #     sdfl_log.debug(f"Post: Iter: [{log_counter}|{i}], y: {y}, step: {accepted_step[i]}, direction: {dir_res.name}, F_y: {F_y}, F_dir: {F_dir}")
        # sdfl_log.debug(f"New point found: {new_point_found}")
        # log_counter += 1
################################################

        if new_point_found:
            minimum[:] = y
            np.maximum(accepted_step, init_step, out = tentative_step)
        else:
            tentative_step = theta * init_step
        max_tentative_step = np.max(tentative_step)

############# Logging ##########################
    # sdfl_log.debug(f"Minimum: {x}")
    # sdfl_log.debug("SDFL: End")
################################################

    return minimum
