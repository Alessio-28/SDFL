import numpy as np
from numpy import float64
from numpy.typing import NDArray
from typing import Callable, TypeAlias, Tuple
from enum import Enum
from dataclasses import dataclass

ObjectiveFunction : TypeAlias = Callable[[ NDArray[float64] ], float64]
Point     : TypeAlias = NDArray[float64]
StepSizes : TypeAlias = NDArray[float64]

@dataclass(frozen = True)
class Parameters:
    theta   : float64
    gamma   : float64
    c       : float64
    eta     : float64
    epsilon : float64

class __FunctionWrapper:
    __obj_func : ObjectiveFunction
    __nF : int

    def __init__(self : __FunctionWrapper, obj_func : ObjectiveFunction) -> None:
        self.__obj_func = obj_func
        self.__nF = 0

    def eval(self : __FunctionWrapper, x : Point) -> float64:
        self.__nF += 1
        return self.__obj_func(x)

    def get_obj_func(self : __FunctionWrapper) -> ObjectiveFunction:
        return self.__obj_func

    def get_nF(self : __FunctionWrapper) -> int:
        return self.__nF


class __DirectionResult(Enum):
    POSITIVE =  1
    NEGATIVE = -1
    FAILURE  =  0

def __compute_bound_coeff(param : Parameters) -> float64:
    return -param.gamma * param.c * param.epsilon

def __compute_bound(coeff : float64, step_size : float64) -> float64:
    return coeff * step_size * step_size

def __compute_direction(F : ObjectiveFunction, y : Point, F_y : float64, step_size : float64, index : int, bound_coeff : float64) -> Tuple[__DirectionResult, float64]:
    y_i : float64 = y[index]
    F_bound : float64 = F_y + __compute_bound(bound_coeff, step_size)

    y[index] += step_size
    F_dir : float64 = F(y)
    if F_dir > F_bound:
        y[index] = y_i - step_size
        F_dir = F(y)

        y[index] = y_i
        if F_dir > F_bound:
            return (__DirectionResult.FAILURE, F_dir)
        else:
            return (__DirectionResult.NEGATIVE, F_dir)

    y[index] = y_i
    return (__DirectionResult.POSITIVE, F_dir)

def __line_search(F : ObjectiveFunction, y : Point, F_dir : float64, direction_sign : int, step_size : float64, index : int, bound_coeff : float64) -> float64:
    iter2 : int = 1
    step  : float64 = step_size * direction_sign
    bound : float64 = __compute_bound(bound_coeff, step_size)

    F_a : float64 = F_dir
    y[index] += 2 * step
    F_b : float64 = F(y)
    while F_b - F_a <= bound * (iter2 * iter2):
        iter2 *= 2                  # step *= 2
        y[index] += iter2 * step    # y[index] = y_i + 2 * step * direction

        F_a = F_b
        F_b = F(y)

    y[index] -= step * iter2          # y[index] = y_i + step * direction
    return step_size * iter2


def SDFL(obj_func : ObjectiveFunction, starting_point : Point, starting_step : StepSizes, param : Parameters) -> Point:
    LIMIT : float64 = float64(1e-6)
    direction_sign : int = 0
    n : int = starting_point.size

    f_wrapper : __FunctionWrapper = __FunctionWrapper(obj_func)
    F : ObjectiveFunction = f_wrapper.eval

    # Creare una classe?
    accepted_step  : StepSizes = np.zeros(n, dtype = float64) # \alpha_k^i
    init_step      : StepSizes = np.zeros(n, dtype = float64) # \bar{\alpha}_k^i
    tentative_step : StepSizes = starting_step.copy()         # \tilde{\alpha}_k^i
    max_tentative_step : float64 = np.max(tentative_step)

    bound_coeff : float64 = __compute_bound_coeff(param)
    eta   : float64 = param.eta
    theta : float64 = param.theta

    F_y : float64 = float64(0)
    x : Point = starting_point.copy()
    y : Point = starting_point.copy()

    while max_tentative_step > LIMIT:
        new_point_found : bool = False
        prev_dir_res : __DirectionResult = __DirectionResult.POSITIVE
        np.maximum(tentative_step, eta * max_tentative_step, out = init_step)

        for i in range(n):

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

        if new_point_found:
            x[:] = y
            np.maximum(accepted_step, init_step, out = tentative_step)
        else:
            tentative_step = theta * init_step
        max_tentative_step = np.max(tentative_step)

    return x
