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

    def compute(self : __FunctionWrapper, x : Point) -> float64:
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

def __compute_bound_coeff(param : Parameters) -> Callable[[float64], float64]:
    coeff : float64 = -param.gamma * param.c * param.epsilon

    def bound(step : float64) -> float64:
        return coeff * (step ** 2)

    return bound

def __compute_direction(F : __FunctionWrapper, y : Point, step_size : float64, index : int, bound : float64) -> Tuple[__DirectionResult, float64]:
    y_i : float64 = y[index]
    F_bound : float64 = F.compute(y) + bound

    y[index] += step_size
    F_dir : float64 = F.compute(y)
    if F_dir > F_bound:
        y[index] = y_i - step_size
        F_dir = F.compute(y)

        y[index] = y_i
        if F_dir > F_bound:
            return (__DirectionResult.FAILURE, F_dir)
        else:
            return (__DirectionResult.NEGATIVE, F_dir)

    y[index] = y_i
    return (__DirectionResult.POSITIVE, F_dir)

def __line_search(F : __FunctionWrapper, y : Point, step_size : float64, index : int, direction_sign : int, F_dir : float64, bound : Callable[[float64], float64]) -> float64:
    y_i : float64 = y[index]

    F_a : float64 = F_dir
    y[index] = y_i + 2 * step_size * direction_sign
    F_b : float64 = F.compute(y)
    while F_b - F_a <= bound(step_size):
        step_size *= 2
        y[index] = y_i + 2 * step_size * direction_sign

        F_a = F_b
        F_b = F.compute(y)

    y[index] = y_i + step_size * direction_sign
    return step_size


def SDFL(obj_func : ObjectiveFunction, x_0 : Point, step_0 : StepSizes, param : Parameters) -> Point:
    LIMIT : float64 = float64(1e-6)
    direction_sign : int = 0
    n : int = x_0.size

    F : __FunctionWrapper = __FunctionWrapper(obj_func)

    # Creare una classe?
    accepted_step  : StepSizes = np.zeros(n, dtype = float64) # \alpha_k^i
    init_step      : StepSizes = np.zeros(n, dtype = float64) # \bar{\alpha}_k^i
    tentative_step : StepSizes = step_0.copy()                # \tilde{\alpha}_k^i

    bound : Callable[[float64], float64] = __compute_bound_coeff(param)
    max_tentative_step : float64 = np.max(tentative_step)
    x : Point = x_0.copy()
    y : Point = x_0.copy()
    while max_tentative_step > LIMIT:
        success : bool = False
        init_step = np.maximum(tentative_step, param.eta * max_tentative_step)
        for i in range(n):

            # Direction expansion
            dir_res : __DirectionResult
            F_dir : float64
            (dir_res , F_dir) = __compute_direction(F, y, init_step[i], i, bound(init_step[i]))

            if dir_res == __DirectionResult.FAILURE:
                accepted_step[i] = 0
            else:
                # Line search
                direction_sign = dir_res.value
                accepted_step[i] = __line_search(F, y, init_step[i], i, direction_sign, F_dir, bound)
                success = True

        if success:
            x = y.copy()
            tentative_step = np.maximum(accepted_step, init_step)
        else:
            tentative_step = param.theta * init_step
        max_tentative_step = np.max(tentative_step)

    return x
