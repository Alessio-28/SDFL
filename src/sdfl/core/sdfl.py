import numpy as np
import numpy.typing as npt
from enum import Enum
from typing import override
import logging

from .typing import Point, ObjectiveFunction
from .parameters import Parameters
from .._utils._logging import _start_default_logging, _stop_default_logging

sdfl_logger: logging.Logger = logging.getLogger(__name__)
"""Logger of `SDFL`

Initialised at import. Its parent has `NullHandler` attached.
Logging level for `SDFL`: `INFO`.

If `SDFL` has `verbose == True` and no other handler is attached to `sdfl_logger`,
`StreamHandler` is attached lazily using `QueueHandler` and `QueueListener`
and it is detached after the algorithm terminates.
"""

def SDFL(obj_fun: ObjectiveFunction, starting_point: Point, starting_step: npt.NDArray[np.float64], params: Parameters, max_eval: int, min_step: np.float64, verbose: bool = False) -> SDFLResult:
    """Stochastic Derivative-Free Linesearch-based algorithm.

    Implementation of SDFL algorithm from:
    `https://arxiv.org/abs/2508.00495v1`

    `Arguments`
    --------
    `obj_fun` : `ObjectiveFunction`
        Objective function.
    `starting_point` : `Point`
        Starting point of the algorithm.
        It must be a one dimensional array.
         `starting_point.size` must be equal to `starting_step.size`.
    `starting_step` : `ndarray[float64]`
        List of step values for the first iteration of the algorithm.
        It must be a one dimensional array.
         `starting_step.size` must be equal to `starting_point.size`.
    `params` : `Parameters`
    `max_eval` : `int`
        Maximum number of evaluations of the objective function.
    `min_step` : `float64`
        Minimum value of maximum of steps.
    `verbose` : `bool` (default: `False)`
        Toggles logging of intermediate and end calculations.

    `Return`
    --------
    `result` : `SDFLResult`
        Contains the result of the algorithm.
    """
    _validate_sdfl_args(starting_point, starting_step, max_eval, min_step)

    n: int = starting_point.size

    f_wrapper: _FunctionWrapper = _FunctionWrapper(obj_fun)
    F: ObjectiveFunction = f_wrapper.eval

    # init_step: npt.NDArray[np.float64] = np.zeros(n, dtype=np.float64) # \bar{\alpha}
    accepted_step: npt.NDArray[np.float64] = np.zeros(n, dtype=np.float64) # \alpha
    tentative_step: npt.NDArray[np.float64] = starting_step.copy()         # \tilde{\alpha}
    max_tentative_step: np.float64 = np.max(tentative_step)

    eta: np.float64 = params.eta
    theta: np.float64 = params.theta

    current_point: Point = starting_point.copy()
    fun_eval_at_cur_point: np.float64 = F(current_point)
    prev_dir_res: _DirectionResult = _DirectionResult.FAILURE

    if verbose:
        _start_default_logging(sdfl_logger)
        sdfl_logger.info("x = %s\nF(x) = %g\nStep = %s\n", current_point, fun_eval_at_cur_point, tentative_step)

    try:
        while f_wrapper.get_nfev() < max_eval and max_tentative_step >= min_step:
            new_point_found: bool = False
            np.maximum(tentative_step, eta * max_tentative_step, out=tentative_step)

            for i in range(n):
                if prev_dir_res != _DirectionResult.FAILURE:
                    fun_eval_at_cur_point = F(current_point)

                step: np.float64 = tentative_step[i]
                bound: np.float64 = params.compute_bound(step)

                dir_res, fun_eval_at_direction = _compute_direction(F, current_point, fun_eval_at_cur_point, step, i, bound)

                if dir_res == _DirectionResult.FAILURE:
                    accepted_step[i] = 0
                else:
                    accepted_step[i] = _line_search(F, current_point, fun_eval_at_direction, dir_res.value, step, i, bound)
                    new_point_found = True

                prev_dir_res = dir_res

            if verbose:
                sdfl_logger.info("x = %s\nF(x) = %g\nStep = %s\n", current_point, fun_eval_at_cur_point, tentative_step)

            if new_point_found:
                np.maximum(accepted_step, tentative_step, out = tentative_step)
            else:
                tentative_step *= theta
            max_tentative_step = np.max(tentative_step)

        result: SDFLResult = SDFLResult(current_point, fun_eval_at_cur_point, f_wrapper._nfev)

        if verbose:
            sdfl_logger.info("Result:\n%s\n", result)
    finally:
        _stop_default_logging(sdfl_logger)

    return result

def _compute_direction(obj_fun: ObjectiveFunction, point: Point, fun_eval_at_point: np.float64, step_size: np.float64, index: int, bound: np.float64) -> tuple[_DirectionResult, np.float64]:
    elem: np.float64 = point[index]
    F_bound: np.float64 = fun_eval_at_point + bound

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

def _line_search(obj_fun: ObjectiveFunction, point: Point, fun_eval_at_point: np.float64, direction_sign: int, step_size: np.float64, index: int, bound: np.float64) -> np.float64:
    elem: np.float64 = point[index]
    step: np.float64 = step_size * direction_sign
    step_aux: np.float64 = step * 2

    iter2: int = 1

    F_a: np.float64 = fun_eval_at_point
    point[index] = elem + step_aux
    F_b: np.float64 = obj_fun(point)
    while F_b - F_a <= bound * iter2 * iter2:
        iter2 *= 2
        point[index] = elem + iter2*step_aux

        F_a, F_b = F_b, obj_fun(point)

    # Correct point
    point[index] = elem + iter2*step

    return step_size * iter2

class _FunctionWrapper:
    """Objective function wrapper, counts function evaluations.

    `Attributes`
    --------
    `_obj_fun` : `ObjectiveFunction`
        Objecive function.
    `_nfev` : `int`
        Counter of objective function evaluations.
        Gets initialised to `0` by the constructor.

    `Methods`
    --------
    `eval` : `(Point) -> float64`
        Evaluates the objective function at the given point.
        Increases the evaluation counter by `1`.
    `get_obj_fun` : `() -> ObjectiveFunction`
    `get_nfev` : `() -> int`
    """

    _obj_fun: ObjectiveFunction
    _nfev: int

    def __init__(self: _FunctionWrapper, obj_fun: ObjectiveFunction) -> None:
        """Initialises the wrapper and sets the counter to `0`.

        Arguments
        --------
        `obj_fun` : `ObjectiveFunction`
            Function to assign to the wrapper.
        """
        self._obj_fun = obj_fun
        self._nfev = 0

    def eval(self: _FunctionWrapper, x: Point) -> np.float64:
        """Evaluates the objective function.

        It evaluates the objective function at `x`
        and increases the evaluations counter by `1`.

        `Arguments`
        --------
        `x` : `Point`
            The point at which the objective function gets evaluated.

        `Return`
        --------
        `result` : `float64`
            The result of the evaluation.
        """
        self._nfev += 1
        return self._obj_fun(x)

    def get_obj_fun(self: _FunctionWrapper) -> ObjectiveFunction:
        return self._obj_fun

    def get_nfev(self: _FunctionWrapper) -> int:
        return self._nfev

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
