"""Main `SDFL` module.

See: `https://arxiv.org/abs/2508.00495v1`

`Public functions`
--------
- `SDFL`: Stochastic Derivative-Free Linesearch-based algorithm.

`Public classes`
--------
- `SDFLResult`

`Public objects`
--------
- `sdfl_logger`: logger for `SDFL` function.
- `sdfl_logging_helper`: sets log messages format.

`Logging`
--------
This module has a fallback logging utility
but other ways of logging can be defined using:
- `sdfl_logging_helper`
"""
import numpy as np
import numpy.typing as npt
from enum import Enum
from typing import override
from logging import getLogger

from .typing import Point, ObjectiveFunction
from .parameters import Parameters
from ..utils.logging import _fallback_logging as fl
from ..utils.logging.sdfl_logging_helper import SDFLLoggingHelper

sdfl_logging_helper: SDFLLoggingHelper = SDFLLoggingHelper(getLogger(__name__))
"""Logger helper for `SDFL`.

Initialised at import.
`sdfl_logging_helper` has this module's logger as an attribute.
Its parent has `NullHandler` attached.
--------
Logging level: `INFO`.
If `SDFL` argument `verbose == True` and no other handler is attached to the logger,
`StreamHandler` will be attached using `QueueHandler` and `QueueListener`
and will be detached after the algorithm terminates.
--------
Intermediate logging messages contain (in this order):
the `current minimum` point, the `objective function` evaluated
at the current minimum point, and the `current step values`.
The last logging message contains (in this order):
the `minimum` point, the `objective function` evaluated
at the minimum point, and the number of evaluations.
--------
To change the message format refer to class `SDFLLoggingHelper`.
At import attributes `msg = ''` and `end_msg = ''`.
"""

def SDFL(obj_fun: ObjectiveFunction, starting_point: Point, max_eval: int, min_step: np.float64, params: Parameters, starting_step: npt.NDArray[np.float64] | None = None, verbose: bool = False) -> SDFLResult:
    """Stochastic Derivative-Free Linesearch-based algorithm.

    Implementation of `SDFL` algorithm from `https://arxiv.org/abs/2508.00495v1`

    If preconditions are not met, a `ValueError` is raised.
    `FloatingPointError` is raise according to `numpy.seterr(all='raise', under='ignore')`.

    `Arguments`
    --------
    `obj_fun` : `ObjectiveFunction`
        Objective function.
    `starting_point` : `Point`
        Starting point of the algorithm.
        Preconditions:
            It must be a one dimensional array.
            If `starting_step is not None`,
            then `starting_point.size` must be equal to `starting_step.size`.
    `params` : `Parameters`
        See `Parameters` class.
    `max_eval` : `int`
        Maximum number of evaluations of the objective function.
        Precondition: `max_eval > 0`.
    `min_step` : `float64`
        Minimum value of maximum of steps.
        Precondition: `min_step > 0`.
    `starting_step` : `ndarray[float64] | None` (default: `None`)
        List of step values for the first iteration of the algorithm.
        If `starting_step is None`, it gets initialised appropriately as an array of `1`s.
        Preconditions:
            If `starting_step is not None`,
            then it must be a one dimensional array and
            `starting_step.size` must be equal to `starting_point.size`.
    `verbose` : `bool` (default: `False`)
        Toggles logging of intermediate and end results.

    `Return`
    --------
    `result` : `SDFLResult`
        Contains the result of the algorithm.
    """
    _validate_sdfl_args(starting_point, max_eval, min_step, starting_step)

    n: int = starting_point.size
    if starting_step is None:
        starting_step = np.ones(n, dtype=np.float64)

    f_wrapper: _FunctionWrapper = _FunctionWrapper(obj_fun)
    F: ObjectiveFunction = f_wrapper.eval

    # init_step: npt.NDArray[np.float64] = np.zeros(n, dtype=np.float64)     # \bar{\alpha}
    accepted_step: npt.NDArray[np.float64] = np.zeros(n, dtype=np.float64) # \alpha
    tentative_step: npt.NDArray[np.float64] = starting_step.copy()         # \tilde{\alpha}
    max_tentative_step: np.float64 = np.max(tentative_step)

    eta: np.float64 = params.eta
    theta: np.float64 = params.theta

    current_point: Point = starting_point.copy()
    prev_dir_res: _DirectionResult = _DirectionResult.FAILURE

    olderr = np.seterr(all="raise", under="ignore")
    try:
        fun_eval_at_cur_point: np.float64 = F(current_point)

        if verbose and fl.use_fallback_logging(sdfl_logging_helper):
            fl.start_fallback_logging(sdfl_logging_helper)

        while f_wrapper.nfev < max_eval and max_tentative_step >= min_step:
            new_point_found: bool = False
            np.maximum(tentative_step, eta * max_tentative_step, out=tentative_step)

            if verbose:
                sdfl_logging_helper.log_msg(current_point, fun_eval_at_cur_point, tentative_step)

            for i in range(n):
                if prev_dir_res is not _DirectionResult.FAILURE:
                    fun_eval_at_cur_point = F(current_point)

                step: np.float64 = tentative_step[i]
                bound: np.float64 = params.compute_bound(step)

                dir_res, fun_eval_at_direction = _choose_direction(F, current_point, fun_eval_at_cur_point, step, i, bound)

                if dir_res is _DirectionResult.FAILURE:
                    accepted_step[i] = 0
                else:
                    current_point[i], accepted_step[i] = _line_search(F, current_point, fun_eval_at_direction, dir_res.value, step, i, bound)
                    new_point_found = True

                prev_dir_res = dir_res

            if new_point_found:
                np.maximum(accepted_step, tentative_step, out=tentative_step)
            else:
                tentative_step *= theta
            max_tentative_step = np.max(tentative_step)

        result: SDFLResult = SDFLResult(current_point, fun_eval_at_cur_point, f_wrapper.nfev)

        if verbose:
                sdfl_logging_helper.log_end_msg(result.x, result.f, result.nfev)
    finally:
        fl.stop_fallback_logging(sdfl_logging_helper)
        np.seterr(**olderr)

    return result

def _choose_direction(obj_fun: ObjectiveFunction, point: Point, fun_eval_at_point: np.float64, step_size: np.float64, axis: int, bound: np.float64) -> tuple[_DirectionResult, np.float64]:
    elem: np.float64 = point[axis]
    F_bound: np.float64 = fun_eval_at_point + bound
    dir_res: _DirectionResult = _DirectionResult.POSITIVE

    # Try POSITIVE direction
    point[axis] = elem + step_size
    fun_eval_at_direction: np.float64 = obj_fun(point)

    if fun_eval_at_direction > F_bound:
        # Try NEGATIVE direction
        point[axis] = elem - step_size
        fun_eval_at_direction = obj_fun(point)

        if fun_eval_at_direction > F_bound:
            dir_res = _DirectionResult.FAILURE 
        else:
            dir_res = _DirectionResult.NEGATIVE 

    # Restore changes
    point[axis] = elem
    return (dir_res, fun_eval_at_direction)

def _line_search(obj_fun: ObjectiveFunction, point: Point, fun_eval_at_point: np.float64, direction_sign: int, step_size: np.float64, axis: int, bound: np.float64) -> tuple[np.float64, np.float64]:
    elem: np.float64 = point[axis]
    step: np.float64 = step_size * direction_sign
    step_2: np.float64 = step * 2

    power2: int = 1
    prev_value: np.float64 = elem + step
    point[axis] = elem + step_2

    F_a: np.float64 = fun_eval_at_point
    F_b: np.float64 = obj_fun(point)
    while F_b - F_a <= bound * power2 * power2:
        power2 *= 2
        prev_value, point[axis] = point[axis], elem + power2*step_2
        F_a, F_b = F_b, obj_fun(point)

    # Restore changes
    point[axis] = elem
    return prev_value, step_size * power2 

class SDFLResult:
    """Contains the result of `SDFL`.

    `Attributes`
    --------
    `x` : `Point`
        Minimum found by `SDFL`.
    `f` : `float64`
        Objective function evaluated at `x`.
    `nfev` : `int`
        How many objective function evaluations have been computed.
    """

    x: Point
    f: np.float64
    nfev: int

    def __init__(self: SDFLResult, x: Point, f: np.float64, nfev: int) -> None:
        self.x = x
        self.f = f
        self.nfev = nfev

    @override
    def __str__(self: SDFLResult) -> str:
        return (f"x = {self.x}\n"
                f"f(x) = {self.f}\n"
                f"nfev = {self.nfev}\n")

def _validate_sdfl_args(starting_point: Point, max_eval: int, min_step: np.float64, starting_step: npt.NDArray[np.float64] | None = None) -> None:
    """Precondition checks for `SDFL`."""

    if not isinstance(starting_point, np.ndarray):
        raise ValueError("starting_point must be a ndarray.")
    if len(starting_point.shape) != 1:
        raise ValueError("starting_point must be a 1-dimensional array.")

    if starting_step is not None:
        if not isinstance(starting_step, np.ndarray):
            raise ValueError("starting_step must be a ndarray.")
        if len(starting_step.shape) != 1:
            raise ValueError("starting_step must be a 1-dimensional array.")
        if starting_point.size != starting_step.size:
            raise ValueError("starting_point and starting_step must have the same size.")
        if np.any(starting_step <= 0):
            raise ValueError("starting_step must be an array of positive real numbers.")

    if max_eval <= 0:
        raise ValueError("max_eval must be a positive integer.")
    if min_step <= 0:
        raise ValueError("min_step must be a positive real number.")

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
    """

    obj_fun: ObjectiveFunction
    nfev: int

    def __init__(self: _FunctionWrapper, obj_fun: ObjectiveFunction) -> None:
        """Initialises the wrapper and sets the counter to `0`.

        Arguments
        --------
        `obj_fun` : `ObjectiveFunction`
            Function to assign to the wrapper.
        """

        self.obj_fun = obj_fun
        self.nfev = 0

    def eval(self: _FunctionWrapper, x: Point) -> np.float64:
        """Evaluates the objective function.

        Evaluates the objective function at `x`
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

        self.nfev += 1
        return self.obj_fun(x)

class _DirectionResult(Enum):
    """Result of `_choose_direction()`.

    `Values`
    --------
    `POSITIVE` = `1`
        Result of `_choose_direction()` found on positive direction along the axis.
    `NEGATIVE` = `-1`
        Result of `_choose_direction()` found on negative direction along the axis.
    `FAiLURE` = `0`
        `_choose_direction()` terminated without finding a suitable result.
    """

    POSITIVE =  1
    NEGATIVE = -1
    FAILURE  =  0
