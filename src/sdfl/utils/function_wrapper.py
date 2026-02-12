import numpy as np
from ..core.typing import Point, ObjectiveFunction

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
        _obj_fun getter.
    `get_nfev` : `() -> int`
        _nfev getter.
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

        Evaluates the objective function at `x`
        and increases the evaluations counter by `1`.
        Raises `RuntimeError` if the result of evaluation is `NaN` or `inf`.

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
        res: np.float64 = self._obj_fun(x)

        if not np.isfinite(res):
            if np.isnan(res):
                raise RuntimeError(f"Evaluation of objective function at {x} results in NaN. Evaluation {self._nfev}")
            elif np.isposinf(res):
                raise RuntimeError(f"Evaluation of objective function at {x} results in +inf. Evaluation {self._nfev}")
            elif np.isneginf(res):
                raise RuntimeError(f"Evaluation of objective function at {x} results in -inf. Evaluation {self._nfev}")

        return res

    def get_obj_fun(self: _FunctionWrapper) -> ObjectiveFunction:
        """_obj_fun getter."""
        return self._obj_fun

    def get_nfev(self: _FunctionWrapper) -> int:
        """_nfev getter."""
        return self._nfev
