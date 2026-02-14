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
