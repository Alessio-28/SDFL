import numpy as np
import numpy.typing as npt

from ..test import problems
from ..sdfl.core import parameters, sdfl

class SDFLData:
    function: problems.Problem
    starting_step: npt.NDArray[np.float64]
    max_eval: int
    min_step: np.float64
    params: parameters.Parameters

    def __init__(self: SDFLData, function: problems.Problem, max_eval: int, min_step: np.float64, params: parameters.Parameters, starting_step: npt.NDArray[np.float64] | None = None) -> None:
        sdfl._validate_sdfl_args(function.starting_point, max_eval, min_step, starting_step)
        self.function = function
        self.max_eval = max_eval
        self.min_step = min_step
        self.params = params
        if starting_step is None:
            self.starting_step = np.ones(function.n, dtype=np.float64)
        else:
            self.starting_step = starting_step
