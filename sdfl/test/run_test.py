import numpy as np
import numpy.typing as npt

from ..sdfl.core.sdfl import SDFL
from ..sdfl.core.parameters import Parameters
from ..sdfl.core.typing import Point, ObjectiveFunction
from . import problems

import logging

test_logger: logging.Logger = logging.getLogger(__name__)


def setup_tests_and_run(functions: list[str], starting_point: Point, starting_step: npt.NDArray[np.float64], param: Parameters, limit_eval: int = LIMIT_EVAL, limit_step: np.float64 = LIMIT_STEP, log: bool = False) -> None:
    prob_collection: dict[str, problems.Problem] = problems.set_problems(functions)

    for prob in functions:
        run(
            prob_collection[prob].name,
            prob_collection[prob].feval,
            starting_point,
            starting_step,
            param,
            limit_eval,
            limit_step,
            log
        )

def run(name: str, obj_fun: ObjectiveFunction, starting_point: Point, starting_step: npt.NDArray[np.float64], param: Parameters, limit_eval: int, limit_step: np.float64, log : bool = False) -> None:
    if log:
        test_logger.info(f"Function: %s", name)
        print(f"Function: {name}")

    result = SDFL(obj_fun, starting_point, starting_step, param, limit_eval, limit_step, log)
