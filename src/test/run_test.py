import logging
import typing

from ..sdfl.core.sdfl import SDFL
from ..sdfl.core.typing import ObjectiveFunction
from ..scripts.sdfl_data_manager import SDFLData
from . import problems

logger: logging.Logger = logging.getLogger(__name__)
handler: logging.StreamHandler[typing.TextIO] = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def setup_tests_and_run(functions: list[str], data: SDFLData, verbose: bool = False) -> None:
    prob_collection: dict[str, problems.Problem] = problems.set_problems(functions)

    for prob in functions:
        if data.starting_point.size != prob_collection[prob].n or data.starting_step.size != prob_collection[prob].n:
            raise ValueError(f"Problem {prob} is defined in {prob_collection[prob].n} dimensions")
        run(
            prob_collection[prob].name,
            prob_collection[prob].feval,
            data,
            verbose
        )

def run(name: str, obj_fun: ObjectiveFunction, data: SDFLData, verbose: bool = False) -> None:
    if verbose:
        logger.info(f"Function: %s", name)

    result = SDFL(obj_fun, data.starting_point, data.starting_step, data.params, data.max_eval, data.min_step, verbose)

    if not verbose:
        print(f"Function: {name}\nResult:\n{result}")
