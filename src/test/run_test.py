import logging
from typing import TextIO

from ..sdfl.core.sdfl import SDFL
from ..scripts.sdfl_data import SDFLData

logger: logging.Logger = logging.getLogger(__name__)
handler: logging.StreamHandler[TextIO] = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def run(data: SDFLData, verbose: bool = False) -> None:
    if data.problem.starting_point.size != data.problem.n or data.starting_step.size != data.problem.n:
        raise ValueError(f"Problem {data.problem.name} requires {data.problem.n}-dimensional starting point and starting step values.")

    if verbose:
        logger.info(f"Problem: %s", data.problem.name)

    result = SDFL(data.problem.feval, data.problem.starting_point, data.max_eval, data.min_step, data.params, data.starting_step, verbose)

    if not verbose:
        print(f"Problem: {data.problem.name}\nResult:\n{result}")
