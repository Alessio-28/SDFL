import logging
import typing

from ..sdfl.core.sdfl import SDFL
from ..scripts.sdfl_data import SDFLData

logger: logging.Logger = logging.getLogger(__name__)
handler: logging.StreamHandler[typing.TextIO] = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def run(data: SDFLData, verbose: bool = False) -> None:
    if data.function.starting_point.size != data.function.n or data.starting_step.size != data.function.n:
        raise ValueError(f"Problem {data.function.name} is defined in {data.function.n} dimensions.")

    if verbose:
        logger.info(f"Function: %s", data.function.name)

    result = SDFL(data.function.feval, data.function.starting_point, data.max_eval, data.min_step, data.params, data.starting_step, verbose)

    if not verbose:
        print(f"Function: {data.function.name}\nResult:\n{result}")
