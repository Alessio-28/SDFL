import logging
from typing import TextIO
import sys

from ..sdfl.core.sdfl import SDFL, sdfl_logging_helper
from ..sdfl.utils.queue_handler_helper import QueueHandlerHelper
from ..sdfl.utils.logging._fallback_logging import _default_info
from ..scripts.sdfl_data import SDFLData

logger: logging.Logger = logging.getLogger(__name__)

def _setup_logging() -> QueueHandlerHelper:
    level: int = logging.INFO

    handler: logging.StreamHandler[TextIO] = logging.StreamHandler(sys.stdout)
    sdfl_handler: logging.StreamHandler[TextIO] = logging.StreamHandler(sys.stdout)

    handler.setLevel(level)
    sdfl_handler.setLevel(level)

    logger.setLevel(level)
    sdfl_logging_helper.logger.setLevel(level)

    sdfl_logging_helper.msg = _default_info.msg
    sdfl_logging_helper.end_msg = _default_info.end_msg

    return QueueHandlerHelper((sdfl_logging_helper.logger, sdfl_handler), (logger, handler))


def run(data: SDFLData, verbose: bool = False) -> None:
    if data.problem.starting_point.size != data.problem.n or data.starting_step.size != data.problem.n:
        raise ValueError(f"Problem {data.problem.name} requires starting_point and starting_step of size {data.problem.n}.")

    q: QueueHandlerHelper | None = None
    if verbose:
        q = _setup_logging()
        q.start()
        logger.info(f"Problem: %s", data.problem.name)

    result = SDFL(data.problem.feval, data.problem.starting_point, data.max_eval, data.min_step, data.params, data.starting_step, verbose)

    if verbose:
        q.stop_and_close() # pyright: ignore[reportOptionalMemberAccess]
    else:
        print(f"Problem: {data.problem.name}\nResult:\n{result}")
