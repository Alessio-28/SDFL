import logging
import typing
import sys

from ..sdfl.core.sdfl import SDFL, sdfl_logger
from ..sdfl.utils.queue_handler_helper import QueueHandlerHelper
from ..scripts.sdfl_data import SDFLData

logger: logging.Logger = logging.getLogger(__name__)


def _setup_logging() -> QueueHandlerHelper:
    level: int = logging.INFO

    handler: logging.StreamHandler[typing.TextIO] = logging.StreamHandler(sys.stdout)
    sdfl_handler: logging.StreamHandler[typing.TextIO] = logging.StreamHandler(
        sys.stdout
    )

    handler.setLevel(level)
    sdfl_handler.setLevel(level)

    logger.setLevel(level)
    sdfl_logger.setLevel(level)

    return QueueHandlerHelper(
        (sdfl_logger, sdfl_handler),
        (logger, handler),
    )


def run(data: SDFLData, verbose: bool = False) -> None:
    if (
        data.problem.starting_point.size != data.problem.n
        or data.starting_step.size != data.problem.n
    ):
        raise ValueError(
            f"Problem {data.problem.name} requires starting_point and starting_step of size {data.problem.n}."
        )

    q: QueueHandlerHelper | None = None
    if verbose:
        q = _setup_logging()
        q.start()
        logger.info("Problem: %s", data.problem.name)

    result = SDFL(
        data.problem.feval,
        data.problem.starting_point,
        data.max_eval,
        data.min_step,
        data.params,
        data.starting_step,
        verbose,
    )

    if verbose:
        q.stop_and_close()  # ty: ignore[unresolved-attribute]
    else:
        print(
            f"Problem: {data.problem.name}\n"
            f"Result:\n"
            f"\tx = {result.x}\n"
            f"\tf(x) = {result.f}\n"
            f"\tnfev = {result.nfev}"
        )
