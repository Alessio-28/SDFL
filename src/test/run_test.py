import logging
import sys
import typing

from ..scripts.sdfl_data import SDFLData
from ..sdfl.core.sdfl import SDFL, SDFLResult
from ..utils.queue_handler_helper import QueueHandlerHelper

logger: logging.Logger = logging.getLogger(__name__)


def _setup_logging() -> QueueHandlerHelper:
    level: int = logging.INFO

    handler: logging.StreamHandler[typing.TextIO] = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    logger.setLevel(level)

    return QueueHandlerHelper((logger, handler))


def logging_callback(res: SDFLResult) -> None:
    logger.log(
        logger.getEffectiveLevel(),
        "x = %s\nf(x) = %g\nstep = %s\nnfev = %d",
        res.x,
        res.f,
        res.step,
        res.nfev,
    )


def run(data: SDFLData, verbose: bool = False) -> None:
    if (
        data.problem.starting_point.size != data.problem.n
        or data.starting_step.size != data.problem.n
    ):
        raise ValueError(
            f"Problem {data.problem.name} requires"
            f"starting_point and starting_step of size {data.problem.n}."
        )

    q: QueueHandlerHelper | None = None
    callback = None
    if verbose:
        q = _setup_logging()
        callback = logging_callback
        q.start()
        logger.info("Problem: %s", data.problem.name)

    result: SDFLResult = SDFL(
        data.problem.feval,
        data.problem.starting_point,
        data.max_eval,
        data.min_step,
        data.params,
        data.starting_step,
        callback=callback,
    )

    if q is not None:
        q.stop_and_close()

    print(
        f"Problem: {data.problem.name}\n"
        "Result:\n"
        # f"\tx = {result.x}\n"
        # f"\tf(x) = {result.f}\n"
        # f"\tstep = {result.step}\n"
        # f"\tnfev = {result.nfev}"
        f"{result}"
    )
