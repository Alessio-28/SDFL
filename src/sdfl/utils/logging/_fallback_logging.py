import logging
import sys

from ..queue_handler_helper import QueueHandlerHelper

_logging_level: int = logging.INFO

_q_helper: QueueHandlerHelper | None = None
_running: bool = False
_prev_logging_level: int | None = None


def use_fallback_logging(logger: logging.Logger) -> bool:
    return len(logger.handlers) == 0


def start_fallback_logging(logger: logging.Logger) -> None:
    global _running, _q_helper
    if _running:
        return
    _running = True

    _prepare_logger(logger)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logger.getEffectiveLevel())

    _q_helper = QueueHandlerHelper((logger, handler))
    _q_helper.start()


def stop_fallback_logging(logger: logging.Logger) -> None:
    global _running, _q_helper
    if not _running:
        return
    _running = False

    if _q_helper is None:
        raise RuntimeError("Queue handler helper is None.")

    _q_helper.stop_and_close()
    _q_helper = None

    _restore_helper(logger)


def _prepare_logger(logger: logging.Logger) -> None:
    global _prev_logging_level
    _prev_logging_level = logger.getEffectiveLevel()

    logger.setLevel(_logging_level)


def _restore_helper(logger: logging.Logger) -> None:
    global _prev_logging_level
    if _prev_logging_level is None:
        raise RuntimeError("No previous log info found.")

    logger.setLevel(_prev_logging_level)
    _prev_logging_level = None
