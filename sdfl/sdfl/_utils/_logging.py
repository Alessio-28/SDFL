import logging
from logging.handlers import QueueHandler, QueueListener
import typing
import sys
import queue

_handler: logging.StreamHandler[typing.TextIO] | None = None
_q_handler: QueueHandler | None = None
_q_listener: QueueListener | None = None

def _enable_default_logging(logger: logging.Logger) -> None:
    if logger.hasHandlers() and (len(logger.handlers) == 0):
        global _handler
        global _q_handler
        global _q_listener
        logger.setLevel(logging.INFO)

        q: queue.Queue[logging.LogRecord] = queue.Queue(-1)
        _q_handler = QueueHandler(q)
        _q_handler.setLevel(logging.INFO)

        _handler = logging.StreamHandler(sys.stdout)
        _handler.setLevel(logging.INFO)
        _q_listener = QueueListener(q, _handler)

        logger.addHandler(_q_handler)

        _q_listener.start()

def _disable_default_logging(logger: logging.Logger) -> None:
    global _q_handler
    if _q_handler:
        global _handler
        global _q_listener
        _q_listener.stop() # pyright: ignore[reportOptionalMemberAccess]
        logger.removeHandler(_q_handler)
        _handler, _q_handler, _q_listener = None, None, None
