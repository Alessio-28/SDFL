"""Logging module for `SDFL`.

`Public classes`
--------
- `SDFLLoggingHelper`
"""
import logging
from logging.handlers import QueueHandler, QueueListener
import typing
import sys
import queue

class SDFLLoggingHelper:
    """Class used to make changes to logging messages.

    `Attributes`
    --------
    `msg` : `str`
        Log message for intermediate `SDFL` computations.
    `end_msg` : `str`
        Log message for end result of `SDFL`.

    `Public methods`
    --------
    `set_msg` : `(str) -> None`
    `set_end_msg` : `(str) -> None`
    """
    msg: str
    end_msg: str

    def __init__(self: SDFLLoggingHelper, msg: str = "", end_msg: str = "") -> None:
        self.msg = msg
        self.end_msg = end_msg

    def _log(self: SDFLLoggingHelper, logger: logging.Logger, *args: object, **kwargs: typing.Any) -> None:
        logger.log(logger.level, self.msg, *args, **kwargs)

    def _end_log(self: SDFLLoggingHelper, logger: logging.Logger, *args: object, **kwargs: typing.Any) -> None:
        logger.log(logger.level, self.msg, *args, **kwargs)

    def set_msg(self: SDFLLoggingHelper, msg: str) -> None:
        self.msg = msg

    def set_end_msg(self: SDFLLoggingHelper, end_msg: str) -> None:
        self.end_msg = end_msg

class _SDFLDefaultLogging:
    _handler: logging.StreamHandler[typing.TextIO] | None = None
    _q_handler: QueueHandler | None = None
    _q_listener: QueueListener | None = None
    _logging_level: int = logging.INFO
    _msg: str = "x = %s\nf(x) = %g\nSteps = %s\n"
    _end_msg: str = "Result:\nx = %s\nf(x) = %g\nnfev = %d\n"

    _default_logging_running: bool = False

    @classmethod
    def start_default_logging(cls, logger: logging.Logger, helper: SDFLLoggingHelper) -> None:
        if len(logger.handlers) > 0:
            return

        cls._default_logging_running = True

        logger.setLevel(cls._logging_level)
        helper.set_msg(cls._msg)
        helper.set_end_msg(cls._end_msg)

        q: queue.Queue[logging.LogRecord] = queue.Queue()
        cls._q_handler = QueueHandler(q)
        cls._q_handler.setLevel(cls._logging_level)

        cls._handler = logging.StreamHandler(sys.stdout)
        cls._handler.setLevel(cls._logging_level)
        cls._q_listener = QueueListener(q, cls._handler)

        logger.addHandler(cls._q_handler)

        cls._q_listener.start()

    @classmethod
    def stop_default_logging(cls, logger: logging.Logger, helper: SDFLLoggingHelper) -> None:
        if not cls._default_logging_running:
            return

        if cls._q_listener is not None:
            cls._q_listener.stop()
            cls._q_listener = None

        if cls._q_handler is not None:
            logger.removeHandler(cls._q_handler)
            cls._q_handler = None

        if cls._handler is not None:
            cls._handler = None

        logger.setLevel(logging.NOTSET)
        helper.set_msg("")
        helper.set_end_msg("")
