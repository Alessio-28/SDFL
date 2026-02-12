"""Logging module for `SDFL`.

`Public classes`
--------
- `QueueHandlerHelper`
- `SDFLLoggingHelper`
"""
from collections.abc import Mapping
import logging
from logging.handlers import QueueHandler, QueueListener
import typing
import sys
import queue


class SDFLLoggingHelper:
    """Helper class for logging.

    `Attributes`
    --------
    `logger` : `logging.Logger`
    `msg` : `str`
        Log message for intermediate `SDFL` computations.
    `end_msg` : `str`
        Log message for end result of `SDFL`.

    `Public methods`
    --------
    `log` : Same signature as `logging.Logger.log()`, except for the argument `level`.
        Logs a message with `logger` at level `logger.level`.
    `log_msg` : Same signature as `logging.Logger.log()`, except for arguments `level` and `msg`.
        Logs `msg` with `logger` at level `logger.level`.
    `log_end_msg` : Same signature as `logging.Logger.log()`, except for arguments `level` and `msg`.
        Logs `end_msg` with `logger` at level `logger.level`.
    `get_logger` : `() -> logging.Logger`
    `get_msg` : `() -> str`
    `get_end_msg` : `() -> str`
    `set_msg` : `(str) -> None`
    `set_end_msg` : `(str) -> None`
    """
    logger: logging.Logger
    msg: str
    end_msg: str

    def __init__(self: SDFLLoggingHelper, logger: logging.Logger, msg: str = "", end_msg: str = "") -> None:
        self.logger = logger
        self.msg = msg
        self.end_msg = end_msg

    def log(self: SDFLLoggingHelper, msg: object, *args: object, exc_info: logging._ExcInfoType = None, stack_info: bool = False, stacklevel: int = 1, extra: Mapping[str, object] | None = None) -> None:
        self.logger.log(
            self.logger.level,
            msg=msg,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra
        )

    def log_msg(self: SDFLLoggingHelper, *args: object, exc_info: logging._ExcInfoType = None, stack_info: bool = False, stacklevel: int = 1, extra: Mapping[str, object] | None = None) -> None:
        self.log(
            msg=self.msg,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra
        )

    def log_end_msg(self: SDFLLoggingHelper, *args: object, exc_info: logging._ExcInfoType = None, stack_info: bool = False, stacklevel: int = 1, extra: Mapping[str, object] | None = None) -> None:
        self.log(
            msg=self.end_msg,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra
        )

    def get_logger(self: SDFLLoggingHelper) -> logging.Logger:
        return self.logger

    def get_msg(self: SDFLLoggingHelper) -> str:
        return self.msg

    def get_end_msg(self: SDFLLoggingHelper) -> str:
        return self.end_msg

    def set_msg(self: SDFLLoggingHelper, msg: str) -> None:
        self.msg = msg

    def set_end_msg(self: SDFLLoggingHelper, end_msg: str) -> None:
        self.end_msg = end_msg

class QueueHandlerHelper:
    _q: queue.Queue[logging.LogRecord]
    _handler: logging.Handler
    _q_handler: QueueHandler
    _q_listener: QueueListener
    _logger: logging.Logger

    def __init__(self: QueueHandlerHelper, logging_level: int, handler: logging.Handler, logger: logging.Logger) -> None:
        self._q = queue.Queue()

        self._handler = handler
        self._q_handler = QueueHandler(self._q)
        self._q_listener = QueueListener(self._q, self._handler)
        self._logger = logger
        
        self._handler.setLevel(logging_level)
        self._q_handler.setLevel(logging_level)
        self._logger.setLevel(logging_level)
        self._logger.addHandler(self._q_handler)

    def start(self: QueueHandlerHelper) -> None:
        self._q_listener.start()

    def stop(self: QueueHandlerHelper) -> None:
        self._q_listener.stop()

    def close(self: QueueHandlerHelper) -> None:
        self._logger.removeHandler(self._q_handler)
        self._q_handler.close()
        self._handler.close()
        self._q.shutdown()

class _SDFLFallbackLogging:
    _prev_log_level: int = 0
    _prev_msg: str = ""
    _prev_end_msg: str = ""

    _q_helper: QueueHandlerHelper | None = None

    _msg: str = "x = %s\nf(x) = %g\nSteps = %s\n"
    _end_msg: str = "Result:\nx = %s\nf(x) = %g\nnfev = %d\n"
    _logging_level: int = logging.INFO
    _handler: logging.StreamHandler[typing.TextIO] = logging.StreamHandler(sys.stdout)

    @classmethod
    def use_fallback_logging(cls, helper: SDFLLoggingHelper) -> bool:
        return len(helper.get_logger().handlers) == 0

    @classmethod
    def start_fallback_logging(cls, helper: SDFLLoggingHelper) -> None:
        cls._prepare_helper(helper)
        cls._q_helper = QueueHandlerHelper(cls._logging_level, cls._handler, helper.get_logger())
        cls._q_helper.start()

    @classmethod
    def stop_fallback_logging(cls, helper: SDFLLoggingHelper) -> None:
        if cls._q_helper is None:
            return

        cls._q_helper.stop()
        cls._q_helper.close()

        cls._q_helper = None
        cls._restore_helper(helper)

    @classmethod
    def _prepare_helper(cls, helper: SDFLLoggingHelper) -> None:
        cls._prev_log_level = helper.get_logger().level
        cls._prev_msg = helper.get_msg()
        cls._prev_end_msg = helper.get_end_msg()
        
        helper.get_logger().setLevel(cls._logging_level)
        helper.set_msg(cls._msg)
        helper.set_end_msg(cls._end_msg)

    @classmethod
    def _restore_helper(cls, helper: SDFLLoggingHelper) -> None:
        helper.get_logger().setLevel(cls._prev_log_level)
        helper.set_msg(cls._prev_msg)
        helper.set_end_msg(cls._prev_end_msg)
        
        cls._prev_log_level = 0
        cls._prev_msg = ""
        cls._prev_end_msg = ""
