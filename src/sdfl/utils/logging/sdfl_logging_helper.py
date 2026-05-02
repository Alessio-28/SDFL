from collections.abc import Mapping

# import string
import logging

import numpy as np
import numpy.typing as npt

# class MessageConfig:
#     msg: str
#     end_msg: str

#     def __init__(self: MessageConfig, msg: str = "", end_msg: str = "") -> None:
#         self.msg = msg
#         self.end_msg = end_msg

#     @staticmethod
#     def _validate_messages(msg: str, allowed: frozenset[str]) -> None:
#         pass


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
    `log` : Same signature as `logging.Logger.log()`, except for argument `level`.
        Logs a message with `logger` at level `logger.level`.
    `log_msg` : `(ndarray[float64], float64, ndarray[float64], **kwargs) -> None`
        `**kwargs` same as `logging.Logger.log()`
        Logs `msg` with `logger` at level `logger.level`.
    `log_end_msg` : Same signature as `logging.Logger.log()`, except for arguments `level` and `msg`.
        Logs `end_msg` with `logger` at level `logger.level`.
    """

    logger: logging.Logger
    msg: str
    end_msg: str

    def __init__(
        self: SDFLLoggingHelper,
        logger: logging.Logger,
        msg: str = "",
        end_msg: str = "",
    ) -> None:
        self.logger = logger
        self.msg = msg
        self.end_msg = end_msg

    def log(
        self: SDFLLoggingHelper,
        msg: object,
        *args: object,
        exc_info: logging._ExcInfoType = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:

        self.logger.log(
            self.logger.getEffectiveLevel(),
            msg,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
        )

    def log_msg(
        self: SDFLLoggingHelper,
        x: npt.NDArray[np.float64],
        fx: np.float64,
        steps: npt.NDArray[np.float64],
        exc_info: logging._ExcInfoType = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:

        self.log(
            self.msg,
            x,
            fx,
            steps,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
        )

    def log_end_msg(
        self: SDFLLoggingHelper,
        x: npt.NDArray[np.float64],
        fx: np.float64,
        nfev: int,
        exc_info: logging._ExcInfoType = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:

        self.log(
            self.end_msg,
            x,
            fx,
            nfev,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
        )
