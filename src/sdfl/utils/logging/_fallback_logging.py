import logging
import sys

from .sdfl_logging_helper import SDFLLoggingHelper
from ..queue_handler_helper import QueueHandlerHelper

class _SDFLLogInfo:
    level: int
    msg: str
    end_msg: str

    def __init__(self: _SDFLLogInfo, level: int = 0, msg: str = "", end_msg: str = "") -> None:
        self.level = level
        self.msg = msg
        self.end_msg = end_msg

_default_info: _SDFLLogInfo = _SDFLLogInfo(
    level=logging.INFO,
    msg="x = %s\nf(x) = %g\nSteps = %s\n",
    end_msg="Result:\n\tx = %s\n\tf(x) = %g\n\tnfev = %d\n"
)

_prev_info: _SDFLLogInfo | None = None
_q_helper: QueueHandlerHelper | None = None
_running: bool = False

def use_fallback_logging(helper: SDFLLoggingHelper) -> bool:
    return len(helper.logger.handlers) == 0

def start_fallback_logging(helper: SDFLLoggingHelper) -> None:
    global _running, _prev_info, _q_helper
    if _running:
        return
    _running = True

    _prev_info = _prepare_helper(helper, _default_info)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(helper.logger.getEffectiveLevel())
    _q_helper = QueueHandlerHelper((helper.logger, handler))
    _q_helper.start()

def stop_fallback_logging(helper: SDFLLoggingHelper) -> None:
    global _running, _q_helper, _prev_info
    if not _running:
        return
    _running = False

    if _q_helper is None:
        raise RuntimeError("Queue handler helper is None.")

    _q_helper.stop_and_close()
    _q_helper = None

    if _prev_info is None:
        raise RuntimeError("No previous log info found.")

    _restore_helper(helper, _prev_info)
    _prev_info = None

def _prepare_helper(helper: SDFLLoggingHelper, info: _SDFLLogInfo) -> _SDFLLogInfo:
    prev: _SDFLLogInfo = _SDFLLogInfo(
        level = helper.logger.getEffectiveLevel(),
        msg = helper.msg,
        end_msg = helper.end_msg
    )
    
    helper.logger.setLevel(info.level)
    helper.msg = info.msg
    helper.end_msg = info.end_msg

    return prev

def _restore_helper(helper: SDFLLoggingHelper, prev_info: _SDFLLogInfo) -> None:
    helper.logger.setLevel(prev_info.level)
    helper.msg = prev_info.msg
    helper.end_msg = prev_info.end_msg
