import logging
from logging.handlers import QueueHandler, QueueListener
import queue

class QueueHandlerHelper:
    logger: logging.Logger
    handler: logging.Handler
    q: queue.Queue[logging.LogRecord]
    q_handler: QueueHandler
    q_listener: QueueListener

    _listening: bool

    def __init__(self: QueueHandlerHelper, logger: logging.Logger, handler: logging.Handler) -> None:
        self.logger = logger
        self.handler = handler
        self.q = queue.Queue()
        self.q_handler = QueueHandler(self.q)
        self.q_listener = QueueListener(self.q, self.handler)

        self.handler.setLevel(logger.getEffectiveLevel())
        self.q_handler.setLevel(logger.getEffectiveLevel())
        self.logger.addHandler(self.q_handler)

        self._listening = False

    def start(self: QueueHandlerHelper) -> None:
        if self._listening:
            return
        self._listening = True
        self.q_listener.start()

    def stop(self: QueueHandlerHelper) -> None:
        if not self._listening:
            return
        self._listening = False
        self.q_listener.stop()

    def close(self: QueueHandlerHelper) -> None:
        if self._listening:
            raise RuntimeError("The queue is still listening.")
        self.logger.removeHandler(self.q_handler)
        self.q_handler.close()
        self.handler.close()
        self.q.shutdown()

    def stop_and_close(self: QueueHandlerHelper) -> None:
        self.stop()
        self.close()

    def is_listening(self: QueueHandlerHelper) -> bool:
        return self._listening
