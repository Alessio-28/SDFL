from itertools import starmap
from logging import Logger, Handler, Filter, LogRecord
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from typing import override

class QueueHandlerHelper:
    q: Queue[LogRecord]
    q_handler: QueueHandler
    q_listener: QueueListener

    _q_objects: tuple[_QueueObjects, ...]

    _listening: bool

    def __init__(self: QueueHandlerHelper, *logger_handler_pairs: tuple[Logger, Handler]) -> None:
        self.q = Queue()
        self.q_handler = QueueHandler(self.q)

        def create_queue_objects(logger: Logger, handler: Handler) -> _QueueObjects:
            return _QueueObjects(logger, handler, self.q_handler)
        self._q_objects = tuple(starmap(create_queue_objects, logger_handler_pairs))

        self.q_listener = QueueListener(self.q, *[p.handler for p in self._q_objects])
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
        self.q_handler.close()
        self.q.shutdown()

        def close_objects(q_obj: _QueueObjects) -> None:
            q_obj.close()
        map(close_objects, self._q_objects)

    def stop_and_close(self: QueueHandlerHelper) -> None:
        self.stop()
        self.close()

    def is_listening(self: QueueHandlerHelper) -> bool:
        return self._listening

class _QueueFilter(Filter):
    def __init__(self: _QueueFilter, name: str = "") -> None:
        super().__init__(name)

    @override
    def filter(self: _QueueFilter, record: LogRecord) -> bool | LogRecord:
        return self.name == record.name

class _QueueObjects:
    logger: Logger
    handler: Handler
    q_handler: QueueHandler
    _filter: _QueueFilter

    def __init__(self: _QueueObjects, logger: Logger, handler: Handler, queue_handler: QueueHandler) -> None:
        self.logger = logger
        self.handler = handler
        self.q_handler = queue_handler
        self._filter = _QueueFilter(self.logger.name)

        self.logger.addHandler(self.q_handler)
        self.handler.addFilter(self._filter)

    def close(self: _QueueObjects) -> None:
        self.logger.removeHandler(self.q_handler)
        self.handler.removeFilter(self._filter)
