from itertools import starmap
from logging import Logger, Handler, Filter, LogRecord
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from typing import override


class QueueHandlerHelper:
    """Handles `QueueHandler` and `QueueListener`.

    `QueueHandlerHelper` can handle multiple loggers and handlers.
    The constructor accepts pairs made of `(Logger, Handler)`.

    `Handler` gets attached to `Logger` when calling the constructor
    and then detached when calling `close()`.
    `QueueHandlerHelper` does not set any logging level for `Logger` and `Handler`.
    A `Handler` will log only messages received from the `Logger` it was paired with.

    `Methods`
    --------
    `start` : `() -> None`
        Starts `QueueListener`.
        If `QueueHandlerHelper` has already been closed, it raises a `RuntimeError`.
        If `start()` is called again without calling `stop()` first,
        it terminates without performing other operations.
    `stop` : `() -> None`
        Stops `QueueListener`.
        If `QueueHandlerHelper` has already been closed, it raises a `RuntimeError`.
        If `stop()` is called without calling `start()` first,
        it terminates without performing other operations.
    `close` : `() -> None`
        Calls `close()` on `QueueHandler`, `shutdown()` on `Queue`,
        and detaches `Handlers` from `Loggers`.
        If `QueueHandlerHelper` has already been closed, it raises a `RuntimeError`.
        If `close()` is called without calling `stop()` first,
        it raises a `RuntimeError`.
    `stop_and_close` : `() -> None`
        Calls both `stop()` and `close()`.
    `is_listening` : `() -> bool`
        Returns `True` if `start()` was called.
        Returns `False` otherwise.
    `is_closed` : `() -> bool`
        Returns `True` if `close()` was called.
        Returns `False` otherwise.
    """

    _q: Queue[LogRecord]
    _q_handler: QueueHandler
    _q_listener: QueueListener

    _q_objects: tuple[_QueueObjects, ...]

    _listening: bool
    _closed: bool

    def __init__(
        self: QueueHandlerHelper, *logger_handler_pairs: tuple[Logger, Handler]
    ) -> None:
        """Initialises QueueHandlerHelper.

        It does not set a logging level for `Logger` or `Handler`.

        `Arguments`
        --------
        `*logger_handler_pairs` : `tuple[Logger, Handler]`
            Each `Handler` gets attached to the `Logger` it is paired with.
        """
        self._q = Queue()
        self._q_handler = QueueHandler(self._q)

        def create_queue_objects(logger: Logger, handler: Handler) -> _QueueObjects:
            return _QueueObjects(logger, handler, self._q_handler)

        self._q_objects = tuple(starmap(create_queue_objects, logger_handler_pairs))

        self._q_listener = QueueListener(
            self._q,
            *[p._handler for p in self._q_objects],
        )
        self._listening = False
        self._closed = False

    def start(self: QueueHandlerHelper) -> None:
        """Starts the `QueueListener`

        Raises `RuntimeError` if `close()` was called.
        If `is_listening() == True`, it terminates immediately.
        """
        if self._closed:
            raise RuntimeError("QueueHandlerHelper has already been closed.")
        if self._listening:
            return
        self._listening = True
        self._q_listener.start()

    def stop(self: QueueHandlerHelper) -> None:
        """Stops the `QueueListener`

        Raises `RuntimeError` if `close()` was called.
        If `is_listening() == False`, it terminates immediately.
        """
        if self._closed:
            raise RuntimeError("QueueHandlerHelper has already been closed.")
        if not self._listening:
            return
        self._listening = False
        self._q_listener.stop()

    def close(self: QueueHandlerHelper) -> None:
        """Calls `close()` on `QueueHandler`,
        `shutdown()` on `Queue`, and detaches `Handlers` from `Loggers`.

        Raises `RuntimeError` if `close()` has been already called
        or if `is_listening() == True`.
        """
        if self._closed:
            raise RuntimeError("QueueHandlerHelper has already been closed.")
        if self._listening:
            raise RuntimeError("QueueHandlerHelper has not been stopped yet.")
        self._closed = True
        self._q_handler.close()
        self._q.shutdown()

        for q_obj in self._q_objects:
            q_obj.close()

    def stop_and_close(self: QueueHandlerHelper) -> None:
        """Calls both `stop()` and `close()`."""
        if self._closed:
            raise RuntimeError("QueueHandlerHelper has already been closed.")
        self.stop()
        self.close()

    def is_listening(self: QueueHandlerHelper) -> bool:
        """Returns `True` if `QueueHandlerHelper` has been started and not stopped yet.
        Returns `False` otherwise.
        """
        return self._listening

    def is_closed(self: QueueHandlerHelper) -> bool:
        """Returns `True` if `close()` was called.
        Returns `False` otherwise.
        """
        return self._closed


class _QueueFilter(Filter):
    def __init__(self: _QueueFilter, name: str = "") -> None:
        super().__init__(name)

    @override
    def filter(self: _QueueFilter, record: LogRecord) -> bool | LogRecord:
        return self.name == record.name


class _QueueObjects:
    _logger: Logger
    _handler: Handler
    _q_handler: QueueHandler
    _filter: _QueueFilter

    def __init__(
        self: _QueueObjects,
        logger: Logger,
        handler: Handler,
        queue_handler: QueueHandler,
    ) -> None:
        self._logger = logger
        self._handler = handler
        self._q_handler = queue_handler
        self._filter = _QueueFilter(self._logger.name)

        self._logger.addHandler(self._q_handler)
        self._handler.addFilter(self._filter)

    def close(self: _QueueObjects) -> None:
        self._logger.removeHandler(self._q_handler)
        self._handler.removeFilter(self._filter)
