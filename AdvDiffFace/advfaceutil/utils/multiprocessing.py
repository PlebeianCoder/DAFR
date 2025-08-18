from logging import getLogger
from logging import Logger
from logging.handlers import QueueHandler
from multiprocessing import Queue
from threading import Thread
from typing import Union


class LoggingThread(Thread):
    def __init__(
        self,
        queue: Queue,
        group=None,
        target=None,
        name=None,
        args=(),
        kwargs=None,
        *,
        daemon=None,
    ):
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.queue = queue
        self._previous_records = []
        self._previous_records_maxsize = 10

    def run(self) -> None:
        try:
            for record in iter(self.queue.get, None):
                if record not in self._previous_records:
                    logger = getLogger(record.name)
                    logger.handle(record)

                    self._previous_records.append(record)

                    if len(self._previous_records) > self._previous_records_maxsize:
                        self._previous_records.pop()

        except EOFError as e:
            getLogger("multiprocessing").exception(e)


def configure_loggers_on_worker(log_level: Union[int, str], log_queue: Queue):
    # Use a queue handler to transmit all log messages to the queue
    handler = QueueHandler(log_queue)

    # By setting the properties on the root level, we apply to all subsequent loggers
    root = getLogger()
    root.setLevel(log_level)

    if handler not in root.handlers:
        root.addHandler(handler)

    # Add the queue handler to all previously created loggers
    for logger in root.manager.loggerDict.values():
        if isinstance(logger, Logger):
            if handler not in logger.handlers:
                logger.addHandler(handler)
