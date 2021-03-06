import logging

from .task_executor import TaskExecutor

class RXPipe:
    """Class that handles the queuing of offloaded tasks data, and passes them to the application-specific task
    executor instance.

    User is not expected to implement a custom class in normal use cases, but instead use implementations provided by
    the framework. Currently only one such implementation exists, namely this class.
    """

    def __init__(self,
                 task_executor : TaskExecutor,
                 listen_address : str,
                 listen_port : int):
        self.listen_address = listen_address
        self.listen_port = listen_port
        self.task_executor = task_executor

    def set_logger(self, logger : logging.Logger):
        self.logger = logger

    def start(self):
        pass

def get_supported_rx_pipe(task_executor : TaskExecutor, listen_address : str, listen_port : int, legacy_asyncio : bool = False):
    if legacy_asyncio:
        from .rx_pipe_legacy import RXPipeLegacy
        return RXPipeLegacy(task_executor, listen_address, listen_port)
    else:
        from .rx_pipe_latest import RXPipeLatest
        return RXPipeLatest(task_executor, listen_address, listen_port)
