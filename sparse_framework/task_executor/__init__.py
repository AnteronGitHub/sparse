import logging

class TaskExecutor:
    """Common base class for task execution logic. This class implements potentially hardware-accelerated computations
    that are offloaded into worker nodes.

    User is expected to implement the computation logic by defining a custom execute_task() function. Additionally it
    is possible to implement custom initialization code by overriding optional start() hook.
    """
    def __init__(self):
        self.task_deployer = None

    def start(self):
        self.logger.debug("Starting task executor")

    def set_benchmark(self, benchmark):
        self.benchmark = benchmark

    def set_logger(self, logger : logging.Logger = logging.getLogger("sparse")):
        self.logger = logger

    async def execute_task(self, input_data : bytes) -> bytes:
        raise "Task executor not implemented! See documentation on how to implement your own executor"
