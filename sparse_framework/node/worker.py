from .node import Node
from .master import Master

from ..rx_pipe import RXPipe
from ..task_executor import TaskExecutor

class Worker(Node):
    def __init__(self, task_executor : TaskExecutor, rx_pipe : RXPipe = None, benchmark_log_file_prefix = 'benchmark_sparse'):
        Node.__init__(self)
        self.task_executor = task_executor
        self.task_executor.set_logger(self.logger)
        self.task_executor.set_node(self)

        if rx_pipe:
            self.rx_pipe = rx_pipe
        else:
            self.rx_pipe = RXPipe(benchmark_log_file_prefix = benchmark_log_file_prefix)
        self.rx_pipe.set_node(self)

        if isinstance(self, Master):
            self.task_executor.task_deployer = self.task_deployer

    def start(self):
        self.task_executor.start()
        self.rx_pipe.start()
