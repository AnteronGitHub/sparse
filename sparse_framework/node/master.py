from .node import Node
from ..task_deployer import TaskDeployer

class Master(Node):
    def __init__(self,
                 upstream_host : str = '127.0.0.1',
                 upstream_port : int = 50007,
                 task_deployer : TaskDeployer = None,
                 benchmark : bool = True):
        Node.__init__(self, benchmark=benchmark)
        if task_deployer:
            self.task_deployer = task_deployer
        else:
            self.task_deployer = TaskDeployer()

        self.task_deployer.set_node(self)
