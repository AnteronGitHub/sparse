from .node import Node
from ..task_deployer import TaskDeployer

class Master(Node):
    def __init__(self, **kwargs):
        Node.__init__(self, **kwargs)
