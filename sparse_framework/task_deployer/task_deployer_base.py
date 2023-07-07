import logging

class TaskDeployerBase:
    """Class that handles network connections to available worker nodes.
    """

    def __init__(self, upstream_host : str, upstream_port : int):
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self.node = None

    def set_logger(self, logger : logging.Logger):
        self.logger = logger

    def set_node(self, node):
        self.node = node

    def deploy_task(self, input_data : bytes):
        pass
