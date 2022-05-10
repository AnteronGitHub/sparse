import logging

class TaskDeployer:
    """Class that handles network connections to available worker nodes.
    """

    def __init__(self, upstream_host : str, upstream_port : int):
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port

    def set_logger(self, logger : logging.Logger):
        self.logger = logger

    def deploy_task(self, input_data : bytes):
        pass

def get_supported_task_deployer(upstream_host : str, upstream_port : int, legacy_asyncio : bool = False):
    if legacy_asyncio:
        from .task_deployer_legacy import TaskDeployerLegacy
        return TaskDeployerLegacy(upstream_host=upstream_host, upstream_port=upstream_port)
    else:
        from .task_deployer_latest import TaskDeployerLatest
        return TaskDeployerLatest(upstream_host=upstream_host, upstream_port=upstream_port)
