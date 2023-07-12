from sparse_framework.networking import TCPClient

class TaskDeployer(TCPClient):
    """Class that handles network connections to available worker nodes.
    """

    def __init__(self):
        super().__init__(server_address = None, server_port = None)

        self.node = None
        self.logger = None

    def set_node(self, node):
        self.node = node
        self.logger = node.logger
        self.server_address = self.node.config_manager.upstream_host
        self.server_port = self.node.config_manager.upstream_port

        self.logger.info(f"Task deployer using upstream {self.server_address}:{self.server_port}")

    def broken_pipe_error(self):
        if self.node.monitor_client is not None:
            self.node.monitor_client.broken_pipe_error()

    async def deploy_task(self, input_data : dict) -> dict:
        return await self._create_request(input_data)
