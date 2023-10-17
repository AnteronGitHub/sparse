import asyncio
import logging
import pickle

from sparse_framework.networking import TCPClient

class TaskDeployer(TCPClient):
    """Class that handles network connections to available worker nodes.
    """

    def __init__(self):
        super().__init__(server_address = None, server_port = None)

        self.node = None
        self.logger = logging.getLogger("sparse")

    def set_node(self, node):
        self.node = node
        self.server_address = self.node.config_manager.upstream_host
        self.server_port = self.node.config_manager.upstream_port

        self.logger.info(f"Task deployer using upstream {self.server_address}:{self.server_port}")

    def broken_pipe_error(self):
        if self.node.monitor_client is not None:
            self.node.monitor_client.broken_pipe_error()

    async def create_connection(self, protocol_factory):
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_connection(protocol_factory, self.server_address, self.server_port)

    def deploy_task(self, input_data : dict) -> dict:
        self.logger.info(f"Deploying task")
        self.transport.write(pickle.dumps(input_data))
