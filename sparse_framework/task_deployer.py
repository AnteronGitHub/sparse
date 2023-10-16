import asyncio
import logging
import pickle

from sparse_framework.networking import TCPClient

class EchoClientProtocol(asyncio.Protocol):
    def __init__(self, on_con_lost, on_done):
        self.on_con_lost = on_con_lost
        self.on_done = on_done
        self.logger = logging.getLogger("sparse")

    def connection_made(self, transport):
        self.transport = transport

    def data_received(self, data):
        self.on_done(pickle.loads(data))

    def connection_lost(self, exc):
        self.on_con_lost.set_result(True)

class TaskDeployer(TCPClient):
    """Class that handles network connections to available worker nodes.
    """

    def __init__(self):
        super().__init__(server_address = None, server_port = None)

        self.node = None
        self.logger = logging.getLogger("sparse")

    def set_node(self, node):
        self.node = node
        self.logger = node.logger
        self.server_address = self.node.config_manager.upstream_host
        self.server_port = self.node.config_manager.upstream_port

        self.logger.info(f"Task deployer using upstream {self.server_address}:{self.server_port}")

    def broken_pipe_error(self):
        if self.node.monitor_client is not None:
            self.node.monitor_client.broken_pipe_error()

    def on_result(self, result_data):
        self.result_data = result_data

    async def deploy_task(self, input_data : dict) -> dict:
        loop = asyncio.get_running_loop()

        on_con_lost = loop.create_future()

        transport, protocol = await loop.create_connection(
            lambda: EchoClientProtocol(on_con_lost, self.on_result), self.server_address, self.server_port)
        try:
            transport.write(pickle.dumps(input_data))
            await on_con_lost
        finally:
            transport.close()

        return self.result_data
