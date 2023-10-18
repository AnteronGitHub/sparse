import asyncio

from sparse_framework import Master

from ..protocols import ModelServeClientProtocol

class ModelServeClient(Master):
    async def start(self, dataset, model_meta_data, no_samples):
        loop = asyncio.get_running_loop()
        on_con_lost = loop.create_future()
        await loop.create_connection(lambda: ModelServeClientProtocol(self.node_id, dataset, model_meta_data, on_con_lost),
                                     self.config_manager.upstream_host,
                                     self.config_manager.upstream_port)
        await on_con_lost

