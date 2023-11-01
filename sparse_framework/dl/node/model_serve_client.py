import asyncio

from sparse_framework import Master

from ..protocols import ModelServeClientProtocol

class ModelServeClient(Master):
    def __init__(self, dataset, model_meta_data, no_samples, **kwargs):
        super().__init__(**kwargs)
        self.protocol_factory = lambda on_con_lost, stats_queue: lambda: ModelServeClientProtocol(self.node_id, dataset, model_meta_data, on_con_lost, no_samples, stats_queue=stats_queue)

    def get_futures(self):
        futures = super().get_futures()

        loop = asyncio.get_running_loop()
        on_con_lost = loop.create_future()
        futures.append(loop.create_connection(self.protocol_factory(on_con_lost, self.stats_queue),
                                              self.config_manager.upstream_host,
                                              self.config_manager.upstream_port))
        futures.append(on_con_lost)

        return futures

