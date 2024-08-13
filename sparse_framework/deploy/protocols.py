import asyncio
import io
import logging
import os
import pickle
import shutil
import struct
import uuid

from ..protocols import SparseProtocol

class SparseAppDeployerProtocol(SparseProtocol):
    """Sparse network protocol for sending an application and its module archive to a cluster.

    Application is deployed in two phases. First its DAG is deployed as a dictionary, and then the application modules
    are deployed as a ZIP archive.
    """
    def __init__(self, app : dict, archive_path : str, on_con_lost : asyncio.Future, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.on_con_lost = on_con_lost
        self.app = app
        self.archive_path = archive_path

    def connection_made(self, transport):
        super().connection_made(transport)

        self.deploy_app(self.app)

    def connection_lost(self, exc):
        app_name = self.app["name"]
        self.logger.info(f"Deployed application '{app_name}' successfully.")
        if self.on_con_lost is not None:
            self.on_con_lost.set_result(True)

        super().connection_lost(exc)

    def object_received(self, obj : dict):
        self.migrate_app_module(self.archive_path)

class DownstreamConnectorProtocol(SparseProtocol):
    def __init__(self, on_con_lost : asyncio.Future, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.on_con_lost = on_con_lost

    def connection_made(self, transport):
        super().connection_made(transport)
        self.node.stream_manager_slice.add_upstream_node(self)

        self.send_payload({"op": "connect_downstream"})

