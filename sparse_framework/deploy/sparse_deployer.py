import asyncio
import os
import pickle
import shutil
import tempfile

from ..node import SparseSlice
from ..protocols import SparseAppDeployerProtocol

class SparseDeployer(SparseSlice):
    """Sparse Deployer is a utility class for packing and deploying Sparse application. Sparse applications comprise of
    software modules defining the sources, operators and sinks, as well as Directed Asyclic Graphs that describe the
    data flow among the sources, operators and sinks.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app_name = None
        self.app = None

    def data_received(self, protocol):
        payload_type, data = protocol.data_type.decode(), protocol.data_buffer.getvalue()
        if payload_type == "f":
            self.file_received(protocol, data)
        elif payload_type == "o":
            try:
                self.object_received(protocol, pickle.loads(data))
            except pickle.UnpicklingError:
                self.logger.error(f"Deserialization error. {len(data)} payload size, {self.payload_buffer.getbuffer().nbytes} buffer size.")

    def archive_filepath(self):
        return os.path.join(tempfile.gettempdir(), f"{self.app_name}.zip")

    def archive_application(self, app_dir : str = '.'):
        self.logger.info("Archiving application")
        shutil.make_archive(os.path.join(tempfile.gettempdir(), self.app_name), 'zip', app_dir)

    def unpack_application(self, app_repo_dir : str = "applications"):
        shutil.unpack_archive(self.archive_filepath(), os.path.join(app_repo_dir, app_name))

    def connected(self, protocol):
        protocol.send_file(self.archive_filepath())

    def object_received(self, protocol, obj : dict):
        if obj["type"] == "ack":
            protocol.send_payload({"type": "app", "data": self.app})

    async def upload_to_server(self):
        self.archive_application()
        loop = asyncio.get_running_loop()
        on_con_lost = loop.create_future()

        while True:
            try:
                self.logger.info(f"Connecting to root server on sparse-worker:{self.config.root_server_port}.")
                await loop.create_connection(lambda: SparseAppDeployerProtocol(self, on_con_lost), \
                                             "sparse-worker", \
                                             self.config.root_server_port)
                await on_con_lost
                break
            except ConnectionRefusedError:
                self.logger.warn("Connection refused. Re-trying in 5 seconds.")
                await asyncio.sleep(5)

    def deploy(self, app : dict):
        self.app_name = app["name"]
        self.app = app

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.upload_to_server())
        except RuntimeError:
            asyncio.run(self.upload_to_server())
