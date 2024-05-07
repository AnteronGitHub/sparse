import asyncio
import io
import logging
import os
import pickle
import shutil
import struct
import tempfile
import uuid

class SparseProtocol(asyncio.Protocol):
    """Common base class for all Sparse network protocols. Provides low-level implementations for sending byte files
    and Python objects.
    """
    def __init__(self):
        self.connection_id = str(uuid.uuid4())
        self.logger = logging.getLogger("sparse")
        self.transport = None

        self.data_buffer = io.BytesIO()
        self.receiving_data = False
        self.data_type = None
        self.data_size = 0

    def clear_buffer(self):
        self.data_buffer = io.BytesIO()
        self.receiving_data = False
        self.data_type = None
        self.data_size = 0

    def connection_made(self, transport):
        self.transport = transport
        peername = self.transport.get_extra_info('peername')
        self.logger.debug(f"Connected to {peername}.")

    def connection_lost(self, exc):
        peername = self.transport.get_extra_info('peername')
        self.logger.debug(f"{peername} disconnected.")

    def data_received(self, data : bytes):
        if self.receiving_data:
            payload = data
        else:
            self.receiving_data = True
            header = data[:9]
            [self.data_type, self.data_size] = struct.unpack("!sQ", header)
            payload = data[9:]

        self.data_buffer.write(payload)

        if self.data_buffer.getbuffer().nbytes >= self.data_size:
            self.message_received(self.data_type.decode(), self.data_buffer.getvalue())
            self.clear_buffer()

    def message_received(self, payload_type : str, data : bytes):
        if payload_type == "f":
            self.file_received(data)
        elif payload_type == "o":
            try:
                self.object_received(pickle.loads(data))
            except pickle.UnpicklingError:
                self.logger.error(f"Deserialization error. {len(data)} payload size, {self.payload_buffer.getbuffer().nbytes} buffer size.")

    def file_received(self, data : bytes):
        pass

    def object_received(self, obj : dict):
        pass

    def send_file(self, file_path):
        with open(file_path, "rb") as f:
            data_bytes = f.read()
            file_size = len(data_bytes)

            self.transport.write(struct.pack("!sQ", b"f", file_size))
            self.transport.write(data_bytes)

    def send_payload(self, payload : dict):
        payload_data = pickle.dumps(payload)
        payload_size = len(payload_data)

        self.transport.write(struct.pack("!sQ", b"o", payload_size))
        self.transport.write(payload_data)

class SparseAppReceiverProtocol(SparseProtocol):
    def __init__(self, migrator_slice, app_repo_path):
        super().__init__()

        self.migrator_slice = migrator_slice
        self.app_repo_path = app_repo_path

        self.app_name = None
        self.app_dag = None

    def file_received(self, data : bytes):
        self.transport.close()

        app_archive_path = f"/tmp/{self.app_name}.zip"
        with open(app_archive_path, "wb") as f:
            f.write(data)

        shutil.unpack_archive(app_archive_path, os.path.join(self.app_repo_path, self.app_name))
        self.migrator_slice.deploy_app(self.app_name, self.app_dag)

    def object_received(self, obj : dict):
        self.send_payload({"type": "ack"})

        app = obj["app"]
        self.app_name = "sparseapp_" + app["name"]
        self.app_dag = app["dag"]
        self.logger.info(f"Received app '{self.app_name}'")

class SparseAppDeployerProtocol(SparseProtocol):
    def __init__(self, app : dict, on_con_lost : asyncio.Future, app_dir : str = '.'):
        super().__init__()

        self.on_con_lost = on_con_lost
        self.app = app
        self.app_dir = app_dir

    def archive_application(self):
        app_name = self.app["name"]
        self.logger.debug("Archiving application")
        shutil.make_archive(os.path.join(tempfile.gettempdir(), app_name), 'zip', self.app_dir)
        return os.path.join(tempfile.gettempdir(), f"{app_name}.zip")

    def connection_made(self, transport):
        super().connection_made(transport)

        self.send_payload({"app": self.app})

    def connection_lost(self, exc):
        app_name = self.app["name"]
        self.logger.info(f"Deployed application '{app_name}' successfully.")
        if self.on_con_lost is not None:
            self.on_con_lost.set_result(True)

        super().connection_lost(exc)

    def object_received(self, obj : dict):
        archive_filepath = self.archive_application()
        self.send_file(archive_filepath)

