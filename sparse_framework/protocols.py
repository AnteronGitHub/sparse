import asyncio
import io
import logging
import pickle
import struct
import uuid

class SparseProtocol(asyncio.Protocol):
    """Sparse protocols provide transport for transmitting both dictionary data and files over network.
    """
    def __init__(self):
        self.connection_id = str(uuid.uuid4())
        self.logger = logging.getLogger("sparse")
        self.transport = None

        self.data_buffer = io.BytesIO()
        self.receiving_data = False
        self.data_type = None
        self.data_size = 0

        self.app_name = None
        self.app_dag = None

    def clear_buffer(self):
        self.data_buffer = io.BytesIO()
        self.receiving_data = False
        self.data_type = None
        self.data_size = 0

    def connection_made(self, transport):
        self.transport = transport
        peer_ip = self.transport.get_extra_info('peername')[0]

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

    def send_data_tuple(self, stream_id : str, data_tuple):
        """Initiates app deployment process by uploading the app dag.
        """
        self.send_payload({"op": "data_tuple", "stream_id": stream_id, "tuple": data_tuple })

    def deploy_app(self, app : dict):
        """Initiates app deployment process by uploading the app dag.
        """
        self.send_payload({"op": "deploy_app", "app": app})

    def migrate_app_module(self, archive_path : str):
        self.logger.debug("Migrating app module '%s'", archive_path)
        self.send_file(archive_path)

class ClusterClientProtocol(SparseProtocol):
    """Cluster client protocol creates an egress connection to another cluster node.
    """
    def __init__(self, on_con_lost : asyncio.Future, node):
        super().__init__()

        self.node = node
        self.on_con_lost = on_con_lost

    def connection_made(self, transport):
        super().connection_made(transport)
        self.node.stream_router.add_cluster_connection(self, direction="egress")

        self.send_payload({"op": "connect_downstream"})

    def connection_lost(self, exc):
        self.node.stream_router.remove_cluster_connection(self.transport)
        peername = self.transport.get_extra_info('peername')
        self.logger.debug(f"{peername} disconnected.")

    def file_received(self, data : bytes):
        self.send_payload({"op": "transfer_file", "type": "ack"})
        self.logger.debug("Received module for app '%s'", self.app_name)
        app_archive_path = f"/tmp/{self.app_name}.zip"
        with open(app_archive_path, "wb") as f:
            f.write(data)

        self.node.module_repo.add_app_module(self.app_name, app_archive_path)
        self.node.stream_router.create_deployment(self, self.app_dag)

    def object_received(self, obj : dict):
        if obj["op"] == "deploy_app":
            self.send_payload({"op": "deploy_app", "type": "ack"})

            app = obj["app"]
            self.app_name = "sparseapp_" + app["name"]
            self.app_dag = app["dag"]
        elif obj["op"] == "data_tuple":
            self.node.runtime.tuple_received(obj["stream_id"], obj["tuple"])

class ClusterServerProtocol(SparseProtocol):
    """Cluster client protocol creates an ingress connection to another cluster node.
    """
    def __init__(self, node):
        super().__init__()

        self.node = node

    def connection_lost(self, exc):
        self.node.stream_router.remove_cluster_connection(self.transport)
        peername = self.transport.get_extra_info('peername')
        self.logger.debug(f"{peername} disconnected.")

    def file_received(self, data : bytes):
        self.send_payload({"op": "transfer_file", "type": "ack"})
        self.logger.debug("Received module for app '%s'", self.app_name)
        app_archive_path = f"/tmp/{self.app_name}.zip"
        with open(app_archive_path, "wb") as f:
            f.write(data)

        self.node.module_repo.add_app_module(self.app_name, app_archive_path)
        self.node.stream_router.create_deployment(self, self.app_dag)

    def object_received(self, obj : dict):
        if "type" in obj and obj["type"] == "ack":
            return
        if obj["op"] == "deploy_app":
            self.send_payload({"op": "deploy_app", "type": "ack"})

            app = obj["app"]
            self.app_name = "sparseapp_" + app["name"]
            self.app_dag = app["dag"]
        elif obj["op"] == "data_tuple":
            self.node.runtime.tuple_received(obj["stream_id"], obj["tuple"])
        elif obj["op"] == "connect_downstream":
            self.node.stream_router.add_cluster_connection(self, "ingress")

class AppUploaderProtocol(SparseProtocol):
    """App uploader protocol uploads a Sparse module including an application deployment to an open Sparse API.

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

    def object_received(self, obj : dict):
        if obj["op"] == "deploy_app":
            self.migrate_app_module(self.archive_path)
        elif obj["op"] == "transfer_file":
            self.transport.close()
