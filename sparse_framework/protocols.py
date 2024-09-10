import asyncio
import io
import logging
import pickle
import struct
import uuid

from .module_repo import SparseModule

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

    def clear_buffer(self):
        self.data_buffer = io.BytesIO()
        self.receiving_data = False
        self.data_type = None
        self.data_size = 0

    def connection_made(self, transport):
        self.transport = transport

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

    def create_deployment(self, app : dict):
        self.send_payload({"op": "create_deployment", "app": app})

    def create_source_stream(self, stream_type : str, stream_id : str = None):
        self.send_payload({"op": "create_source_stream", "stream_type": stream_type, "stream_id": stream_id})

    def send_data_tuple(self, stream_id : str, data_tuple):
        """Initiates app deployment process by uploading the app dag.
        """
        self.send_payload({"op": "data_tuple", "stream_id": stream_id, "tuple": data_tuple })

class ClusterProtocol(SparseProtocol):
    def __init__(self, node):
        super().__init__()
        self.node = node

        self.app_name = None
        self.app_dag = None

        self.transferring_module = None
        self.receiving_module_name = None

    def connection_lost(self, exc):
        self.node.stream_router.remove_cluster_connection(self.transport)
        peername = self.transport.get_extra_info('peername')
        self.logger.debug(f"{peername} disconnected.")

    def object_received(self, obj : dict):
        if obj["op"] == "init_module_transfer":
            if "status" in obj:
                if obj["status"] == "accepted":
                    self.send_file(self.transferring_module.zip_path)
                else:
                    self.logger.error("Module transfer initialization ended in status '%s'", obj["status"])
            else:
                if self.receiving_module_name is None:
                    self.receiving_module_name = obj["module_name"]
                    self.send_payload({"op": "init_module_transfer", "status": "accepted"})
                else:
                    self.send_payload({"op": "init_module_transfer", "status": "rejected"})
        elif obj["op"] == "create_deployment":
            if "status" in obj:
                if obj["status"] == "success":
                    self.logger.info("Created deployment")
                else:
                    self.logger.info("Unable to create a deployment")
            else:
                app = obj["app"]
                self.node.stream_router.create_deployment(self, app["dag"])

                self.send_payload({"op": "create_deployment", "status": "success"})
        elif obj["op"] == "data_tuple":
            self.node.runtime.tuple_received(obj["stream_id"], obj["tuple"])
        else:
            super().object_received(obj)

    def transfer_module(self, module : SparseModule):
        self.transferring_module = module

        self.send_payload({ "op": "init_module_transfer", "module_name": self.transferring_module.name })

    def file_received(self, data : bytes):
        app_archive_path = f"/tmp/{self.app_name}.zip"
        with open(app_archive_path, "wb") as f:
            f.write(data)

        module = self.node.module_repo.add_app_module(self.receiving_module_name, app_archive_path)
        self.node.stream_router.distribute_module(self, module)
        self.receiving_module_name = None

        self.send_payload({"op": "transfer_file", "status": "success"})

class ClusterClientProtocol(ClusterProtocol):
    """Cluster client protocol creates an egress connection to another cluster node.
    """
    def __init__(self, on_con_lost : asyncio.Future, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.on_con_lost = on_con_lost

    def connection_made(self, transport):
        super().connection_made(transport)

        self.send_payload({"op": "connect_downstream"})

    def object_received(self, obj : dict):
        if obj["op"] == "connect_downstream":
            if "status" in obj and obj["status"] == "success":
                self.node.stream_router.add_cluster_connection(self, direction="egress")
        else:
            super().object_received(obj)

class ClusterServerProtocol(ClusterProtocol):
    """Cluster client protocol creates an ingress connection to another cluster node.
    """
    def object_received(self, obj : dict):
        if obj["op"] == "connect_downstream":
            self.node.stream_router.add_cluster_connection(self, "ingress")
            self.send_payload({"op": "connect_downstream", "status": "success"})
        elif obj["op"] == "create_source_stream":
            stream_type = obj["stream_type"]
            if "stream_id" in obj.keys():
                stream_id = obj["stream_id"]
            else:
                stream_id = None

            stream = self.node.stream_router.add_source_stream(stream_type, self, stream_id)
            self.send_payload({"op": "create_source_stream", "status": "success", "stream_id": stream.stream_id})
        else:
            super().object_received(obj)

class ModuleUploaderProtocol(SparseProtocol):
    """App uploader protocol uploads a Sparse module including an application deployment to an open Sparse API.

    Application is deployed in two phases. First its DAG is deployed as a dictionary, and then the application modules
    are deployed as a ZIP archive.
    """
    def __init__(self, module_name : str, archive_path : str, on_con_lost : asyncio.Future, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.on_con_lost = on_con_lost
        self.module_name = module_name
        self.archive_path = archive_path

    def connection_made(self, transport):
        super().connection_made(transport)

        self.send_payload({ "op": "init_module_transfer", "module_name": self.module_name })

    def connection_lost(self, exc):
        if self.on_con_lost is not None:
            self.on_con_lost.set_result(True)

    def object_received(self, obj : dict):
        if obj["op"] == "init_module_transfer":
            if obj["status"] == "accepted":
                self.send_file(self.archive_path)
        elif obj["op"] == "transfer_file":
            if obj["status"] == "success":
                self.logger.info("Uploaded module '%s' successfully.", self.module_name)
                self.transport.close()
        else:
            super().object_received(obj)

class DeploymentPostProtocol(SparseProtocol):
    """App uploader protocol uploads a Sparse module including an application deployment to an open Sparse API.

    Application is deployed in two phases. First its DAG is deployed as a dictionary, and then the application modules
    are deployed as a ZIP archive.
    """
    def __init__(self, deployment : dict, on_con_lost : asyncio.Future, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.on_con_lost = on_con_lost
        self.deployment = deployment

    def connection_made(self, transport):
        super().connection_made(transport)
        self.send_payload({"op": "create_deployment", "app": self.deployment})

    def connection_lost(self, exc):
        if self.on_con_lost is not None:
            self.on_con_lost.set_result(True)

    def object_received(self, obj : dict):
        if obj["op"] == "create_deployment":
            if obj["status"] == "success":
                self.logger.info("Deployed application '%s' successfully.", self.deployment)
                self.transport.close()
        else:
            super().object_received(obj)

class SourceProtocol(SparseProtocol):
    """Source protocol connects to a cluster end point and receives a stream id that can be used to transmit data
    tuples using the established connection.
    """
    def __init__(self, stream_type : str, on_stream_initialized : asyncio.Future):
        super().__init__()
        self.stream_type = stream_type
        self.on_stream_initialized = on_stream_initialized

    def connection_made(self, transport):
        super().connection_made(transport)
        self.create_source_stream(self.stream_type)

    def object_received(self, obj : dict):
        if obj["op"] == "create_source_stream":
            if obj["status"] == "success":
                stream_id = obj["stream_id"]
                from .stream_api import SparseStream
                stream = SparseStream(self.stream_type, stream_id)
                stream.add_protocol(self)
                self.on_stream_initialized.set_result(stream)
        else:
            super().object_received(obj)
