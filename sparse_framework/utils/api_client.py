import asyncio
import logging
import os
import shutil
import tempfile

from ..protocols import SparseProtocol

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

        self.send_init_module_transfer(self.module_name)

    def init_module_transfer_ok_received(self):
        self.send_file(self.archive_path)

    def transfer_file_ok_received(self):
        self.logger.info("Uploaded module '%s' successfully.", self.module_name)
        self.transport.close()

    def connection_lost(self, exc):
        if self.on_con_lost is not None:
            self.on_con_lost.set_result(True)

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
        self.send_create_deployment(self.deployment)

    def connection_lost(self, exc):
        if self.on_con_lost is not None:
            self.on_con_lost.set_result(True)

    def create_deployment_ok_received(self):
        self.logger.info("Deployed application '%s' successfully.", self.deployment)
        self.transport.close()

class SparseAPIClient:
    """Sparse API client can be used to communicate with the Sparse API to upload applications.

    Sparse applications comprise of software modules defining the sources, operators and sinks, as well as Directed
    Asyclic Graphs that describe the data flow among the sources, operators and sinks.
    """
    def __init__(self, api_host : str, api_port : int = 50006):
        self.logger = logging.getLogger("sparse")
        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=logging.INFO)
        self.api_host = api_host
        self.api_port = api_port

    def archive_application(self, app : dict, app_dir : str):
        app_name = app["name"]
        self.logger.info("Creating Sparse module from directory %s", app_dir)
        shutil.make_archive(os.path.join(tempfile.gettempdir(), app_name), 'zip', app_dir)
        return os.path.join(tempfile.gettempdir(), f"{app_name}.zip")

    async def upload_to_server(self, app : dict, archive_path : str):
        loop = asyncio.get_running_loop()
        on_con_lost = loop.create_future()

        while True:
            try:
                self.logger.debug("Connecting to root server on %s:%s.", self.api_host, self.api_port)
                await loop.create_connection(lambda: ModuleUploaderProtocol(app["name"], archive_path, on_con_lost), \
                                             self.api_host, \
                                             self.api_port)
                await on_con_lost
                await asyncio.sleep(1)
                await loop.create_connection(lambda: DeploymentPostProtocol(app, on_con_lost), \
                                             self.api_host, \
                                             self.api_port)
                break
            except ConnectionRefusedError:
                self.logger.warn("Connection refused. Re-trying in 5 seconds.")
                await asyncio.sleep(5)

    def upload_app(self, app : dict, app_dir : str = '.'):
        """Archives and deploys a Sparse application. Uses the running task loop or creates one if one is not already
        running.
        """
        archive_path = self.archive_application(app, app_dir)

        asyncio.run(self.upload_to_server(app, archive_path))
