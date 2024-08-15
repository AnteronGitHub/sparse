import asyncio
import logging
import os
import shutil
import tempfile

from .protocols import AppUploaderProtocol

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
        self.logger.debug("Archiving application")
        shutil.make_archive(os.path.join(tempfile.gettempdir(), app_name), 'zip', app_dir)
        return os.path.join(tempfile.gettempdir(), f"{app_name}.zip")

    async def upload_to_server(self, app : dict, archive_path : str):
        loop = asyncio.get_running_loop()
        on_con_lost = loop.create_future()

        while True:
            try:
                self.logger.debug("Connecting to root server on %s:%s.", self.api_host, self.api_port)
                await loop.create_connection(lambda: AppUploaderProtocol(app, archive_path, on_con_lost), \
                                             self.api_host, \
                                             self.api_port)
                await on_con_lost
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
