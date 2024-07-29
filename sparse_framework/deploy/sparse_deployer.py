import asyncio
import os
import tempfile
import shutil

from ..node import SparseSlice

from .protocols import SparseAppDeployerProtocol

class SparseDeployer(SparseSlice):
    """Sparse Deployer is a utility class for packing and deploying Sparse application.

    Sparse applications comprise of software modules defining the sources, operators and sinks, as well as Directed
    Asyclic Graphs that describe the data flow among the sources, operators and sinks.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

#    def unpack_application(self, app_repo_dir : str = "applications"):
#        shutil.unpack_archive(self.archive_filepath(), os.path.join(app_repo_dir, app_name))

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
                self.logger.debug("Connecting to root server on %s:%s.",
                                  self.config.root_server_address,
                                  self.config.root_server_port)
                await loop.create_connection(lambda: SparseAppDeployerProtocol(app, archive_path, on_con_lost), \
                                             self.config.root_server_address, \
                                             self.config.root_server_port)
                await on_con_lost
                break
            except ConnectionRefusedError:
                self.logger.warn("Connection refused. Re-trying in 5 seconds.")
                await asyncio.sleep(5)

    def deploy(self, app : dict, app_dir : str = '.'):
        """Archives and deploys a Sparse application. Uses the running task loop or creates one if one is not already
        running.
        """
        archive_path = self.archive_application(app, app_dir)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.upload_to_server(app, archive_path))
        except RuntimeError:
            asyncio.run(self.upload_to_server(app, archive_path))
