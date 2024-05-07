import asyncio

from ..node import SparseSlice

from .protocols import SparseAppDeployerProtocol

class SparseDeployer(SparseSlice):
    """Sparse Deployer is a utility class for packing and deploying Sparse application. Sparse applications comprise of
    software modules defining the sources, operators and sinks, as well as Directed Asyclic Graphs that describe the
    data flow among the sources, operators and sinks.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

#    def unpack_application(self, app_repo_dir : str = "applications"):
#        shutil.unpack_archive(self.archive_filepath(), os.path.join(app_repo_dir, app_name))

    async def upload_to_server(self, app : dict):
        loop = asyncio.get_running_loop()
        on_con_lost = loop.create_future()

        while True:
            try:
                self.logger.debug(f"Connecting to root server on sparse-worker:{self.config.root_server_port}.")
                await loop.create_connection(lambda: SparseAppDeployerProtocol(app, on_con_lost), \
                                             self.config.root_server_address, \
                                             self.config.root_server_port)
                await on_con_lost
                break
            except ConnectionRefusedError:
                self.logger.warn("Connection refused. Re-trying in 5 seconds.")
                await asyncio.sleep(5)

    def deploy(self, app : dict):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.upload_to_server(app))
        except RuntimeError:
            asyncio.run(self.upload_to_server(app))
