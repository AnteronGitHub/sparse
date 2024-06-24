import asyncio
import io
import logging
import os
import pickle
import shutil
import struct
import uuid

from ..protocols import SparseProtocol

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
    def __init__(self, app : dict, archive_path : str, on_con_lost : asyncio.Future):
        super().__init__()

        self.on_con_lost = on_con_lost
        self.app = app
        self.archive_path = archive_path

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
        self.send_file(self.archive_path)

