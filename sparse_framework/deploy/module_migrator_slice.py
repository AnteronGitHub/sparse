import asyncio
import importlib
import os
import pickle
import shutil

from ..node import SparseSlice
from ..protocols import SparseProtocol

class SparseApp:
    """A sparse app is a Python module that provides a set of sources, operators, and sinks.
    """
    def __init__(self, name : str, zip_path : str):
        self.name = name
        self.zip_path = zip_path
        self.app_module = None

    def load(self, app_repo_path : str):
        if self.app_module is None:
            shutil.unpack_archive(self.zip_path, os.path.join(app_repo_path, self.name))
            self.app_module = importlib.import_module(f".{self.name}", package="sparse_framework.apps")

        return self.app_module

class UpstreamNode:
    def __init__(self, protocol : SparseProtocol):
        self.protocol = protocol

    def push_app(self, app : SparseApp):
        self.protocol.migrate_app_module(app.zip_path)

class SparseModuleMigratorSlice(SparseSlice):
    """Sparse Module Migrator Slice migrates software modules for sources, operators and sinks over the network so that
    stream applications can be distributed in the cluster.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apps = set()

    def add_app_module(self, name : str, zip_path : str):
        self.apps.add(SparseApp(name, zip_path))

    def get_app(self, name : str):
        for app in self.apps:
            if app.name == name:
                return app
        return None

