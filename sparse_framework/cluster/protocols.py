import asyncio
import io
import logging
import pickle
import struct
import uuid

from ..protocols import SparseProtocol

class DataPlaneProtocol(SparseProtocol):
    def __init__(self):
        super().__init__()

    def object_received(self, obj : dict):
        self.logger.info("Received object")

