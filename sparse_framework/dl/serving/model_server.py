import asyncio
import logging
import pickle

from sparse_framework.networking import TCPServer

from ..utils import count_model_parameters
from .model_repository import DiskModelRepository

class ModelServer(asyncio.Protocol):
    """TCP server for deploying models with latest trained parameters.
    """
    def __init__(self, model_repository : DiskModelRepository):
        self.logger = logging.getLogger("sparse")
        self.model_repository = model_repository

    def get_model(self, input_data : dict):
        model_meta_data = input_data["model_meta_data"]

        model, loss_fn, optimizer = self.model_repository.get_model(model_meta_data)

        return { "model": model, "loss_fn": loss_fn, "optimizer": optimizer }

    def save_model(self, input_data : dict):
        model, model_meta_data = input_data["model"], input_data["model_meta_data"]

        self.model_repository.save_model(model, model_meta_data)

        return { "status": "ok" }

    def request_processed(self, request_context : dict, processing_time : float):
        if request_context["method"] == "get_model":
            served_model = request_context["model"]
            num_parameters = count_model_parameters(served_model)
            self.logger.info(f"Served model '{type(served_model).__name__}' with {num_parameters} parameters in {processing_time} seconds.")
        elif request_context["method"] == "save_model":
            saved_model = request_context["model"]
            num_parameters = count_model_parameters(saved_model)
            self.logger.info(f"Saved model '{type(saved_model).__name__}' with {num_parameters} parameters in {processing_time} seconds.")

    def connection_made(self, transport):
        peername = transport.get_extra_info('peername')
        self.logger.info('Connection from {}'.format(peername))
        self.transport = transport

    def data_received(self, data):
        input_data = pickle.loads(data)
        method = input_data["method"]

        if method == "get_model":
            self.transport.write(pickle.dumps(self.get_model(input_data)))
        elif method == "save_model":
            self.transport.write(pickle.dumps(self.save_model(input_data)))
        else:
            self.logger.error(f"Received unreckognized method '{method}'.")
            self.transport.write(f"Received unreckognized method '{method}'.")
        self.transport.close()
        self.logger.info(f"Processed request.")
