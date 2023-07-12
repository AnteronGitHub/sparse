from torch.nn import Module

from sparse_framework.networking import TCPServer

from ..utils import count_model_parameters
from .model_repository import ModelRepository

class ModelServer(TCPServer):
    """TCP server for deploying models with latest trained parameters.
    """
    def __init__(self, model_repository : ModelRepository, **args):
        super().__init__(**args)

        self.model_repository = model_repository

    def request_processed(self, request_context : dict, processing_time : float):
        served_model = request_context["model"]
        num_parameters = count_model_parameters(served_model)
        self.logger.info(f"Served model '{type(served_model).__name__}' with {num_parameters} parameters in {processing_time} seconds.")

    async def handle_request(self, input_data : dict, context : dict) -> dict:
        model_name, partition = input_data["model_name"], input_data["partition"]

        model, loss_fn, optimizer = self.model_repository.get_model(model_name, partition)

        return { "model": model, "loss_fn": loss_fn, "optimizer": optimizer }, { "model" : model }

