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

    def get_model(self, input_data : dict, context : dict):
        model_name, partition = input_data["model_name"], input_data["partition"]

        model, loss_fn, optimizer = self.model_repository.get_model(model_name, partition)

        return { "model": model, "loss_fn": loss_fn, "optimizer": optimizer }, { "method": "get_model", "model" : model }

    def save_model(self, input_data : dict, context : dict):
        model, model_name, partition = input_data["model"], input_data["model_name"], input_data["partition"]

        self.model_repository.save_model(model, model_name, partition)

        return { "status": "ok" }, { "method": "save_model", "model" : model }

    def request_processed(self, request_context : dict, processing_time : float):
        if request_context["method"] == "get_model":
            served_model = request_context["model"]
            num_parameters = count_model_parameters(served_model)
            self.logger.info(f"Served model '{type(served_model).__name__}' with {num_parameters} parameters in {processing_time} seconds.")
        elif request_context["method"] == "save_model":
            saved_model = request_context["model"]
            num_parameters = count_model_parameters(saved_model)
            self.logger.info(f"Saved model '{type(saved_model).__name__}' with {num_parameters} parameters in {processing_time} seconds.")

    async def handle_request(self, input_data : dict, context : dict) -> dict:
        method = input_data["method"]
        context["method"] = method

        if method == "get_model":
            return self.get_model(input_data, context)
        elif method == "save_model":
            return self.save_model(input_data, context)
        else:
            self.logger.error(f"Received unreckognized method '{method}'.")
            return input_data, context

