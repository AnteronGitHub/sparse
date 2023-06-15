
import torch
from torch.autograd import Variable
import numpy as np

from sparse_framework.task_executor import TaskExecutor

from .serialization import decode_offload_inference_request, \
                           encode_offload_inference_request, \
                           decode_offload_inference_response, \
                           encode_offload_inference_response, \
                           encode_offload_inference_request_pruned
from .serialization import decode_offload_inference_request_pruned


from .utils import get_device
from .models.model_loader import ModelLoader

class InferenceCalculatorPruning(TaskExecutor):
    def __init__(self, model_name : str, partition : str, compressionProps : dict, use_compression : bool):
        super().__init__()
        self.device = get_device()
        self.model_name = model_name
        self.partition = partition
        self.compressionProps = compressionProps
        self.use_compression = use_compression

        self.model = None


    def start(self):
        """Initialize executor by transferring the model to the processor memory."""
        super().start()

        model_loader = ModelLoader(self.node.config_manager.model_server_address,
                                   self.node.config_manager.model_server_port)

        self.model, self.loss_fn, self.optimizer = model_loader.load_model(self.model_name,
                                                                           self.partition,
                                                                           self.compressionProps,
                                                                           self.use_compression)
        self.logger.info(f"Downloaded model '{self.model_name}' partition '{self.partition}' with compression props '{self.compressionProps}' and using compression '{self.use_compression}'")

        num_parameters = 0
        for param in self.model.parameters():
            num_parameters += param.nelement()
        self.logger.info(f"Inferring with model {type(self.model).__name__} with {num_parameters} parameters using {self.device} for processing")
        self.model.to(self.device)
        self.logger.info(f"Task executor using {self.device} for processing")

        self.model.to(self.device)

    def compress_with_pruneFilter(self, pred, prune_filter, budget, serverFlag = False):

        if serverFlag:
            mask = prune_filter
        else:
            mask = torch.square(torch.sigmoid(prune_filter.squeeze()))
        topk = torch.topk(mask, budget)
        compressedPred = torch.index_select(pred, 1, topk.indices.sort().values)

        return compressedPred, mask

    def decompress_with_pruneFilter(self, pred, mask, budget):

        a = torch.mul(mask.repeat([128,1]).t(), torch.eye(128).to(self.device))
        b = a.index_select(1, mask.topk(budget).indices.sort().values)
        b = torch.where(b>0.0, 1.0, 0.0).to(self.device)
        decompressed_pred = torch.einsum('ij,bjlm->bilm', b, pred)

        return decompressed_pred

    async def execute_task(self, input_data: bytes) -> bytes:
        """Execute a single forward computation for the offloaded layers."""

        with torch.no_grad():
            if self.task_deployer:
                split_layer, prune_filter, budget = decode_offload_inference_request_pruned(input_data)
                split_layer = Variable(split_layer, requires_grad=False).to(self.device)

                model_return = self.model(split_layer)
                pred = model_return[0].to("cpu").detach()   #partial model output
                #quantization/compression TBD
                prune_filter = model_return[1].to("cpu").detach()   #the prune filter in training

                pred, mask = self.compress_with_pruneFilter(pred, prune_filter, budget)

                self.logger.debug("Deploying to the next worker further")

                offload_input_data = encode_offload_inference_request_pruned(pred.to("cpu").detach(), mask, budget)
                result_data = await self.task_deployer.deploy_task(offload_input_data)
            else:
                split_layer, prune_filter, budget = decode_offload_inference_request_pruned(input_data)
                split_layer = Variable(split_layer, requires_grad=False).to(self.device)

                if prune_filter == None:
                    # self.logger.error(
                    #     "This version of sparse does not support unsplit inference")
                    pred = self.model(split_layer)
                    result_data = encode_offload_inference_response(
                        pred.to("cpu").detach())
                else:
                    self.logger.debug("Not deploying task any further")
                    prune_filter = prune_filter.to(self.device)
                    split_layer = self.decompress_with_pruneFilter(
                        split_layer, prune_filter, budget)
                    model_return = self.model(split_layer)
                    pred = model_return[0].to("cpu").detach()
                    result_data = encode_offload_inference_response(pred.to("cpu").detach())

        return result_data
