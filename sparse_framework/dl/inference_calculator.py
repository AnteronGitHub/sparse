
import torch 
import numpy as np

from ..task_executor import TaskExecutor

from .serialization import decode_offload_inference_request, \
                           encode_offload_inference_request, \
                           decode_offload_inference_response, \
                           encode_offload_inference_response, \
                           encode_offload_inference_request_pruned
from .serialization import decode_offload_inference_request_pruned


from .utils import get_device
from torch.autograd import Variable


class InferenceCalculatorOD(TaskExecutor):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = get_device()

    def start(self):
        """Initialize executor by transferring the model to the processor memory."""
        super().start()
        self.logger.info(f"Task executor using {self.device} for processing")
        self.model.to(self.device)

    async def execute_task(self, input_data: bytes) -> bytes:
        """Execute a single forward computation for the offloaded layers."""
        split_layer = decode_offload_inference_request(input_data).to(self.device)
        pred = self.model(split_layer)

        if self.task_deployer:
            self.logger.debug("Deploying to the next worker further")

            offload_input_data = encode_offload_inference_request(pred.to("cpu").detach())
            result_data = await self.task_deployer.deploy_task(offload_input_data)
        else:
            self.logger.debug("Not deploying task any further")
            result_data = encode_offload_inference_response(pred.to("cpu").detach())

        return result_data


class InferenceCalculator(TaskExecutor):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = get_device()


    def start(self):
        """Initialize executor by transferring the model to the processor memory."""
        super().start()
        self.logger.info(f"Task executor using {self.device} for processing")
        self.model.to(self.device)
        
    def compress_with_pruneFilter(self, pred, prune_filter, budget):
        
        compressedPred = torch.tensor([])
        mask = torch.square(torch.sigmoid(prune_filter.squeeze())).to('cpu')
        masknp = mask.detach().numpy()
        partitioned = np.partition(masknp, -budget)[-budget]
        for entry in range(len(mask)):
            if mask[entry] >= partitioned:
                 predRow = pred[:,entry,:,:].unsqueeze(dim=1) 
                 compressedPred = torch.cat((compressedPred, predRow), 1)
                
        return compressedPred, mask    
        
    def decompress_with_pruneFilter(self, pred, mask, budget):
        
        decompressed_pred = torch.tensor([]).to(self.device)
        a_row = pred[:,0,:,:].unsqueeze(dim=1)
        zeroPad = torch.zeros(a_row.shape).to(self.device)
        masknp = mask.to('cpu').detach().numpy()
        partitioned = np.partition(masknp, -budget)[-budget]
        count = 0
        for entry in range(len(mask)):
            if mask[entry] >= partitioned: 
                predRow = pred[:,count,:,:].unsqueeze(dim=1)
                decompressed_pred = torch.cat((decompressed_pred, predRow), 1)
                count += 1
            else:
                decompressed_pred = torch.cat((decompressed_pred, zeroPad), 1) 
        
        return decompressed_pred

    async def execute_task(self, input_data: bytes) -> bytes:
        """Execute a single forward computation for the offloaded layers."""

        with torch.no_grad():
            if self.task_deployer:
                split_layer, prune_filter, budget = decode_offload_inference_request_pruned(input_data)
                split_layer = Variable(split_layer, requires_grad=False).to(self.device)
                
                model_return = self.model(split_layer, local=True)
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
                    self.logger.error(
                        "This version of sparse does not support unsplit inference")
                    self.logger.info(f"split_layer is {split_layer}")
                    pred = self.model(split_layer)
                    result_data = encode_offload_inference_response(
                        pred.to("cpu").detach())
                else:
                    self.logger.debug("Not deploying task any further")
                    prune_filter = prune_filter.to(self.device)
                    split_layer = self.decompress_with_pruneFilter(
                        split_layer, prune_filter, budget)
                    model_return = self.model(split_layer, local=False)
                    pred = model_return[0].to("cpu").detach()
                    result_data = encode_offload_inference_response(pred.to("cpu").detach())

        return result_data
