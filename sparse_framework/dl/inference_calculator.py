
import torch 
import numpy as np

from ..task_executor import TaskExecutor

from .serialization import decode_offload_inference_request, \
                           encode_offload_inference_request, \
                           decode_offload_inference_response, \
                           encode_offload_inference_response                         
from .serialization import decode_offload_inference_request_pruned


from .utils import get_device


class InferenceCalculatorYOLO(TaskExecutor):
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
    def __init__(self, model, depruneProps):
        super().__init__()
        self.model = model
        self.device = get_device()
        self.depruneProps = depruneProps

    def start(self):
        """Initialize executor by transferring the model to the processor memory."""
        super().start()
        self.logger.info(f"Task executor using {self.device} for processing")
        self.model.to(self.device)
        
    def compress_with_pruneFilter(self, pred, prune_filter, budget):
        
        compressedPred = torch.tensor()
        mask = torch.square(torch.sigmoid(prune_filter.squeeze())).to('cpu')
        masknp = mask.detach().numpy()
        partitioned = np.partition(masknp, -budget)[-budget]
        for entry in range(len(mask)):
            if mask[entry] >= partitioned: 
                 compressedPred = torch.cat((compressedPred, pred[:,entry,:,:]), 1)
                
        return compressedPred, mask    
        
    def decompress_with_pruneFilter(self, pred, mask, budget):
        
        decompressed_pred = torch.tensor()
        zeroPad = torch.zeros(torch.shape(pred[:,0,:,:]))
        masknp = mask.detach().numpy()
        partitioned = np.partition(masknp, -budget)[-budget]
        for entry in range(len(mask)):
            if mask[entry] >= partitioned: 
                decompressed_pred = torch.cat((decompressed_pred, pred[:,entry,:,:]), 1)
            else:
                decompressed_pred = torch.cat((decompressed_pred, zeroPad), 1) 
        
        return decompressed_pred

    async def execute_task(self, input_data: bytes) -> bytes:
        """Execute a single forward computation for the offloaded layers."""

        with torch.no_grad():
            if self.task_deployer:
                split_layer, budget = decode_offload_inference_request_pruned(input_data).to(self.device)
                
                model_return = self.model(split_layer, local=True)
                pred = model_return[0].to("cpu").detach()   #partial model output
                #quantization/compression TBD
                prune_filter = pred[1].to("cpu").detach()   #the prune filter in training
                
                pred = self.decompress_with_pruneFilter(pred, prune_filter, budget)
                
                self.logger.debug("Deploying to the next worker further")

                offload_input_data = encode_offload_inference_request(pred.to("cpu").detach())
                result_data = await self.task_deployer.deploy_task(offload_input_data)
            else:
                split_layer, prune_filter, budget = decode_offload_inference_request_pruned(input_data).to(self.device)
                
                pred = self.model(split_layer)
                
                self.logger.debug("Not deploying task any further")
                result_data = encode_offload_inference_response(pred.to("cpu").detach())

        return result_data
