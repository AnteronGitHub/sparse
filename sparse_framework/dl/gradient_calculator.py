import asyncio
from torch.autograd import Variable
from torch.nn import Module
import torch
import numpy as np


from ..task_executor import TaskExecutor

from .serialization import decode_offload_request, encode_offload_request, decode_offload_response, encode_offload_response
from .serialization import encode_offload_request_pruned, decode_offload_request_pruned
from .utils import get_device

class GradientCalculator(TaskExecutor):
    def __init__(self, model : Module, loss_fn, optimizer):
        super().__init__()
        self.device = get_device()
        self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer

    def start(self):
        """Initialize executor by transferring the model to the processor memory."""
        super().start()
        num_parameters = 0
        for param in self.model.parameters():
            num_parameters += param.nelement()
        self.logger.info(f"Training {type(self.model).__name__} model with {num_parameters} parameters using {self.device} for processing")
        self.model.to(self.device)
        self.model.train()

    async def execute_task(self, input_data: bytes) -> bytes:
        """Execute a single gradient computation for the offloaded layers."""
        # Input de-serialization
        split_layer, labels = decode_offload_request(input_data)
        split_layer, labels = Variable(split_layer, requires_grad=True).to(
            self.device
        ), labels.to(self.device)
        split_layer.retain_grad()

        # Local forward pass
        pred = self.model(split_layer)

        if self.task_deployer:
            self.logger.debug("Deploying to the next worker further")

            # Offloaded layers
            input_data = encode_offload_request(pred.to("cpu").detach(), labels.to("cpu"))
            result_data = await self.task_deployer.deploy_task(input_data)

            # Local back propagation
            split_grad, loss = decode_offload_response(result_data)
            split_grad = split_grad.to(self.device)
            self.optimizer.zero_grad()
            pred.backward(split_grad)
            self.optimizer.step()
        else:
            self.logger.debug("Not deploying task any further")
            # Start back propagation
            loss = self.loss_fn(pred, labels)
            self.logger.debug("Computed loss")
            self.optimizer.zero_grad()
            loss.backward()
            self.logger.debug("Updated parameters")
            self.optimizer.step()
            self.logger.debug("Updated optimizer")
            loss = loss.item()

        # Result serialization
        result_data = encode_offload_response(split_layer.grad.to("cpu").detach(), loss)

        self.logger.debug("Executed task")
        return result_data



class GradientCalculatorPruneStep(TaskExecutor):
    def __init__(self, model : Module, loss_fn, optimizer, depruneProps):
        super().__init__()
        self.device = get_device()
        self.loss_fn = loss_fn
        #prune loss functions
        self.model = model
        self.optimizer = optimizer
        self.depruneProps = depruneProps

    def start(self):
        """Initialize executor by transferring the model to the processor memory."""
        super().start()
        num_parameters = 0
        for param in self.model.parameters():
            num_parameters += param.nelement()
        self.logger.info(f"Training {type(self.model).__name__} model with {num_parameters} parameters using {self.device} for processing")
        self.model.to(self.device)
        self.model.train()
        
        #send budget and prune loss function here? or has to be passed each time?
        
    def compress_with_pruneFilter(self, pred, prune_filter, budget, serverFlag = False):
        
        compressedPred = torch.tensor([])
        if serverFlag:
            mask = prune_filter.to('cpu')
        else:
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
        
    def prune_loss_fn(self, loss_fn, pred, y, prune_filter, budget, delta = 0.1, epsilon=1000):
        prune_filter_squeezed = prune_filter.squeeze()
        prune_filter_control = torch.exp( delta * (sum(torch.square(torch.sigmoid(prune_filter_squeezed)))-budget)   )
        entropyLoss = loss_fn(pred,y)
        diff = entropyLoss + epsilon * prune_filter_control
        return diff
        
        

    async def execute_task(self, input_data: bytes) -> bytes:
        """Execute a single gradient computation for the offloaded layers."""


        if self.task_deployer:
            self.logger.debug("Deploying to the next worker further")
            
            # Input de-serialization
            split_layer, labels, prune_filter, budget = decode_offload_request_pruned(input_data)
            split_layer, labels = Variable(split_layer, requires_grad=True).to(
                self.device
            ), labels.to(self.device)
            split_layer.retain_grad()

            # Local forward pass
            model_return = self.model(split_layer, local=True)
        
            pred = model_return[0].to("cpu").detach()   #partial model output
            ############################
            #quantization/compression TBD
            ############################
            prune_filter = model_return[1].to("cpu").detach()   #the prune filter in training

            # Offloaded layers
            upload_data, filter_to_send = self.compress_with_pruneFilter(pred, prune_filter, budget)
            
            input_data = encode_offload_request_pruned(upload_data, labels.to("cpu"), filter_to_send, budget)
            result_data = await self.task_deployer.deploy_task(input_data)

            # Local back propagation
            split_grad, loss = decode_offload_response(result_data)
            split_grad = split_grad.to(self.device)
            
            split_layer = self.decompress_with_pruneFilter(split_layer, prune_filter, budget)
            
            self.optimizer.zero_grad()
            model_return[0].backward(split_grad)
            self.optimizer.step()
        else:
            self.logger.debug("Not deploying task any further")
            
            # Input de-serialization
            split_layer, labels, prune_filter, budget = decode_offload_request_pruned(input_data)
            split_layer, labels  = Variable(split_layer, requires_grad=True).to(self.device), labels.to(self.device)
            if prune_filter == None:
                # self.logger.info("This version of sparse does not support unsplit training")
                split_layer.retain_grad()
                pred = self.model(split_layer)
                loss = self.loss_fn(pred, labels)
                self.logger.debug("Computed loss")
                self.optimizer.zero_grad()
                loss.backward()
                self.logger.debug("Updated parameters")
                self.optimizer.step()
                self.logger.debug("Updated optimizer")
                loss = loss.item()
            else:
                prune_filter = prune_filter.to(self.device)
                split_layer = self.decompress_with_pruneFilter(split_layer, prune_filter, budget)
                split_layer.retain_grad()
                pred = self.model(split_layer)   #partial model output
                loss = self.prune_loss_fn(self.loss_fn, pred, labels, prune_filter, budget, delta = 0.1, epsilon=1000)
                self.logger.debug("Computed loss")
                self.optimizer.zero_grad()
                loss.backward()
                self.logger.debug("Updated parameters")
                self.optimizer.step()
                self.logger.debug("Updated optimizer")
                loss = loss.item()
                split_layer, _ = self.compress_with_pruneFilter(split_layer, prune_filter, budget, serverFlag=True)
                # Result serialization
        result_data = encode_offload_response(split_layer.grad.to("cpu").detach(), loss)

        self.logger.debug("Executed task")
        return result_data