import asyncio
from torch.autograd import Variable
from torch.nn import Module
import torch
import numpy as np

from .model_executor import ModelExecutor

from .serialization import decode_offload_request, encode_offload_request, decode_offload_response, encode_offload_response
from .serialization import encode_offload_request_pruned, decode_offload_request_pruned
from .utils import get_device
from .models.model_loader import ModelLoader

class GradientCalculatorPruneStep(ModelExecutor):
    def start(self):
        super().start()
        self.model.train()

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

    def prune_loss_fn(self, pred, y, prune_filter, budget, delta = 0.1, epsilon=1000):
        prune_filter_squeezed = prune_filter.squeeze()
        prune_filter_control_1 = torch.exp( delta * (sum(torch.square(torch.sigmoid(prune_filter_squeezed)))-budget))
        prune_filter_control_2 = torch.exp(-delta * (sum(torch.square(torch.sigmoid(prune_filter_squeezed)))-budget))
        prune_filter_control = prune_filter_control_1 + prune_filter_control_2
        entropyLoss = self.loss_fn(pred,y)
        diff = entropyLoss + epsilon * prune_filter_control
        return diff, entropyLoss

    async def execute_task(self, input_data: bytes) -> bytes:
        """Execute a single gradient computation for the offloaded layers."""
        split_layer, labels, prune_filter, budget = decode_offload_request_pruned(input_data)
        split_layer = Variable(split_layer, requires_grad=True).to(self.device)

        if self.task_deployer is None:
            split_layer = self.decompress_with_pruneFilter(split_layer, prune_filter, budget)

        split_layer.retain_grad()

        if self.task_deployer:
            pred, prune_filter = self.model(split_layer)

            upload_data, filter_to_send = self.compress_with_pruneFilter(pred, prune_filter, budget)
            input_data = encode_offload_request_pruned(upload_data, labels.to(self.device), filter_to_send, budget)
            result_data = await self.task_deployer.deploy_task(input_data)
            split_grad, reported_loss = decode_offload_response(result_data)
            split_grad = self.decompress_with_pruneFilter(split_grad, filter_to_send, budget)

            split_grad = split_grad.to(self.device)
            self.optimizer.zero_grad()
            pred.backward(split_grad)
            self.optimizer.step()
        else:
            pred = self.model(split_layer)

            loss, reported_loss = self.prune_loss_fn(pred, labels.to(self.device), prune_filter, budget)
            reported_loss = reported_loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            split_layer, _ = self.compress_with_pruneFilter(split_layer.grad, prune_filter, budget, serverFlag=True)

        result_data = encode_offload_response(split_layer.to("cpu").detach(), reported_loss)

        return result_data
