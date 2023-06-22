import torch
from torch.autograd import Variable

from .model_executor import ModelExecutor

from .serialization import decode_offload_inference_request, \
                           encode_offload_inference_request, \
                           decode_offload_inference_response, \
                           encode_offload_inference_response, \
                           encode_offload_inference_request_pruned
from .serialization import decode_offload_inference_request_pruned


class InferenceCalculatorPruning(ModelExecutor):
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
        split_layer, prune_filter, budget = decode_offload_inference_request_pruned(input_data)
        split_layer = Variable(split_layer, requires_grad=False).to(self.device)

        with torch.no_grad():
            if self.task_deployer:
                pred, prune_filter = self.model(split_layer)

                pred, mask = self.compress_with_pruneFilter(pred, prune_filter, budget)
                offload_input_data = encode_offload_inference_request_pruned(pred.to("cpu").detach(), mask, budget)
                result_data = await self.task_deployer.deploy_task(offload_input_data)
            else:
                prune_filter = prune_filter.to(self.device)
                split_layer = self.decompress_with_pruneFilter(split_layer, prune_filter, budget)

                pred = self.model(split_layer)
                result_data = encode_offload_inference_response(pred.to("cpu").detach())

        return result_data
