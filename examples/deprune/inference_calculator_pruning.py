import torch
from torch.autograd import Variable

from sparse_framework.dl import ModelExecutor

from compression_utils import compress_with_pruneFilter, decompress_with_pruneFilter, prune_loss_fn
from serialization import decode_offload_inference_request, \
                          decode_offload_inference_request_pruned, \
                          encode_offload_inference_request, \
                          decode_offload_inference_response, \
                          encode_offload_inference_response, \
                          encode_offload_inference_request_pruned

class InferenceCalculatorPruning(ModelExecutor):
    async def execute_task(self, input_data: bytes) -> bytes:
        """Execute a single forward computation for the offloaded layers."""
        split_layer, prune_filter, budget = decode_offload_inference_request_pruned(input_data)
        split_layer = Variable(split_layer, requires_grad=False).to(self.device)

        with torch.no_grad():
            if self.task_deployer:
                pred, prune_filter = self.model(split_layer)

                pred, mask = compress_with_pruneFilter(pred, prune_filter, budget)
                offload_input_data = encode_offload_inference_request_pruned(pred.to("cpu").detach(), mask, budget)
                result_data = await self.task_deployer.deploy_task(offload_input_data)
            else:
                prune_filter = prune_filter.to(self.device)
                split_layer = decompress_with_pruneFilter(split_layer, prune_filter, budget, self.device)

                pred = self.model(split_layer)
                result_data = encode_offload_inference_response(pred.to("cpu").detach())

        return result_data
