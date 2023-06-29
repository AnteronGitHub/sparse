import torch
from torch.autograd import Variable

from sparse_framework.dl import ModelExecutor

from compression_utils import compress_with_pruneFilter, decompress_with_pruneFilter, prune_loss_fn
from models.compression_utils_vgg import DecodingUnit, EncodingUnit
from serialization import decode_offload_inference_request, \
                          decode_offload_inference_request_pruned, \
                          encode_offload_inference_request, \
                          decode_offload_inference_response, \
                          encode_offload_inference_response, \
                          encode_offload_inference_request_pruned

class InferenceCalculatorPruning(ModelExecutor):
    def __init__(self, model_name, partition, compressionProps):
        super().__init__(model_name, partition)
        self.compressionProps = compressionProps

    def start(self):
        super().start()

        if self.task_deployer:
            self.encoder = EncodingUnit(self.compressionProps, in_channel=128)
        else:
            self.decoder = DecodingUnit(self.compressionProps, out_channel=128)

        self.logger.info(f"Inferring with the model.")

    async def execute_task(self, input_data: bytes) -> bytes:
        """Execute a single forward computation for the offloaded layers."""
        with torch.no_grad():
            split_layer, prune_filter, budget = decode_offload_inference_request_pruned(input_data)
            split_layer = Variable(split_layer, requires_grad=False).to(self.device)

            if prune_filter is None:
                pred = split_layer
            else:
                split_layer = decompress_with_pruneFilter(split_layer, prune_filter, budget, self.device)
                pred = self.decoder(split_layer)

            pred = self.model(pred)

            if self.task_deployer:
                pred, prune_filter = self.encoder(pred)
                payload, mask = compress_with_pruneFilter(pred.to('cpu').detach(), prune_filter.to('cpu').detach(), budget)

                input_data = encode_offload_inference_request_pruned(payload, mask, budget)
                result_data = await self.task_deployer.deploy_task(input_data)
            else:
                result_data = encode_offload_inference_response(pred.to("cpu").detach())

            return result_data
