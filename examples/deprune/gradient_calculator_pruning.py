from torch.autograd import Variable
import torch

from sparse_framework.dl.utils import get_device
from sparse_framework.dl import ModelExecutor

from compression_utils import compress_with_pruneFilter, decompress_with_pruneFilter, prune_loss_fn
from serialization import decode_offload_request, \
                          decode_offload_request_pruned, \
                          encode_offload_request, \
                          encode_offload_request_pruned, \
                          decode_offload_response, \
                          encode_offload_response
from models.compression_utils_vgg import DecodingUnit, EncodingUnit

class GradientCalculatorPruneStep(ModelExecutor):
    def __init__(self, model_name, partition, compressionProps):
        super().__init__(model_name, partition)
        self.compressionProps = compressionProps

    def start(self):
        super().start()

        if self.task_deployer:
            self.encoder = EncodingUnit(self.compressionProps, in_channel=128).to(self.device)
        else:
            self.decoder = DecodingUnit(self.compressionProps, out_channel=128).to(self.device)

        self.logger.info(f"Training the model.")
        self.model.train()

    async def execute_task(self, input_data: bytes) -> bytes:
        """Execute a single gradient computation for the offloaded layers."""
        split_layer, labels, prune_filter, budget = decode_offload_request_pruned(input_data)
        split_layer = Variable(split_layer, requires_grad=True).to(self.device)

        if prune_filter is None:
            pred = split_layer
        else:
            split_layer = decompress_with_pruneFilter(split_layer, prune_filter, budget, self.device)
            pred = self.decoder(split_layer)

        split_layer.retain_grad()
        pred = self.model(pred)

        if self.task_deployer:
            pred, prune_filter = self.encoder(pred)
            upload_data, filter_to_send = compress_with_pruneFilter(pred, prune_filter, budget)

            input_data = encode_offload_request_pruned(upload_data, labels.to(self.device), filter_to_send, budget)
            result_data = await self.task_deployer.deploy_task(input_data)
            split_grad, reported_loss = decode_offload_response(result_data)
            split_grad = decompress_with_pruneFilter(split_grad, filter_to_send, budget, self.device)

            split_grad = split_grad.to(self.device)
            self.optimizer.zero_grad()
            pred.backward(split_grad)
            self.optimizer.step()
        else:
            loss, reported_loss = prune_loss_fn(self.loss_fn, pred, labels.to(self.device), prune_filter, budget)
            reported_loss = reported_loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            split_layer, _ = compress_with_pruneFilter(split_layer.grad, prune_filter, budget, serverFlag=True)

        result_data = encode_offload_response(split_layer.to("cpu").detach(), reported_loss)

        return result_data
