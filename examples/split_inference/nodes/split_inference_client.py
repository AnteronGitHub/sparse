import asyncio
import time

import torch
import numpy as np
from torch import nn
from tqdm import tqdm

from sparse_framework.node.master import Master
from sparse_framework.stats.monitor_client import MonitorClient

from datasets.yolov3 import YOLOv3Dataset
from models.yolov3 import YOLOv3_local
from utils import get_device, ImageLoading, non_max_suppression, save_detection
from torch.utils.data import DataLoader
from sparse_framework.dl.serialization import encode_offload_inference_request, encode_offload_inference_request_pruned

class SplitInferenceClient(Master):
    def __init__(self, dataset, model, benchmark = True):
        super().__init__()
        compressionProps = {}
        compressionProps['feature_compression_factor'] = 4
        compressionProps['resolution_compression_factor'] = 1
        self.model = model
        self.dataset = dataset
        self.device = get_device()
        if benchmark:
            self.monitor_client = MonitorClient()
        else:
            self.monitor_client = None

    def compress_with_pruneFilter(self, pred, prune_filter, budget):
        compressedPred = torch.tensor([])
        mask = torch.square(torch.sigmoid(prune_filter.squeeze())).to('cpu')
        masknp = mask.detach().numpy()
        partitioned = np.partition(masknp, -budget)[-budget]
        for entry in range(len(mask)):
            if mask[entry] >= partitioned:
                predRow = pred[:, entry, :, :].unsqueeze(dim=1)
                compressedPred = torch.cat((compressedPred, predRow), 1)

        return compressedPred, mask

    async def infer(self, batch_size, batches, depruneProps,
                    inferences_to_be_run = 100, save_result = False, log_file_prefix = None):
        if self.monitor_client is not None:
            self.monitor_client.start_benchmark(log_file_prefix)

        self.logger.info(
            f"Starting inference using {self.device} for local computations"
        )

        # Transfer model to device
        model = self.model.to(self.device)

        inferences_to_be_run = batch_size * batches
        progress_bar = tqdm(total=inferences_to_be_run,
                            unit='inferences',
                            unit_scale=True)
        pruneState = depruneProps['pruneState']
        budget = depruneProps['budget']
        with torch.no_grad():
            self.logger.info(f"--------- inferring ----------")

            for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                X, y = X.to(self.device), y.to(self.device)

                model_return = self.model(X, local=True)
                pred = model_return[0].to(
                    "cpu").detach()  # partial model output
                # quantization/compression TBD
                prune_filter = model_return[1].to(
                    "cpu").detach()  # the prune filter in training

                pred, mask = self.compress_with_pruneFilter(
                    pred, prune_filter, budget)

                if pruneState:
                    input_data = encode_offload_inference_request_pruned(
                        pred, mask, budget)
                else:
                    input_data = encode_offload_inference_request(X)

                self.logger.debug("Deploying to the next worker further")

                offload_input_data = encode_offload_inference_request_pruned(
                    pred.to("cpu").detach(), mask, budget)
                result_data = await self.task_deployer.deploy_task(offload_input_data)

                if self.monitor_client is not None:
                    self.monitor_client.batch_processed(len(X))
                progress_bar.update(len(X))

                if batch + 1 >= batches:
                    break

        progress_bar.close()
        self.logger.info("Done!")
        if self.monitor_client is not None:
            self.logger.info("Waiting for the benchmark client to finish sending messages")
            await asyncio.sleep(1)

if __name__ == "__main__":
    compressionProps = {}
    compressionProps['feature_compression_factor'] = 4
    compressionProps['resolution_compression_factor'] = 1

    split_inference_client = SplitInferenceClient(YOLOv3_local(compressionProps))
    asyncio.run(split_inference_client.infer())
