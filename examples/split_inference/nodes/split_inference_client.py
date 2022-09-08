import asyncio
import time

import torch
from torch import nn
from tqdm import tqdm

from sparse.node.master import Master
from sparse.dl.serialization import encode_offload_inference_request, \
                                    decode_offload_inference_response
from sparse.stats.monitor_client import MonitorClient

from datasets.yolov3 import YOLOv3Dataset
from models.yolov3 import YOLOv3_local
from utils import get_device, ImageLoading, non_max_suppression, save_detection

class SplitInferenceClient(Master):
    def __init__(self, benchmark = True):
        super().__init__()
        compressionProps = {}
        compressionProps['feature_compression_factor'] = 4
        compressionProps['resolution_compression_factor'] = 1

        self.model = YOLOv3_local(compressionProps)
        self.dataset = YOLOv3Dataset()
        self.device = get_device()
        if benchmark:
            self.monitor_client = MonitorClient()
        else:
            self.monitor_client = None

    async def infer(self, inferences_to_be_run = 100, save_result = False):
        if self.monitor_client is not None:
            self.monitor_client.start_benchmark()

        self.logger.info(
            f"Starting inference using {self.device} for local computations"
        )

        # Transfer model to device
        model = self.model.to(self.device)

        progress_bar = tqdm(total=inferences_to_be_run,
                            unit='inferences',
                            unit_scale=True)
        with torch.no_grad():
            self.logger.info(f"--------- inferring ----------")

            for t in range(inferences_to_be_run):
                X = self.dataset.get_sample(self.model.img_size).to(self.device).to(self.device)

                # Local forward propagation
                split_vals = model(X)

                # Offloaded layers
                input_data = encode_offload_inference_request(split_vals.to("cpu").detach())

                result_data = await self.task_deployer.deploy_task(input_data)

                pred = decode_offload_inference_response(result_data)

                if save_result:
                    #post process layers
                    conf_thres = 0.95
                    nms_thres = 0.3
                    detection = non_max_suppression(pred, conf_thres, nms_thres)
                    save_detection(X, imagePath, detection)

                if self.monitor_client is not None:
                    self.monitor_client.task_processed()
                progress_bar.update(1)

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
