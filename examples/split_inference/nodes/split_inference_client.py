import asyncio
import time

import torch
from torch import nn

from sparse.node.master import Master

from models.yolov3 import YOLOv3_local
from serialization import encode_offload_request, decode_offload_response
from utils import get_device, ImageLoading, non_max_suppression, save_detection

class SplitInferenceClient(Master):
    def __init__(self):
        super().__init__()
        compressionProps = {}
        compressionProps['feature_compression_factor'] = 4
        compressionProps['resolution_compression_factor'] = 1

        self.model = YOLOv3_local(compressionProps)
        self.device = get_device()

    async def infer(self, epochs: int = 5):
        self.logger.info(
            f"Starting inference using {self.device} for local computations"
        )

        # Transfer model to device
        model = self.model.to(self.device)

        with torch.no_grad():
            self.logger.info(f"--------- inferring ----------")

            while True:
                imagePath = "data/samples/dog.jpg"

                # Load image to processor memory
                img = ImageLoading(imagePath, self.model.img_size)
                X = img.to(self.device)

                # Local forward propagation
                split_vals = model(X)

                # Offloaded layers
                input_data = encode_offload_request(split_vals.to("cpu").detach())

                result_data = await self.task_deployer.deploy_task(input_data)

                pred = decode_offload_response(result_data)

                #post process layers
                conf_thres = 0.95
                nms_thres = 0.3
                detection = non_max_suppression(pred, conf_thres, nms_thres)

                img = X
                save_detection(img, imagePath, detection)

        self.logger.info("Done!")

if __name__ == "__main__":
    compressionProps = {}
    compressionProps['feature_compression_factor'] = 4
    compressionProps['resolution_compression_factor'] = 1

    split_inference_client = SplitInferenceClient(YOLOv3_local(compressionProps))
    asyncio.run(split_inference_client.infer())
