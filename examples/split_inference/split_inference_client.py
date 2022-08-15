import asyncio
import time

import torch
from torch import nn

from sparse.node.master import Master

from models import NeuralNetwork_local
from serialization import encode_offload_request, decode_offload_response
from utils import get_device, ImageLoading, non_max_suppression, save_detection

class SplitInferenceClient(Master):
    def __init__(self, img_size, config_path_local, compressionProps, weight_local):
        super().__init__()
        self.device = get_device()
        self.model = NeuralNetwork_local(config_path_local, compressionProps)
        self.model.load_state_dict(torch.load(weight_local))
        self.img_size = img_size


    async def infer(self, epochs: int = 5):
        self.logger.info(
            f"Starting inference using {self.device} for local computations"
        )

        # Transfer model to device
        model = self.model.to(self.device)

        with torch.no_grad():
            self.logger.info(f"--------- inferring ----------")

            while True:
                imagePath = "data/dog.jpg"
                start_time = time.time()

                # Load image to processor memory
                img = ImageLoading(imagePath, self.img_size)
                X = img.to(self.device)

                # Local forward propagation
                split_vals = model(X)

                # Offloaded layers
                input_data = encode_offload_request(split_vals.to("cpu").detach())
                result_data = await self.task_deployer.deploy_task(input_data)
                pred = decode_offload_response(result_data)

                #post process layers
                conf_thres = 0.95 #object confidence threshold
                nms_thres = 0.3 #iou thresshold for non-maximum suppression
                detection = non_max_suppression(pred, conf_thres, nms_thres)

                img = X
                save_detection(img, imagePath, detection)
                self.logger.info(f"Run object detection for image '{imagePath}' in {time.time() - start_time} seconds.")

        self.logger.info("Done!")


if __name__ == "__main__":
    config_path_local = "config/yolov3_local.cfg"
    compressionProps = {} ###
    compressionProps['feature_compression_factor'] = 4 ### resolution compression factor, compress by how many times
    compressionProps['resolution_compression_factor'] = 1 ###layer compression factor, reduce by how many times TBD

    weight_local = "weights/yolov3_local.paths"
    img_size = 416
    split_inference_client = SplitInferenceClient(img_size, config_path_local, compressionProps, weight_local)
    asyncio.run(split_inference_client.infer())
