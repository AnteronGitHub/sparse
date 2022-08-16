import asyncio
import time

import torch
from torch import nn

from sparse.node.master import Master

from models.yolov3 import YOLOv3_local
from serialization import encode_offload_request, decode_offload_response
from utils import get_device, ImageLoading, non_max_suppression, save_detection

class SplitInferenceClient(Master):
    def __init__(self, model):
        super().__init__()
        self.device = get_device()
        self.model = model

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
                img = ImageLoading(imagePath, self.model.img_size)
                X = img.to(self.device)
                load_time = time.time() - start_time

                # Local forward propagation
                split_vals = model(X)
                local_processing_time = time.time() - start_time - load_time

                # Offloaded layers
                input_data = encode_offload_request(split_vals.to("cpu").detach())
                encoding_time = time.time() - start_time - load_time - local_processing_time

                result_data = await self.task_deployer.deploy_task(input_data)
                server_processing_time = time.time() - start_time - load_time - local_processing_time - encoding_time

                pred = decode_offload_response(result_data)
                decoding_time = time.time() - start_time - load_time - local_processing_time - encoding_time - server_processing_time

                #post process layers
                conf_thres = 0.95 #object confidence threshold
                nms_thres = 0.3 #iou thresshold for non-maximum suppression
                detection = non_max_suppression(pred, conf_thres, nms_thres)
                post_processing_time = time.time() - start_time - load_time - local_processing_time - encoding_time - server_processing_time - decoding_time

                img = X
                save_detection(img, imagePath, detection)
                self.logger.info(f"Run object detection for image '{imagePath}'. Wall clock time: {time.time() - start_time:.4f}s, loading time: {load_time:.4f}s, local inference time: {local_processing_time:.4f}s, encoding time: {encoding_time:.4f}s, offloaded processing time: {server_processing_time:.4f}s, decoding time: {decoding_time:.4f}s, post processing time: {post_processing_time:.4f}s.")

        self.logger.info("Done!")


if __name__ == "__main__":
    compressionProps = {} ###
    compressionProps['feature_compression_factor'] = 4 ### resolution compression factor, compress by how many times
    compressionProps['resolution_compression_factor'] = 1 ###layer compression factor, reduce by how many times TBD

    split_inference_client = SplitInferenceClient(YOLOv3_local(compressionProps))
    asyncio.run(split_inference_client.infer())
