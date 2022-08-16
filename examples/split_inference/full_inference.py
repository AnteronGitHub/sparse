import asyncio
import time

import torch
from torch import nn

from models.yolov3 import YOLOv3
from utils import get_device, ImageLoading, non_max_suppression, save_detection

if __name__ == "__main__":
    compressionProps = {} ###
    compressionProps['feature_compression_factor'] = 4 ### resolution compression factor, compress by how many times
    compressionProps['resolution_compression_factor'] = 1 ###layer compression factor, reduce by how many times TBD

    model = YOLOv3(compressionProps)
    device = get_device()
    model.to(device)

    with torch.no_grad():
        print(f"--------- inferring ----------")

        while True:
            imagePath = "data/dog.jpg"
            start_time = time.time()

            # Load image to processor memory
            X = ImageLoading(imagePath, self.model.img_size).to(device)
            load_time = time.time() - start_time

            # Local forward propagation
            pred = model(X)
            inference_time = time.time() - start_time - load_time

            #post process layers
            conf_thres = 0.95 #object confidence threshold
            nms_thres = 0.3 #iou thresshold for non-maximum suppression
            detection = non_max_suppression(pred, conf_thres, nms_thres)
            post_processing_time = time.time() - start_time - load_time - inference_time

            save_detection(img, imagePath, detection)
            print(f"Run object detection for image '{imagePath}'. Wall clock time: {time.time() - start_time:.4f}s, loading time: {load_time:.4f}s, local inference time: {local_processing_time:.4f}s, encoding time: {encoding_time:.4f}s, offloaded processing time: {server_processing_time:.4f}s, decoding time: {decoding_time:.4f}s, post processing time: {post_processing_time:.4f}s.")
