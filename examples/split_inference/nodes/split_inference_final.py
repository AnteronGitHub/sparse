from sparse.node.worker import Worker
from sparse.dl.inference_calculator import InferenceCalculator

from models.yolov3 import YOLOv3_server

class SplitInferenceFinal(Worker):
    def __init__(self):
        compressionProps = {}
        compressionProps['feature_compression_factor'] = 4
        compressionProps['resolution_compression_factor'] = 1

        Worker.__init__(self,
                        task_executor = InferenceCalculator(YOLOv3_server(compressionProps)))
