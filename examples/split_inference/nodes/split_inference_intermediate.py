from sparse_framework.node.master import Master
from sparse_framework.node.worker import Worker
from sparse_framework.dl.inference_calculator import InferenceCalculator

from models.yolov3 import YOLOv3_local

class SplitInferenceIntermediate(Master, Worker):
    def __init__(self):
        compressionProps = {}
        compressionProps['feature_compression_factor'] = 4
        compressionProps['resolution_compression_factor'] = 1

        Master.__init__(self)
        Worker.__init__(self, task_executor = InferenceCalculator(model=YOLOv3_local(compressionProps)))
