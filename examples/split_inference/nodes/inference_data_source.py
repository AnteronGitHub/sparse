from tqdm import tqdm

from sparse.dl.serialization import encode_offload_inference_request
from sparse.node.master import Master
from sparse.stats.monitor_client import MonitorClient

from utils import ImageLoading

class InferenceDataSource(Master):
    def __init__(self):
        super().__init__()

    async def start(self, inferences_to_be_run = 100, img_size=416):
        for t in range(inferences_to_be_run):
            imagePath = "data/dog.jpg"
            X = ImageLoading(imagePath, img_size).to('cpu')

            result_data = await self.task_deployer.deploy_task(encode_offload_inference_request(X))
