from tqdm import tqdm

from sparse.dl.serialization import encode_offload_inference_request
from sparse.node.master import Master
from sparse.stats.monitor_client import MonitorClient

class InferenceDataSource(Master):
    def __init__(self):
        super().__init__()

    async def infer(self, inferences_to_be_run = 100):
        for t in range(inferences_to_be_run):
            imagePath = "data/dog.jpg"
            X = ImageLoading(imagePath, self.model.img_size).to('cpu')

            result_data = await self.task_deployer.deploy_task(encode_offload_inference_request(X))
