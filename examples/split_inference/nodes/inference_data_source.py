from tqdm import tqdm

from sparse.dl.serialization import encode_offload_inference_request
from sparse.node.master import Master
from sparse.stats.monitor_client import MonitorClient

from datasets.yolov3 import YOLOv3Dataset

class InferenceDataSource(Master):
    def __init__(self):
        super().__init__()
        self.dataset = YOLOv3Dataset()

    async def start(self, inferences_to_be_run = 100, img_size=416):
        progress_bar = tqdm(total=inferences_to_be_run,
                            unit='inferences',
                            unit_scale=True)
        for t in range(inferences_to_be_run):
            X = self.dataset.get_sample(img_size).to('cpu')

            result_data = await self.task_deployer.deploy_task(encode_offload_inference_request(X))
            progress_bar.update(1)

        progress_bar.close()
