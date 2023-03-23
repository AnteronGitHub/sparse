import asyncio
from tqdm import tqdm
from torch.utils.data import DataLoader

from sparse_framework.dl.serialization import encode_offload_inference_request, encode_offload_inference_request_pruned
from sparse_framework.node.master import Master
from sparse_framework.stats.monitor_client import MonitorClient

from datasets.yolov3 import YOLOv3Dataset

class InferenceDataSourceYOLO(Master):
    def __init__(self, benchmark = True):
        super().__init__()
        self.dataset = YOLOv3Dataset()
        if benchmark:
            self.monitor_client = MonitorClient()
        else:
            self.monitor_client = None

    async def start(self, inferences_to_be_run = 100, img_size=416):
        progress_bar = tqdm(total=inferences_to_be_run,
                            unit='inferences',
                            unit_scale=True)
        if self.monitor_client is not None:
            self.monitor_client.start_benchmark()
        for t in range(inferences_to_be_run):
            X = self.dataset.get_sample(img_size).to('cpu')

            result_data = await self.task_deployer.deploy_task(encode_offload_inference_request(X))
            if self.monitor_client is not None:
                self.monitor_client.batch_processed(len(X))
            progress_bar.update(1)

        progress_bar.close()
        if self.monitor_client is not None:
            self.logger.info("Waiting for the benchmark client to finish sending messages")
            await asyncio.sleep(1)


class InferenceDataSource(Master):
    def __init__(self, dataset, model_name, benchmark = True):
        super().__init__()
        self.dataset = dataset
        #self.classes = classes
        self.model_name = model_name
        if benchmark:
            self.monitor_client = MonitorClient()
        else:
            self.monitor_client = None

    async def start(self, batch_size, batches, depruneProps):
        
        inferences_to_be_run = batch_size * batches
        progress_bar = tqdm(total=inferences_to_be_run,
                            unit='inferences',
                            unit_scale=True)
        if self.monitor_client is not None:
            self.monitor_client.start_benchmark()
            
        pruneState = depruneProps['pruneState'] 
        budget = depruneProps['budget'] 
        for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):

            if pruneState:
                input_data = encode_offload_inference_request_pruned(X, None, budget)
            else:
                input_data = encode_offload_inference_request(X)

            #result_data = await self.task_deployer.deploy_task(encode_offload_inference_request(X))
            if self.monitor_client is not None:
                self.monitor_client.batch_processed(len(X))
            progress_bar.update(1)
            
            if batch + 1 >= batches:
                break

        progress_bar.close()
        if self.monitor_client is not None:
            self.logger.info("Waiting for the benchmark client to finish sending messages")
            await asyncio.sleep(1)