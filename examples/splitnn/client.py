import asyncio
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from sparse_framework import Master, TaskDeployer
from sparse_framework.dl import get_device, DatasetRepository, ModelLoader

from serialization import encode_offload_request, decode_offload_response, encode_offload_request_pruned, decode_offload_inference_response
from utils import parse_arguments, _get_benchmark_log_file_prefix

class SplitNNClient(Master):
    def __init__(self,
                 application : str,
                 dataset,
                 model_name : str,
                 partition : str,
                 benchmark : bool = True):
        Master.__init__(self, benchmark=benchmark)

        self.device = get_device()

        self.dataset = dataset
        self.model_name = model_name
        self.partition = partition
        self.application = application
        self.warmed_up = False

    def load_model(self):
        model_loader = ModelLoader(self.config_manager.model_server_address,
                                   self.config_manager.model_server_port)

        self.model, self.loss_fn, self.optimizer = model_loader.load_model(self.model_name,
                                                                           self.partition)
        self.model = self.model.to(self.device)
        self.logger.info(f"Downloaded model '{self.model_name}' partition '{self.partition}'.")
        if self.is_learning():
            self.logger.info(f"Training the model.")
            self.model.train()
        else:
            self.logger.info(f"Inferring with the model.")

    def is_learning(self):
        return self.application == 'learning'

    async def start(self, batch_size, batches, log_file_prefix, epochs, verbose = False):
        if verbose:
            progress_bar = tqdm(total=batch_size*batches*total_epochs,
                                unit='samples',
                                unit_scale=True)
        else:
            progress_bar = None

        for t in range(epochs):
            offset = 0 if t == 0 else 1 # Ensures that an extra batch is processed in the first epoch since one batch is for warming up
            for batch, (X, y) in enumerate(DataLoader(self.dataset, batch_size)):
                X = X.to(self.device)

                pred = self.model(X)
                input_data = encode_offload_request(pred, y)


                # Offloaded layers
                result_data = await self.task_deployer.deploy_task(input_data)

                if self.is_learning():
                    split_grad, loss = decode_offload_response(result_data)
                    split_grad = split_grad.to(self.device)
                    # Back Propagation
                    self.optimizer.zero_grad()
                    pred.backward(split_grad)
                    self.optimizer.step()
                else:
                    prediction = decode_offload_inference_response(result_data)
                    loss = None

                # Logging
                if progress_bar is not None:
                    progress_bar.update(len(X))
                else:
                    self.logger.info(f"Processed batch of {len(X)} samples. Loss: {loss}")

                # Benchmarks
                if self.monitor_client is not None:
                    if self.warmed_up:
                        self.monitor_client.batch_processed(len(X), loss)
                    else:
                        self.warmed_up = True
                        self.monitor_client.start_benchmark(log_file_prefix)

                if batch + offset >= batches:
                    break

        if progress_bar is not None:
            progress_bar.close()
        if self.monitor_client is not None:
            self.monitor_client.stop_benchmark()
            self.logger.info("Waiting for the benchmark client to finish sending messages")
            await asyncio.sleep(1)

if __name__ == '__main__':
    args = parse_arguments()

    partition = "client"
    dataset, classes = DatasetRepository().get_dataset(args.dataset)
    log_file_prefix = _get_benchmark_log_file_prefix(args, "datasource")

    splitnn_client = SplitNNClient(application=args.application,
                                   dataset=dataset,
                                   model_name=args.model_name,
                                   partition=partition)
    splitnn_client.load_model()
    asyncio.run(splitnn_client.start(args.batch_size,
                                     args.batches,
                                     log_file_prefix=log_file_prefix,
                                     epochs=int(args.epochs)))
