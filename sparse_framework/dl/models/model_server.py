import asyncio
import logging
import pickle

from torch.nn import Module

from .model_repository import ModelRepository

def encode_model_request(model_name : str, partition):
    return pickle.dumps({
        'model_name': model_name,
        'partition': partition
    })

def decode_offload_request(data : bytes):
    payload = pickle.loads(data)
    return payload['model_name'], payload['partition']

def encode_model_reply(model : Module, loss_fn, optimizer):
    return pickle.dumps({
        'model': model,
        'loss_fn': loss_fn,
        'optimizer': optimizer
    })

def decode_offload_reply(data : bytes):
    payload = pickle.loads(data)
    return payload['model'], payload['loss_fn'], payload['optimizer']

class ModelServer():
    """TCP server for deploying models with latest trained parameters.
    """
    def __init__(self,
                 model_repository : ModelRepository,
                 listen_address : str = '0.0.0.0',
                 listen_port : int = 50006):
        logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=logging.INFO)
        self.logger = logging.getLogger("sparse")

        self.model_repository = model_repository

        self.listen_address = listen_address
        self.listen_port = listen_port

    async def receive_task(self, reader : asyncio.StreamReader, writer : asyncio.StreamWriter) -> None:
        input_data = await reader.read()

        model_name, partition = decode_offload_request(input_data)

        model, loss_fn, optimizer = self.model_repository.get_model(model_name, partition)

        writer.write(encode_model_reply(model, loss_fn, optimizer))
        await writer.drain()
        writer.close()

    async def serve(self):
        server = await asyncio.start_server(self.receive_task, self.listen_address, self.listen_port)
        self.logger.info(f"Model server listening on {self.listen_address}:{self.listen_port}")
        await server.serve_forever()

    def start(self):
        asyncio.run(self.serve())
