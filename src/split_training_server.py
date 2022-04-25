import asyncio

import torch
from torch import nn

from networking import decode_offload_request, encode_offload_response
from models.neural_network import NeuralNetwork_server
from roles.worker import Worker
from training import compute_gradient
from utils import get_device

class SplitTrainingServer(Worker):
    def __init__(self):
        self.device = get_device()
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = NeuralNetwork_server().to('cpu')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.host = '127.0.0.1'
        self.port = 50007

    async def handle_offload_request(self, reader : asyncio.StreamReader, writer : asyncio.StreamWriter) -> None:
        """Callback function used for serving gradient computation requests.

        """
        offload_request = await reader.read()

        activation_layer, labels = decode_offload_request(offload_request)
        gradient, loss = compute_gradient(self.model,
                                          self.loss_fn,
                                          self.optimizer,
                                          self.device,
                                          activation_layer,
                                          labels)

        writer.write(encode_offload_response(gradient.detach(), loss))
        await writer.drain()
        writer.close()
        print("Served gradient offloading request")

    async def serve(self):
        server = await asyncio.start_server(self.handle_offload_request, self.host, self.port)
        addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
        print(f'Serving on {addrs}')
        async with server:
            self.model.train()
            await server.serve_forever()

    def start(self):
        asyncio.run(self.serve())

if __name__ == "__main__":
    SplitTrainingServer().start()
