import asyncio

import torch
from torch import nn

from networking import decode_offload_request, encode_offload_response
from models.neural_network import NeuralNetwork_server
from training import compute_gradient
from utils import get_device

async def handle_offload_request(reader : asyncio.StreamReader, writer : asyncio.StreamWriter) -> None:
    """Callback function used for serving gradient computation requests.

    """
    offload_request = await reader.read()

    activation_layer, labels = decode_offload_request(offload_request)
    gradient, loss = compute_gradient(model, loss_fn, optimizer, device, activation_layer, labels)

    writer.write(encode_offload_response(gradient.detach(), loss))
    await writer.drain()
    writer.close()
    print("Served gradient offloading request")

async def main():
    server = await asyncio.start_server(handle_offload_request, HOST, PORT)
    addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
    print(f'Serving on {addrs}')
    async with server:
        model.train()
        await server.serve_forever()

if __name__ == "__main__":
    # Model training initialization
    loss_fn = nn.CrossEntropyLoss()
    model = NeuralNetwork_server().to('cpu')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    device = get_device()

    # Server initialization
    HOST = '127.0.0.1'
    PORT = 50007

    # Start server
    asyncio.run(main())
