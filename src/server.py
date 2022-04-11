# Echo server program
import asyncio

import socket
from io import BytesIO

import torch
from torch import nn
from torch.autograd import Variable

from networking import decode_offload_request, encode_offload_response
from models.neural_network import NeuralNetwork_server
from utils import get_device

# Model training initialization
loss_fn = nn.CrossEntropyLoss()
model = NeuralNetwork_server().to('cpu')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
device = get_device()

# Server initialization
HOST = '127.0.0.1'
PORT = 50007

async def handle_offload_request(reader, writer):
    offload_request = await reader.read()

    activation_layer, labels = decode_offload_request(offload_request)
    gradient, loss = compute_gradient(activation_layer, labels)

    writer.write(encode_offload_response(gradient.detach(), loss))
    await writer.drain()
    writer.close()
    print("Served gradient offloading request")

def compute_gradient(split_activation, y):
    split_activation = Variable(split_activation, requires_grad=True).to('cpu')
    pred = model(split_activation)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    gradient = split_activation.grad.to(device)
    optimizer.step()

    return gradient, loss.item()

async def main():
    server = await asyncio.start_server(handle_offload_request, HOST, PORT)
    addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
    print(f'Serving on {addrs}')
    async with server:
        model.train()
        await server.serve_forever()

asyncio.run(main())
