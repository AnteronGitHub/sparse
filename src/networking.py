import asyncio
import json

import torch
import numpy as np

def encode_offload_request(activation_layer : torch.tensor, labels : torch.tensor):
    return json.dumps({
        'activation': activation_layer.numpy().tolist(),
        'labels': labels.numpy().tolist()
    }).encode()

def decode_offload_request(request : bytes):
    deserialized = json.loads(request.decode())
    return torch.from_numpy(np.asarray(deserialized['activation'], dtype=np.float32)), \
           torch.from_numpy(np.asarray(deserialized['labels'], dtype=np.long))

def encode_offload_response(gradient : torch.tensor, loss : float):
    return json.dumps({
        'gradient': gradient.numpy().tolist(),
        'loss': loss
    }).encode()

def decode_offload_response(response : bytes):
    deserialized = json.loads(response.decode())
    return torch.from_numpy(np.asarray(deserialized['gradient'], dtype=np.float32)), \
           deserialized['loss']

async def offload_training(np_split_vals : torch.tensor, np_labels : torch.tensor,
                           server_host : str, server_port : int):
    reader, writer = await asyncio.open_connection(server_host, server_port)

    writer.write(encode_offload_request(np_split_vals, np_labels))
    writer.write_eof()
    await writer.drain()

    response = await reader.read()
    writer.close()

    return decode_offload_response(response)

def run_offload_training(np_split_vals : torch.tensor, np_labels : torch.tensor,
                         server_host : str = '127.0.0.1', server_port : int = 50007):
    return asyncio.run(offload_training(np_split_vals,
                                        np_labels,
                                        server_host,
                                        server_port))
