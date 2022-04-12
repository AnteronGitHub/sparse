import asyncio
import pickle

import torch
import numpy as np

def encode_offload_request(activation_layer : torch.tensor, labels : torch.tensor):
    return pickle.dumps({
        'activation': activation_layer,
        'labels': labels
    })

def decode_offload_request(request : bytes):
    payload = pickle.loads(request)
    return payload['activation'], payload['labels']

def encode_offload_response(gradient : torch.tensor, loss : float):
    return pickle.dumps({
        'gradient': gradient,
        'loss': loss
    })

def decode_offload_response(data : bytes):
    payload = pickle.loads(data)
    return payload['gradient'], \
           payload['loss']

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
