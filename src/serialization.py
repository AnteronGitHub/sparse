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
