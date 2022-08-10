import asyncio
import pickle

import torch

def encode_offload_request(activation_layer : torch.tensor):
    return pickle.dumps({
        'activation': activation_layer
    })

def decode_offload_request(request : bytes):
    payload = pickle.loads(request)
    return payload['activation']

def encode_offload_response(prediction : torch.tensor):
    return pickle.dumps({
        'prediction': gradient
    })

def decode_offload_response(data : bytes):
    payload = pickle.loads(data)
    return payload['prediction']
