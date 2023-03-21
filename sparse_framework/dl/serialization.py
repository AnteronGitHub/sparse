import asyncio
import pickle

import torch

def encode_offload_request(activation_layer : torch.tensor, labels : torch.tensor):
    return pickle.dumps({
        'activation': activation_layer,
        'labels': labels
    })
    
def encode_offload_request_pruned(activation_layer_compressed : torch.tensor, labels : torch.tensor, prune_filter: torch.tensor):
    return pickle.dumps({
        'activation': activation_layer_compressed,
        'labels': labels,
        'prune_filter': prune_filter
    })

def decode_offload_request(request : bytes):
    payload = pickle.loads(request)
    return payload['activation'], payload['labels']

def decode_offload_request_pruned(request : bytes):
    payload = pickle.loads(request)
    return payload['activation'], payload['labels'], payload['prune_filters']

def encode_offload_response(gradient : torch.tensor, loss : float):
    return pickle.dumps({
        'gradient': gradient,
        'loss': loss
    })

def decode_offload_response(data : bytes):
    payload = pickle.loads(data)
    return payload['gradient'], \
           payload['loss']

def encode_offload_inference_request(activation_layer : torch.tensor):
    return pickle.dumps({
        'activation': activation_layer
    })

def decode_offload_inference_request(request : bytes):
    payload = pickle.loads(request)
    return payload['activation']

def encode_offload_inference_response(prediction : torch.tensor):
    return pickle.dumps({
        'prediction': prediction
    })

def decode_offload_inference_response(data : bytes):
    payload = pickle.loads(data)
    return payload['prediction']
