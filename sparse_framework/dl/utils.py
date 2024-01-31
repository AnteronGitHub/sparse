import torch

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def count_model_parameters(model):
    num_parameters = 0
    for param in model.parameters():
        num_parameters += param.nelement()
    return num_parameters

