import sys

import torch

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for processing")
    return device

def use_legacy_asyncio():
    if sys.version_info >= (3, 8, 10):
        print("Using latest asyncio implementation.")
        return False
    elif sys.version_info >= (3, 6, 9):
        print("Using legacy asyncio implementation.")
        return True
    else:
        print("WARNING: The used Python interpreter is older than what is officially supported. This may cause some functionalities to break")
        return True

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print("Saved PyTorch Model State to {:s}".format(filepath))
