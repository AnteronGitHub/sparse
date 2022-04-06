import torch

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print("Saved PyTorch Model State to {:s}".format(filepath))
