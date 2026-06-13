import torch

from models.unet import UNet

def load_model(model_path):
    model = UNet()
    model.load_state_dict(torch.load(model_path))
    
    return model