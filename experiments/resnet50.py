import torch
from torch import nn


def load_model(n_activations=2):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

    # print(model.blocks[6])

    # override last layer to fit the given prediction task
    model.blocks[5].proj = nn.Linear(in_features=2048, out_features=n_activations, bias=True)
    return model
