import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

# fix some validation error, while fetching the pretrained models
# https://github.com/pytorch/vision/issues/4156
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


def resnet50(n_activations=2, weights=None):
    """
       Image classification, ResNet-50
    """
    model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2", skip_validation=True)

    # override last layer to fit the given prediction task
    model.fc = nn.Linear(in_features=2048, out_features=n_activations, bias=True)

    if weights is not None:
        print('laded weights', weights)
        model.load_state_dict(torch.load(weights))

    return model


def unet(n_classes=1):
    """
        Image Encoder/Decoder, U-Net
        https://github.com/milesial/Pytorch-UNet#pretrained-model
    """
    model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
    model.outc = nn.Conv2d(64, 1, kernel_size=1)
    print(model.outc)
    # print(dir(model.up4.named_modules))
    return model


def resnet3D(n_activations=2):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

    # print(model.blocks[6])

    # override last layer to fit the given prediction task
    model.blocks[5].proj = nn.Linear(in_features=2048, out_features=n_activations, bias=True)
    return model
