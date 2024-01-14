import torch
import torch.nn as nn
from typing import Tuple
from monai.networks.nets.unet import UNet


def get_model(
    model_type: str,
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    strides: Tuple[int, ...],
    channels: Tuple[int, ...],
    device: torch.device
) -> nn.Module:
    """Select neural network architecture for given run"""

    if model_type == 'unet':
        model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            channels=channels
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    print("Loaded Neural Network Arch.:", model_type)
    return model
