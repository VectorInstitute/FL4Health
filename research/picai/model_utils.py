import torch
import torch.nn as nn
from typing import Sequence, Tuple
from monai.networks.nets.unet import UNet


def get_model(
    device: torch.device,
    model_type: str = "unet",
    spatial_dims: int = 3,
    in_channels: int = 3,
    out_channels: int = 2,
    channels: Sequence[int] = [32, 64, 128, 256, 512, 1024],
    strides: Sequence[Tuple[int, ...]] = [(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)],
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