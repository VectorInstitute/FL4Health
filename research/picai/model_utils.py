from collections.abc import Sequence

import torch
from monai.networks.nets.unet import UNet
from torch import nn


def get_model(
    device: torch.device | None = None,
    model_type: str = "unet",
    spatial_dims: int = 3,
    in_channels: int = 3,
    out_channels: int = 2,
    channels: Sequence[int] = [32, 64, 128, 256, 512, 1024],
    strides: Sequence[tuple[int, ...]] = [(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)],
) -> nn.Module:
    """Select neural network architecture for given run."""
    if model_type == "unet":
        # ignore typing for strides argument because Sequence[tuple[int, ...]] is valid input type
        # https://docs.monai.io/en/stable/networks.html#unet
        model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,  # type: ignore
            channels=channels,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if device is not None:
        model = model.to(device)
    print("Loaded Neural Network Arch.:", model_type)
    return model
