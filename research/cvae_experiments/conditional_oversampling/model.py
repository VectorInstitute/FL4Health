import numpy as np
import torch
import torch.nn as nn

# This Conditional Variational Auto-Encoder structure is backed by U-net encoder and decoder structures.
# Skip connections are removed since we don't start from a noisy image for image generation,
# rather CVAE works by sampling from a standard normal distribution


def EncoderUnit(in_filters: int, out_filters: int, kernel_size: int = 3) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_filters, out_filters, kernel_size=kernel_size, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


def EncoderFinalLayer(in_filters: int, out_filters: int, kernel_size: int = 3) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_filters, out_filters, kernel_size=kernel_size, padding=1),
        nn.ReLU(),
    )


def DecoderUnit(in_filters: int, out_filters: int, kernel_size: int = 3) -> nn.Module:
    return nn.Sequential(
        # Unlike the original U-net structure we don't have concatenation,
        # therefor ConvTranspose can keep the original number of filters.
        nn.ConvTranspose2d(in_filters, in_filters, kernel_size=2, stride=2),
        nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_filters, out_filters, kernel_size=kernel_size, padding=1),
        nn.ReLU(),
    )


def DecoderFinalLayer(in_filters: int, out_filters: int, kernel_size: int = 3) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_filters, out_filters, kernel_size=kernel_size, padding=1),
        nn.ReLU(),
    )


def UnetConditionalAutoencoder() -> torch.Tensor:
    x = torch.Tensor(np.random.rand(1, 3, 32, 32))
    x = EncoderUnit(3, 32, 3)(x)  # Output shape : 32*16*16
    x = EncoderUnit(32, 64, 3)(x)  # Output shape : 64*8*8
    x = EncoderUnit(64, 128, 3)(x)  # Output shape : 128*4*4
    x = EncoderFinalLayer(128, 256, 3)(x)  # Output shape : 256*4*4
    # Flatten
    x = x.view(x.size(0), -1)
    x = nn.Linear(4096, 256 * 4 * 4)(x)
    x = x.view(-1, 256, 4, 4)
    x = DecoderUnit(256, 128)(x)  # Output shape: 128*8*8
    x = DecoderUnit(128, 64)(x)  # Output shape: 64*16*16
    x = DecoderUnit(64, 32)(x)  # Output shape: 32*32*32
    x = DecoderFinalLayer(32, 3)(x)  # Output shape: 3*32*32
    print(x.shape)

    return x


if __name__ == "__main__":
    UnetConditionalAutoencoder()
