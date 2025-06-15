import torch
from flamby.datasets.fed_ixi.model import Decoder, Encoder, EncodingBlock
from torch import nn


class PerFclFeatureExtractor(nn.Module):
    """
    Adapted from https://pypi.org/project/unet/0.7.7/ PyTorch implementation of 2D and 3D U-Net (unet 0.7.7)
    License: MIT License (MIT license)
    Author: Fernando Perez-Garcia
    Requires: Python >=3.6.
    """

    def __init__(
        self,
        in_channels: int = 1,
        dimensions: int = 3,
        num_encoding_blocks: int = 3,
        out_channels_first_layer: int = 8,
        normalization: str | None = "batch",
        pooling_type: str = "max",
        upsampling_type: str = "linear",
        preactivation: bool = False,
        residual: bool = False,
        padding: int = 1,
        padding_mode: str = "zeros",
        activation: str | None = "PReLU",
        initial_dilation: int | None = None,
        dropout: float = 0,
    ):
        super().__init__()
        self.CHANNELS_DIMENSION = 1
        depth = num_encoding_blocks - 1

        # Force padding if residual blocks
        if residual:
            padding = 1

        # Encoder
        self.encoder = Encoder(
            in_channels,
            out_channels_first_layer,
            dimensions,
            pooling_type,
            depth,
            normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=initial_dilation,
            dropout=dropout,
        )

        # Bottom (last encoding block)
        in_channels = self.encoder.out_channels
        out_channels_first = 2 * in_channels if dimensions == 2 else in_channels

        self.bottom_block = EncodingBlock(
            in_channels,
            out_channels_first,
            dimensions,
            normalization,
            pooling_type=None,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=self.encoder.dilation,
            dropout=dropout,
        )

        # Decoder
        if dimensions == 2:
            power = depth - 1
        elif dimensions == 3:
            power = depth
        in_channels = self.bottom_block.out_channels
        in_channels_skip_connection = out_channels_first_layer * 2**power
        num_decoding_blocks = depth
        self.decoder = Decoder(
            in_channels_skip_connection,
            dimensions,
            upsampling_type,
            num_decoding_blocks,
            normalization=normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=self.encoder.dilation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections, encoding = self.encoder(x)
        encoding = self.bottom_block(encoding)
        return self.decoder(skip_connections, encoding)
