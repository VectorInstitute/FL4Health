import torch
import torch.nn.functional as F
from torch import nn

from fl4health.model_bases.parallel_split_models import ParallelFeatureJoinMode, ParallelSplitHeadModule


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.independent_layer = nn.Linear(4, 4, bias=False)
        self.shared_layer = nn.Linear(3, 3, bias=False)


class SingleLayerWithSeed(nn.Module):
    def __init__(self, seed: int = 42, output_size: int = 2) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.linear = nn.Linear(100, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ToyConvNet(nn.Module):
    def __init__(self, include_bn: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5, bias=False)
        self.fc1 = nn.Linear(16 * 4 * 4, 120, bias=False)
        self.fc2 = nn.Linear(120, 64, bias=False)
        if include_bn:
            self.bn1 = nn.BatchNorm1d(10)


class ToyConvNet2(nn.Module):
    def __init__(self, include_bn: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 5, bias=False)
        self.fc1 = nn.Linear(32 * 4 * 4, 120, bias=False)
        self.fc2 = nn.Linear(240, 64, bias=False)
        if include_bn:
            self.bn1 = nn.BatchNorm1d(10)


class SmallCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        return F.relu(self.fc1(x))


class HierarchicalCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.h1_layer1 = nn.ModuleDict(
            {
                "h2_layer1": nn.ModuleDict(
                    {
                        "conv": nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1),
                        "pool": nn.MaxPool2d(kernel_size=2, stride=2),
                    }
                ),
                "h2_layer2": nn.ModuleDict(
                    {
                        "conv": nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1),
                        "pool": nn.MaxPool2d(kernel_size=2, stride=2),
                    }
                ),
            }
        )

        self.h1_layer2 = nn.ModuleDict(
            {
                "h2_layer1": nn.ModuleDict(
                    {
                        "conv": nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
                        "pool": nn.MaxPool2d(kernel_size=2, stride=2),
                    }
                ),
                "h2_layer2": nn.ModuleDict(
                    {
                        "conv": nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
                        "pool": nn.MaxPool2d(kernel_size=2, stride=2),
                    }
                ),
            }
        )

        self.classifier = nn.ModuleDict(
            {"fc": nn.Linear(1 * 4 * 4, 10), "relu": nn.ReLU()}  # Assuming input image size is 64x64
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.h1_layer1["h2_layer1"]["conv"](x)
        x = self.h1_layer1["h2_layer1"]["pool"](x)
        x = self.h1_layer1["h2_layer2"]["conv"](x)
        x = self.h1_layer1["h2_layer2"]["pool"](x)

        x = self.h1_layer2["h2_layer1"]["conv"](x)
        x = self.h1_layer2["h2_layer1"]["pool"](x)
        x = self.h1_layer2["h2_layer2"]["conv"](x)
        x = self.h1_layer2["h2_layer2"]["pool"](x)

        x = torch.flatten(x, 1)
        x = self.classifier["fc"](x)

        return self.classifier["relu"](x)


class FeatureCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(x)
        return x.flatten(start_dim=1)


class HeadCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16 * 4 * 4, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(x)


class FendaHeadCnn(ParallelSplitHeadModule):
    def __init__(self, join_mode: ParallelFeatureJoinMode = ParallelFeatureJoinMode.CONCATENATE) -> None:
        super().__init__(join_mode)
        self.fc1 = nn.Linear(16 * 4 * 4 * 2, 32)

    def parallel_output_join(self, local_x: torch.Tensor, global_x: torch.Tensor) -> torch.Tensor:
        # Assuming tensors are "batch first" so join column-wise
        return torch.concat([local_x, global_x], dim=1)

    def head_forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc1(x))


class LinearTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LinearPredictionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class UNet3D(nn.Module):
    """
    https://pypi.org/project/unet/0.7.7/.

    PyTorch implementation of 2D and 3D U-Net (unet 0.7.7)
    License: MIT License (MIT license)
    Author: Fernando Perez-Garcia
    Requires: Python >=3.6
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_classes: int = 2,
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
        monte_carlo_dropout: float = 0,
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

        # Monte Carlo dropout
        self.monte_carlo_layer = None
        if monte_carlo_dropout:
            dropout_class = getattr(nn, "Dropout{}d".format(dimensions))
            self.monte_carlo_layer = dropout_class(p=monte_carlo_dropout)

        # Classifier
        if dimensions == 2:
            in_channels = out_channels_first_layer
        elif dimensions == 3:
            in_channels = 2 * out_channels_first_layer
        self.classifier = ConvolutionalBlock(dimensions, in_channels, out_classes, kernel_size=1, activation=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections, encoding = self.encoder(x)
        encoding = self.bottom_block(encoding)
        x = self.decoder(skip_connections, encoding)
        if self.monte_carlo_layer is not None:
            x = self.monte_carlo_layer(x)
        x = self.classifier(x)
        return F.softmax(x, dim=self.CHANNELS_DIMENSION)


# Conv


class ConvolutionalBlock(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        normalization: str | None = None,
        kernel_size: int = 3,
        activation: str | None = "ReLU",
        preactivation: bool | None = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        dilation: int | None = None,
        dropout: float = 0,
    ):
        super().__init__()

        block = nn.ModuleList()

        dilation = 1 if dilation is None else dilation
        if padding:
            total_padding = kernel_size + 2 * (dilation - 1) - 1
            padding = total_padding // 2

        class_name = "Conv{}d".format(dimensions)
        conv_class = getattr(nn, class_name)
        no_bias = not preactivation and (normalization is not None)
        conv_layer = conv_class(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            bias=not no_bias,
        )

        norm_layer = None
        if normalization is not None:
            class_name = "{}Norm{}d".format(normalization.capitalize(), dimensions)
            norm_class = getattr(nn, class_name)
            num_features = in_channels if preactivation else out_channels
            norm_layer = norm_class(num_features)

        activation_layer = None
        if activation is not None:
            activation_layer = getattr(nn, activation)()

        if preactivation:
            self.add_if_not_none(block, norm_layer)
            self.add_if_not_none(block, activation_layer)
            self.add_if_not_none(block, conv_layer)
        else:
            self.add_if_not_none(block, conv_layer)
            self.add_if_not_none(block, norm_layer)
            self.add_if_not_none(block, activation_layer)

        dropout_layer = None
        if dropout:
            class_name = "Dropout{}d".format(dimensions)
            dropout_class = getattr(nn, class_name)
            dropout_layer = dropout_class(p=dropout)
            self.add_if_not_none(block, dropout_layer)

        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.dropout_layer = dropout_layer

        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    @staticmethod
    def add_if_not_none(module_list: nn.ModuleList, module: nn.Module | None) -> None:
        if module is not None:
            module_list.append(module)


# Decoding

CHANNELS_DIMENSION = 1
UPSAMPLING_MODES = ("nearest", "linear", "bilinear", "bicubic", "trilinear")


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels_skip_connection: int,
        dimensions: int,
        upsampling_type: str,
        num_decoding_blocks: int,
        normalization: str | None,
        preactivation: bool = False,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        activation: str | None = "ReLU",
        initial_dilation: int | None = None,
        dropout: float = 0,
    ):
        super().__init__()
        upsampling_type = fix_upsampling_type(upsampling_type, dimensions)
        self.decoding_blocks = nn.ModuleList()
        self.dilation = initial_dilation
        for _ in range(num_decoding_blocks):
            decoding_block = DecodingBlock(
                in_channels_skip_connection,
                dimensions,
                upsampling_type,
                normalization=normalization,
                preactivation=preactivation,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.dilation,
                dropout=dropout,
            )
            self.decoding_blocks.append(decoding_block)
            in_channels_skip_connection //= 2
            if self.dilation is not None:
                self.dilation //= 2

    def forward(self, skip_connections: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        zipped = zip(reversed(skip_connections), self.decoding_blocks)
        for skip_connection, decoding_block in zipped:
            x = decoding_block(skip_connection, x)
        return x


class DecodingBlock(nn.Module):
    def __init__(
        self,
        in_channels_skip_connection: int,
        dimensions: int,
        upsampling_type: str,
        normalization: str | None,
        preactivation: bool = True,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        activation: str | None = "ReLU",
        dilation: int | None = None,
        dropout: float = 0,
    ):
        super().__init__()

        self.residual = residual

        if upsampling_type == "conv":
            in_channels = out_channels = 2 * in_channels_skip_connection
            self.upsample = get_conv_transpose_layer(dimensions, in_channels, out_channels)
        else:
            self.upsample = get_upsampling_layer(upsampling_type)
        in_channels_first = in_channels_skip_connection * (1 + 2)
        out_channels = in_channels_skip_connection
        self.conv1 = ConvolutionalBlock(
            dimensions,
            in_channels_first,
            out_channels,
            normalization=normalization,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )
        in_channels_second = out_channels
        self.conv2 = ConvolutionalBlock(
            dimensions,
            in_channels_second,
            out_channels,
            normalization=normalization,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )

        if residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions,
                in_channels_first,
                out_channels,
                kernel_size=1,
                normalization=None,
                activation=None,
            )

    def forward(self, skip_connection: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        skip_connection = self.center_crop(skip_connection, x)
        x = torch.cat((skip_connection, x), dim=CHANNELS_DIMENSION)
        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x += connection
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        return x

    def center_crop(self, skip_connection: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        skip_shape = torch.tensor(skip_connection.shape)
        x_shape = torch.tensor(x.shape)
        crop = skip_shape[2:] - x_shape[2:]
        half_crop = crop // 2
        # If skip_connection is 10, 20, 30 and x is (6, 14, 12)
        # Then pad will be (-2, -2, -3, -3, -9, -9)
        pad = -torch.stack((half_crop, half_crop)).t().flatten()
        return F.pad(skip_connection, pad.tolist())


def get_upsampling_layer(upsampling_type: str) -> nn.Upsample:
    if upsampling_type not in UPSAMPLING_MODES:
        message = 'Upsampling type is "{}" but should be one of the following: {}'
        message = message.format(upsampling_type, UPSAMPLING_MODES)
        raise ValueError(message)
    return nn.Upsample(scale_factor=2, mode=upsampling_type, align_corners=False)


def get_conv_transpose_layer(dimensions: int, in_channels: int, out_channels: int) -> nn.Module:
    class_name = "ConvTranspose{}d".format(dimensions)
    conv_class = getattr(nn, class_name)
    return conv_class(in_channels, out_channels, kernel_size=2, stride=2)


def fix_upsampling_type(upsampling_type: str, dimensions: int) -> str:
    if upsampling_type == "linear":
        if dimensions == 2:
            upsampling_type = "bilinear"
        elif dimensions == 3:
            upsampling_type = "trilinear"
    return upsampling_type


# Encoding


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_first: int,
        dimensions: int,
        pooling_type: str,
        num_encoding_blocks: int,
        normalization: str | None,
        preactivation: bool = False,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        activation: str | None = "ReLU",
        initial_dilation: int | None = None,
        dropout: float = 0,
    ):
        super().__init__()

        self.encoding_blocks = nn.ModuleList()
        self.dilation = initial_dilation
        is_first_block = True
        for _ in range(num_encoding_blocks):
            encoding_block = EncodingBlock(
                in_channels,
                out_channels_first,
                dimensions,
                normalization,
                pooling_type,
                preactivation,
                is_first_block=is_first_block,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.dilation,
                dropout=dropout,
            )
            is_first_block = False
            self.encoding_blocks.append(encoding_block)
            if dimensions == 2:
                in_channels = out_channels_first
                out_channels_first = in_channels * 2
            elif dimensions == 3:
                in_channels = 2 * out_channels_first
                out_channels_first = in_channels
            if self.dilation is not None:
                self.dilation *= 2

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        skip_connections: list[torch.Tensor] = []
        for encoding_block in self.encoding_blocks:
            x, skip_connection = encoding_block(x)
            skip_connections.append(skip_connection)
        return skip_connections, x

    @property
    def out_channels(self) -> int:
        return self.encoding_blocks[-1].out_channels  # type: ignore


class EncodingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_first: int,
        dimensions: int,
        normalization: str | None,
        pooling_type: str | None,
        preactivation: bool | None = False,
        is_first_block: bool = False,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        activation: str | None = "ReLU",
        dilation: int | None = None,
        dropout: float = 0,
    ):
        super().__init__()

        self.preactivation: bool | None = preactivation
        self.normalization: str | None = normalization

        self.residual = residual

        if is_first_block:
            normalization = None
            preactivation = None
        else:
            normalization = self.normalization
            preactivation = self.preactivation

        self.conv1 = ConvolutionalBlock(
            dimensions,
            in_channels,
            out_channels_first,
            normalization=normalization,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )

        if dimensions == 2:
            out_channels_second = out_channels_first
        elif dimensions == 3:
            out_channels_second = 2 * out_channels_first
        self.conv2 = ConvolutionalBlock(
            dimensions,
            out_channels_first,
            out_channels_second,
            normalization=self.normalization,
            preactivation=self.preactivation,
            padding=padding,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )

        if residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions,
                in_channels,
                out_channels_second,
                kernel_size=1,
                normalization=None,
                activation=None,
            )

        self.downsample = None
        if pooling_type is not None:
            self.downsample = get_downsampling_layer(dimensions, pooling_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x += connection
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        if self.downsample is None:
            return x
        skip_connection = x
        x = self.downsample(x)
        return x, skip_connection

    @property
    def out_channels(self) -> int:
        return self.conv2.conv_layer.out_channels


def get_downsampling_layer(dimensions: int, pooling_type: str, kernel_size: int = 2) -> nn.Module:
    class_name = "{}Pool{}d".format(pooling_type.capitalize(), dimensions)
    class_ = getattr(nn, class_name)
    return class_(kernel_size)


# Autoencoder: encoder and decoder units
class VariationalEncoder(nn.Module):
    def __init__(self, embedding_size: int = 2, condition_vector_size: int | None = None) -> None:
        super().__init__()
        if condition_vector_size is not None:
            self.fc_mu = nn.Linear(100 + condition_vector_size, embedding_size)
            self.fc_logvar = nn.Linear(100 + condition_vector_size, embedding_size)
        else:
            self.fc_mu = nn.Linear(100, embedding_size)
            self.fc_logvar = nn.Linear(100, embedding_size)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if condition is not None:
            return self.fc_mu(torch.cat((x, condition), dim=-1)), self.fc_logvar(torch.cat((x, condition), dim=-1))
        return self.fc_mu(x), self.fc_logvar(x)


class VariationalDecoder(nn.Module):
    def __init__(self, embedding_size: int = 2, condition_vector_size: int | None = None) -> None:
        super().__init__()
        if condition_vector_size is not None:
            self.linear = nn.Linear(embedding_size + condition_vector_size, 100)
        else:
            self.linear = nn.Linear(embedding_size, 100)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if condition is not None:
            return self.linear(torch.cat((x, condition), dim=-1))
        return self.linear(x)


class ConstantConvNet(nn.Module):
    def __init__(self, constants: list[float]) -> None:
        assert len(constants) == 4
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.fc1 = nn.Linear(16 * 4 * 4, 16, bias=False)
        self.fc2 = nn.Linear(16, 4, bias=False)

        nn.init.constant_(self.conv1.weight, val=constants[0])
        nn.init.constant_(self.conv2.weight, val=constants[1])

        nn.init.constant_(self.fc1.weight, val=constants[2])
        nn.init.constant_(self.fc2.weight, val=constants[3])


class MnistNetWithBnAndFrozen(nn.Module):
    def __init__(self, freeze_cnn_layer: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.bn = nn.BatchNorm2d(num_features=16)
        self.fc1 = nn.Linear(16 * 4 * 4, 10)

        if freeze_cnn_layer:
            layer_to_freeze = self._modules["conv1"]
            assert layer_to_freeze is not None
            layer_to_freeze.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn(x)
        x = x.view(-1, 16 * 4 * 4)
        return F.relu(self.fc1(x))


class CompositeConvNet(nn.Module):
    def __init__(self) -> None:
        super(CompositeConvNet, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3)
        self.bn1d = nn.BatchNorm1d(16)

        self.conv2d = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.bn2d = nn.BatchNorm2d(16)

        self.conv3d = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3)
        self.bn3d = nn.BatchNorm3d(16)

        self.conv_transpose1d = nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=3)
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3)
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=3)

        self.linear = nn.Linear(1880, 10)

        # LayerNorm applied to 2D shape (batch_size, features)
        self.layer_norm = nn.LayerNorm(10)

    def forward(self, x1d: torch.Tensor, x2d: torch.Tensor, x3d: torch.Tensor) -> torch.Tensor:
        x1d = F.relu(self.conv1d(x1d))
        x1d = self.bn1d(x1d)
        x1d = F.relu(self.conv_transpose1d(x1d))

        x2d = F.relu(self.conv2d(x2d))
        x2d = self.bn2d(x2d)
        x2d = F.relu(self.conv_transpose2d(x2d))

        x3d = F.relu(self.conv3d(x3d))
        x3d = self.bn3d(x3d)
        x3d = F.relu(self.conv_transpose3d(x3d))

        x_flat = torch.cat([x1d.flatten(1), x2d.flatten(1), x3d.flatten(1)], dim=1)
        x_flat = self.linear(x_flat)

        return self.layer_norm(x_flat)


class ModelWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.model = module

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return self.model.forward(*args)
