import torch
import torch.nn.functional as F
from flamby.datasets.fed_ixi.model import ConvolutionalBlock
from torch import nn

from fl4health.model_bases.moon_base import MoonModel
from research.flamby.fed_ixi.moon.moon_feature_extractor import MoonFeatureExtractor
from research.flamby.utils import shutoff_batch_norm_tracking


class HeadClassifier(nn.Module):
    def __init__(self, out_channels_first_layer: int, monte_carlo_dropout: float = 0.0) -> None:
        super().__init__()

        # We're doing 3D segmentation, so hardcode
        dimensions = 3
        # Binary segmentation so out_classes = 2
        out_classes = 2

        # Which dimension are the channels for the tensor inputs.
        self.CHANNELS_DIMENSION = 1

        # Monte Carlo dropout
        self.monte_carlo_layer = None
        if monte_carlo_dropout:
            dropout_class = getattr(nn, "Dropout{}d".format(dimensions))
            self.monte_carlo_layer = dropout_class(p=monte_carlo_dropout)

        # Classifier
        # Standard UNet concatenates the channels from the first conv layer (residual connection) and the upsampled
        # embeddings from the full-forward process of the U-Net.
        single_stack_in_channels = 2 * out_channels_first_layer

        self.classifier = ConvolutionalBlock(
            dimensions,
            in_channels=single_stack_in_channels,
            out_channels=out_classes,
            kernel_size=1,
            activation=None,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.monte_carlo_layer(input_tensor) if self.monte_carlo_layer is not None else input_tensor
        x = self.classifier(x)
        return F.softmax(x, dim=self.CHANNELS_DIMENSION)


class BaseUNetFeatureExtractor(nn.Module):
    """
    Base module: We use a UNet with the classifier head stripped off to extract a set of features on which each
    pixel of the image is classified.
    """

    def __init__(self, turn_off_bn_tracking: bool = False, out_channels_first_layer: int = 8):
        super().__init__()
        self.base_model = MoonFeatureExtractor(out_channels_first_layer=out_channels_first_layer)
        if turn_off_bn_tracking:
            shutoff_batch_norm_tracking(self.base_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


class FedIxiMoonModel(MoonModel):
    def __init__(
        self, turn_off_bn_tracking: bool = False, out_channels_first_layer: int = 12, monte_carlo_dropout: float = 0.0
    ) -> None:
        # FedIXI out_channels_first_layer = 8 is the Baseline model default. So we use it here. The monte carlo dropout
        # is also set to 0 by default for FedIXI

        base_module = BaseUNetFeatureExtractor(turn_off_bn_tracking, out_channels_first_layer)
        head_module = HeadClassifier(out_channels_first_layer, monte_carlo_dropout)
        super().__init__(base_module, head_module)
