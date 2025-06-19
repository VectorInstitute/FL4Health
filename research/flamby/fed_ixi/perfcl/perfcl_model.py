import torch
import torch.nn.functional as F
from flamby.datasets.fed_ixi.model import ConvolutionalBlock
from torch import nn

from fl4health.model_bases.parallel_split_models import ParallelFeatureJoinMode, ParallelSplitHeadModule
from fl4health.model_bases.perfcl_base import PerFclModel
from research.flamby.fed_ixi.perfcl.perfcl_feature_extractor import PerFclFeatureExtractor
from research.flamby.utils import shutoff_batch_norm_tracking


class PerFclClassifier(ParallelSplitHeadModule):
    def __init__(
        self, join_mode: ParallelFeatureJoinMode, out_channels_first_layer: int, monte_carlo_dropout: float = 0.0
    ) -> None:
        super().__init__(join_mode)

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
        # For a PerFCL feature extractor, we take the outputs of the global and local stacks and we concatenate them
        # along the channels. So if each has in_channel dimension of 4, the classifier actually processes 8 channels
        self.classifier = ConvolutionalBlock(
            dimensions,
            in_channels=2 * single_stack_in_channels,
            out_channels=out_classes,
            kernel_size=1,
            activation=None,
        )

    def parallel_output_join(self, local_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        # Assuming tensors are "batch first", we concatenate along the channel dimension
        return torch.concat([local_tensor, global_tensor], dim=self.CHANNELS_DIMENSION)

    def head_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.monte_carlo_layer(input_tensor) if self.monte_carlo_layer is not None else input_tensor
        x = self.classifier(x)
        return F.softmax(x, dim=self.CHANNELS_DIMENSION)


class LocalUNetFeatureExtractor(nn.Module):
    """
    Local PerFCL module: We use a UNet with the classifier head stripped off to extract a set of features on which each
    pixel of the image is classified.
    """

    def __init__(self, turn_off_bn_tracking: bool = False, out_channels_first_layer: int = 8):
        super().__init__()
        self.base_model = PerFclFeatureExtractor(out_channels_first_layer=out_channels_first_layer)
        if turn_off_bn_tracking:
            shutoff_batch_norm_tracking(self.base_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


class GlobalUNetFeatureExtractor(nn.Module):
    """
    Global PerFCL module.
    We use a UNet with the classifier head stripped off to extract a set of features on
    which each pixel of the image is classified.
    """

    def __init__(self, turn_off_bn_tracking: bool = False, out_channels_first_layer: int = 8):
        super().__init__()
        self.base_model = PerFclFeatureExtractor(out_channels_first_layer=out_channels_first_layer)
        if turn_off_bn_tracking:
            shutoff_batch_norm_tracking(self.base_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


class FedIxiPerFclModel(PerFclModel):
    def __init__(
        self, turn_off_bn_tracking: bool = False, out_channels_first_layer: int = 8, monte_carlo_dropout: float = 0.0
    ) -> None:
        # FedIXI out_channels_first_layer = 8 is the Baseline model default. So we use it here. The monte carlo dropout
        # is also set to 0 by default for FedIXI
        local_module = LocalUNetFeatureExtractor(turn_off_bn_tracking, out_channels_first_layer)
        global_module = GlobalUNetFeatureExtractor(turn_off_bn_tracking, out_channels_first_layer)
        model_head = PerFclClassifier(
            ParallelFeatureJoinMode.CONCATENATE, out_channels_first_layer, monte_carlo_dropout
        )
        super().__init__(local_module, global_module, model_head)
