import torch
import torch.nn as nn
from flamby.datasets.fed_isic2019 import Baseline

from research.flamby.utils import shutoff_group_norm_tracking
from opacus.validators import ModuleValidator


class ModifiedBaseline(nn.Module):
    """FedAdam implements server-side momentum in aggregating the updates from each client. For layers that carry state
    that must remain non-negative, like BatchNormalization layers (present in FedIXI U-Net), they may become negative
    due to momentum carrying updates past the origin. For Batch Normalization this means that the variance state
    estimated during training and applied during evaluation may become negative. This blows up the model. In order
    to get around this issue, we modify all batch normalization layers in the FedIXI U-Net to not carry such state by
    setting track_running_stats to false.

    NOTE: We set the out_channels_first_layer to 12 rather than the default of 8. This roughly doubles the size of the
    baseline model to be used (1106520 DOF). This is to allow for a fair parameter comparison with FENDA and APFL
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = ModuleValidator.fix(Baseline())
        # shutoff_group_norm_tracking(self.model)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

import torch 
import torch.nn as nn 
import torch.nn.functional as F

class FedISICImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 17)
        # relu

        self.conv2 = nn.Conv2d(8, 16, 33)
        # relu 
        self.maxpool1 = nn.MaxPool2d(8, 8)

        self.conv3 = nn.Conv2d(16, 32, 5)
        # relu 

        self.conv4 = nn.Conv2d(32, 64, 6)
        # relu 
        self.maxpool2 = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(64*5*5, 300)
        self.drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(300, 30)
        # drop
        self.fc3 = nn.Linear(30, 8)

    def forward(self, x):
        
        # convolutional stack
        x = F.relu(self.conv1(x))
        x = self.maxpool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(F.relu(self.conv4(x)))
        # fully connected stack
        x = torch.flatten(x, 1)
        x = self.drop(self.fc1(x))
        x = self.drop(self.fc2(x))
        x = self.fc3(x)

        return x