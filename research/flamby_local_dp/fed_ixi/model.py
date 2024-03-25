import torch
import torch.nn as nn
from flamby.datasets.fed_ixi import Baseline

# from research.flamby.utils import shutoff_group_norm_tracking, shutoff_batch_norm_tracking
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
        self.model = Baseline()
        # while len(ModuleValidator.validate(self.model, strict=False)) > 0:
        #     self.model = ModuleValidator.fix(self.model)
        # for _, module in self.model.named_modules():
        #     print(module)
        # https://discuss.pytorch.org/t/does-group-norm-maintain-an-running-average-of-mean-and-variance/43959
        # shutoff_batch_norm_tracking(self.model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


import torch 
import torch.nn as nn 
import torch.nn.functional as F


# Fed-IXI batch tensor shape [batch_size, 1 channel, 48, 60, 48]

class UNetForwardBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super().__init__()
        kernel_size=3
        self.conv1 = nn.Conv3d(in_channels, intermediate_channels, kernel_size=kernel_size, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=2, num_channels=intermediate_channels)
        self.conv2 = nn.Conv3d(intermediate_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=2, num_channels=out_channels)
        
    
    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        return x

class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super().__init__()
        self.forward_block = UNetForwardBlock(in_channels, intermediate_channels, out_channels)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        residual = self.forward_block(x)
        return self.maxpool(residual), residual
    
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super().__init__()
        self.forward_block = UNetForwardBlock(in_channels, intermediate_channels, out_channels)
        self.upconv = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.gn = nn.GroupNorm(num_groups=2, num_channels=out_channels)
    
    def forward(self, x):
        x = self.forward_block(x)
        x = self.upconv(x)
        return self.gn(x)
    
class FedIXIUNet(nn.Module):
    # NOTE Numbering convention: by levels from top-to-bottom.

    def __init__(self):
        super().__init__()


        size = 16
        self.db1  = UNetDownBlock(1, size, 2*size)
        self.db2 = UNetDownBlock(2*size, 2*size, 4*size)
        self.bottleneck = UNetUpBlock(4*size, 4*size, 8*size)
        self.ub2 = UNetUpBlock(4*size + 8*size, 4*size, 4*size)
        # self.ub1 = UNetUpBlock(64+128, 64, 64)
        self.final_conv = UNetForwardBlock(2*size+4*size, 2*size, 2*size)
        self.gn = nn.GroupNorm(num_groups=2, num_channels=2*size)
        self.classifier_conv = nn.Conv3d(2*size, 2, kernel_size=1)


        # self.db1  = UNetDownBlock(1, 32, 64)
        # self.db2 = UNetDownBlock(64, 64, 128)
        # self.bottleneck = UNetUpBlock(128, 128, 256)
        # self.ub2 = UNetUpBlock(128+256, 128, 128)
        # # self.ub1 = UNetUpBlock(64+128, 64, 64)
        # self.final_conv = UNetForwardBlock(64+128, 64, 64)
        # self.gn = nn.GroupNorm(num_groups=2, num_channels=64)
        # self.classifier_conv = nn.Conv3d(64, 2, kernel_size=1)
        
    
    def forward(self, x):
        # down
        x, residual_1 = self.db1(x)
        x, residual_2 = self.db2(x)

        # bottom 
        x = self.bottleneck(x)


        x = self.ub2(torch.cat((x, residual_2), 1))
        x = self.final_conv(torch.cat((x, residual_1), 1))
        x = self.gn(x)
        x = self.classifier_conv(x)
        x = F.softmax(x, dim=1)

        return x

def debug(x):
    print(x.shape)
    exit()

if __name__ == '__main__':
    # test_ixi_batch = torch.rand(1,1,48,60,48)
    # unet = FedIXIUNet()

    # out = unet(test_ixi_batch)

    # print(out.shape)
    from research.flamby.utils import summarize_model_info
    summarize_model_info(FedIXIUNet())

    
# if __name__ == '__main__':
#     initial_model = ModifiedBaseline()
#     err = ModuleValidator.validate(initial_model, strict=False)

#     print('error list')
#     print(err)