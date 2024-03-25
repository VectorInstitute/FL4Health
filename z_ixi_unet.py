import torch 
import torch.nn as nn 
import torch.nn.functional as F


# Fed-IXI batch tensor shape [batch_size, 1 channel, 48, 60, 48]

class UNetForwardBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, intermediate_channels, kernel_size=3)
        self.gn1 = nn.GroupNorm(num_groups=2, num_channels=intermediate_channels)
        self.conv2 = nn.Conv3d(intermediate_channels, out_channels, kernel_size=3)
        self.gn2 = nn.GroupNorm(num_groups=2, num_channels=out_channels)
    
    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        return x

class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super().__init__()
        self.forward_block = UNetForwardBlock(in_channels, intermediate_channels, out_channels)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=1, ceil_mode=True)
    
    def forward(self, x):
        return self.maxpool(self.forward_block(x))
    
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super().__init__()
        self.forward_block = UNetForwardBlock(in_channels, intermediate_channels, out_channels)
        self.upconv = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3)
        self.gn = nn.GroupNorm(num_groups=2, num_channels=out_channels)
    
    def forward(self, x):
        return self.gn(self.upconv(self.forward_block(x)))
    
class FedIXIUNet(nn.Module):
    # NOTE Numbering convention: by levels from top-to-bottom.

    def __init__(self):
        super().__init__()

        self.db1 = UNetDownBlock(1, 32, 64)
        self.db2 = UNetDownBlock(64, 64, 128)
        self.db3 = UNetDownBlock(128, 128, 256)
        self.bottleneck = UNetUpBlock(256, 256, 512)
        self.ub3 = UNetUpBlock(256+512, 256, 256)
        self.ub2 = UNetUpBlock(128+256, 128, 128)
        self.ub1 = UNetUpBlock(64+128, 64, 64)
        self.conv = nn.Conv3d(64, 2, kernel_size=2)
        self.gn = nn.GroupNorm(num_groups=2, num_channels=2)
    
    def forward(self, x):
        x = F.pad(x, tuple(8 for _ in range(6)),'constant',0)

        # down
        x = self.db1(x)
        skip1 = x.clone()
        x = self.db2(x)
        skip2 = x.clone()
        # x = self.db3(x)
        # skip3 = x.clone()

        # bottom 
        x = self.bottleneck(x)

        # up 
        # x = self.ub3(torch.cat((x, skip3), 1))


        x = self.ub2(torch.cat((x, skip2), 1))
        x = self.ub1(torch.cat((x, skip1), 1))
        x = self.gn(self.conv(x))

        return x

def debug(x):
    print(x.shape)
    exit()

if __name__ == '__main__':
    test_ixi_batch = torch.rand(4,1,48,60,48)
    unet = FedIXIUNet()
    out = unet(test_ixi_batch)

    print(out.shape)