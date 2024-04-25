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

        self.fc1 = nn.Linear(64*5*5, 1000)
        self.drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(1000, 100)
        # drop
        self.fc3 = nn.Linear(100, 8)

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

if __name__ == '__main__':
    test_isic_batch = torch.rand(4,3,200, 200)
    unet = FedISICImageClassifier()
    out = unet(test_isic_batch)

    print(out.shape)

#############################################################
class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottom=False):
        super().__init__()
        self.do_pool = not bottom

        # stack
        self.conv1 = nn.Conv3d()
        self.gn1 = nn.GroupNorm()
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d()
        self.gn2 = nn.GroupNorm()
        if self.do_pool:
            self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.relu(self.gn1(self.conv1(x)))
        skip = self.relu(self.gn2(self.conv2(skip)))

        down = None
        if self.do_pool:
            down = self.maxpool(skip)
            return down, skip
        return skip
    
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, top=False):
        super().__init__()
        self.predict = top

        self.upconv1 = nn.ConvTranspose3d()
        self.conv1 = nn.Conv3d()
        self.gn1 = nn.GroupNorm()
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d()
        self.gn2 = nn.GroupNorm()

        if self.predict:
            self.conv3 = nn.Conv3d()

    def forward(self, x):
        up = self.upconv1(x)
        up = self.relu(self.gn1(self.conv1(up)))
        up = self.relu(self.gn2(self.conv2(up)))

        if self.predict:
            up = self.conv3(up)
        return up

class FedIXIUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # DownBlock(db) & UpBlock(ub) numbering are sequential, not by level.
        self.db1 = UNetDownBlock()
        self.db2 = UNetDownBlock()
        self.db3 = UNetDownBlock()
        self.bottleneck = UNetDownBlock()
        self.ub1 = UNetUpBlock()
        self.ub2 = UNetUpBlock()
        self.top = UNetUpBlock()

    def forward(self, x):
        # Skip numbering are by level (top-to-bottom), not sequential.
        down, skip_lv1 = self.db1(x)
        down, skip_lv2 = self.db2(down)
        down, skip_lv3 = self.db3(down)
        up = self.bottleneck(down)
        up = self.ub1(torch.cat(up, skip_lv3))
        up = self.ub2(torch.cat(up, skip_lv2))
        return self.top(torch.cat(up, skip_lv1))
        


#############################################################



class FedIXISegmentationFCN____DRAFT____(nn.Module):
    def __init__(self, x=48, y=60, z=48):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 96, 3)
        self.conv2 = nn.Conv3d(96, 256, 3)
        self.conv3 = nn.Conv3d(256, 384, 3)
        self.conv4 = nn.Conv3d(384, 384, 3)
        self.conv5 = nn.Conv3d(384, 256, 3)
        self.conv6 = nn.Conv3d(256, 4096, 3)
        self.conv7 = nn.Conv3d(4096, 4096, 3)
        self.conv7 = nn.Conv3d(4096, 1000, 3)

    
    def forward(self, x):
        pass
    

class FedISICImageClassifier____DRAFT____(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution_stack = nn.Sequential(
            # substack - one
            nn.Conv2d(3, 8, 17),
            F.relu,
            # substack - two
            nn.Conv2d(8, 16, 33),
            F.relu,
            nn.MaxPool2d(8, 8),
            # substack - three
            nn.Conv2d(16, 32, 5),
            F.relu,
            # substack - four
            nn.Conv2d(32, 64, 17),
            F.relu,
            nn.MaxPool2d(2,2),
        )
        self.fully_connected_stack = nn.Sequential(
            nn.Linear(64*9*9, 1000),
            nn.Dropout(p=0.3),
            nn.Linear(1000, 100),
            nn.Dropout(p=0.3),
            nn.Linear(100, 8),
        )


    def forward(self, x):
        x = self.convolution_stack(x)
        x = torch.flatten(x, 1)
        x = self.fully_connected_stack(x)
        return x
