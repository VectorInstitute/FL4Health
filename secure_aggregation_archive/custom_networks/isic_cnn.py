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
