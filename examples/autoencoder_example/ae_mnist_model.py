import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 8, 5)

        self.convT1 = nn.ConvTranspose2d(8, 16, 6, stride = 2 )
        self.convT2 = nn.ConvTranspose2d(16, 1, 6, stride = 2)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        y = self.pool(F.relu(self.conv2(x)))
        # decoder
        y = self.convT1(y)
        y = F.relu(y)
        y = self.convT2(y)
        y = self.sigmoid(y)
        return y
    

    