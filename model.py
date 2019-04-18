import torch.nn as nn
import torch.nn.functional as F
import torch


class LeNet(nn.Module):
    def __init__(self, se1=False, se2=False):
        super(LeNet, self).__init__()
        self.se1 = se1
        self.se2 = se2
        # Input channel, Output Channel, Kernel size, Stride
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv1_se = SqueezeExcitationModule(self.conv1, 20)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv2_se = SqueezeExcitationModule(self.conv2, 50)
        self.fc1 = nn.Linear(50*5*5, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # X --> 3x32x32 (CIFAR10)
        if self.se1:
            x = self.conv1_se(x)
        else:
            x = self.conv1(x)
        F.relu(x, inplace=True)
        # X --> 20x28x28
        x = F.max_pool2d(x, 2, 2)
        # X --> 20x14x14
        if self.se2:
            x = self.conv2_se(x)
        else:
            x = self.conv2(x)
        F.relu(x, inplace=True)
        # X --> 50x10x10
        x = F.max_pool2d(x, 2, 2)
        # X --> 50x5x5
        # Batch, Image Dimension (HxWxC)
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SqueezeExcitationModule(nn.Module):

    def __init__(self, block, C, r=2):
        super().__init__()
        self.block = block
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(C, C//r, bias=False)
        self.fc2 = nn.Linear(C//r, C, bias=False)

    def forward(self, x):
        # x --> B x C x H x W
        output_block = self.block(x)
        # SE BLOCK
        output_se = self.global_pooling(output_block).squeeze()
        output_se = self.fc1(output_se)
        F.relu(output_se, inplace=True)
        output_se = self.fc2(output_se)
        # B x C
        output_se = F.sigmoid(output_se)

        return torch.mul(output_block, output_se.unsqueeze(-1).unsqueeze(-1))
