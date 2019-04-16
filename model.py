import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Input channel, Output Channel, Kernel size, Stride
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(50*5*5, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # X --> 3x32x32 (CIFAR10)
        x = F.relu(self.conv1(x))
        # X --> 20x28x28
        x = F.max_pool2d(x, 2, 2)
        # X --> 20x14x14
        x = F.relu(self.conv2(x))
        # X --> 50x10x10
        x = F.max_pool2d(x, 2, 2)
        # X --> 50x5x5
        # Batch, Image Dimension (HxWxC)
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SqueezeExcitationModule(nn.Module):

    def __init__(self, base_model, C, r):
        super().__init__()
        self.base_model = base_model
        self.global_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(C, C//r, bias=False)
        self.fc2 = nn.Linear(C//r, C, bias=False)

    def forward(self, x):

        output = self.base_model(x)
        output = self.global_pooling(output)
        output = self.fc1(output)
        F.relu(output, inplace=True)
        output = self.fc2(output)
        output = F.sigmoid(output)

        return output
