import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import math

nclasses = 37 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self, no_dp=False, p=0.5):
        super(Net, self).__init__()
        self.no_dp = no_dp
        self.p = p
        self.conv1 = nn.Conv2d(3, 48, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 192, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(192)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(192)
        self.conv5 = nn.Conv2d(192, 384, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(384)
        self.conv6 = nn.Conv2d(384, 384, kernel_size=3)
        self.bn6 = nn.BatchNorm2d(384)

        self.fc1 = nn.Linear(6144, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, nclasses)

        # Initilize the parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        #print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=3, padding=1)
        #print(x.size())
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        #print(x.size())
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=3, padding=1)
        #print(x.size())
        x = x.view(-1, 6144)

        x = F.dropout(F.relu(self.fc1(x)), p=self.p, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), p=self.p, training=self.training)
        x = F.relu(self.fc3(x))
        return x
