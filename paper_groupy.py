import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import math
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M
#from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling

nclasses = 37 # GTSRB as 43 classes


def to_bn(x):
    xs = x.size()
    x = x.view(xs[0], xs[1] * xs[2], xs[3], xs[4])
    # x = x.view(xs[0], xs[1], xs[2], x.size()[2], x.size()[3])
    return x


def from_bn(x, channels=8):
    xs = x.size()
    x = x.view(xs[0], xs[1] / channels, channels, x.size()[2], x.size()[3])
    return x

class Net(nn.Module):
    def __init__(self, no_dp=False, p=0.5):
        super(Net, self).__init__()
        self.no_dp = no_dp
        self.p = p
        self.conv1 = P4MConvZ2(in_channels=3, out_channels=8, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = P4MConvP4M(in_channels=8, out_channels=16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = P4MConvP4M(in_channels=16, out_channels=32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = P4MConvP4M(in_channels=32, out_channels=32, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = P4MConvP4M(in_channels=32, out_channels=64, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = P4MConvP4M(in_channels=64, out_channels=64, kernel_size=3)
        self.bn6 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(8192, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, nclasses)

        # Initilize the parameters
        '''for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        '''



    def forward(self, x):
        #print(x.size())
        x = F.relu(self.bn1(to_bn(self.conv1(x))))
        x = F.max_pool2d(x, kernel_size=3, stride=3, padding=1)
        x = from_bn(x)
        #print(x.size())
        x = F.relu(self.bn2(to_bn(self.conv2(x))))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = from_bn(x)
        #print(x.size())
        x = from_bn(F.relu(self.bn3(to_bn(self.conv3(x)))))
        x = from_bn(F.relu(self.bn4(to_bn(self.conv4(x)))))
        x = from_bn(F.relu(self.bn5(self.conv5(x))))

        x = F.relu(self.bn6(to_bn(self.conv6(x))))
        x = F.max_pool2d(x, kernel_size=3, stride=3, padding=1)
        #print(x.size())
        x = x.view(-1, 8192)

        x = F.dropout(F.relu(self.fc1(x)), p=self.p, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), p=self.p, training=self.training)
        x = F.relu(self.fc3(x))
        return x